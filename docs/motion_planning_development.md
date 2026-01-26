# Motion Planning Development Documentation

## PAPA Project - UR5 Pick and Place Automation

**Author:** Lilla (Control & Task Pipeline)
**Date:** January 2026
**Status:** In Development

---

## 1. Overview

This document describes the development process of the motion planning module for the PAPA (Pick and Place Automation) project. The goal is to implement autonomous pick-and-place operations using a UR5 manipulator arm with a soft gripper in Gazebo simulation.

The motion planner is responsible for:
- Computing inverse kinematics (IK) to convert Cartesian positions to joint angles
- Generating smooth trajectories between configurations
- Executing pick and place sequences with proper gripper control

---

## 2. Starting Point: MATLAB Inverse Kinematics

### 2.1 Source Code

The foundation for the IK implementation was the `ur5Inverse.m` MATLAB script provided by Prof. Luigi Palopoli. This implements the **full analytical inverse kinematics** for the UR5 robot, which produces up to 8 possible joint configurations for any given end-effector pose.

**Location:** `matlabScripts/ur5Inverse.m`

### 2.2 UR5 DH Parameters

The UR5 robot uses standard Denavit-Hartenberg (DH) parameters:

```python
# From motion_planner.py lines 42-44
self.A = np.array([0, -0.425, -0.3922, 0, 0, 0])        # Link lengths (m)
self.D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])  # Link offsets (m)
self.Alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])  # Twist angles (rad)
```

### 2.3 Algorithm Overview

The analytical IK solves for joints in this order:
1. **q1 (base rotation)**: From wrist center position projected onto XY plane
2. **q5 (wrist 2)**: From the geometry of the wrist
3. **q6 (wrist 3)**: From the orientation constraint
4. **q3 (elbow)**: Using the arm triangle geometry
5. **q2 (shoulder)**: From the arm configuration
6. **q4 (wrist 1)**: From the remaining orientation constraint

Each step typically has 2 solutions (e.g., elbow up/down), leading to 2^3 = 8 total configurations.

---

## 3. Porting to Python

### 3.1 Implementation

The MATLAB code was ported to Python in `motion_planner.py`. The main IK function is:

```python
def ur5_inverse(self, p60, R60):
    """
    Full analytical inverse kinematics for UR5.

    Args:
        p60: End-effector position [x, y, z] in robot base (DH) frame
        R60: End-effector rotation matrix (3x3)

    Returns:
        6x8 matrix of joint angles (8 possible solutions)
    """
```

**Location:** `motion_planner.py` lines 93-256

### 3.2 Bug Found in MATLAB Code

During verification, a **copy-paste bug** was discovered in the original MATLAB code:

> **Issue:** In the solution matrix assembly, columns 5 and 7 were identical. The indexing for `th5` and `th6` values was incorrect.

**Python fix:** The solution matrix in `motion_planner.py` (lines 247-254) uses correct indexing:

```python
Th = np.array([
    [th1_1,     th1_1,     th1_1,     th1_1,     th1_2,     th1_2,     th1_2,     th1_2],
    [th2_1_1_1, th2_1_1_2, th2_1_2_1, th2_1_2_2, th2_2_1_1, th2_2_1_2, th2_2_2_1, th2_2_2_2],
    [th3_1_1_1, th3_1_1_2, th3_1_2_1, th3_1_2_2, th3_2_1_1, th3_2_1_2, th3_2_2_1, th3_2_2_2],
    [th4_1_1_1, th4_1_1_2, th4_1_2_1, th4_1_2_2, th4_2_1_1, th4_2_1_2, th4_2_2_1, th4_2_2_2],
    [th5_1_1,   th5_1_1,   th5_1_2,   th5_1_2,   th5_2_1,   th5_2_1,   th5_2_2,   th5_2_2],
    [th6_1_1,   th6_1_1,   th6_1_2,   th6_1_2,   th6_2_1,   th6_2_1,   th6_2_2,   th6_2_2]
])
```

---

## 4. Coordinate System Challenges

### 4.1 The Inverted Robot Problem

The UR5 in our simulation is mounted **inverted** (hanging from a ceiling structure):

- **Robot base position:** World frame (0.5, 0.35, 1.75)
- **DH frame +Z:** Points DOWN in world frame
- **Table surface:** At world Z = 0.85

This caused significant confusion when converting between:
- World frame coordinates
- Robot-relative coordinates (from perception)
- DH frame coordinates (for IK)

### 4.2 The A2 Transform

After extensive testing, the correct coordinate transform was identified as **A2**:

```python
# motion_planner.py, simple_ik() function, lines 695-698
# A2 Transform: Robot is mounted INVERTED (hanging from above)
# DH frame +Z points DOWN in world frame
# So we negate Z: p_dh = (x, y, -z)
p_robot = np.array([x_rel, y_rel, -z_rel])
```

**Why this works:**
- Perception returns positions relative to robot base
- The robot's DH frame has +Z pointing down
- Negating Z converts from "world up" to "DH down" convention

### 4.3 Testing the Transform

Multiple test scripts were created to verify the coordinate transform:
- `coord_test.py` - Tests different transform hypotheses
- `fk_diagnostic.py` - Forward kinematics verification
- `ik_verify.py` - IK roundtrip tests

---

## 5. IK Solution Selection

### 5.1 The Problem

The IK returns 8 solutions, but we need to choose the "best" one. Criteria:
1. Must be within joint limits
2. Should be close to current/home configuration (smooth motion)
3. Should avoid joint limit boundaries
4. Should prevent the arm from reaching "the wrong way"

### 5.2 Solution Normalization

IK solutions may have angles that differ by 360 degrees. Before comparing solutions, we normalize them to be close to a reference configuration:

```python
def _normalize_to_reference(self, solution, reference):
    """
    Normalize solution angles to be close to reference configuration.
    Adds/subtracts 2*pi to minimize distance from reference.
    """
    # motion_planner.py lines 302-340
```

**Example:** If home q1 = -18 degrees and IK returns q1 = 297 degrees, normalization converts it to -63 degrees (297 - 360 = -63).

### 5.3 Q1 Constraint

To prevent the arm from swinging around to reach from the "wrong side," a constraint limits q1 deviation:

```python
# motion_planner.py line 369
max_q1_deviation = np.pi / 3  # 60 degrees from home
```

**Issue discovered:** This constraint was too restrictive for some brick positions, causing valid solutions to be rejected. This is still being tuned.

### 5.4 Joint Limit Avoidance Cost

A cost function penalizes configurations close to joint limits:

```python
def _joint_limit_cost(self, joints):
    """
    w(q) = (1/2n) * sum_i ((q_i - q_middle_i) / (q_max_i - q_min_i))^2
    """
    # motion_planner.py lines 258-282
```

### 5.5 Combined Selection

The `select_best_solution()` function (lines 342-411) combines all criteria:

```python
combined_score = ref_distance + 2.0 * limit_cost
```

---

## 6. Trajectory Generation

### 6.1 Quintic Polynomial Interpolation

For smooth motion, we use quintic (5th order) polynomial interpolation:

```python
def _quintic_interpolation(self, t, duration):
    """
    s(t) = 10*tau^3 - 15*tau^4 + 6*tau^5
    where tau = t/duration
    """
    # motion_planner.py lines 425-441
```

**Properties:**
- s(0) = 0, s(T) = 1
- s'(0) = 0, s'(T) = 0 (zero velocity at endpoints)
- s''(0) = 0, s''(T) = 0 (zero acceleration at endpoints)

This creates smooth S-curve motion profiles.

### 6.2 Move to Joints

The `move_to_joints()` function (lines 443-492) interpolates between configurations:

```python
def move_to_joints(self, target_joints, controller, duration=3.0):
    for i in range(steps + 1):
        alpha = self._quintic_interpolation(t, duration)
        desired_joints = current_joints + alpha * (target_joints - current_joints)
        controller.send_joint_command(desired_joints)
```

---

## 7. Pick and Place Sequences

### 7.1 Pick Object Sequence

The `pick_object()` function (lines 490-585) executes:

1. Move to safe waypoint (30cm above object)
2. Move to approach position (15cm above object)
3. Open gripper
4. Move down to grasp position
5. Close gripper
6. Lift to safe height

```python
def pick_object(self, object_pos, controller, robot_relative=True):
    # ... waypoint calculations ...

    # Table plane constraint to prevent collision
    min_z_robot_relative = -0.89  # 1cm above table surface
```

### 7.2 Place Object Sequence

The `place_object()` function (lines 587-663) executes the reverse:

1. Move to safe waypoint above target
2. Move to place approach position
3. Move down to place position
4. Open gripper
5. Lift up

---

## 8. Errors Encountered and Solutions

### 8.1 Error: Robot Not Moving

**Symptom:** Commands sent but robot stays stationary in Gazebo.

**Root Cause 1 - Joint State Feedback:**
```python
# controller.py get_current_joint_state() always returns zeros
# because the joint state subscriber was never created
```

**Solution:** Track last commanded position instead of reading joint state:
```python
# motion_planner.py
self.last_commanded_joints = np.array(config.home_joint_config)

def move_to_joints(self, ...):
    # Use tracked position instead of get_current_joint_state()
    current_joints = np.concatenate([self.last_commanded_joints, [0.0, 0.0]])
```

**Root Cause 2 - PD Gains Not Set:**
The `ros_impedance_controller` requires PD gains to be configured before it responds to position commands.

**Solution:** Add PD gain setup in controller.py main():
```python
from base_controllers.utils.pidManager import PidManager
p.pid = PidManager(p.joint_names)
p.pid.setPDjoints(
    base_conf.robot_params[p.robot_name]['kp'],
    base_conf.robot_params[p.robot_name]['kd'],
    np.zeros(len(p.joint_names))
)
```

### 8.2 Error: IK Returns None

**Symptom:** "Position in unreachable cylinder" warning.

**Cause:** The UR5 has an inner unreachable cylinder with radius D[3] = 0.1333m around the base.

**Solution:** Spawn bricks at Y = 0.50 (away from robot base at Y = 0.35):
```python
# object_spawner.py spawn_in_line()
start_y = 0.60  # World Y, away from robot
```

### 8.3 Error: "No valid solution within joint limits"

**Symptom:** IK finds 8 solutions but all are rejected.

**Cause:** The q1 constraint (60 degrees from home) was rejecting valid solutions for bricks at certain positions.

**Status:** Under investigation. May need to relax constraint or make it position-dependent.

### 8.4 Error: Grasp Position Below Table

**Symptom:** Robot tries to go through the table.

**Solution:** Clamp grasp Z to table surface:
```python
# motion_planner.py pick_object()
min_z_robot_relative = -0.89  # 1cm above table
if grasp_z < min_z_robot_relative:
    grasp_z = min_z_robot_relative
```

### 8.5 Error: Wrong Perception Method Used

**Symptom:** Task scheduler finds 0 objects even when bricks are spawned.

**Cause:** `get_detected_objects()` requires YOLO camera, but ground truth mode was enabled.

**Solution:** Check config and use appropriate method:
```python
# task_scheduler.py
if self.config.use_ground_truth:
    objects = self.perception.get_ground_truth_objects()
else:
    objects = self.perception.get_detected_objects()
```

### 8.6 Error: Place Position in Wrong Frame

**Symptom:** Robot moves to wrong position when placing.

**Cause:** Target positions calculated in world frame but `place_object()` defaulted to `robot_relative=True`.

**Solution:** Pass correct frame flag:
```python
# task_scheduler.py
self.motion_planner.place_object(target_pos, self.robot_interface, robot_relative=False)
```

---

## 9. Current Status

### Working:
- Full analytical IK with 8 solutions
- IK solution selection with normalization
- Quintic trajectory interpolation
- Pick sequence (approach, grasp, lift)
- Place sequence
- Ground truth perception integration
- Coordinate transforms (A2)

### In Progress:
- PD gains setup for robot control (just added)
- Q1 constraint tuning for full table coverage
- Testing complete pick-and-place cycles

### Known Limitations:
- Target table (Y=1.0) at edge of workspace - may be unreachable
- Joint state feedback not working (using commanded position tracking as workaround)
- Some brick positions fail IK due to q1 constraint

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `motion_planner.py` | Main IK and trajectory generation |
| `controller.py` | ROS node, robot interface |
| `task_scheduler.py` | High-level task state machine |
| `object_spawner.py` | Brick spawning utilities |
| `perception_module.py` | YOLO + ground truth perception |
| `config.py` | All configuration parameters |
| `coord_test.py` | Coordinate transform testing |
| `ik_verify.py` | IK verification tests |
| `fk_diagnostic.py` | Forward kinematics diagnostics |

---

## 11. Lessons Learned

1. **Always verify ported code** - The MATLAB bug would have been hard to find without FK/IK roundtrip tests.

2. **Understand the coordinate frames** - The inverted robot mounting caused days of debugging until the A2 transform was identified.

3. **Check the full pipeline** - The robot not moving wasn't an IK issue, it was missing PD gains in the controller setup.

4. **Test incrementally** - Testing `pick_object()` directly before running the full task sequence helped isolate issues.

5. **Track what you command** - When feedback is broken, track your own commands to enable proper interpolation.
