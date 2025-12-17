# PAPA Module Interfaces

This document defines the interfaces between the three main modules: **Perception**, **Motion Planning**, and **Task Scheduling**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│          PAPA Controller                                │
│  (controller.py)                                        │
│  - Main orchestrator                                     │
│  - ROS node initialization                              │
│  - Robot interface                                       │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
       ┌───────▼──────┐  ┌───▼──────┐  ┌───▼───────────┐
       │  Perception  │  │  Motion  │  │ Task Scheduler│
       │   Module     │  │ Planner  │  │               │
       └──────────────┘  └──────────┘  └───────────────┘
```

---

## 1. Perception Module (`perception_module.py`)

### Purpose

Detects and localizes objects in the workspace using either ground truth (Gazebo) or camera-based perception.

### Interface

#### Initialization

```python
perception = PerceptionModule(config)
```

**Parameters:**

- `config`: Configuration object with attributes:
  - `use_ground_truth` (bool): If True, use Gazebo model states
  - `camera_topic` (str): ROS topic for point cloud (if not using ground truth)
  - `min_object_points` (int): Minimum points to consider an object
  - `segmentation_threshold` (float): Table segmentation threshold

#### Key Methods

##### `update_ground_truth(gazebo_model_states)`

Update detected objects from Gazebo ground truth.

**Input:**

- `gazebo_model_states` (gazebo_msgs/ModelStates): Gazebo model states message

**Returns:** List of detected objects

**Called by:** Controller's model_states_callback

---

##### `get_detected_objects()`

Get list of currently detected objects.

**Returns:** List of dictionaries, each containing:

```python
{
    'name': str,           # Object name (e.g., 'cube_red')
    'position': np.array,  # [x, y, z] in world frame (meters)
    'class': str          # Object class (e.g., 'cube_red', 'cylinder_yellow')
}
```

**Example:**

```python
objects = perception.get_detected_objects()
# Returns: [
#   {'name': 'cube_red', 'position': array([0.35, 0.5, 0.895]), 'class': 'cube_red'},
#   {'name': 'cube_green', 'position': array([0.5, 0.5, 0.895]), 'class': 'cube_green'},
#   ...
# ]
```

**Called by:** TaskScheduler.detect_and_plan()

---

### Extension Points for Camera-Based Perception

To implement camera-based perception, modify these methods:

1. **`point_cloud_callback(msg)`** - Process incoming point cloud
2. **`segment_table(points)`** - Segment table from point cloud
3. **`cluster_objects(points)`** - Cluster object point clouds
4. **`classify_object(obj)`** - Classify object by shape/color

---

## 2. Motion Planner (`motion_planner.py`)

### Purpose

Plans and executes robot arm movements in joint space.

### Interface

#### Initialization

```python
motion_planner = MotionPlanner(controller, config)
```

**Parameters:**

- `controller`: PickAndPlaceController instance
- `config`: Configuration object with:
  - `home_joint_config` (np.array): Home joint positions [6]
  - `approach_height` (float): Height above object for approach
  - `lift_height` (float): Height to lift after grasp

#### Key Methods

##### `move_to_joints(target_joints, controller, duration=3.0)`

Move robot to target joint configuration with linear interpolation.

**Input:**

- `target_joints` (np.array): Target joint angles [6] in radians
- `controller`: Controller instance
- `duration` (float): Movement duration in seconds

**Returns:** bool (True if successful)

**Called by:** TaskScheduler at various stages

---

##### `pick_object(object_pos, controller)`

Execute pick sequence: approach → grasp → lift.

**Input:**

- `object_pos` (np.array): Object position [x, y, z] in meters
- `controller`: Controller instance

**Returns:** bool (True if successful)

**Called by:** TaskScheduler.execute_pick()

**Sequence:**

1. Move to approach position (above object)
2. Move down to grasp position
3. Close gripper
4. Lift object

---

##### `place_object(target_pos, controller)`

Execute place sequence: approach → place → release → lift.

**Input:**

- `target_pos` (np.array): Target position [x, y, z] in meters
- `controller`: Controller instance

**Returns:** bool (True if successful)

**Called by:** TaskScheduler.execute_place()

**Sequence:**

1. Move to approach position (above target)
2. Move down to place position
3. Open gripper
4. Lift after placing

---

### Extension Points for IK-Based Motion Planning

Current implementation uses hardcoded joint waypoints. To add IK:

1. **`compute_ik(position, orientation)`** - Compute inverse kinematics
2. Replace hardcoded waypoints with IK solutions based on actual object positions
3. Add collision checking and path planning

---

## 3. Task Scheduler (`task_scheduler.py`)

### Purpose

High-level task planning and execution using a state machine.

### Interface

#### Initialization

```python
task_scheduler = TaskScheduler(perception, motion_planner, robot_interface, config)
```

**Parameters:**

- `perception`: PerceptionModule instance
- `motion_planner`: MotionPlanner instance
- `robot_interface`: Controller instance
- `config`: Configuration with object_classes dictionary

#### Key Methods

##### `detect_and_plan()`

Detect objects and create pick-place task sequence.

**Returns:** List of task dictionaries:

```python
[
    {
        'object': dict,      # Object from perception
        'target': np.array,  # Target position [x, y, z]
        'class': str        # Object class
    },
    ...
]
```

**Called by:** execute_task_sequence()

---

##### `execute_task_sequence()`

Execute the full pick-and-place sequence for all objects.

**Returns:** bool (True if all tasks completed successfully)

**Called by:** Controller.start_task()

**Flow:**

1. DETECTING_OBJECTS: Call detect_and_plan()
2. PLANNING_SEQUENCE: Validate tasks
3. For each task:
   - MOVING_TO_OBJECT: Navigate to object
   - PICKING: Execute pick sequence
   - MOVING_TO_TARGET: Navigate to target
   - PLACING: Execute place sequence
4. RETURNING_HOME: Return to home position
5. COMPLETED: Finish

---

### State Machine

```
IDLE
  ↓
DETECTING_OBJECTS
  ↓
PLANNING_SEQUENCE
  ↓
MOVING_TO_OBJECT ←──┐
  ↓                  │
PICKING              │
  ↓                  │
MOVING_TO_TARGET     │ (repeat for
  ↓                  │  each object)
PLACING              │
  ↓                  │
(next object?) ──────┘
  ↓
RETURNING_HOME
  ↓
COMPLETED
```

---

## 4. Controller Interface (`pick_and_place_controller.py`)

The controller provides robot interface methods used by motion planner and task scheduler:

### Key Methods

##### `send_joint_command(joints, velocities=None)`

Send joint position command to robot.

**Input:**

- `joints` (np.array): Joint positions [6] or [8] (with gripper)
- `velocities` (np.array, optional): Joint velocities

---

##### `send_gripper_command(width)`

Control gripper opening.

**Input:**

- `width` (float): Gripper opening in meters (0.0 = closed, 0.085 = open)

---

##### `get_current_joint_state()`

Get current robot joint positions and velocities.

**Returns:** tuple (positions, velocities), each np.array[8]

---

## 5. Configuration (`pick_and_place_conf.py`)

Shared configuration used by all modules:

```python
# Robot parameters
robot_name = 'ur5'
q0 = np.array([...])  # Home joint config

# Object definitions
object_classes = {
    'cube_red': {
        'type': 'cube',
        'initial_pos': np.array([x, y, z]),
        'final_position': np.array([x, y, z]),
        'priority': int
    },
    ...
}

# Perception settings
use_ground_truth = True
camera_topic = '/ur5/zed_node/point_cloud/cloud_registered'

# Motion parameters
approach_height = 0.15  # meters
grasp_height_offset = 0.02
lift_height = 0.20
```

---

## Development Guidelines

### Team Responsibilities

#### Perception Team:

- Implement camera-based object detection
- Add object classification (color, shape)
- Improve robustness to lighting/occlusions
- **Interface to maintain:** `get_detected_objects()` return format

#### Motion Planning Team:

- Implement inverse kinematics
- Add trajectory optimization
- Implement collision avoidance
- **Interface to maintain:** Method signatures for `move_to_joints()`, `pick_object()`, `place_object()`

#### Task Scheduling Team:

- Add error recovery logic
- Implement task constraints
- Add dynamic replanning
- **Interface to maintain:** `execute_task_sequence()` workflow

### Testing Individual Modules

Each module can be tested independently:

```python
# Start the system
bash run_papa.sh

# In Python interactive mode:
# Test perception
>>> objects = p.perception.get_detected_objects()
>>> print(objects)

# Test motion planning
>>> p.motion_planner.move_to_joints(p.config.home_joint_config, p)

# Test task scheduling
>>> p.task_scheduler.execute_task_sequence()
```

---

## Communication Protocol

- **Perception → Task Scheduler**: Object list via `get_detected_objects()`
- **Task Scheduler → Motion Planner**: Target positions and method calls
- **Motion Planner → Controller**: Joint commands via `send_joint_command()`
- **Controller → Perception**: Ground truth via `update_ground_truth()` callback

All modules should:

- Handle errors gracefully
- Log status using `rospy.loginfo/warn/err()`
- Return boolean success indicators where applicable
- Maintain backward compatibility with existing interfaces
