# PAPA Module Interfaces

This document defines the technical interfaces between the three main modules: **Perception**, **Motion Planning**, and **Task Scheduling**.

For high-level system overview, see [README.md](README.md).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│          PAPA Controller (controller.py)                │
│  - Main orchestrator                                    │
│  - ROS node initialization                              │
│  - Robot interface                                      │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
       ┌───────▼──────┐   ┌───▼──────┐   ┌───▼───────────┐
       │  Perception  │   │  Motion  │   │     Task      │
       │    Module    │   │ Planner  │   │   Scheduler   │
       └──────────────┘   └──────────┘   └───────────────┘
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
    'name': str,              # Unique object identifier (e.g., 'brick_1_X1_Y2_Z2')
    'position': np.array,     # [x, y, z] in world frame (meters)
    'orientation': np.array,  # [qx, qy, qz, qw] quaternion rotation
    'class': str             # Semantic class name (e.g., 'rectangle', 'small_cube', 'filleted_rectangle')
}
```

**Example:**

```python
objects = perception.get_detected_objects()
# Returns: [
#   {
#     'name': 'brick_1_X1_Y4_Z2',
#     'position': array([0.5552, 0.43569, 0.87]),
#     'orientation': array([-0.00003, 0.00006, 0.94782, 0.31881]),
#     'class': 'very_long_rectangle'
#   },
#   {
#     'name': 'brick_2_X1_Y2_Z2',
#     'position': array([0.49638, 0.28011, 0.87]),
#     'orientation': array([0.00004, 0.00003, 0.85883, 0.51226]),
#     'class': 'rectangle'
#   },
#   ...
# ]
```

**Available semantic classes** (defined in `brick_classes.py`):

- `small_cube` - X1-Y1-Z2 (8×8×16mm)
- `flat_rectangle` - X1-Y2-Z1 (8×16×8mm)
- `rectangle` - X1-Y2-Z2 (8×16×16mm)
- `long_rectangle` - X1-Y3-Z2 (8×24×16mm)
- `very_long_rectangle` - X1-Y4-Z2 (8×32×16mm)
- `large_cube` - X2-Y2-Z2 (16×16×16mm)
- `chamfered_rectangle` - X1-Y2-Z2-CHAMFER
- `filleted_rectangle` - X1-Y2-Z2-TWINFILLET

**Current Implementation:** Ground truth from Gazebo (automatic, real-time updates)

**Future Implementation:** Camera-based vision (same data structure)

**Called by:** TaskScheduler.detect_and_plan()

---

### **Using Perception Data in Your Module**

**Access detected objects:**

```python
# In task_scheduler.py or motion_planner.py
objects = self.perception.get_detected_objects()

for obj in objects:
    # Get position for motion planning
    target_x, target_y, target_z = obj['position']

    # Get orientation for grasp planning
    qx, qy, qz, qw = obj['orientation']

    # Get object identifier
    object_name = obj['name']
    object_type = obj['class']
```

**Check number of objects:**

```python
num_objects = len(self.perception.get_detected_objects())
```

**Find specific object:**

```python
def find_object_by_name(name_pattern):
    for obj in self.perception.get_detected_objects():
        if name_pattern in obj['name']:
            return obj
    return None
```

---

### ⚠️ **Notes for Teammates**

1. **Data is always available** - Perception updates automatically in ground truth mode
2. **Position is in meters** - World frame coordinates
3. **Orientation is quaternion** - [x, y, z, w] format for grasp planning
4. **Semantic class names** - The `'class'` field contains human-readable names (e.g., `'rectangle'`, `'small_cube'`) defined in `brick_classes.py`
5. **Data structure is stable** - Will remain the same when switching to camera-based perception
6. **Shared definitions** - Object classes are defined in `brick_classes.py`, shared between spawner and perception modules

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
        'class': str         # Object class
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
MOVING_TO_OBJECT ←───┐
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

## 5. Brick Classes (`brick_classes.py`)

Shared brick type definitions used by object spawner and perception modules.

**Purpose:** Single source of truth for all brick types with their properties.

**Structure:**

```python
BRICK_CLASSES = {
    'X1-Y1-Z2': {
        'mesh': 'X1-Y1-Z2.stl',
        'size': np.array([0.008, 0.008, 0.016]),  # meters
        'mass': 0.05,  # kg
        'class': 'small_cube',  # Semantic class name
    },
    # ... 7 more brick types
}
```

**Usage:**

```python
from brick_classes import BRICK_CLASSES

# Get properties
brick_info = BRICK_CLASSES['X1-Y2-Z2']
semantic_name = brick_info['class']  # 'rectangle'
mesh_file = brick_info['mesh']  # 'X1-Y2-Z2.stl'
```

---

## 6. Configuration (`config.py`)

Shared configuration used by all modules:

```python
# Robot parameters
robot_name = 'ur5'
q0 = np.array([...])  # Home joint config

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

#### Perception Team

- Implement camera-based object detection
- Point cloud segmentation
- Add object classification (color, shape)
- **Interface to maintain:** `get_detected_objects()` return format

#### Motion Planning Team

- Implement inverse kinematics
- Add trajectory optimization
- Implement collision avoidance
- **Interface to maintain:** Method signatures for `move_to_joints()`, `pick_object()`, `place_object()`

#### Task Scheduling Team

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

# Test full sequence
>>> p.start_task()
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
