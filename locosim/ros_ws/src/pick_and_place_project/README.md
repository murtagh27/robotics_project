# Pick and Place Project

Autonomous robotic manipulation system for sorting objects using UR5 manipulator with soft gripper.

## Quick Start

### Launch the System

```bash
bash /home/ubuntu/ros_ws/run_papa.sh
```

This will:

1. Set up the ROS environment
2. Start Gazebo with the UR5 robot and objects
3. Launch the interactive controller

### Basic Commands

In the Python console:

```python
p.start_task()   # Start pick and place sequence
p.stop()         # Emergency stop
p.reset()        # Reset to initial state
p.go_home()      # Move to home position
```

## System Overview

Modular pick-and-place system with three main components working together:

1. **Perception Module** - Detects and localizes objects
2. **Motion Planner** - Plans and executes robot movements
3. **Task Scheduler** - Coordinates high-level task execution

### Current Status

✅ **Working:**

- Ground truth object detection (4 objects: 3 cubes, 1 cylinder)
- Task planning and sequencing
- Joint-space motion execution
- Gripper control
- Full end-to-end pipeline

⚠️ **In Progress:**

- Inverse kinematics for accurate positioning
- Camera-based perception
- Collision avoidance
- Error recovery

## Project Structure

```
pick_and_place_project/
├── README.md                       # Project documentation
├── INTERFACES.md                   # Module interface specifications
├── run_papa.sh                     # Launch script
│
├── controller.py                   # Main controller & robot interface
├── perception_module.py            # Object detection module
├── motion_planner.py               # Motion planning (joint space)
├── task_scheduler.py               # High-level task coordination
├── pick_and_place_conf.py          # Configuration parameters
├── pick_and_place.world            # Gazebo world file
│
├── archive/                        # Archived/unused code
│   ├── pick_and_place_gazebo.py
│   ├── pick_and_place_main.py
│   └── motion_planner.py
│
└── config/
    └── params.py                   # Legacy config (not used)
```

## Team Collaboration

### Module Responsibilities

See [INTERFACES.md](INTERFACES.md) for detailed interface specifications.

**Perception Team:**

- Object detection from camera
- Point cloud segmentation
- Object classification
- Interface: `get_detected_objects()` → list of object dicts

**Motion Planning Team:**

- Inverse kinematics
- Trajectory optimization
- Collision avoidance
- Interface: `move_to_joints()`, `pick_object()`, `place_object()`

**Task Scheduling Team:**

- Error recovery
- Task constraints
- Dynamic replanning
- Interface: `execute_task_sequence()` workflow

### Development Workflow

1. **Read [INTERFACES.md](INTERFACES.md)** - Understand module boundaries
2. **Branch per module** - e.g., `feature/perception-camera`, `feature/motion-ik`
3. **Test independently** - Each module has test commands
4. **Integration** - Test full pipeline after changes
5. **Document changes** - Update INTERFACES.md if signatures change

### Testing Individual Modules

```python
# Start system
bash run_papa.sh

# Test perception
>>> objects = p.perception.get_detected_objects()
>>> print(f"Found {len(objects)} objects")

# Test motion
>>> p.motion_planner.move_to_joints(p.config.home_joint_config, p)

# Test full sequence
>>> p.start_task()
```

## Configuration

Edit `pick_and_place_conf.py` to adjust:

- **Object definitions** - Initial and target positions
- **Motion parameters** - Speeds, heights, thresholds
- **Perception settings** - Ground truth vs camera, topics
- **Gripper settings** - Open/close values

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          Pick and Place Controller                      │
│  - ROS node initialization                              │
│  - Robot interface (joints, gripper)                    │
│  - Gazebo simulation management                         │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
       ┌───────▼──────┐  ┌───▼──────┐  ┌───▼───────────┐
       │  Perception  │  │  Motion  │  │ Task Scheduler│
       │              │  │ Planner  │  │               │
       │ - Detect     │  │ - IK     │  │ - State       │
       │ - Localize   │  │ - Plan   │  │   machine     │
       │ - Classify   │  │ - Execute│  │ - Sequence    │
       └──────────────┘  └──────────┘  └───────────────┘
```

## Key Features

### Perception Module (`perception_module.py`)

- **Ground truth mode**: Uses Gazebo `/gazebo/model_states` topic
- **Vision mode**: Point cloud processing (to be implemented)
- **Output**: List of detected objects with positions and classes

### Motion Planner (`motion_planner.py`)

- **Current**: Hardcoded joint waypoints with linear interpolation
- **Planned**: IK-based motion planning with actual object coordinates
- **Methods**: `move_to_joints()`, `pick_object()`, `place_object()`

### Task Scheduler (`task_scheduler.py`)

- **State machine** for coordinating pick-place workflow
- **Sequence**: Detect → Plan → Pick → Place → Repeat
- **States**: IDLE, DETECTING, PLANNING, MOVING_TO_OBJECT, PICKING, MOVING_TO_TARGET, PLACING, RETURNING_HOME, COMPLETED

## Troubleshooting

### Robot not moving

- Check if Gazebo is running: `rosnode list | grep gazebo`
- Verify `/command` topic: `rostopic info /command`
- Check for errors: `rostopic echo /rosout | grep ERROR`

### Objects not detected

- Verify ground truth mode: Check `use_ground_truth = True` in config
- Check model states: `rostopic echo /gazebo/model_states`
- Test perception: `p.perception.get_detected_objects()`

### Simulation crashes

- Increase Gazebo's real-time factor in world file
- Reduce control loop frequency
- Check system resources: `htop`

## Development Tips

### Adding New Objects

1. Edit `pick_and_place.world` - Add model definition
2. Edit `pick_and_place_conf.py` - Add to `object_classes` dict
3. Restart simulation

### Tuning Motion

1. Adjust waypoints in `motion_planner.py`
2. Change durations for smoother/faster motion
3. Test with single object first

### Debugging

```python
# Enable debug logging
>>> import rospy
>>> rospy.set_param('/rosout/level', 'DEBUG')

# Check current state
>>> p.task_scheduler.current_state
>>> p.perception.get_detected_objects()
>>> p.get_current_joint_state()
```

## References

- [Locosim Documentation](https://github.com/mfocchi/locosim)
- [UR5 Robot Specs](https://www.universal-robots.com/products/ur5-robot/)
- ROS Noetic: http://wiki.ros.org/noetic
- Gazebo Classic: http://gazebosim.org/

## License

Educational project for robotics course.
