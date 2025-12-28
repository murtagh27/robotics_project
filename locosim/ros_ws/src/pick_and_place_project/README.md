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

and an extra component to set up the task:

4. **Object Spawner** - Dynamically spawns objects with known geometries

### Features

✅ **Automatic Object Spawning**: Randomly spawns objects from 8 different brick types at startup  
✅ **Multiple Object Classes**: Different geometries defined in STL files  
✅ **Ground Truth Detection**: Uses Gazebo model states for object localization  
✅ **Position & Orientation**: Full 6-DOF pose information (position + quaternion)  
✅ **Dynamic TF Broadcasting**: Automatic transforms for all spawned objects

See [OBJECT_SPAWNING.md](OBJECT_SPAWNING.md) for object spawning details.  
See [INTERFACES.md](INTERFACES.md) for detailed interface specifications.

### Current Status

✅ **Working:**

- Automatic random object spawning with 8 brick types
- Ground truth object detection (multiple object classes)
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
├── README.md                       # Project overview and quick start
├── INTERFACES.md                   # Detailed API specifications
├── OBJECT_SPAWNING.md              # Object spawning system documentation
├── project_task.md                 # Original project requirements
│
├── controller.py                   # Main controller & robot interface
├── perception_module.py            # Object detection module
├── motion_planner.py               # Motion planning module
├── task_scheduler.py               # High-level task coordination
├── object_spawner.py               # Dynamic object spawning system
├── config.py                       # Configuration parameters
├── papa.world                      # Gazebo world file
│
└── archive/                        # Archived/unused code
    ├── README_OLD.md
    ├── pick_and_place_gazebo.py
    ├── pick_and_place_main.py
    ├── motion_planner.py
    └── config/
        └── params.py

Launch script: ../run_papa.sh       # Located in ros_ws/
```

## Team Collaboration

### Module Responsibilities

See [INTERFACES.md](INTERFACES.md) for detailed interface specifications.

### Development Workflow

1. **Read [INTERFACES.md](INTERFACES.md)** - Understand module boundaries
2. **Branch per module** - e.g., `feature/perception-camera`, `feature/motion-ik`
3. **Test independently** - Each module has test commands
4. **Integration** - Test full pipeline after changes
5. **Document changes** - Update INTERFACES.md if signatures change

## Configuration

Edit `config.py` to adjust:

- **Object definitions** - Initial and target positions
- **Motion parameters** - Speeds, heights, thresholds
- **Perception settings** - Ground truth vs camera, topics
- **Gripper settings** - Open/close values

## Module Overview

The system uses a modular architecture with three main modules coordinated by a central controller. See [INTERFACES.md](INTERFACES.md) for detailed technical specifications.

**Key Modules:**

### 1. Perception Module

- **Current**: Ground truth detection from Gazebo model states
- **Future**: Camera-based point cloud processing
- **Output**: Object positions (x, y, z) and orientations (quaternion)
- **Interface**: `perception.get_detected_objects()` returns list of detected objects

### 2. Motion Planner

- **Current**: Joint-space motion with linear interpolation
- **Future**: IK-based motion planning and trajectory optimization
- **Interface**: `pick_object(pos)`, `place_object(pos)`, `move_to_joints(joints)`

### 3. Task Scheduler

- **Function**: State machine coordinating pick-place workflow
- **Sequence**: Detect → Plan → Pick → Place → Repeat
- **Interface**: `execute_task_sequence()` runs full automation

## Testing Your Module

### Test Perception

```python
# Check what objects are detected
objects = p.perception.get_detected_objects()
print(f"Detected {len(objects)} objects")

# View object details
for obj in objects:
    print(f"{obj['name']}: pos={obj['position']}, ori={obj['orientation']}")

# Spawn additional objects for testing
p.object_spawner.spawn_random_objects(3)
import time; time.sleep(1)
objects = p.perception.get_detected_objects()
```

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
- [ROS Noetic](http://wiki.ros.org/noetic)
- [Gazebo Classic](http://gazebosim.org/)

## License

Educational project for robotics course.
