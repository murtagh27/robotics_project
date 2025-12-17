# Pick and Place Project

Robotic manipulation project for sorting objects using UR5 manipulator arm.

## Quick Start

### In VNC Desktop (http://localhost:6080)

Open a terminal and run:

```bash
bash /home/ubuntu/ros_ws/run_pickandplace.sh
```

This will:
1. Set up the environment
2. Start Gazebo with the UR5 robot and pick-and-place world
3. Launch the interactive controller

Then in the Python console that appears:
```python
p.start_task()   # Start pick and place sequence
p.stop()         # Stop current task
p.reset()        # Reset to initial position
```

## Project Overview

This project implements an autonomous pick-and-place system where a robotic manipulator (UR5) picks objects from an initial table and places them in specific positions on a final table based on their class/type.

### Features

- **Object Detection**: Ground-truth from Gazebo or vision-based perception
- **Motion Planning**: Inverse kinematics and smooth trajectory generation
- **Task Scheduling**: High-level state machine for pick-place operations
- **Gripper Control**: Support for soft gripper (optional)

## Project Structure

```
pick_and_place_project/
├── README.md                      # This file
├── pick_and_place_controller.py   # Main controller
├── perception_module.py           # Object detection
├── motion_planner.py             # Motion planning and IK
├── task_scheduler.py             # High-level task planning
├── config/
│   └── params.py                 # Configuration parameters
├── worlds/
│   └── pick_place.world          # Gazebo world file
└── objects/
    └── (STL files for objects)
```

## Installation

### Prerequisites

- ROS Noetic
- Locosim framework
- Python 3.8+
- Required packages: numpy, pinocchio

### Setup

1. Ensure locosim is installed and working:
   ```bash
   source /home/ubuntu/ros_ws/src.sh
   ```

2. The project is located in:
   ```
   /home/ubuntu/ros_ws/src/locosim/robot_control/lab_exercises/pick_and_place_project/
   ```

## Usage

### Running the Simulation

1. Open a terminal in the VNC desktop (http://localhost:6080)

2. Source the environment:
   ```bash
   source /home/ubuntu/ros_ws/src.sh
   ```

3. Navigate to project directory:
   ```bash
   cd $LOCOSIM_DIR/robot_control/lab_exercises/pick_and_place_project
   ```

4. Run the controller:
   ```bash
   python3 -i pick_and_place_controller.py
   ```

5. In the Python interactive console, start the task:
   ```python
   p.start_task()
   ```

### Available Commands

- `p.start_task()` - Start the pick and place sequence
- `p.stop()` - Emergency stop
- `p.reset()` - Reset to initial state
- `p.go_home()` - Move robot to home position
- `p.perception.get_detected_objects()` - See detected objects
- `p.task_scheduler.get_progress()` - Check task progress

## Configuration

Edit `config/params.py` to customize:

- Object classes and their target positions
- Table dimensions and positions
- Motion planning parameters (speed, acceleration)
- Perception settings
- Gripper type

### Example Configuration

```python
# Define objects
object_classes = {
    'cube_red': {
        'size': [0.05, 0.05, 0.05],
        'final_position': [0.4, 0.2, 0.02]
    },
    # Add more objects...
}

# Motion parameters
approach_height = 0.15  # Height above object before grasping
max_velocity = 0.5      # Maximum end-effector velocity
```

## Development

### Adding New Objects

1. Define object class in `config/params.py`:
   ```python
   'my_object': {
       'color': [R, G, B, A],
       'size': [x, y, z],
       'final_position': [x, y, z]
   }
   ```

2. Add STL model to `objects/` folder (optional)

3. Update perception module to detect/classify new object

### Implementing Camera-Based Perception

Currently uses ground truth from Gazebo. To use camera:

1. Set in `config/params.py`:
   ```python
   use_ground_truth = False
   ```

2. Implement clustering in `perception_module.py`:
   - Use DBSCAN or Euclidean clustering
   - Extract object features (color, geometry)
   - Classify based on features

3. Add camera sensor to world file

### Improving Motion Planning

The current IK solver is basic. For better performance:

- Use MoveIt! for collision-aware planning
- Implement trajectory optimization
- Add velocity/acceleration profiling

## Testing

### Unit Tests

Test individual components:

```python
# Test perception
python3 -c "from perception_module import PerceptionModule; ..."

# Test motion planning
python3 -c "from motion_planner import MotionPlanner; ..."
```

### Integration Testing

Run full pick-place sequence and verify:
- All objects are detected correctly
- Robot reaches each object successfully
- Objects are placed in correct positions
- No collisions occur

## Troubleshooting

### Common Issues

**Robot doesn't move:**
- Check ROS master is running: `rostopic list`
- Verify joint commands: `rostopic echo /ur5/joint_group_pos_controller/command`

**Objects not detected:**
- Check Gazebo models are spawned: `rostopic echo /gazebo/model_states`
- Verify perception module: `p.perception.get_detected_objects()`

**IK fails:**
- Target might be out of reach
- Check joint limits in params.py
- Verify end-effector orientation

**Gripper doesn't close:**
- Check gripper type in config matches simulation
- Implement gripper control in `send_gripper_command()`

## Report Guidelines

Your project report should include:

### 1. Perception (1-2 pages)
- Object detection algorithm
- Segmentation method
- Classification approach
- Results with sample images

### 2. Motion Planning (1-2 pages)
- Inverse kinematics approach
- Trajectory generation method
- Collision avoidance strategy
- Sample trajectories with plots

### 3. Task Planning (1-2 pages)
- High-level algorithm
- State machine description
- Sequencing logic
- Performance metrics

### 4. Results (1 page)
- Success rate
- Execution time
- Error analysis
- Future improvements

## Documentation

Generate Doxygen documentation:

```bash
cd /path/to/pick_and_place_project
doxygen Doxyfile  # (create Doxyfile first)
```

## Contributors

- [Your Name]
- [Team Member 2]
- [Team Member 3]
- [Team Member 4]

## License

Educational project for Fundamental Robotics course.

## References

- Locosim: https://github.com/idra-lab/locosim
- Pinocchio: https://github.com/stack-of-tasks/pinocchio
- ROS Noetic: http://wiki.ros.org/noetic
