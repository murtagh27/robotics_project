# PAPA (Pick and Place Automation)

Autonomous robotic manipulation system for sorting objects using the UR5 manipulator.

## Quick Start

- Start the docker
- Enter the container
- Inside the container (best the VM) run:

```bash
roscore &
bash /home/ubuntu/ros_ws/run_papa.sh
```

**Basic Commands:**

```python
p.start_task()   # Start pick and place sequence
p.stop()         # Emergency stop
p.go_home()      # Move to home position
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│          PAPA Controller (controller.py)                │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
       ┌───────▼──────┐   ┌───▼──────┐   ┌───▼───────────┐
       │  Perception  │   │  Motion  │   │     Task      │
       │ (YOLOv8-OBB) │   │ (Joints) │   │(State Machine)│
       └──────────────┘   └──────────┘   └───────────────┘
```

- **Perception** - YOLOv8-OBB object detection with RGB-D localization
- **Motion Planner** - Joint-space trajectory generation
- **Task Scheduler** - Pick-place workflow coordination
- **Object Spawner** - Test environment setup

---

## Modules

### 1. Perception Module

The perception module detects and localizes objects in 3D space using a YOLOv8-OBB model trained on a custom generated dataset of the brick objects. It processes RGB-D camera data from Gazebo to identify object types, positions, and orientations.

**Data Flow:**

1. Subscribe to RGB and depth camera topics
2. Run YOLO OBB inference on RGB image
3. Extract depth values for detected bounding boxes
4. Project 2D detections + depth to 3D world coordinates
5. Height-based class correction (distinguishes Z1 vs Z2 bricks)
6. Extract orientation from OBB rotation angle
7. Duplicate filtering (2.5cm threshold)
8. Return list of detected objects with poses

**Model Performance:**

- YOLOv8-OBB trained on 500 synthetic images (100 epochs, Google Colab)
- mAP50: 0.933, mAP50-95: 0.804
- Inference: 2.2ms per image
- Position accuracy: <1cm, Orientation accuracy: <10° for most bricks

#### Interface

```python
def get_detected_objects() -> List[Dict]
```

Returns list of detected objects with:

- `name`: Object identifier (e.g., "X1-Y3-Z2-FILLET_0")
- `class`: Trivial brick name (e.g., "long_rectangle_filleted")
- `position`: 3D position [x, y, z] in world frame (numpy array)
- `orientation`: Quaternion [x, y, z, w] (numpy array)
- `dimensions`: Physical size [width, depth, height] (numpy array)
- `conf`: Detection confidence score (0.0 to 1.0, from YOLO)

**Example:**

```python
objects = p.perception.get_detected_objects()
# [{'name': 'X1-Y3-Z2-FILLET_0',
#   'class': 'long_rectangle_filleted',
#   'position': array([-0.206, 0.241, -0.905]),
#   'orientation': array([0, 0, -0.397, 0.918]),
#   'dimensions': array([0.03, 0.1, 0.06]),
#   'conf': 0.9815678000450134}]
```

**ROS Topics:** `/camera/rgb/image_raw`, `/camera/depth/image_raw`, `/camera/rgb/camera_info`

**Config:** Set `use_ground_truth = True` in `config.py` to use Gazebo ground truth instead of YOLO

#### Testing

```python
# Check detection status
objects = p.perception.get_detected_objects()
print(f"Detected {len(objects)} objects")

# View detailed object info
for obj in objects:
    pos = obj['position']
    print(f"{obj['class']}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], conf={obj['conf']:.3f}")

# Check ground truth objects for comparison
objects_gt = p.perception.get_ground_truth_objects()
```

**Test Script:**

```bash
cd ~/ros_ws/src/pick_and_place_project
python test_perception.py  # Shows live detections with bounding boxes and arrows
```

**Training Dataset:**

- Generated using `dataset_generator.py` (500 images from Gazebo)
- OBB annotations with 4 corner coordinates
- Config in `training_data/dataset_obb.yaml`
- Google Colab training notebook: [training_data/yolo_training.ipynb](training_data/yolo_training.ipynb)
- Regenerate: `python3 dataset_generator.py`

---

### 2. Motion Planner

TODO

### 3. Task Scheduler

TODO

### 4. Object Spawner

The object spawner dynamically creates brick objects in Gazebo for testing and simulation.
There are 11 brick classes defined in `brick_classes.py`.
On spawning bricks are assigned a random position, a random rotation around the Z-axis and a random color.
All meshes include visual and collision geometry (STL format)

#### Interface

```python
def spawn_object(brick_type, position=None, rotation=None) -> Dict
def spawn_one_of_each(allowed_types=None) -> List[Dict]
def spawn_random_objects(count, allowed_types=None) -> List[Dict]
def get_spawned_objects() -> List[Dict]
def clear_all_objects() -> None
```

**Example:**

```python
# Spawn all brick types (one of each)
p.object_spawner.spawn_one_of_each()

# Spawn specific object at custom position
spawned = p.object_spawner.spawn_object(
    brick_type='X2-Y2-Z2',
    position=np.array([0.6, 0.6, 0.87]),
    rotation=np.array([0.0, 0.0, 0.707, 0.707])  # 90° rotation
)

# Spawn 5 random objects
p.object_spawner.spawn_random_objects(5)

# Check what's spawned
objects = p.object_spawner.get_spawned_objects()
for obj in objects:
    print(f"{obj['name']}: pos={obj['position']}")

# Clear everything
p.object_spawner.clear_all_objects()
```

**Config:** Edit spawn area and allowed types in `config.py`:

```python
auto_spawn_objects = True
spawn_area_center = [0.5, 0.5]  # Center [x, y] in meters
spawn_area_size = [0.5, 0.5]    # Size [width, depth] in meters
allowed_brick_types = None      # None = all types, or specify list
```

**In Progress**

[ ] Collision avoidance on spawning

---

## Configuration & Testing

**All parameters in `config.py`:** robot joints, motion speeds, perception topics, spawn areas, target positions

**Testing:**

```python
# Perception
python test_perception.py

# Motion
p.go_home()
p.motion_planner.move_to_joints([0.5, -1.2, -1.8, -1.0, 1.57, 0.0], p)

# Spawner
p.object_spawner.spawn_one_of_each()

# Dataset generation
python dataset_generator.py --count 1000
```

---

## Project Structure

```
pick_and_place_project/
├── README.md                       # This file
├── controller.py                   # Main controller & ROS interface
├── perception_module.py            # YOLOv8-OBB object detection
├── motion_planner.py               # Joint-space motion planning
├── task_scheduler.py               # State machine coordinator
├── object_spawner.py               # Gazebo object spawning
│
├── config.py                       # System configuration
├── brick_classes.py                # Brick definitions (11 types)
├── papa.world                      # Gazebo world file
│
├── dataset_generator.py            # Training data generation
├── test_perception.py              # Perception testing script
│
├── training_data/                  # Generated datasets
│   ├── images/                     # Training images
│   ├── labels/                     # OBB annotations
│   └── dataset_obb.yaml            # YOLO config
└── weights/
    └── best.pt                     # Trained YOLOv8-OBB model

Launch script: ../run_papa.sh
```

---

## References

**Project Dependencies:**

- [Locosim](https://github.com/mfocchi/locosim) - Simulation framework
- [UR5 Robot](https://www.universal-robots.com/products/ur5-robot/) - Manipulator specs
- [ROS Noetic](http://wiki.ros.org/noetic) - Robot Operating System
- [Gazebo Classic](http://gazebosim.org/) - Physics simulator
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection

**Brick Objects:**
The object spawner is based on:

- Repository: [brick_description](https://github.com/mfocchi/brick_description) by Michele Focchi
- License: BSD 3-Clause

## License

Educational project for robotics course.
