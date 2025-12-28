# Object Spawning System

## Overview

The PAPA project now includes automatic spawning of random objects from the `brick_description` package. Objects can belong to different classes but have known geometry (coded in STL files).

## Features

- **Automatic Spawning**: Random objects are spawned at startup
- **Multiple Object Classes**: 8 different brick types with varying geometries
- **Random Orientations**: Objects spawn with random rotation around Z-axis (yaw) to stay upright
- **Dynamic TF Broadcasting**: Automatic TF transforms for spawned objects (position + orientation)
- **Configurable**: Easy configuration of spawn area, number of objects, and allowed types
- **Ground Truth Integration**: Spawned objects are automatically detected via Gazebo model states

## Object Classes

The following brick types are available from the `brick_description` package:

| Type                  | Mesh File               | Approximate Size | Class Name          |
| --------------------- | ----------------------- | ---------------- | ------------------- |
| `X1-Y1-Z2`            | X1-Y1-Z2.stl            | 8×8×16mm         | small_cube          |
| `X1-Y2-Z1`            | X1-Y2-Z1.stl            | 8×16×8mm         | flat_rectangle      |
| `X1-Y2-Z2`            | X1-Y2-Z2.stl            | 8×16×16mm        | rectangle           |
| `X1-Y3-Z2`            | X1-Y3-Z2.stl            | 8×24×16mm        | long_rectangle      |
| `X1-Y4-Z2`            | X1-Y4-Z2.stl            | 8×32×16mm        | very_long_rectangle |
| `X2-Y2-Z2`            | X2-Y2-Z2.stl            | 16×16×16mm       | large_cube          |
| `X1-Y2-Z2-CHAMFER`    | X1-Y2-Z2-CHAMFER.stl    | 8×16×16mm        | chamfered_rectangle |
| `X1-Y2-Z2-TWINFILLET` | X1-Y2-Z2-TWINFILLET.stl | 8×16×16mm        | filleted_rectangle  |

All meshes include visual and collision geometry defined in STL format.

## Configuration

Edit `config.py` to configure object spawning:

```python
# Automatic object spawning configuration
auto_spawn_objects = True  # Enable/disable automatic spawning
num_objects_to_spawn = 5  # Number of objects to spawn
spawn_area_center = [0.5, 0.5]  # Center of spawning area [x, y]
spawn_area_size = [0.5, 0.5]  # Size of spawning area [width, depth]

# Allowed brick types (None = all types allowed)
allowed_brick_types = None
# Or specify a subset:
# allowed_brick_types = ['X1-Y1-Z2', 'X1-Y2-Z2', 'X2-Y2-Z2']
```

## Usage

### Automatic Spawning (Default)

Objects are automatically spawned when the controller starts:

```bash
./run_papa.sh
```

The controller will:

1. Start Gazebo simulation
2. Initialize robot and controllers
3. Automatically spawn configured number of random objects
4. Start TF broadcasting for each object
5. Objects become available to perception module

### Manual Spawning

You can also manually spawn objects from the Python interpreter:

```python
# After starting the controller
p.spawn_objects()  # Spawn objects according to config

# Spawn specific object type with random position and rotation
p.object_spawner.spawn_object(brick_type='X1-Y2-Z2')

# Spawn at specific position
spawned = p.object_spawner.spawn_object(
    brick_type='X1-Y2-Z2',
    position=np.array([0.5, 0.5, 0.87])
)

# Spawn with specific position and rotation
spawned = p.object_spawner.spawn_object(
    brick_type='X2-Y2-Z2',
    position=np.array([0.6, 0.6, 0.87]),
    rotation=np.array([0.0, 0.0, 0.707, 0.707])  # 90° rotation (quaternion)
)

# Get list of spawned objects
objects = p.object_spawner.get_spawned_objects()
for obj in objects:
    print(f"{obj['name']}: {obj['position']}, rotation: {obj['rotation']}")
```

### Standalone Testing

Test the spawner independently:

```bash
# In Docker container, after sourcing ROS
source /home/ubuntu/ros_ws/install/setup.bash
cd /home/ubuntu/ros_ws/src/pick_and_place_project
python3 object_spawner.py
```

## Implementation Details

### Object Spawning Process

1. **URDF Generation**: Each object's URDF is dynamically generated with:

   - Reference to STL mesh file
   - Physical properties (mass, inertia)
   - Random color for visual diversity (both RViz and Gazebo)
   - Collision geometry matching visual

2. **Random Orientation**: Objects spawn with random Z-axis rotation (yaw) to keep them upright while varying orientation

3. **ROS Parameter Upload**: URDF is uploaded to ROS parameter server as `{object_name}_description`

4. **Gazebo Spawning**: Uses `gazebo_ros/spawn_model` with position (x, y, z) and orientation (roll, pitch, yaw)

5. **TF Broadcasting**: Continuous TF transform broadcast between `/world` and `/{object_name}` frames, including rotation

### Integration with Perception

Spawned objects are automatically detected by the perception module when using ground truth mode:

```python
# Objects appear in model_states callback
detected_objects = p.perception.get_detected_objects()
# Each object contains: name, position, orientation (quaternion), class, size
```

## Files

- `object_spawner.py`: Main spawning module with ObjectSpawner class
- `config.py`: Configuration parameters for spawning
- `controller.py`: Integration with main controller
- `brick_description/`: Package with STL meshes and URDF templates
  - `meshes/`: STL files for all brick types
  - `urdf/`: URDF templates (not used by dynamic spawner)
  - `scripts/spawnModel.py`: Original spawning example

## Troubleshooting

### Objects Not Appearing

1. Check Gazebo is running: Look for Gazebo GUI window
2. Verify brick_description package is built:

   ```bash
   rospack find brick_description
   ```

3. Check spawning is enabled in config.py:

   ```python
   auto_spawn_objects = True
   ```

4. Look for error messages in terminal output

### Objects Fall Through Table

- Ensure `table_height` in config.py matches your world file
- Default: `table_height = 0.85`
- Objects spawn slightly above table to avoid collision

### TF Transform Errors

- TF broadcasting runs in background thread
- Check with: `rosrun tf tf_echo /world /brick_1_X1_Y1_Z2`
- Restart controller if TF is not broadcasting

## Credits

This implementation is based on the `brick_description` package by Michele Focchi:

- Repository: `https://github.com/mfocchi/brick_description`
- License: BSD 3-Clause

## Future Enhancements

- [ ] Collision checking to prevent objects spawning inside each other
- [ ] Delete/respawn objects dynamically during runtime
- [ ] Support for custom object classes beyond brick_description package
