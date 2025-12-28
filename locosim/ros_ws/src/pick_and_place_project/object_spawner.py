#!/usr/bin/env python3
"""
Object Spawner Module for PAPA
Automatically spawns random objects from brick_description package
Supports multiple object classes with different geometries (STL files)
"""

import rospy
import roslaunch
import tf
import numpy as np
import random
from threading import Thread
from brick_classes import BRICK_CLASSES


class ObjectSpawner:
    """
    Handles spawning of objects from brick_description package.

    Supports multiple object classes with known geometries defined in STL files.
    Automatically uploads URDF descriptions to ROS parameter server and spawns
    objects in Gazebo at random or specified positions.
    """

    def __init__(self, table_height=0.85, spawn_area_center=[0.5, 0.5], spawn_area_size=[0.3, 0.3]):
        """
        Initialize the object spawner.

        Args:
            table_height (float): Z-coordinate of table surface
            spawn_area_center (list): [x, y] center of spawning area
            spawn_area_size (list): [width, depth] of spawning area
        """
        self.table_height = table_height
        self.spawn_area_center = np.array(spawn_area_center)
        self.spawn_area_size = np.array(spawn_area_size)

        self.spawned_objects = []  # List of spawned object info
        self.tf_broadcasters = []  # TF broadcasters for each object
        self.tf_thread = None
        self.tf_thread_running = False

        rospy.loginfo("ObjectSpawner initialized")
        rospy.loginfo(f"  Table height: {table_height}m")
        rospy.loginfo(f"  Spawn area: {spawn_area_center} ± {spawn_area_size}")
        rospy.loginfo(f"  Available classes: {len(BRICK_CLASSES)}")

    def generate_urdf(self, brick_type, object_name):
        """
        Generate URDF description for a brick type.

        Args:
            brick_type (str): Type from BRICK_CLASSES keys
            object_name (str): Unique name for this object instance

        Returns:
            str: URDF XML string
        """
        if brick_type not in BRICK_CLASSES:
            raise ValueError(f"Unknown brick type: {brick_type}")

        brick_info = BRICK_CLASSES[brick_type]
        mesh_file = brick_info['mesh']
        mass = brick_info['mass']

        # Generate random color for visual diversity (both RViz and Gazebo)
        color_rgb, gazebo_material = self._random_color()

        urdf_template = f'''<?xml version="1.0" encoding="utf-8"?>
            <robot name="{object_name}" xmlns:xacro="http://ros.org/wiki/xacro">

            <material name="{object_name}_material">
                <color rgba="{color_rgb[0]} {color_rgb[1]} {color_rgb[2]} 1.0"/>
            </material>

            <link name="{object_name}">
                <inertial>
                    <mass value="{mass}"/>
                    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
                </inertial>
                
                <visual>
                    <origin xyz="0 0 0" rpy="0 0 0"/>
                    <geometry>
                        <mesh filename="package://brick_description/meshes/{mesh_file}" scale="1 1 1"/>
                    </geometry>
                    <material name="{object_name}_material"/>
                </visual>
                
                <collision>
                    <origin xyz="0 0 0" rpy="0 0 0"/>
                    <geometry>
                        <mesh filename="package://brick_description/meshes/{mesh_file}" scale="1 1 1"/>
                    </geometry>
                </collision>
            </link>

            <gazebo reference="{object_name}">
                <material>{gazebo_material}</material>
                <mu1>0.8</mu1>
                <mu2>0.8</mu2>
                <kp>1000000.0</kp>
                <kd>1.0</kd>
            </gazebo>

            </robot>
            '''
        return urdf_template

    def _random_color(self):
        """
        Generate a random color for object visualization.
        Returns matching RGB values for RViz and Gazebo material name.

        Returns:
            tuple: (rgb_list, gazebo_material_string)
        """
        # Define colors with matching RGB and Gazebo material names
        color_options = [
            ([0.8, 0.2, 0.2], 'Gazebo/Red'),
            ([0.2, 0.8, 0.2], 'Gazebo/Green'),
            ([0.2, 0.2, 0.8], 'Gazebo/Blue'),
            ([0.8, 0.8, 0.2], 'Gazebo/Yellow'),
            ([0.8, 0.2, 0.8], 'Gazebo/Purple'),
            ([0.2, 0.8, 0.8], 'Gazebo/Turquoise'),
            ([0.9, 0.5, 0.2], 'Gazebo/Orange'),
            ([0.9, 0.9, 0.9], 'Gazebo/White'),
        ]
        return random.choice(color_options)

    def _random_rotation(self):
        """
        Generate random rotation as quaternion.
        Generates random rotation around Z axis (yaw) to keep objects upright.

        Returns:
            np.ndarray: [x, y, z, w] quaternion
        """
        # Random yaw angle (rotation around Z axis)
        yaw = random.uniform(0, 2 * np.pi)

        # Convert to quaternion (only Z-axis rotation to keep objects upright)
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)

        return np.array([qx, qy, qz, qw])

    def _random_position(self):
        """
        Generate random position within spawn area.

        Returns:
            np.ndarray: [x, y, z] position on table surfaces
        """
        half_size = self.spawn_area_size / 2.0
        x = self.spawn_area_center[0] + random.uniform(-half_size[0], half_size[0])
        y = self.spawn_area_center[1] + random.uniform(-half_size[1], half_size[1])
        z = self.table_height + 0.1  # Slightly above table to avoid spawn collisions
        return np.array([x, y, z])

    def spawn_object(self, brick_type=None, position=None, rotation=None, object_name=None):
        """
        Spawn a single object in Gazebo.

        Args:
            brick_type (str, optional): Type from BRICK_CLASSES. Random if None.
            position (np.ndarray, optional): [x, y, z] position. Random if None.
            rotation (np.ndarray, optional): [x, y, z, w] quaternion. Random if None.
            object_name (str, optional): Unique name. Auto-generated if None.

        Returns:
            dict: Information about spawned object
        """
        # Select random brick type if not specified
        if brick_type is None:
            brick_type = random.choice(list(BRICK_CLASSES.keys()))

        # Generate unique object name
        if object_name is None:
            object_id = len(self.spawned_objects) + 1
            object_name = f"brick_{object_id}_{brick_type.replace('-', '_')}"

        # Generate random position if not specified
        if position is None:
            position = self._random_position()

        # Generate random rotation if not specified
        if rotation is None:
            rotation = self._random_rotation()

        rospy.loginfo(f"Spawning {object_name} (type: {brick_type}) at {position}")

        try:
            # Generate and upload URDF to parameter server
            urdf_content = self.generate_urdf(brick_type, object_name)
            param_name = f'{object_name}_description'
            rospy.set_param(param_name, urdf_content)
            rospy.logdebug(f"Uploaded URDF to parameter: {param_name}")

            # Spawn model in Gazebo using roslaunch
            package = 'gazebo_ros'
            executable = 'spawn_model'
            node_name = f'spawn_{object_name}'
            namespace = '/'

            # Convert quaternion to yaw angle for spawn command
            yaw = np.arctan2(
                2.0 * (rotation[3] * rotation[2] + rotation[0] * rotation[1]),
                1.0 - 2.0 * (rotation[1] ** 2 + rotation[2] ** 2),
            )

            args = f'-urdf -param {param_name} -model {object_name} -x {position[0]} -y {position[1]} -z {position[2]} -R 0 -P 0 -Y {yaw}'

            node = roslaunch.core.Node(
                package, executable, node_name, namespace, args=args, output="screen"
            )

            launch = roslaunch.scriptapi.ROSLaunch()
            launch.start()
            process = launch.launch(node)

            rospy.sleep(0.5)  # Wait for spawning to complete

            # Create TF broadcaster for this object (with rotation)
            broadcaster = tf.TransformBroadcaster()
            self.tf_broadcasters.append((broadcaster, object_name, position, rotation))

            # Store object information
            object_info = {
                'name': object_name,
                'type': brick_type,
                'class': BRICK_CLASSES[brick_type]['class'],
                'position': position.copy(),
                'rotation': rotation.copy(),
                'size': BRICK_CLASSES[brick_type]['size'].copy(),
                'mesh': BRICK_CLASSES[brick_type]['mesh'],
                'param_name': param_name,
            }
            self.spawned_objects.append(object_info)

            rospy.loginfo(f"✓ Successfully spawned {object_name}")
            return object_info

        except Exception as e:
            rospy.logerr(f"Failed to spawn {object_name}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def spawn_random_objects(self, num_objects=5, allowed_types=None):
        """
        Spawn multiple random objects.

        Args:
            num_objects (int): Number of objects to spawn
            allowed_types (list, optional): List of allowed brick types. All types if None.

        Returns:
            list: List of spawned object info dicts
        """
        rospy.loginfo(f"Spawning {num_objects} random objects...")

        if allowed_types is None:
            allowed_types = list(BRICK_CLASSES.keys())

        spawned = []
        for i in range(num_objects):
            brick_type = random.choice(allowed_types)
            obj_info = self.spawn_object(brick_type=brick_type)
            if obj_info:
                spawned.append(obj_info)
            rospy.sleep(0.3)  # Small delay between spawns

        rospy.loginfo(f"Spawned {len(spawned)}/{num_objects} objects successfully")

        # Start TF broadcasting thread
        self._start_tf_broadcast()

        return spawned

    def _start_tf_broadcast(self):
        """Start background thread to broadcast TF transforms for all objects."""
        if self.tf_thread_running:
            return

        self.tf_thread_running = True

        def broadcast_loop():
            rate = rospy.Rate(100)  # 100 Hz
            while self.tf_thread_running and not rospy.is_shutdown():
                current_time = rospy.Time.now()
                for item in self.tf_broadcasters:
                    broadcaster, object_name, position = item[0], item[1], item[2]
                    rotation = item[3] if len(item) > 3 else (0.0, 0.0, 0.0, 1.0)
                    broadcaster.sendTransform(
                        position,
                        tuple(rotation),  # Use actual rotation quaternion
                        current_time,
                        f'/{object_name}',
                        '/world',
                    )
                rate.sleep()

        self.tf_thread = Thread(target=broadcast_loop, daemon=True)
        self.tf_thread.start()
        rospy.loginfo("Started TF broadcast thread for spawned objects")

    def stop_tf_broadcast(self):
        """Stop the TF broadcasting thread."""
        self.tf_thread_running = False
        if self.tf_thread:
            self.tf_thread.join(timeout=1.0)

    def get_spawned_objects(self):
        """
        Get list of all spawned objects.

        Returns:
            list: List of object info dicts
        """
        return self.spawned_objects.copy()

    def clear_all_objects(self):
        """Remove all spawned objects from Gazebo (future implementation)."""
        # This would require gazebo_ros delete_model service
        rospy.logwarn("clear_all_objects not yet implemented")


def main():
    """
    Standalone test of object spawner.
    """
    rospy.init_node('object_spawner_test', anonymous=True)

    rospy.loginfo("Testing ObjectSpawner...")

    # Create spawner
    spawner = ObjectSpawner(
        table_height=0.85, spawn_area_center=[0.5, 0.5], spawn_area_size=[0.3, 0.3]
    )

    # Spawn 3 random objects
    objects = spawner.spawn_random_objects(num_objects=3)

    rospy.loginfo(f"\nSpawned {len(objects)} objects:")
    for obj in objects:
        rospy.loginfo(f"  - {obj['name']}: {obj['type']} at {obj['position']}")

    rospy.loginfo("\nPress Ctrl+C to exit")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
