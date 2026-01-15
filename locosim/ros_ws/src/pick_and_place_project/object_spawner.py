#!/usr/bin/env python3
"""
Object Spawner Module for PAPA
Automatically spawns random objects from brick_description package
Supports multiple object classes with different geometries (STL files)
"""

import rospy
import tf
import numpy as np
import random
from threading import Thread
from typing import Optional, List, Dict, Any
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion
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

        self.spawned_objects: List[Dict[str, Any]] = []  # List of spawned object info
        self.tf_broadcasters: List[tuple] = []  # TF broadcasters for each object
        self.tf_thread: Optional[Thread] = None
        self.tf_thread_running = False

        # Single Broadcaster instance
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Placeholder for the service client (Connected lazily)
        self.spawn_client: Optional[rospy.ServiceProxy] = None

        rospy.loginfo("ObjectSpawner initialized")

    def _ensure_connection(self):
        """
        Connect to Gazebo service only when needed.
        This prevents 'Deadlock' if the spawner is created before Gazebo starts.
        """
        if self.spawn_client is not None:
            return True

        service_name = '/gazebo/spawn_urdf_model'
        try:
            rospy.loginfo(f"ObjectSpawner connecting to {service_name}...")
            rospy.wait_for_service(service_name, timeout=5.0)
            self.spawn_client = rospy.ServiceProxy(service_name, SpawnModel)
            rospy.loginfo("ObjectSpawner connected to Gazebo!")
            return True
        except rospy.ROSException:
            rospy.logwarn("ObjectSpawner could not connect to Gazebo! Is simulation running?")
            return False

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
        # 1. Ensure connection before trying to spawn
        if not self._ensure_connection():
            return None

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

        rospy.logdebug(f"Spawning {object_name}...")

        try:
            # Generate and upload URDF to parameter server
            urdf_content = self.generate_urdf(brick_type, object_name)
            param_name = f'{object_name}_description'
            rospy.set_param(param_name, urdf_content)

            pose = Pose(Point(*position), Quaternion(*rotation))

            # Type guard: ensure spawn_client is not None
            assert self.spawn_client is not None, "Spawn client not initialized"
            self.spawn_client(object_name, urdf_content, "/", pose, "world")

            self.tf_broadcasters.append((object_name, position, rotation))

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
            return object_info

        except Exception as e:
            rospy.logerr(f"Failed to spawn {object_name}: {e}")
            return None

    def spawn_one_of_each(self, allowed_types=None):
        """
        Spawn exactly one instance of each object type.
        This is the recommended default for comprehensive testing.

        Args:
            allowed_types (list, optional): List of allowed brick types. All types if None.

        Returns:
            list: List of spawned object info dicts
        """
        if allowed_types is None:
            allowed_types = list(BRICK_CLASSES.keys())

        rospy.loginfo(f"Spawning one of each object type ({len(allowed_types)} total)...")

        spawned = []
        for brick_type in allowed_types:
            obj_info = self.spawn_object(brick_type=brick_type)
            if obj_info:
                spawned.append(obj_info)
            rospy.sleep(0.3)  # Small delay between spawns

        rospy.loginfo(f"Spawned {len(spawned)}/{len(allowed_types)} object types successfully")

        # Start TF broadcasting thread
        self._start_tf_broadcast()

        return spawned

    def spawn_random_objects(self, num_objects=5, allowed_types=None):
        """
        Spawn multiple random objects (may include duplicates).

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

        rospy.loginfo(f"Spawned {len(spawned)} objects")

        # Start TF broadcasting thread
        self._start_tf_broadcast()

        return spawned

    def _start_tf_broadcast(self):
        """Start background thread to broadcast TF transforms for all objects."""
        if self.tf_thread_running:
            return

        self.tf_thread_running = True

        def broadcast_loop():
            rate = rospy.Rate(10)
            while self.tf_thread_running and not rospy.is_shutdown():
                current_time = rospy.Time.now()
                for item in list(self.tf_broadcasters):
                    if len(item) == 3:
                        name, pos, rot = item
                        self.tf_broadcaster.sendTransform(
                            pos, tuple(rot), current_time, f'/{name}', '/world'
                        )
                rate.sleep()

        self.tf_thread = Thread(target=broadcast_loop, daemon=True)
        self.tf_thread.start()

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
        # Normal shutdown on ROS interrupt (e.g., Ctrl+C)
        rospy.loginfo("ObjectSpawner node interrupted, shutting down.")
