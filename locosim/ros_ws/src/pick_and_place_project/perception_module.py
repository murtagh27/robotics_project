"""
Perception module for object detection and localization
Handles object detection from camera/sensor data
"""

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
import sensor_msgs.point_cloud2 as pc2
from brick_classes import BRICK_CLASSES


class PerceptionModule:
    """
    Handles object detection and pose estimation
    """

    def __init__(self, config):
        self.config = config
        self.detected_objects = []
        self._last_object_count = 0  # Track object count to avoid log spam

        if not config.use_ground_truth:
            # Subscribe to camera point cloud
            self.pc_sub = rospy.Subscriber(
                config.camera_topic, PointCloud2, self.point_cloud_callback
            )

    def point_cloud_callback(self, msg):
        """Process incoming point cloud data"""
        # Convert ROS point cloud to numpy array
        points = self.ros_to_numpy_pointcloud(msg)

        # Segment table plane
        table_points, object_points = self.segment_table(points)

        # Cluster objects
        self.detected_objects = self.cluster_objects(object_points)

    def ros_to_numpy_pointcloud(self, cloud_msg):
        """Convert ROS PointCloud2 to numpy array"""
        points = []
        for point in pc2.read_points(cloud_msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])
        return np.array(points)

    def segment_table(self, points):
        """
        Segment table plane from point cloud using RANSAC
        Returns: (table_points, object_points)
        """
        if len(points) == 0:
            return np.array([]), np.array([])

        # Simple plane segmentation (RANSAC-like)
        # Find points close to table height
        table_height = self.config.table_initial['position'][2]
        threshold = self.config.segmentation_threshold

        # Points on table
        table_mask = np.abs(points[:, 2] - table_height) < threshold
        table_points = points[table_mask]

        # Points above table (objects)
        object_mask = points[:, 2] > (table_height + threshold)
        object_points = points[object_mask]

        return table_points, object_points

    def cluster_objects(self, points):
        """
        Cluster point cloud into individual objects
        Returns list of detected objects with positions
        """
        if len(points) < self.config.min_object_points:
            return []

        # Simple clustering based on spatial proximity
        # In practice, use DBSCAN or Euclidean clustering
        objects = []

        # For now, return centroid of all points as single object
        # TODO: Implement proper clustering algorithm
        centroid = np.mean(points, axis=0)

        obj = {
            'position': centroid,
            'points': points,
            'class': 'unknown',  # Will be classified later
        }
        objects.append(obj)

        return objects

    def classify_object(self, obj):
        """
        Classify object based on geometry/color
        Returns: object class name
        """
        # Analyze object geometry
        points = obj['points']

        # Calculate bounding box
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        dimensions = max_pt - min_pt

        # Simple classification based on shape
        # Check if cube-like or cylinder-like
        x, y, z = dimensions

        if abs(x - y) < 0.01 and abs(y - z) < 0.01:
            # Cube-like
            return 'cube'
        elif abs(x - y) < 0.01:
            # Cylinder-like
            return 'cylinder'
        else:
            return 'unknown'

    def get_detected_objects(self):
        """Return list of detected objects"""
        return self.detected_objects

    def get_object_pose(self, object_class):
        """
        Get pose of specific object class
        Returns: Pose message or None
        """
        for obj in self.detected_objects:
            if obj.get('class') == object_class:
                pose = Pose()
                pose.position.x = obj['position'][0]
                pose.position.y = obj['position'][1]
                pose.position.z = obj['position'][2]
                # Orientation (assuming upright)
                pose.orientation.w = 1.0
                return pose
        return None

    def _extract_brick_type_from_name(self, name):
        """
        Extract brick type from spawned object name.

        Args:
            name (str): Object name (e.g., 'brick_2_X1_Y2_Z2')

        Returns:
            str or None: Brick type (e.g., 'X1-Y2-Z2') or None if not extractable
        """
        if name.startswith('brick_'):
            parts = name.split('_')
            if len(parts) >= 4:  # brick, number, type parts
                # Rejoin type parts with hyphens (X1, Y2, Z2 -> X1-Y2-Z2)
                return '-'.join(parts[2:])
        return None

    def update_ground_truth(self, gazebo_model_states):
        """
        Update object positions from Gazebo ground truth
        Useful for testing without camera
        """
        self.detected_objects = []

        # Parse Gazebo model states
        for i, name in enumerate(gazebo_model_states.name):
            # Detect spawned brick objects (brick_0, brick_1, etc.)
            # Also support legacy names for backwards compatibility
            if name.startswith('brick_') or 'cube' in name or 'cylinder' in name:
                pose = gazebo_model_states.pose[i]

                # Extract brick type and look up semantic class name from BRICK_CLASSES
                brick_class = 'unknown'
                if name.startswith('brick_'):
                    brick_type = self._extract_brick_type_from_name(name)
                    if brick_type and brick_type in BRICK_CLASSES:
                        brick_class = BRICK_CLASSES[brick_type]['class']
                    elif brick_type:
                        brick_class = brick_type  # Fallback to brick type if not in BRICK_CLASSES

                obj = {
                    'name': name,
                    'position': np.array([pose.position.x, pose.position.y, pose.position.z]),
                    'orientation': np.array(
                        [
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]
                    ),
                    'class': brick_class,
                }
                self.detected_objects.append(obj)
                rospy.logdebug(f"Added object: {name} at position {obj['position']}")

        # Only log when object count changes to avoid spam
        if len(self.detected_objects) != self._last_object_count:
            rospy.loginfo(f"Ground truth perception: detected {len(self.detected_objects)} objects")
            self._last_object_count = len(self.detected_objects)

        return self.detected_objects

    def use_ground_truth_positions(self, gazebo_model_states):
        """Alias for update_ground_truth for backwards compatibility"""
        return self.update_ground_truth(gazebo_model_states)
