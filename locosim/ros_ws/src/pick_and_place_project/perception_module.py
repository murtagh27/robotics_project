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

    def get_detected_objects(self):
        """Return list of detected objects"""
        return self.detected_objects

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
