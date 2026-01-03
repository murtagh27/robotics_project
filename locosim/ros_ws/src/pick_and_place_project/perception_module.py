"""
Perception module for object detection and localization
Handles object detection from camera/sensor data
"""

import numpy as np

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from sklearn.cluster import DBSCAN

from brick_classes import BRICK_CLASSES


class PerceptionModule:
    """
    Handles object detection and pose estimation
    """

    def __init__(self, config):
        self.config = config
        self.detected_objects = []
        self._last_object_count = 0  # Track object count to avoid log spam

        # RGB-D camera processing
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None

        # Subscribe to camera topics
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)

        rospy.loginfo("PerceptionModule: Subscribed to RGB-D camera topics")

    # ============================================================================
    # RGB-D CAMERA CALLBACKS
    # ============================================================================

    def rgb_callback(self, msg):
        """
        Convert ROS Image message to OpenCV format
        """
        try:
            # Convert ROS Image to OpenCV format (BGR8)
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg):
        """
        Convert ROS depth image to NumPy array
        """
        try:
            # Convert ROS Image to NumPy array
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            # Handle invalid depth values (NaN, inf)
            self.latest_depth = np.nan_to_num(self.latest_depth, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            rospy.logerr(f"Failed to convert depth image: {e}")

    def camera_info_callback(self, msg):
        """
        Store camera intrinsics for 3D unprojection
        Only needs to run once as parameters don't change
        """
        if self.camera_info is None:
            # Extract camera intrinsics from K matrix (3x3)
            # K = [fx  0  cx]
            #     [ 0 fy  cy]
            #     [ 0  0   1]
            self.camera_info = {
                'fx': msg.K[0],  # Focal length X
                'fy': msg.K[4],  # Focal length Y
                'cx': msg.K[2],  # Principal point X
                'cy': msg.K[5],  # Principal point Y
                'width': msg.width,
                'height': msg.height,
            }
            rospy.loginfo(
                f"Camera intrinsics: "
                f"fx={self.camera_info['fx']:.2f}, "
                f"fy={self.camera_info['fy']:.2f}, "
                f"cx={self.camera_info['cx']:.2f}, "
                f"cy={self.camera_info['cy']:.2f}"
            )

    # ============================================================================
    # IMAGE PROCESSING & 3D RECONSTRUCTION
    # ============================================================================

    def process_rgbd_frame(self):
        """
        TODO: Main processing pipeline - call this periodically
        - Check if latest_rgb and latest_depth are available
        - Call create_point_cloud()
        - Call segment_table_plane()
        - Call cluster_objects()
        - Call classify_objects()
        - Update self.detected_objects with results
        """
        pass

    def create_point_cloud(self, rgb, depth, camera_info):
        """
        Convert RGB-D images to 3D point cloud

        Returns Nx6 array: [X, Y, Z, R, G, B]
        """
        height, width = depth.shape
        fx = camera_info['fx']
        fy = camera_info['fy']
        cx = camera_info['cx']
        cy = camera_info['cy']

        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Unproject to 3D
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Filter out invalid depths
        valid_mask = (Z > 0) & (Z < 5.0)  # Only points between 0 and 5 meters
        points_3d = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=1)

        # Extract corresponding RGB colors (OpenCV uses BGR, convert to RGB)
        colors = rgb[valid_mask][:, [2, 1, 0]]

        # Combine into Nx6 array [X, Y, Z, R, G, B]
        point_cloud = np.hstack([points_3d, colors])

        rospy.logdebug(f"Created point cloud with {len(point_cloud)} points")
        return point_cloud

    # ============================================================================
    # TABLE SEGMENTATION
    # ============================================================================

    def segment_table_plane(self, point_cloud):
        """
        TODO: Remove table plane using RANSAC
        - RANSAC algorithm:
            1. Randomly sample 3 points
            2. Fit plane equation: ax + by + cz + d = 0
            3. Count inliers (points within threshold distance)
            4. Repeat many iterations, keep best plane
        - Filter points: keep only points ABOVE table
        - Return: point cloud without table (Mx6 array)

        Hints:
        - Plane equation from 3 points using cross product
        - Distance point-to-plane: |ax + by + cz + d| / sqrt(a² + b² + c²)
        - Typical threshold: 0.01m (1cm)
        """
        pass

    # ============================================================================
    # OBJECT CLUSTERING
    # ============================================================================

    def cluster_objects(self, point_cloud):
        """
        TODO: Group points into individual objects using DBSCAN
        - DBSCAN parameters:
            eps: maximum distance between points in same cluster (~0.02m)
            min_samples: minimum points per cluster (~50-100)
        - Use only XYZ coordinates for clustering (not RGB)
        - Return: list of clusters, each cluster is Kx6 array

        Example:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.02, min_samples=50).fit(point_cloud[:, :3])
            labels = clustering.labels_
        """
        pass

    # ============================================================================
    # OBJECT CLASSIFICATION
    # ============================================================================

    def classify_objects(self, clusters):
        """
        TODO: Classify each cluster into brick types
        For each cluster:
        1. Extract features:
            - Bounding box dimensions (length, width, height)
            - Dominant color from RGB values
            - Number of points (size indicator)

        2. Match to BRICK_CLASSES:
            - Compare dimensions to known brick sizes
            - Use color to distinguish between types
            - Return class name (e.g., 'X1-Y2-Z2')

        3. Compute pose:
            - Position: centroid of cluster points
            - Orientation: use PCA or assume upright (identity quaternion)

        Return: list of dicts with 'class', 'position', 'orientation', 'name'
        """
        pass

    # ============================================================================
    # Public interface
    # ============================================================================

    def get_detected_objects(self):
        """Return list of detected objects"""
        return self.detected_objects

    # ============================================================================
    # Helpers & Ground truth
    # ============================================================================

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
            # TODO: does this have to be >=5 ?
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
