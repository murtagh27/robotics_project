"""
Perception module for object detection and localization
Handles object detection from camera/sensor data

main function calls:

- process_rgbs_frame()
- get_detected_objects()
(- update_ground_truth())
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
        Main processing pipeline - processes latest RGB-D frame
        Should be called from task_sceduler when necasssary
        """
        # Check if we have all required data
        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            rospy.logwarn_throttle(
                5.0, "Waiting for camera data (RGB, Depth, or CameraInfo not available)"
            )
            return

        try:
            # 1. Create 3D point cloud from RGB-D images
            point_cloud = self.create_point_cloud(
                self.latest_rgb, self.latest_depth, self.camera_info
            )

            if len(point_cloud) == 0:
                rospy.logwarn("Point cloud is empty")
                return

            # 2. Remove table plane (keep only objects above table)
            objects_cloud = self.segment_table_plane(point_cloud)

            if objects_cloud is None or len(objects_cloud) == 0:
                rospy.logdebug("No objects found above table")
                self.detected_objects = []
                return
            else:
                rospy.loginfo("starting clustering…")

            # 3. Cluster points into individual objects
            clusters = self.cluster_objects(objects_cloud)

            if not clusters:
                rospy.logdebug("No object clusters found")
                self.detected_objects = []
                return

            # 4. Classify each cluster and compute pose
            self.detected_objects = self.classify_objects(clusters)

            # Log results
            if len(self.detected_objects) != self._last_object_count:
                rospy.loginfo(f"Vision perception: detected {len(self.detected_objects)} objects")
                self._last_object_count = len(self.detected_objects)

        except Exception as e:
            rospy.logerr(f"Error in RGB-D processing pipeline: {e}")

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
        Remove table plane using RANSAC

        Returns point cloud with only objects (points above table)
        """
        if len(point_cloud) < 3:
            rospy.logwarn("Not enough points for plane fitting")
            return None

        # RANSAC parameters
        max_iterations = 100
        distance_threshold = 0.01  # 1cm - points closer than this are inliers
        min_inliers_ratio = 0.3  # At least 30% of points should be on table

        xyz_points = point_cloud[:, :3]  # Extract XYZ coordinates
        n_points = len(xyz_points)

        best_plane = None
        best_inliers = 0

        # RANSAC iterations
        for iteration in range(max_iterations):
            # Step 1: Randomly sample 3 points
            sample_indices = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = xyz_points[sample_indices]

            # Step 2: Fit plane through the 3 points
            plane_coeffs = self._fit_plane_from_points(p1, p2, p3)

            if plane_coeffs is None:
                continue  # Points were collinear, try again

            # Step 3: Count inliers (points close to plane)
            distances = self._point_to_plane_distance(xyz_points, plane_coeffs)
            inliers = np.abs(distances) < distance_threshold
            n_inliers = np.sum(inliers)

            # Step 4: Keep track of best plane
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_plane = plane_coeffs

        # Check if we found a valid table plane
        if best_plane is None or best_inliers < min_inliers_ratio * n_points:
            rospy.logwarn(f"Could not find table plane (only {best_inliers}/{n_points} inliers)")
            return point_cloud  # Return all points if no table found

        # Step 5: Filter points - keep only those above the table
        a, b, c, d = best_plane

        # Calculate distances
        distances = self._point_to_plane_distance(xyz_points, best_plane)

        # Camera is fixed looking down: Y-axis points toward table
        # -> Table plane has normal along Y (b ≈ ±1)
        # -> Objects are closer to camera (negative Y side)
        # -> So we want: distances < -0.005 (negative = toward camera = above table)
        above_table_mask = distances < -0.005
        objects_cloud = point_cloud[above_table_mask]

        rospy.loginfo(
            f"RANSAC: Found table plane with {best_inliers} inliers, "
            f"kept {len(objects_cloud)} object points"
        )

        return objects_cloud

    def _fit_plane_from_points(self, p1, p2, p3):
        """
        Fit plane equation ax + by + cz + d = 0 from 3 points

        Returns:
            tuple: (a, b, c, d) plane coefficients, or None if points are collinear
        """
        # Create two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1

        # Normal vector = cross product
        normal = np.cross(v1, v2)

        # Check if points are collinear (cross product ≈ 0)
        if np.linalg.norm(normal) < 1e-6:
            return None

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Calculate parameters using Point-Normal Form
        a, b, c = normal
        d = -np.dot(normal, p1)

        return a, b, c, d

    def _point_to_plane_distance(self, points, plane_coeffs):
        """
        Calculate signed distance from points to plane

        Args:
            points: Nx3 array of points
            plane_coeffs: (a, b, c, d) plane equation

        Returns:
            N-length array of signed distances
        """
        a, b, c, d = plane_coeffs

        # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
        # Since normal is normalized, denominator = 1
        distances = points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d

        return distances

    # ============================================================================
    # OBJECT CLUSTERING
    # ============================================================================

    def cluster_objects(self, point_cloud):
        """
        Cluster objects using XYZ position AND RGB color.
        """
        # A color_weight of 0.05 means a full color shift (0 to 255)
        # adds 5cm of "distance" to the clustering calculation.
        COLOR_WEIGHT = 0.05
        EPSILON = 0.02  # 2cm search radius
        MIN_SAMPLES = 120  # Density threshold for a valid cluster

        if len(point_cloud) == 0:
            return []

        # 1. Combine Spatial and Chromatic data
        # We normalize RGB to [0, 1] and scale by weight to align with Metric units
        xyz = point_cloud[:, :3]
        rgb_normalized = (point_cloud[:, 3:6] / 255.0) * COLOR_WEIGHT
        # Create a 6D feature space (X, Y, Z, R', G', B')
        features = np.hstack([xyz, rgb_normalized])

        # 2. Density-Based Clustering (DBSCAN)
        db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(features)
        labels = db.labels_

        # 3. Extract Clusters
        # DBSCAN returns -1 for "noise" points, we skip those
        unique_labels = np.unique(labels)
        clusters = []
        for label in unique_labels:
            if label == -1:
                continue

            # Create mask for this specific object
            cluster_mask = labels == label
            cluster_points = point_cloud[cluster_mask]
            clusters.append(cluster_points)

        rospy.loginfo(f"Clustering: Found {len(clusters)} objects")
        return clusters

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
