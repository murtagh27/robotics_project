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
from perception_viz import PerceptionVisualizer


class PerceptionModule:
    """
    Handles object detection and pose estimation
    """

    def __init__(self, config):
        self.config = config
        self.detected_objects = []
        self.ground_truth_objects = []
        self._last_object_count = 0

        # --- CAMERA POSE (Relative to Robot Base) ---
        cam_pos_relative = np.array([0.0, 0.40, 0.05])

        # Rotation: Optical Frame to Robot Base Frame
        R_optical_to_base = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])

        self.camera_pose = {
            'position': cam_pos_relative,
            'rotation': R_optical_to_base,
        }

        rospy.loginfo(
            f"Camera transform loaded (Relative to Base). Position: {self.camera_pose['position']}"
        )

        # Initialize the Visualizer
        self.viz = PerceptionVisualizer()

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

            # 2. Remove table plane
            objects_cloud = self.segment_table_plane(point_cloud)

            # VISUALIZATION: Publish filtered cloud
            if objects_cloud is not None and len(objects_cloud) > 0:
                self.viz.publish_point_cloud(objects_cloud, self.camera_pose)

            if objects_cloud is None or len(objects_cloud) == 0:
                rospy.logdebug("No objects found above table")
                self.detected_objects = []
                return

            # 3. Cluster points into individual objects
            clusters = self.cluster_objects(objects_cloud)

            if not clusters:
                rospy.logdebug("No object clusters found")
                self.detected_objects = []
                return

            # 4. Classify and Pose Estimation
            self.detected_objects = self.classify_objects(clusters)

            # VISUALIZATION: Publish result markers
            self.viz.publish_detected_objects(self.detected_objects)

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
        # Only points between 0 and 1.75 meters to filter out floor # TODO: dynamicly depending on robot hight?
        valid_mask = (Z > 0) & (Z < 1.75)
        points_3d = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=1)

        # Extract corresponding RGB colors (OpenCV uses BGR, convert to RGB)
        colors = rgb[valid_mask][:, [2, 1, 0]]

        # Combine into Nx6 array [X, Y, Z, R, G, B]
        point_cloud = np.hstack([points_3d, colors])
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
        vertical_threshold = 0.9  # normal must be roughly parallel to Z

        xyz_points = point_cloud[:, :3]  # Extract XYZ coordinates
        n_points = len(xyz_points)
        best_plane = None
        best_inliers = 0

        # RANSAC Loop (Find the dominant plane)
        for _ in range(max_iterations):

            # Step 1: Randomly sample 3 points to form a candidate plane
            sample_indices = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = xyz_points[sample_indices]

            # Step 2: Calculate Plane Geometry (Normal Vector)
            # Create two vectors on the plane and cross-product them
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            # Safety check for collinear points (length is near zero)
            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)  # Normalize

            # If the normal isn't pointing mostly Up/Down (Z-axis), skip it.
            if abs(normal[2]) < vertical_threshold:
                continue

            d = -np.dot(normal, p1)  # Calculate distance offset

            # Step 3: Evaluate the Plane
            # Calculate distance from ALL points to this plane
            dists = np.abs(xyz_points.dot(normal) + d)
            # Check how many points belong to the plane
            inliers = np.sum(dists < distance_threshold)

            # Step 4: Update Best Fit
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal[0], normal[1], normal[2], d)

        if best_plane is None:
            return point_cloud

        # Global Cut (Delete everything below the table)

        # Unpack plane
        nx, ny, nz, d = best_plane

        # We need the normal to point TOWARDS the camera (Up) to define "Above".
        # In the Optical Frame, Z points DOWN. So "Up" means Negative Z.
        # If normal Z is positive, it points down (away). Flip it.
        if nz > 0:
            nx, ny, nz, d = -nx, -ny, -nz, -d

        corrected_plane = (nx, ny, nz, d)

        # Calculate signed distances
        distances = self._point_to_plane_distance(xyz_points, corrected_plane)

        # Keep points that are 0.5cm "above" the table (closer to camera)
        above_table_mask = distances > 0.005
        objects_cloud = point_cloud[above_table_mask]

        rospy.loginfo(
            f"RANSAC: Found table plane with {best_inliers} inliers, "
            f"kept {len(objects_cloud)} object points"
        )

        return objects_cloud

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
        detected_objects = []
        for cluster in clusters:
            points_xyz = cluster[:, :3]
            centroid_cam = np.mean(points_xyz, axis=0)

            # Size check
            if len(cluster) < 50:
                continue

            # Transform to World Frame (Relative to Robot Base)
            centroid_world = self._transform_camera_to_world(centroid_cam)

            obj = {
                'name': f"obj_{len(detected_objects)}",
                'class': 'unknown',
                'position': centroid_world,
                'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
                'num_points': len(cluster),
            }
            detected_objects.append(obj)
        return detected_objects

    def _transform_camera_to_world(self, point_camera):
        return self.camera_pose['rotation'] @ point_camera + self.camera_pose['position']

    # ============================================================================
    # Public interface
    # ============================================================================

    def get_detected_objects(self):
        """Return list of detected objects"""
        return self.detected_objects

    def get_gt_objects(self):
        return self.ground_truth_objects

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
        self.ground_truth_objects = []

        robot_pose = None
        for i, name in enumerate(gazebo_model_states.name):
            if 'ur5' in name.lower() or name == 'robot':
                robot_pose = gazebo_model_states.pose[i]
                break

        if robot_pose is None:
            return []

        for i, name in enumerate(gazebo_model_states.name):
            if name.startswith('brick_') or 'cube' in name or 'cylinder' in name:
                pose = gazebo_model_states.pose[i]

                # Manual Transform: Gazebo World -> Robot Base Frame
                rel_pos = np.array(
                    [
                        pose.position.x - robot_pose.position.x,
                        pose.position.y - robot_pose.position.y,
                        pose.position.z - robot_pose.position.z,
                    ]
                )

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
                    'position': rel_pos,
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
                self.ground_truth_objects.append(obj)

        # VISUALIZATION: Publish GT
        self.viz.publish_ground_truth(self.ground_truth_objects)

        return self.ground_truth_objects
