"""
@file perception_module.py
@brief Perception module for object detection and localization.
@details This module handles computer vision tasks for detecting and localizing objects.
         It uses YOLO combined with an RGB-D camera. Camera images are processed to detect
         all objects and calculate their position and rotation.
@author Benjamin Krech
@date January 2026
"""

import numpy as np
import rospy
import cv2
import os
import math
from collections import deque
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from brick_classes import BRICK_CLASSES

# Freshness and filtering constants
FRESHNESS_THRESHOLD = 2.0  # seconds - how long detection is considered fresh
EMA_ALPHA = 0.3  # Exponential Moving Average smoothing factor


class PerceptionModule:
    """@class
    @brief Main perception class handling YOLO-based object detection and 3D localization.
    @details This class manages the complete perception pipeline including YOLO model initialization,
             ROS subscribers, the actual object detection as a combination of YOLO and the depth data,
             class correction based on depth measurements, and ground truth tracking from Gazebo.
             It provides two main interfaces giving access to two object lists: (1) get_detected_objects
             and (2) get_ground_truth_objects.
    """

    def __init__(self, config=None):
        """@brief Constructor that initializes the module with configuration data.
        @param config Optional configuration data, currently not in use.
        """
        self.config = config
        self.detected_objects = []
        self.ground_truth_objects = []
        self.robot_pose = None
        
        # Freshness tracking
        self.last_detection_time = None
        self.last_ground_truth_time = None
        
        # Position history for EMA filtering
        self.position_history = {}  # object_name -> deque of positions

        # Configuration
        weights_path = os.path.join(os.path.dirname(__file__), 'weights/best.pt')
        if not os.path.exists(weights_path):
            rospy.logerr(f"YOLO weights not found at {weights_path}!")
            self.model = None
        else:
            rospy.loginfo(f"Loading YOLO from {weights_path}...")
            self.model = YOLO(weights_path)

        self.id_to_class = {v['id']: k for k, v in BRICK_CLASSES.items()}
        self.bridge = CvBridge()

        self.latest_rgb = None
        self.latest_depth = None
        self.camera_info = None

        # Extrinsics
        self.cam_world_pos = np.array([0.5, 0.65, 1.7])

        # Calibration
        self.calibration_offset = np.array([0.012, 0.010, 0.003])

        rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)

    def get_detection_age(self):
        """Get time since last detection"""
        if self.last_detection_time is None:
            return float('inf')
        return (rospy.Time.now() - self.last_detection_time).to_sec()
    
    def is_fresh(self):
        """Check if last detection is fresh"""
        return self.get_detection_age() < FRESHNESS_THRESHOLD
    
    def _apply_ema_filter(self, object_name, new_position):
        """Apply Exponential Moving Average filter to position"""
        if object_name not in self.position_history:
            self.position_history[object_name] = deque(maxlen=5)
            self.position_history[object_name].append(new_position)
            return new_position
        
        history = self.position_history[object_name]
        if len(history) == 0:
            history.append(new_position)
            return new_position
        
        # EMA: filtered = alpha * new + (1-alpha) * old
        old_position = history[-1]
        filtered = EMA_ALPHA * new_position + (1 - EMA_ALPHA) * old_position
        history.append(filtered)
        return filtered
    
    def clear_position_history(self, object_name=None):
        """Clear position history for object or all objects"""
        if object_name:
            if object_name in self.position_history:
                del self.position_history[object_name]
        else:
            self.position_history.clear()

    def rgb_callback(self, msg):
        """@brief ROS callback for RGB camera images.
        @param msg ROS message containing the camera frame.
        @return None
        """
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"RGB Error: {e}")

    def depth_callback(self, msg):
        """@brief ROS callback for camera depth images.
        @param msg ROS message containing the depth data.
        @return None
        """
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth = np.nan_to_num(depth, nan=0.0)
        except Exception as e:
            rospy.logerr(f"Depth Error: {e}")

    def camera_info_callback(self, msg):
        """@brief ROS callback for camera info.
        @param msg ROS message containing the camera information.
        @return None
        """
        if self.camera_info is None:
            self.camera_info = msg

    def get_detected_objects(self):
        """@brief Main interface for returning the detected objects.
        @return List of detected objects containing information about class, position, orientation,
                dimensions, and prediction confidence.
        """
        self.last_detection_time = rospy.Time.now()
        
        # If using ground truth mode, return ground truth objects instead of camera-based detection
        if self.config is not None and getattr(self.config, 'use_ground_truth', False):
            return self.ground_truth_objects
        
        # Otherwise use camera-based detection
        if self.model is None:
            return []
        self._detect_and_process()
        return self.detected_objects

    def get_ground_truth_objects(self):
        """@brief Interface for returning the Gazebo ground truth objects.
        @return List of all objects in the world containing information about name, class, position,
                and orientation.
        """
        return self.ground_truth_objects

    def _detect_and_process(self):
        """@brief Main processing pipeline that detects and classifies all objects.
        @details Takes the YOLO predictions as a base truth and processes them:
                 1. The height of the object is calculated via the depth information of the RGB-D camera.
                 2. The position relative to the robot is calculated by transforming the detected position.
                 3. The object height is calculated, and if it doesn't match with the YOLO prediction,
                    the class is changed. (Since the camera is looking straight down, it is very hard
                    for YOLO to classify objects that only differ in height.)
                 4. The bottom position of the object is calculated via the known height of the bricks.
                 5. The rotation of the object is calculated via minimum area rectangle fitting and
                    longest edge detection.
        @return None
        """
        if (
            self.latest_rgb is None
            or self.latest_depth is None
            or self.camera_info is None
            or self.model is None
        ):
            return

        if self.robot_pose is None:
            current_robot_pos = np.zeros(3)
        else:
            current_robot_pos = self.robot_pose

        image = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        h_img, w_img = depth.shape

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx_cam = self.camera_info.K[2]
        cy_cam = self.camera_info.K[5]

        # Get YOLO prediction
        results = self.model.predict(image, conf=0.25, verbose=False, save=False)
        result = results[0]
        new_objects = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Get name from YOLO id or set to "unknown" when not in brick_classes
            current_name = self.id_to_class.get(cls_id, "unknown")

            # 1. LOOKUP DEPTH
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Define region of interest as a 10x10 array around center
            roi_x1 = max(0, cx - 5)
            roi_x2 = min(w_img, cx + 5)
            roi_y1 = max(0, cy - 5)
            roi_y2 = min(h_img, cy + 5)

            depth_roi = depth[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depths = depth_roi[(depth_roi > 0.1) & (depth_roi < 2.0)]

            if valid_depths.size == 0:
                continue
            # Use 10th percentile to get top of brick and eliminate noise
            z_surface_depth = np.percentile(valid_depths, 10)

            # 2. TRANSFORM IMAGE COORDINATES
            X_cam_metric = (cx - cx_cam) * z_surface_depth / fx
            Y_cam_metric = (cy - cy_cam) * z_surface_depth / fy

            # Transform to world frame (x and y are swapped since camera is looking straight down)
            pos_world_abs_surface = np.array(
                [
                    self.cam_world_pos[0] - Y_cam_metric,
                    self.cam_world_pos[1] - X_cam_metric,
                    self.cam_world_pos[2] - z_surface_depth,
                ]
            )
            # Transform to robot relative coordinates including CALIBRATION
            pos_surface_rel = pos_world_abs_surface - current_robot_pos - self.calibration_offset

            # 3. CORRECT CLASSES BASED ON HEIGHT
            final_name = self._correct_classes_by_height(current_name, pos_surface_rel[2])

            # 4. CALCULATE BOTTOM POSITION
            if final_name in BRICK_CLASSES:
                dims = BRICK_CLASSES[final_name]['size']
                brick_height = dims[2]
                class_str = BRICK_CLASSES[final_name]['class']
            else:
                dims = np.array([0.05, 0.05, 0.05])
                brick_height = 0.05
                class_str = 'unknown'

            # Shift from top surface to bottom
            pos_final = pos_surface_rel.copy()
            pos_final[2] -= brick_height

            # 5. CALCULATE ORIENTATION
            angle = self._calculate_angle_longest_edge(image, box)
            qz = np.sin(angle / 2.0)
            qw = np.cos(angle / 2.0)
            
            # Apply EMA filter to position
            object_name = f"{final_name}_{len(new_objects)}"
            filtered_pos = self._apply_ema_filter(object_name, pos_final)

            obj = {
                'name': object_name,
                'class': class_str,
                'brick_type': final_name,  # Original brick type for target lookup
                'position': filtered_pos,
                'orientation': np.array([0.0, 0.0, qz, qw]),
                'dimensions': dims,
                'conf': conf,
                'timestamp': rospy.Time.now(),
                'is_fresh': True,
            }
            new_objects.append(obj)

        self.detected_objects = self._filter_duplicates(new_objects)

    def _correct_classes_by_height(self, current_name, z_height):
        """@brief Changes the YOLO predicted class if the depth information disagrees.
        @param current_name The YOLO predicted class.
        @param z_height The z-coordinate of the brick detected by the camera.
        @return The final class, either corrected or still the same as YOLO.
        """
        # Threshold Splits Z2(High) and Z1(Low)
        HEIGHT_THRESHOLD = -0.835
        potential_name = current_name

        if z_height > HEIGHT_THRESHOLD:
            if "Z1" in current_name:
                # Try to change to Z2
                check_name = current_name.replace("Z1", "Z2").replace("flat_", "")
                if check_name in BRICK_CLASSES:
                    potential_name = check_name
        else:
            if "Z2" in current_name:
                # Try to change to Z1
                check_name = current_name.replace("Z2", "Z1")
                if check_name in BRICK_CLASSES:
                    potential_name = check_name

        return potential_name

    def _calculate_angle_longest_edge(self, image, box):
        """@brief Calculates the angle of a brick by fitting a rectangle to its contour and
                 finding the longest edge.
        @param image The captured RGB image.
        @param box The YOLO bounding box.
        @return The angle (in radians) by which the longest side is rotated.
        """
        # Expand the bounding box to capture full object
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad = 5
        h_img, w_img, _ = image.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)

        # Crop out the region from the image
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        # Convert image to black and white
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Smooth out noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Invert and convert to pure black and white
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50:
            return 0.0

        # Fit minimum rectangle around the contour
        rect = cv2.minAreaRect(largest_contour)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int0(box_pts)

        # Find the longest edge & calculate the angle of the brick
        d1 = np.linalg.norm(box_pts[0] - box_pts[1])
        d2 = np.linalg.norm(box_pts[1] - box_pts[2])

        if d1 > d2:
            dx = box_pts[1][0] - box_pts[0][0]
            dy = box_pts[1][1] - box_pts[0][1]
            angle = math.atan2(dy, dx)
        else:
            dx = box_pts[2][0] - box_pts[1][0]
            dy = box_pts[2][1] - box_pts[1][1]
            angle = math.atan2(dy, dx)

        return angle

    def _filter_duplicates(self, objects, threshold=0.025):
        """@brief Filters out duplicates from a list of detected objects.
        @param objects List of detected objects to filter.
        @param threshold Distance threshold (in meters) determining when two objects are the same.
        @return The filtered list without duplicates.
        """
        if not objects:
            return []
        sorted_objects = sorted(objects, key=lambda x: x['conf'], reverse=True)
        unique_objects = []

        for obj in sorted_objects:
            is_duplicate = False
            for kept_obj in unique_objects:
                dist = np.linalg.norm(obj['position'] - kept_obj['position'])
                if dist < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_objects.append(obj)

        return unique_objects

    def _extract_brick_type_from_name(self, name):
        """@brief Extracts the brick type from a Gazebo object name.
        @param name The Gazebo object name (e.g., 'brick_0_X2-Y2-Z2').
        @return The extracted brick type (e.g., 'X2-Y2-Z2') or None if extraction failed.
        """
        if name.startswith('brick_'):
            parts = name.split('_')
            if len(parts) >= 4:
                return '-'.join(parts[2:])
        return None

    def update_ground_truth(self, gazebo_model_states):
        """@brief Updates ground truth object list from Gazebo simulation.
        @param gazebo_model_states ROS ModelStates message containing all models in the simulation
                                    with their poses.
        @return None
        """
        self.last_ground_truth_time = rospy.Time.now()
        self.ground_truth_objects = []
        robot_pose_obj = None

        # Find robot
        for i, name in enumerate(gazebo_model_states.name):
            if 'ur5' in name.lower() or name == 'robot':
                robot_pose_obj = gazebo_model_states.pose[i]
                self.robot_pose = np.array(
                    [
                        robot_pose_obj.position.x,
                        robot_pose_obj.position.y,
                        robot_pose_obj.position.z,
                    ]
                )
                break

        if robot_pose_obj is None:
            return

        # Find all bricks
        for i, name in enumerate(gazebo_model_states.name):
            if name.startswith('brick_'):
                pose = gazebo_model_states.pose[i]
                # Use world coordinates (not robot-relative)
                world_pos = np.array(
                    [
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                    ]
                )

                brick_type = self._extract_brick_type_from_name(name)
                if brick_type in BRICK_CLASSES:
                    brick_class = BRICK_CLASSES[brick_type]['class']
                else:
                    brick_class = 'unknown'

                obj = {
                    'name': name,
                    'position': world_pos,
                    'orientation': np.array(
                        [
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]
                    ),
                    'class': brick_class,
                    'brick_type': brick_type,  # Original brick type for target lookup
                }
                self.ground_truth_objects.append(obj)
