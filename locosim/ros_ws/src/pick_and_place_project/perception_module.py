"""
Perception module - FINAL
Features:
1. Validates class names against BRICK_CLASSES before renaming.
2. Prevents downgrading 'Large Cubes' or 'Small Cubes' to non-existent flat versions.
3. Uses dictionary dimensions for perfect center-point calculation.
"""

import numpy as np
import rospy
import cv2
import os
import math
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from brick_classes import BRICK_CLASSES


class PerceptionModule:
    def __init__(self, config=None):
        self.config = config
        self.detected_objects = []
        self.ground_truth_objects = []
        self.robot_pose = None

        # --- CONFIGURATION ---
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

        # --- EXTRINSICS ---
        self.cam_world_pos = np.array([0.5, 0.65, 1.7])

        # --- CALIBRATION ---
        # Tuned offsets for your simulation environment
        self.calibration_offset = np.array([0.012, 0.010, 0.003])

        rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.camera_info_callback)

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"RGB Error: {e}")

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth = np.nan_to_num(depth, nan=0.0)
        except Exception as e:
            rospy.logerr(f"Depth Error: {e}")

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg

    def get_detected_objects(self):
        if self.model is None:
            return []
        self._detect_and_process()
        return self.detected_objects

    def get_ground_truth_objects(self):
        return self.ground_truth_objects

    def _detect_and_process(self):
        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            return

        if self.robot_pose is None:
            current_robot_pos = np.zeros(3)
        else:
            current_robot_pos = self.robot_pose

        image = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        h_img, w_img = depth.shape

        # 1. INFERENCE
        results = self.model.predict(image, conf=0.25, verbose=False, save=False)
        result = results[0]
        new_objects = []

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx_cam = self.camera_info.K[2]
        cy_cam = self.camera_info.K[5]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Initial Name from YOLO
            current_name = self.id_to_class.get(cls_id, "unknown")

            # 2. DEPTH LOOKUP (10th Percentile for Top Surface)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            roi_x1 = max(0, cx - 5)
            roi_x2 = min(w_img, cx + 5)
            roi_y1 = max(0, cy - 5)
            roi_y2 = min(h_img, cy + 5)

            depth_roi = depth[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depths = depth_roi[(depth_roi > 0.1) & (depth_roi < 2.0)]

            if valid_depths.size == 0:
                continue
            z_surface_depth = np.percentile(valid_depths, 10)

            # 3. TRANSFORM (To Top Surface)
            X_cam_metric = (cx - cx_cam) * z_surface_depth / fx
            Y_cam_metric = (cy - cy_cam) * z_surface_depth / fy

            pos_world_abs_surface = np.array(
                [
                    self.cam_world_pos[0] - Y_cam_metric,
                    self.cam_world_pos[1] - X_cam_metric,
                    self.cam_world_pos[2] - z_surface_depth,
                ]
            )

            pos_surface_rel = pos_world_abs_surface - current_robot_pos - self.calibration_offset

            # 4. INTELLIGENT CLASS CORRECTION
            # Threshold: -0.835 (Splits Z2/High and Z1/Low)
            HEIGHT_THRESHOLD = -0.835
            z_height_val = pos_surface_rel[2]

            potential_name = current_name

            # LOGIC: Only swap if the target class actually exists in the dictionary!

            if z_height_val > HEIGHT_THRESHOLD:
                # MEASURED: HIGH (Z2)
                if "Z1" in current_name:
                    # Try to upgrade to Z2
                    check_name = current_name.replace("Z1", "Z2").replace("flat_", "")
                    if check_name in BRICK_CLASSES:
                        potential_name = check_name
                    # If check_name NOT in dictionary (e.g. X1-Y1-Z2 upgrade?), keep original.
            else:
                # MEASURED: LOW (Z1)
                if "Z2" in current_name:
                    # Try to downgrade to Z1
                    check_name = current_name.replace("Z2", "Z1")
                    # Special handling for "flat_" prefix if your keys use it?
                    # Based on your dict, keys are just 'X1-Y2-Z1', not 'flat_X1...'
                    # But the 'class' field has 'flat_'. The ID map uses KEYS.

                    if check_name in BRICK_CLASSES:
                        potential_name = check_name
                    # Critical: If X2-Y2-Z1 doesn't exist, we KEEP X2-Y2-Z2
                    # despite the low height reading. Better to trust YOLO than create ghosts.

            final_name = potential_name

            # 5. GET DIMENSIONS & CALCULATE BOTTOM POSITION
            if final_name in BRICK_CLASSES:
                # Use exact dimensions from the Single Source of Truth
                dims = BRICK_CLASSES[final_name]['size']
                brick_height = dims[2]
                class_str = BRICK_CLASSES[final_name]['class']
            else:
                # Fallback (should rarely happen now)
                dims = np.array([0.05, 0.05, 0.05])
                brick_height = 0.05
                class_str = 'unknown'

            # Shift from Top Surface to Bottom (Ground Truth)
            pos_final = pos_surface_rel.copy()
            pos_final[2] -= brick_height

            # 6. ORIENTATION
            angle = self._calculate_angle_longest_edge(image, box)
            qz = np.sin(angle / 2.0)
            qw = np.cos(angle / 2.0)

            obj = {
                'name': f"{final_name}_{len(new_objects)}",
                'class': class_str,
                'position': pos_final,
                'orientation': np.array([0.0, 0.0, qz, qw]),
                'dimensions': dims,
                'conf': conf,
            }
            new_objects.append(obj)

        self.detected_objects = self._filter_duplicates(new_objects)

    def _calculate_angle_longest_edge(self, image, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad = 5
        h_img, w_img, _ = image.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50:
            return 0.0

        rect = cv2.minAreaRect(largest_contour)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int0(box_pts)

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

    # --- GROUND TRUTH METHODS ---
    def _extract_brick_type_from_name(self, name):
        if name.startswith('brick_'):
            parts = name.split('_')
            if len(parts) >= 4:
                return '-'.join(parts[2:])
        return None

    def update_ground_truth(self, gazebo_model_states):
        self.ground_truth_objects = []
        robot_pose_obj = None

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

        for i, name in enumerate(gazebo_model_states.name):
            if name.startswith('brick_') or 'cube' in name:
                pose = gazebo_model_states.pose[i]
                rel_pos = np.array(
                    [
                        pose.position.x - robot_pose_obj.position.x,
                        pose.position.y - robot_pose_obj.position.y,
                        pose.position.z - robot_pose_obj.position.z,
                    ]
                )

                brick_class = 'unknown'
                if name.startswith('brick_'):
                    brick_type = self._extract_brick_type_from_name(name)
                    if brick_type in BRICK_CLASSES:
                        brick_class = BRICK_CLASSES[brick_type]['class']

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
