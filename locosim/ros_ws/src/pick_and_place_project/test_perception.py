#!/usr/bin/env python3
"""
@file test_perception.py
@brief Perception testing and visualization tool.
@details Subscribes to camera feed and visualizes detected bricks with oriented bounding boxes
         and orientation arrows. Used for testing and verifying YOLO OBB model performance.
@author Benjamin Krech
@date January 2026
"""

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from brick_classes import BRICK_CLASSES


class OrientationVisualizer:
    """
    @class OrientationVisualizer
    @brief Visualizes brick detections with oriented bounding boxes and arrows.
    """

    def __init__(self):
        """
        @brief Initializes the visualizer with YOLO model and ROS subscribers.
        """
        rospy.init_node('orientation_visualizer', anonymous=True)
        self.bridge = CvBridge()

        model_path = '/home/ubuntu/ros_ws/src/pick_and_place_project/weights/best.pt'
        self.model = YOLO(model_path)
        rospy.loginfo(f"Loaded OBB model from {model_path}")

        self.latest_rgb = None
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        rospy.loginfo("Orientation Visualizer started. Press 'q' to quit.")

    def rgb_callback(self, msg):
        """
        @brief ROS callback for RGB camera images.
        @param msg ROS Image message.
        """
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"RGB Error: {e}")

    def draw_obb_box(self, image, corners, color=(0, 255, 0), thickness=2):
        """
        @brief Draws oriented bounding box on image.
        @param image Target image to draw on.
        @param corners Array of 4 corner points.
        @param color Box color in BGR format.
        @param thickness Line thickness.
        """
        corners = corners.astype(np.int32)
        cv2.polylines(image, [corners], isClosed=True, color=color, thickness=thickness)

    def draw_orientation_arrow(self, image, cx, cy, angle, length=50, color=(0, 255, 0)):
        """
        @brief Draws orientation arrow showing brick angle.
        @param image Target image to draw on.
        @param cx Center x coordinate.
        @param cy Center y coordinate.
        @param angle Orientation angle in radians.
        @param length Arrow length in pixels.
        @param color Arrow color in BGR format.
        """
        end_x = int(cx + length * math.cos(angle))
        end_y = int(cy + length * math.sin(angle))
        cv2.arrowedLine(image, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    def run(self):
        """
        @brief Main loop processing camera feed and displaying detections.
        @details Runs at 5Hz, displays live feed with OBB boxes, orientation arrows,
                 and class labels. Press 'q' to quit.
        """
        rate = rospy.Rate(5)
        cv2.namedWindow('Brick Orientations', cv2.WINDOW_NORMAL)

        while not rospy.is_shutdown():
            if self.latest_rgb is not None:
                vis_image = self.latest_rgb.copy()

                # Run detection
                results = self.model(vis_image, verbose=False)

                num_detections = 0
                if len(results) > 0 and results[0].obb is not None:
                    obb_data = results[0].obb
                    num_detections = len(obb_data.xywhr)

                    # Get class names
                    class_names = results[0].names

                    for idx in range(num_detections):
                        cx, cy, w, h, angle_rad = obb_data.xywhr[idx].cpu().numpy()
                        confidence = obb_data.conf[idx].cpu().numpy()
                        class_id = int(obb_data.cls[idx].cpu().numpy())
                        class_name = class_names[class_id]

                        # Get the 4 corner points
                        if hasattr(obb_data, 'xyxyxyxy'):
                            corners = obb_data.xyxyxyxy[idx].cpu().numpy().reshape(4, 2)
                        else:
                            # Calculate corners manually from center, size, and angle
                            cos_a = math.cos(angle_rad)
                            sin_a = math.sin(angle_rad)
                            w_half = w / 2
                            h_half = h / 2

                            corners = np.array(
                                [
                                    [-w_half, -h_half],
                                    [w_half, -h_half],
                                    [w_half, h_half],
                                    [-w_half, h_half],
                                ]
                            )

                            # Rotate
                            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                            corners = corners @ rot_matrix.T

                            # Translate
                            corners[:, 0] += cx
                            corners[:, 1] += cy

                        self.draw_obb_box(vis_image, corners, color=(0, 255, 0), thickness=2)
                        self.draw_orientation_arrow(
                            vis_image, int(cx), int(cy), angle_rad, length=50, color=(0, 255, 0)
                        )

                        brick_info = list(BRICK_CLASSES.values())[class_id]
                        trivial_name = brick_info['class']
                        label = f"{trivial_name}: {confidence:.2f}"
                        text_pos = (int(cx) - 50, int(cy) - 25)

                        cv2.putText(
                            vis_image,
                            label,
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            vis_image,
                            label,
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (139, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    vis_image,
                    f"Detected: {num_detections} bricks",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_image,
                    "Green arrows show detected orientation",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow('Brick Orientations', vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        visualizer = OrientationVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
