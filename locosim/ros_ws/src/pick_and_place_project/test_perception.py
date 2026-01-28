#!/usr/bin/env python3
"""
@file test_perception.py
@brief Perception testing and visualization tool.
@details Subscribes to camera feed and visualizes detected bricks from the PerceptionModule.
         Shows what the actual perception pipeline outputs (including depth processing,
         class correction, and duplicate filtering).
@author Benjamin Krech
@date January 2026
"""

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from perception_module import PerceptionModule
from brick_classes import BRICK_CLASSES


class PerceptionVisualizer:
    """
    @class PerceptionVisualizer
    @brief Visualizes perception module output with 3D positions and orientations.
    """

    def __init__(self):
        """
        @brief Initializes the visualizer with PerceptionModule and ROS subscribers.
        """
        rospy.init_node('perception_visualizer', anonymous=True)
        self.bridge = CvBridge()

        # Use actual perception module
        self.perception = PerceptionModule()
        rospy.loginfo("Initialized PerceptionModule")

        self.latest_rgb = None
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        rospy.loginfo("Perception Visualizer started. Press 'q' to quit.")

    def rgb_callback(self, msg):
        """
        @brief ROS callback for RGB camera images.
        @param msg ROS Image message.
        """
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"RGB Error: {e}")

    def project_3d_to_2d(self, position_3d):
        """
        @brief Projects 3D world position to 2D pixel coordinates.
        @param position_3d 3D position [x, y, z] in world frame.
        @return Tuple of (pixel_x, pixel_y) or None if projection fails.
        """
        if self.perception.camera_info is None:
            return None

        # Get camera intrinsics
        fx = self.perception.camera_info.K[0]
        fy = self.perception.camera_info.K[4]
        cx_cam = self.perception.camera_info.K[2]
        cy_cam = self.perception.camera_info.K[5]

        # Transform from world to camera frame (reverse of perception transform)
        cam_pos = self.perception.cam_world_pos
        Y_cam_metric = cam_pos[0] - position_3d[0]
        X_cam_metric = cam_pos[1] - position_3d[1]
        z_depth = cam_pos[2] - position_3d[2]

        if z_depth <= 0:
            return None

        # Project to image plane
        pixel_x = int(cx_cam + (X_cam_metric * fx / z_depth))
        pixel_y = int(cy_cam + (Y_cam_metric * fy / z_depth))

        return (pixel_x, pixel_y)

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
        cv2.arrowedLine(image, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    def run(self):
        """
        @brief Main loop processing camera feed and displaying detections.
        @details Runs at 1Hz, gets detections from PerceptionModule and visualizes them.
                 Press 'q' to quit.
        """
        rate = rospy.Rate(1)
        cv2.namedWindow('Perception Output', cv2.WINDOW_NORMAL)

        while not rospy.is_shutdown():
            if self.latest_rgb is not None:
                vis_image = self.latest_rgb.copy()

                # Get detections from actual perception module
                detected_objects = self.perception.get_detected_objects()
                num_detections = len(detected_objects)

                # Visualize each detected object
                for obj in detected_objects:
                    # Project 3D position to 2D image coordinates
                    pixel_coords = self.project_3d_to_2d(obj['position'])
                    if pixel_coords is None:
                        continue

                    cx, cy = pixel_coords

                    # Extract orientation (quaternion to yaw angle)
                    qz, qw = obj['orientation'][2], obj['orientation'][3]
                    angle_rad = 2 * math.atan2(qz, qw)

                    # Draw orientation arrow
                    self.draw_orientation_arrow(
                        vis_image, cx, cy, angle_rad, length=50, color=(0, 255, 0)
                    )

                    # Create label with class name and confidence
                    label = f"{obj['class']}: {obj['conf']:.2f}"
                    text_pos = (cx - 50, cy - 25)

                    # Draw label in dark blue
                    cv2.putText(
                        vis_image,
                        label,
                        text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (139, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Display detection count
                cv2.putText(
                    vis_image,
                    f"Detected: {num_detections} bricks",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (139, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Display instructions
                cv2.putText(
                    vis_image,
                    "Green arrows show detected orientation (from PerceptionModule)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (139, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow('Perception Output', vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        visualizer = PerceptionVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
