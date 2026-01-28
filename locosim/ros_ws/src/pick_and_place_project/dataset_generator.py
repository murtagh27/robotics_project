#!/usr/bin/env python3
"""
@file dataset_generator.py
@brief Dataset generator for creating a synthetic YOLO OBB training dataset.
@details Uses the object spawner to generate a random setup of bricks on the table, then captures
         an RGB image. 3D brick corners taken from Gazebo model states are projected to 2D
         bounding boxes and then transformed to YOLO OBB format
         (class_id x_center y_center width height rotation_radians)
         and are saved as labels. Also creates debug images with drawn bounding boxes and rotation.
@author Benjamin Krech
@date January 2026
"""

import rospy
import cv2
import numpy as np
import tf.transformations as tr
import random
import math
from pathlib import Path
from typing import Optional, Tuple

from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteModel
from cv_bridge import CvBridge, CvBridgeError

from brick_classes import BRICK_CLASSES
from object_spawner import ObjectSpawner


class DatasetGenerator:
    """
    @class DatasetGenerator
    @brief Main class for generating the YOLO training dataset.
    @details Manages the complete dataset generation pipeline including scene randomization,
             camera data capture, 3D-to-2D projection of brick bounding boxes, and saving images
             with YOLO annotations. Creates three outputs per frame:
             training images, YOLO label files, and debug visualizations.
    """

    def __init__(self, output_dir: str = "training_data"):
        """
        @brief Initializes the DatasetGenerator, configures settings, and checks all folders exist.
        @param output_dir The folder in which the dataset is placed.
        """
        rospy.init_node('dataset_generator', anonymous=False)

        self.bridge = CvBridge()
        self.output_dir = Path(output_dir)

        # Ensure directories exist
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug_images").mkdir(parents=True, exist_ok=True)

        # State storage
        self.latest_image: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.model_states: Optional[ModelStates] = None

        # Configuration
        self.spawn_settle_time = 2.0
        self.deletion_wait_time = 0.5

        # Subscribers & Services
        rospy.Subscriber('/camera/rgb/image_raw', Image, self._img_cb)
        rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self._info_cb)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self._state_cb)

        rospy.wait_for_service('/gazebo/delete_model')
        self.delete_model_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # Spawner setup
        self.spawner = ObjectSpawner(
            table_height=0.85, spawn_area_center=[0.5, 0.5], spawn_area_size=[0.6, 0.4]
        )

        rospy.loginfo("Dataset Generator initialized. Waiting for camera stream...")
        rospy.sleep(1.0)

    def _img_cb(self, msg: Image):
        """
        @brief ROS callback for RGB camera image.
        @param msg ROS image containing the image frame.
        @return None
        """
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def _info_cb(self, msg: CameraInfo):
        """
        @brief ROS callback for camera info.
        @param msg ROS message containing the camera information.
        @return None
        """
        if self.camera_info is None:
            self.camera_info = msg

    def _state_cb(self, msg: ModelStates):
        """
        @brief ROS callback for receiving all Gazebo model states.
        @param msg ROS message containing model states.
        @return None
        """
        self.model_states = msg

    def randomize_scene(self):
        """
        @brief Creates a new random scene of objects.
        @details Deletes all old bricks, resets spawner state, spawns 3-8 new objects, and waits
                 for physics to settle.
        @return None
        """
        # 1. Stop broadcasting TFs for old objects to prevent errors
        self.spawner.stop_tf_broadcast()

        # 2. Delete existing bricks
        if self.model_states:
            for name in self.model_states.name:
                if name.startswith("brick_"):
                    try:
                        self.delete_model_srv(name)
                    except rospy.ServiceException as e:
                        rospy.logwarn(f"Failed to delete {name}: {e}")

        rospy.sleep(self.deletion_wait_time)

        # 3. Reset spawner and generate new objects
        self.spawner.spawned_objects = []
        self.spawner.tf_broadcasters = []

        num_bricks = random.randint(3, 8)
        self.spawner.spawn_random_objects(num_bricks)

        # 4. Wait for physics to settle and image buffers to flush
        rospy.sleep(self.spawn_settle_time)

    def get_camera_extrinsics(self) -> Optional[np.ndarray]:
        """
        @brief Computes the World-to-Camera transformation matrix.
        @return 4×4 numpy array representing the World-to-Camera transformation, or None if camera
                is not found.
        """
        if self.model_states is None:
            return None

        try:
            idx = self.model_states.name.index("rgbd_camera")
            pose = self.model_states.pose[idx]
        except ValueError:
            return None

        # Build World -> Link transform
        trans = [pose.position.x, pose.position.y, pose.position.z]
        rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        T_world_link = tr.quaternion_matrix(rot)
        T_world_link[0:3, 3] = trans

        # Correction for Optical Frame (Standard OpenCV: z-forward, x-right, y-down)
        # Gazebo camera links are x-forward
        T_link_optical = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

        # We need World -> Camera so use Inverse of Camera -> World
        return np.linalg.inv(T_world_link @ T_link_optical)

    def project_point(
        self, point_3d: np.ndarray, T_cw: np.ndarray, K: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        @brief Projects 3D point to 2D pixel coordinates with pinhole projection.
        @param point_3d 3D point in world coordinates.
        @param T_cw 4×4 World-to-Camera transformation matrix.
        @param K 3×3 matrix containing camera intrinsics.
        @return Tuple of pixel coordinates (u, v), or None if point is behind camera.
        """
        p_h = np.append(point_3d, 1)  # Homogeneous
        p_cam = T_cw @ p_h

        # Check if point is behind the camera
        if p_cam[2] <= 0:
            return None

        # Pinhole projection
        u = (K[0, 0] * p_cam[0]) / p_cam[2] + K[0, 2]
        v = (K[1, 1] * p_cam[1]) / p_cam[2] + K[1, 2]

        return (int(u), int(v))

    def capture_frame(self, frame_id: int):
        """
        @brief Captures and processes a single frame, generating YOLO OBB annotations for it.
        @details 1. Validates all data is available.
                 2. Loops through all brick models found in Gazebo.
                 3. Calculates all the corners of the brick.
                 4. Converts corners to 2D.
                 5. Extracts rotation angle from Gazebo orientation (Z-axis rotation).
                 6. Takes maximum coordinates and transforms them to YOLO OBB format.
                 7. Saves image, labels, and debugging image.
        @param frame_id Integer numbering the frame for filename.
        @return None - saves files to disk.
        """
        if self.model_states is None or self.latest_image is None or self.camera_info is None:
            rospy.logwarn_throttle(2, "Waiting for topics...")
            return

        img_h, img_w = self.latest_image.shape[:2]
        K = np.array(self.camera_info.K).reshape(3, 3)
        T_cw = self.get_camera_extrinsics()

        if T_cw is None:
            return

        labels = []
        save_img = self.latest_image.copy()
        debug_img = self.latest_image.copy()

        for i, name in enumerate(self.model_states.name):
            if not name.startswith("brick_"):
                continue

            # Parse brick type from name (e.g., brick_0_X1-Y2-Z2)
            parts = name.split('_')
            brick_type = '-'.join(parts[2:]) if len(parts) >= 3 else None

            if brick_type not in BRICK_CLASSES:
                continue

            # Get 3D Pose
            pose = self.model_states.pose[i]
            pos = [pose.position.x, pose.position.y, pose.position.z]
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

            T_obj = tr.quaternion_matrix(quat)
            T_obj[0:3, 3] = pos

            # Extract Z-axis rotation (yaw) from quaternion
            euler = tr.euler_from_quaternion(quat, 'sxyz')
            z_rotation = euler[2]  # Z-axis rotation from Gazebo

            # Get 3D Corners
            dims = BRICK_CLASSES[brick_type]['size']
            class_id = BRICK_CLASSES[brick_type]['id']
            dx, dy, dz = dims[0] / 2, dims[1] / 2, dims[2] / 2

            local_corners = [
                [dx, dy, dz],
                [dx, dy, -dz],
                [dx, -dy, dz],
                [dx, -dy, -dz],
                [-dx, dy, dz],
                [-dx, dy, -dz],
                [-dx, -dy, dz],
                [-dx, -dy, -dz],
            ]

            # Project to 2D
            pixel_coords = []
            for c in local_corners:
                c_world = (T_obj @ np.append(c, 1))[:3]
                uv = self.project_point(c_world, T_cw, K)
                if uv:
                    pixel_coords.append(uv)

            if not pixel_coords:
                continue

            pixel_coords = np.array(pixel_coords)

            # Clamp to image bounds
            min_x, min_y = np.min(pixel_coords, axis=0)
            max_x, max_y = np.max(pixel_coords, axis=0)

            min_x, min_y = max(0, min_x), max(0, min_y)
            max_x, max_y = min(img_w, max_x), min(img_h, max_y)

            # Skip invalid boxes
            if max_x <= min_x or max_y <= min_y:
                continue

            # Get the 2D projected corners to calculate oriented bounding box
            if len(pixel_coords) >= 4:
                # Fit oriented bounding box to the projected corners
                pixel_coords_for_rect = np.array(pixel_coords, dtype=np.float32)
                rect = cv2.minAreaRect(pixel_coords_for_rect)
                box_corners = cv2.boxPoints(rect)  # Get 4 corner points

                # Normalize corner coordinates
                box_corners_norm = box_corners.copy()
                box_corners_norm[:, 0] /= img_w  # Normalize x
                box_corners_norm[:, 1] /= img_h  # Normalize y

                # Extract angle for visualization
                (_, _), (w_rect, h_rect), angle_deg = rect
                rotation_radians = math.radians(angle_deg)

                # YOLO OBB Format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized corner coordinates)
                # Flatten the 4 corner points into 8 values
                obb_coords = box_corners_norm.flatten()
                labels.append(
                    f"{class_id} {obb_coords[0]:.6f} {obb_coords[1]:.6f} {obb_coords[2]:.6f} {obb_coords[3]:.6f} {obb_coords[4]:.6f} {obb_coords[5]:.6f} {obb_coords[6]:.6f} {obb_coords[7]:.6f}"
                )
            else:
                # Fallback: not enough corners, skip this object
                continue

            # Draw Debug Visuals
            # Draw the oriented bounding box (not axis-aligned)
            if len(pixel_coords) >= 4:
                box_corners_int = box_corners.astype(int)
                cv2.drawContours(debug_img, [box_corners_int], 0, (0, 255, 0), 2)

                # Draw orientation arrow from center
                center_x = int(np.mean(box_corners[:, 0]))
                center_y = int(np.mean(box_corners[:, 1]))
                arrow_length = min(max_x - min_x, max_y - min_y) * 0.4
                arrow_end_x = int(center_x + arrow_length * math.cos(rotation_radians))
                arrow_end_y = int(center_y + arrow_length * math.sin(rotation_radians))
                cv2.arrowedLine(
                    debug_img,
                    (center_x, center_y),
                    (arrow_end_x, arrow_end_y),
                    (255, 0, 0),
                    2,
                    tipLength=0.3,
                )

                cv2.putText(
                    debug_img,
                    f"{brick_type} ({math.degrees(rotation_radians):.0f}deg)",
                    (int(min_x), int(min_y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        # Save to disk
        file_id = f"{frame_id:05d}"

        cv2.imwrite(str(self.output_dir / "images" / f"{file_id}.jpg"), save_img)
        cv2.imwrite(str(self.output_dir / "debug_images" / f"{file_id}_debug.jpg"), debug_img)

        if labels:
            with open(self.output_dir / "labels" / f"{file_id}.txt", "w") as f:
                f.write("\n".join(labels))
            rospy.loginfo(f"Saved frame {file_id}: {len(labels)} objects")

    def run(self, num_frames: int = 500):
        """
        @brief Main function that runs the frame generation in a loop.
        @param num_frames The number of frames to be generated.
        @return None
        """
        rospy.loginfo(f"Starting generation of {num_frames} frames...")

        for i in range(num_frames):
            if rospy.is_shutdown():
                break
            self.randomize_scene()
            self.capture_frame(i)

        # Cleanup
        self.spawner.stop_tf_broadcast()
        rospy.loginfo("Dataset generation complete.")


if __name__ == '__main__':
    try:
        gen = DatasetGenerator()
        gen.run(num_frames=500)
    except rospy.ROSInterruptException:
        pass
