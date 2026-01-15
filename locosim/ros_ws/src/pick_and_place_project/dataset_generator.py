"""
Dataset Generator for the perception module.
Generates a synthetic YOLO dataset by taking pictures or randomized Gazebo scenes
and attaching brick location data to them.
"""

import rospy
import cv2
import numpy as np
import tf.transformations as tr
import random
from pathlib import Path
from typing import Optional, Tuple

from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteModel
from cv_bridge import CvBridge, CvBridgeError

from brick_classes import BRICK_CLASSES
from object_spawner import ObjectSpawner


class DatasetGenerator:
    def __init__(self, output_dir: str = "training_data"):
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
        """Callback for camera RGB image topic. Converts ROS image to OpenCV format."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def _info_cb(self, msg: CameraInfo):
        """Callback for camera info topic. Caches camera intrinsic parameters (K matrix)."""
        if self.camera_info is None:
            self.camera_info = msg

    def _state_cb(self, msg: ModelStates):
        """Callback for Gazebo model states. Updates positions and orientations of all models."""
        self.model_states = msg

    def randomize_scene(self):
        """Resets the scene by deleting old bricks and spawning new ones."""
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
        """Calculates World-to-Camera transformation matrix."""
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
        # Gazebo camera links are usually x-forward
        T_link_optical = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

        # We need World -> Camera so use Inverse of Camera -> World
        return np.linalg.inv(T_world_link @ T_link_optical)

    def project_point(
        self, point_3d: np.ndarray, T_cw: np.ndarray, K: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Projects a 3D world point to 2D pixel coordinates."""
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

            # YOLO Format: class x_center y_center width height (normalized)
            bw = (max_x - min_x) / img_w
            bh = (max_y - min_y) / img_h
            bx = (min_x + max_x) / 2.0 / img_w
            by = (min_y + max_y) / 2.0 / img_h

            labels.append(f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

            # Draw Debug Visuals
            cv2.rectangle(
                debug_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2
            )
            cv2.putText(
                debug_img,
                brick_type,
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
        """Main generation loop."""
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
