#!/usr/bin/env python3
"""
Perception Visualizer - Handles all RViz visualization logic
Separated from the main logic module for cleanliness.
"""

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg


class PerceptionVisualizer:
    """
    Helper class to manage ROS publishers for visualization.
    """

    def __init__(self):
        # Publishers
        self.marker_pub = rospy.Publisher("/perception/object_markers", MarkerArray, queue_size=1)
        self.pointcloud_pub = rospy.Publisher("/perception/point_cloud", PointCloud2, queue_size=1)
        self.gt_marker_pub = rospy.Publisher(
            "/perception/ground_truth_markers", MarkerArray, queue_size=1
        )

        rospy.loginfo("PerceptionVisualizer initialized")

    def publish_detected_objects(self, objects, frame_id="world"):
        """
        Publishes spheres and text labels for detected objects.
        """
        marker_array = MarkerArray()

        # Cleanup command
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # Standard settings
        common_marker = Marker()
        common_marker.header.frame_id = frame_id
        common_marker.header.stamp = rospy.Time.now()
        common_marker.pose.orientation.w = 1.0
        common_marker.lifetime = rospy.Duration(5)

        for i, obj in enumerate(objects):
            # 1. Sphere Marker (The detected object)
            sphere = Marker()
            sphere.header = common_marker.header
            sphere.ns = "object_spheres"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = obj['position'][0]
            sphere.pose.position.y = obj['position'][1]
            sphere.pose.position.z = obj['position'][2]
            sphere.pose.orientation.w = 1.0

            # Scale (5cm sphere)
            sphere.scale.x = 0.05
            sphere.scale.y = 0.05
            sphere.scale.z = 0.05

            # Colors (Cyclic)
            colors = [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
            ]
            c = colors[i % len(colors)]
            sphere.color.r, sphere.color.g, sphere.color.b = c
            sphere.color.a = 0.8  # Slightly transparent
            sphere.lifetime = rospy.Duration(5)
            marker_array.markers.append(sphere)

            # 2. Text Label
            text = Marker()
            text.header = common_marker.header
            text.ns = "object_labels"
            text.id = i + 100
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = obj['position'][0]
            text.pose.position.y = obj['position'][1]
            text.pose.position.z = obj['position'][2] + 0.08
            text.pose.orientation.w = 1.0
            text.scale.z = 0.05  # Text height

            # Color: BLACK
            text.color.r = 0.0
            text.color.g = 0.0
            text.color.b = 0.0
            text.color.a = 1.0

            text.lifetime = rospy.Duration(5)
            text.text = f"{obj['name']}"
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)

    def publish_point_cloud(self, point_cloud, camera_pose, frame_id="world"):
        """
        Transforms and publishes the point cloud to RViz.
        """
        if point_cloud is None or len(point_cloud) == 0:
            return

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # Pre-calculate transform
        rot = camera_pose['rotation']
        pos = camera_pose['position']

        # Vectorized Transform: (R * p') + T
        xyz_cam = point_cloud[:, :3]
        xyz_world = (rot @ xyz_cam.T).T + pos

        points = []
        for i in range(len(point_cloud)):
            x, y, z = xyz_world[i]
            r, g, b = point_cloud[i, 3:6]
            rgb = int(r) << 16 | int(g) << 8 | int(b)
            points.append([x, y, z, rgb])

        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, points)
        self.pointcloud_pub.publish(cloud_msg)

    def publish_ground_truth(self, objects, frame_id="world"):
        """
        Publishes:
        1. Flat Green Square (Base)
        2. Green Arrow pointing UP
        3. Black Text Label
        """
        marker_array = MarkerArray()

        # Cleanup
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        timestamp = rospy.Time.now()

        for i, obj in enumerate(objects):
            # 1. Flat Square (Cube with small Z)
            square = Marker()
            square.header.frame_id = frame_id
            square.header.stamp = timestamp
            square.ns = "gt_bases"
            square.id = i
            square.type = Marker.CUBE
            square.action = Marker.ADD
            square.pose.position.x = obj['position'][0]
            square.pose.position.y = obj['position'][1]
            square.pose.position.z = obj['position'][2]
            square.pose.orientation.w = 1.0

            # Dimensions: 4cm x 4cm x 2mm (Flat)
            square.scale.x = 0.04
            square.scale.y = 0.04
            square.scale.z = 0.002

            # Color: Semi-transparent Green
            square.color.r = 0.0
            square.color.g = 1.0
            square.color.b = 0.0
            square.color.a = 0.5
            square.lifetime = rospy.Duration(1)
            marker_array.markers.append(square)

            # 2. Arrow (Pointing UP)
            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.header.stamp = timestamp
            arrow.ns = "gt_arrows"
            arrow.id = i + 100
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            # Define Start and End points for the arrow
            p_start = Point()
            p_start.x, p_start.y, p_start.z = obj['position']

            p_end = Point()
            p_end.x, p_end.y = obj['position'][0], obj['position'][1]
            p_end.z = obj['position'][2] + 0.10  # 10cm tall arrow

            arrow.points = [p_start, p_end]

            # Arrow properties
            arrow.scale.x = 0.005  # Shaft diameter
            arrow.scale.y = 0.01  # Head diameter
            arrow.scale.z = 0.0  # Head length (0 = auto)

            # Color: Solid Green
            arrow.color.r = 0.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0
            arrow.lifetime = rospy.Duration(1)
            marker_array.markers.append(arrow)

            # 3. Label
            label = Marker()
            label.header.frame_id = frame_id
            label.header.stamp = timestamp
            label.ns = "gt_labels"
            label.id = i + 200
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = obj['position'][0]
            label.pose.position.y = obj['position'][1]
            label.pose.position.z = obj['position'][2] - 0.05  # Below the object
            label.pose.orientation.w = 1.0
            label.scale.z = 0.04

            # Color: Solid Green
            arrow.color.r = 0.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0

            label.lifetime = rospy.Duration(1)
            label.text = f"GT: {obj['class']}"  # Showing Class instead of Name for clarity
            marker_array.markers.append(label)

        self.gt_marker_pub.publish(marker_array)
