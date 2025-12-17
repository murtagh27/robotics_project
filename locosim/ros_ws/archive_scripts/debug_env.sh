#!/bin/bash

echo "Checking environment setup..."
echo "ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
echo ""
echo "Checking if solo_description package can be found:"
rospack find solo_description
echo ""
echo "Checking mesh files:"
ls -la /home/ubuntu/ros_ws/src/locosim/robot_descriptions/solo_description/meshes/stl/with_foot/
