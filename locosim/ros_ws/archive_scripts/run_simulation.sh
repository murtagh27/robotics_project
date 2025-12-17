#!/bin/bash

# First time setup - install pinocchio (only needed once per container)
# Uncomment these lines if running for the first time:
# apt-get update
# apt-get install -y ros-noetic-pinocchio

# Source ROS setup directly
source /opt/ros/noetic/setup.bash
source /home/ubuntu/ros_ws/devel/setup.bash

# Set Python paths
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3.8/site-packages

# Set LOCOSIM_DIR
export LOCOSIM_DIR=/home/ubuntu/ros_ws/src/locosim

# Add robot_control to Python path
export PYTHONPATH=$PYTHONPATH:$LOCOSIM_DIR/robot_control

# Set ROS package path - THIS IS CRITICAL
export ROS_PACKAGE_PATH=/home/ubuntu/ros_ws/src:$ROS_PACKAGE_PATH

echo "Environment configured!"
echo "ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"

# Verify package can be found
echo "Checking solo_description package..."
rospack find solo_description

# Run the base controller
python3 -i $LOCOSIM_DIR/robot_control/base_controllers/base_controller.py
