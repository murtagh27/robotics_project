#!/bin/bash
#Firs time: apt-get update
#apt-get install -y ros-noetic-pinocchio
# Source ROS and workspace setup
source /opt/ros/noetic/setup.bash
source /home/ubuntu/ros_ws/devel/setup.bash

# Set Python paths
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3.8/site-packages

# Set LOCOSIM_DIR
export LOCOSIM_DIR=/home/ubuntu/ros_ws/src/locosim

# Add robot_control to Python path
export PYTHONPATH=$PYTHONPATH:$LOCOSIM_DIR/robot_control

# Set ROS package path to include workspace and locosim
export ROS_PACKAGE_PATH=/home/ubuntu/ros_ws/src:$LOCOSIM_DIR:$ROS_PACKAGE_PATH

echo "Environment configured!"
echo "LOCOSIM_DIR: $LOCOSIM_DIR"
echo ""
echo "To view the simulation GUI, open: http://localhost:6080 in your browser"
echo ""
echo "To run a controller example:"
echo "  python3 -i \$LOCOSIM_DIR/robot_control/base_controllers/base_controller.py"