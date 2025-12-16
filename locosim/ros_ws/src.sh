#!/bin/bash
#First time: apt-get update
#apt-get install -y ros-noetic-pinocchio

# Source ROS and workspace setup (use install space, not devel)
source /opt/ros/noetic/setup.bash
source /home/ubuntu/ros_ws/install/setup.bash

# Set LOCOSIM_DIR
export LOCOSIM_DIR=/home/ubuntu/ros_ws/src/locosim

# Add robot_control to Python path (for base_controllers imports)
export PYTHONPATH=$PYTHONPATH:$LOCOSIM_DIR/robot_control

# Add ROS Python packages to PYTHONPATH (for pinocchio, etc.)
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:/opt/ros/noetic/lib/python3.8/site-packages:$PYTHONPATH

echo "Environment configured!"
echo "LOCOSIM_DIR: $LOCOSIM_DIR"
echo ""
echo "To view the simulation GUI, open: http://localhost:6080 in your browser"
echo ""
echo "To run a controller example:"
echo "  python3 -i \$LOCOSIM_DIR/robot_control/base_controllers/base_controller.py"
