#!/bin/bash
# Run Pick and Place Simulation
# This script starts all necessary components for the pick and place project
#
# Usage: bash run_pickandplace.sh
#
# This will:
# 1. Set up the ROS environment
# 2. Start the pick and place controller (which launches Gazebo automatically)

echo "=============================================="
echo "  Pick and Place Simulation Launcher"
echo "=============================================="

# Set up environment
source /opt/ros/noetic/setup.bash

# Check if workspace is built
if [ -f "/home/ubuntu/ros_ws/devel/setup.bash" ]; then
    source /home/ubuntu/ros_ws/devel/setup.bash
else
    echo "Warning: Workspace not built. Running catkin_make..."
    cd /home/ubuntu/ros_ws
    catkin_make -j2
    source /home/ubuntu/ros_ws/devel/setup.bash
fi

export LOCOSIM_DIR=/home/ubuntu/ros_ws/src/locosim
export PYTHONPATH=$LOCOSIM_DIR/robot_control:$PYTHONPATH
export ROS_PACKAGE_PATH=/home/ubuntu/ros_ws/src:$ROS_PACKAGE_PATH

# Navigate to project directory
cd $LOCOSIM_DIR/robot_control/lab_exercises/pick_and_place_project

echo ""
echo "Starting pick and place controller..."
echo "(This will automatically start Gazebo)"
echo ""

# Run the controller interactively
python3 -i pick_and_place_gazebo.py
