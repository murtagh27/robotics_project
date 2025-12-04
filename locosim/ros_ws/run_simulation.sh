#!/bin/bash

# First time setup - install pinocchio (only needed once per container)
# Uncomment these lines if running for the first time:
# apt-get update
# apt-get install -y ros-noetic-pinocchio

# Source environment
source /home/ubuntu/ros_ws/src.sh

# Run the base controller
python3 -i $LOCOSIM_DIR/robot_control/base_controllers/base_controller.py
