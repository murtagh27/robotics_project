#!/bin/bash

# PAPA (Pick and Place Automation) Launch Script
# This script sets up the environment and launches the controller

echo "=========================================="
echo "PAPA (Pick and Place Automation) Launcher"
echo "=========================================="

# Source ROS environment
source /opt/ros/noetic/setup.bash

# Source workspace
if [ -f ~/ros_ws/devel/setup.bash ]; then
    source ~/ros_ws/devel/setup.bash
    echo "✓ ROS workspace sourced"
else
    echo "✗ Error: ROS workspace not found!"
    exit 1
fi

# Set environment variables
export LOCOSIM_DIR=~/ros_ws/src/locosim
export PYTHONPATH=$LOCOSIM_DIR:$PYTHONPATH
export PYTHONPATH=$LOCOSIM_DIR/robot_control:$PYTHONPATH
export PYTHONPATH=~/ros_ws/src/pick_and_place_project:$PYTHONPATH
echo "✓ Environment variables set"

# Copy world file to locosim worlds directory
WORLD_SRC=~/ros_ws/src/pick_and_place_project/papa.world
WORLD_DST=$LOCOSIM_DIR/ros_impedance_controller/worlds/papa.world

if [ -f "$WORLD_SRC" ]; then
    cp "$WORLD_SRC" "$WORLD_DST"
    echo "✓ World file copied to locosim"
else
    echo "⚠ Warning: World file not found at $WORLD_SRC"
fi

# Navigate to project directory
cd ~/ros_ws/src/pick_and_place_project

echo "=========================================="
echo "Launching PAPA Controller"
echo "=========================================="
echo ""
echo "Commands available in interactive mode:"
echo "  p.start_task()  - Start automation task"
echo "  p.stop()        - Emergency stop"
echo "  p.reset()       - Reset controller"
echo "  p.go_home()     - Move to home position"
echo ""
echo "=========================================="

# Launch the controller in interactive mode
python3 -i controller.py
