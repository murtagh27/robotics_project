#!/bin/bash

# PAPA (Pick and Place Automation) Launch Script
# This script sets up the environment and launches the controller

echo "=========================================="
echo "PAPA (Pick and Place Automation) Launcher"
echo "=========================================="

# Check and install dependencies if needed
DEPS_MARKER=/tmp/papa_deps_installed
if [ ! -f "$DEPS_MARKER" ]; then
    echo "First run detected - installing dependencies..."
    if [ -f /home/ubuntu/ros_ws/install_dependencies.sh ]; then
        bash /home/ubuntu/ros_ws/install_dependencies.sh
        touch "$DEPS_MARKER"
    else
        echo "⚠ Warning: install_dependencies.sh not found, skipping dependency installation"
    fi
fi

# Source ROS environment
source /opt/ros/noetic/setup.bash

# Source workspace
if [ -f /home/ubuntu/ros_ws/devel/setup.bash ]; then
    source /home/ubuntu/ros_ws/devel/setup.bash
    echo "✓ ROS workspace sourced"
else
    echo "✗ Error: ROS workspace not found at /home/ubuntu/ros_ws/devel/setup.bash"
    exit 1
fi

# Set environment variables
export LOCOSIM_DIR=/home/ubuntu/ros_ws/src/locosim
export PYTHONPATH=/opt/ros/noetic/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export PYTHONPATH=$LOCOSIM_DIR:$PYTHONPATH
export PYTHONPATH=$LOCOSIM_DIR/robot_control:$PYTHONPATH
export PYTHONPATH=/home/ubuntu/ros_ws/src/pick_and_place_project:$PYTHONPATH
echo "✓ Environment variables set"
echo "  LOCOSIM_DIR: $LOCOSIM_DIR"

# Copy world file to locosim worlds directory
WORLD_SRC=/home/ubuntu/ros_ws/src/pick_and_place_project/papa.world
WORLD_DST=$LOCOSIM_DIR/ros_impedance_controller/worlds/papa.world

if [ -f "$WORLD_SRC" ]; then
    cp "$WORLD_SRC" "$WORLD_DST"
    echo "✓ World file copied to locosim"
else
    echo "⚠ Warning: World file not found at $WORLD_SRC"
fi

# Navigate to project directory
cd /home/ubuntu/ros_ws/src/pick_and_place_project

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
