#!/bin/bash

# All-in-One Launch Script for Pick and Place Project
# Starts roscore, Gazebo, and controller

echo "=========================================="
echo "Pick and Place - Complete Launcher"
echo "=========================================="

# Source ROS environment
source /opt/ros/noetic/setup.bash
source ~/ros_ws/devel/setup.bash

# Set environment variables
export LOCOSIM_DIR=~/ros_ws/src/locosim
export PYTHONPATH=$LOCOSIM_DIR:$PYTHONPATH
export PYTHONPATH=$LOCOSIM_DIR/robot_control:$PYTHONPATH
export PYTHONPATH=~/ros_ws/src/pick_and_place_project:$PYTHONPATH

echo "✓ Environment configured"

# Cleanup function
cleanup() {
    echo ""
    echo "=========================================="
    echo "Shutting down all processes..."
    echo "=========================================="
    
    # Kill all child processes
    pkill -P $$
    killall gzserver gzclient roscore rosmaster 2>/dev/null
    
    echo "✓ Cleanup complete"
    exit 0
}

# Register cleanup on exit
trap cleanup SIGINT SIGTERM EXIT

# Check if roscore is already running
if ! pgrep -x "roscore" > /dev/null && ! pgrep -x "rosmaster" > /dev/null; then
    echo "Starting ROS master..."
    roscore &
    ROSCORE_PID=$!
    sleep 3
    echo "✓ ROS master started (PID: $ROSCORE_PID)"
else
    echo "✓ ROS master already running"
fi

# Copy world file
WORLD_SRC=~/ros_ws/src/pick_and_place_project/world/pick_and_place.world
WORLD_DST=$LOCOSIM_DIR/ros_impedance_controller/worlds/pick_and_place.world

if [ -f "$WORLD_SRC" ]; then
    cp "$WORLD_SRC" "$WORLD_DST"
    echo "✓ World file copied"
fi

# Launch Gazebo
echo "Starting Gazebo simulation..."
roslaunch locosim ur5.launch world_name:=pick_and_place.world &
GAZEBO_PID=$!
echo "✓ Gazebo started (PID: $GAZEBO_PID)"
echo "  Waiting for Gazebo to initialize (15 seconds)..."
sleep 15

# Launch controller
echo "=========================================="
echo "Starting Pick and Place Controller"
echo "=========================================="
echo ""
echo "Commands available in interactive mode:"
echo "  p.start_task()  - Start pick and place task"
echo "  p.stop()        - Emergency stop"
echo "  p.reset()       - Reset controller"
echo "  p.go_home()     - Move to home position"
echo ""
echo "Press Ctrl+C to shut down everything"
echo "=========================================="

cd ~/ros_ws/src/pick_and_place_project
python3 -i pick_and_place_controller.py

# Cleanup will be called automatically when script exits
