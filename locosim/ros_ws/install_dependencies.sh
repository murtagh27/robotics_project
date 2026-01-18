#!/bin/bash

# Install additional dependencies required for PAPA project
# This script should be run once when the Docker container is first started

echo "Installing additional dependencies..."

# 1. Install ROS Pinocchio
echo "Installing ros-noetic-pinocchio..."
apt-get update && apt-get install -y ros-noetic-pinocchio

# 2. Install Scikit-Learn
echo "Installing scikit-learn..."
pip3 install scikit-learn

# 3. Install Ultralytics for YOLO
echo "Installing ultralytics..."
# Install CPU-only PyTorch first to avoid CUDA bloat
pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Then install ultralytics
pip3 install --no-cache-dir ultralytics

echo "All dependencies installed successfully!"
