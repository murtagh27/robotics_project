# -*- coding: utf-8 -*-
"""
PAPA (Pick and Place Automation) - Configuration
All parameters for robot, objects, and motion planning
"""

import numpy as np

# Robot configuration
robot_name = 'ur5'

# End-effector frame name
frame_name = 'tool0'

# Home joint configuration (safe position) - from locosim params
q0 = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])

# Velocity and acceleration
qd0 = np.zeros(6)
qdd0 = np.zeros(6)

# Simulation parameters
dt = 0.001  # Control loop period (1000 Hz) - matches base params
exp_duration = 300.0  # Experiment duration in seconds

# Table heights (tavolo surface is at z=0.85)
table_height = 0.85  # Surface height of main tavolo
target_table_height = 0.85  # Target table also at 0.85

# Source table center (tavolo - where objects start)
source_table_pos = np.array([0.5, 0.5, table_height])

# Target table center (where objects go)
target_table_pos = np.array([0.5, 1.0, target_table_height])

# Motion planning parameters
approach_height = 0.15  # Height above object for approach
grasp_height = 0.02  # Height offset for grasping
lift_height = 0.20  # Height to lift after grasping
place_height = 0.05  # Height above target before placing

# Velocity limits
max_joint_velocity = 1.0  # rad/s
max_ee_velocity = 0.3  # m/s

# Control gains for Cartesian control
kp = np.array([300, 300, 300, 30, 30, 1])  # Position gains
kd = np.array([20, 20, 20, 5, 5, 0.5])  # Velocity gains

# Gripper parameters (if using gripper)
gripper_open_pos = 0.04
gripper_close_pos = 0.0
gripper_force = 10.0

# Use torque control or position control
use_torque_control = True

# World file to load
world_name = 'papa.world'

# Perception mode
use_ground_truth = True  # Use Gazebo model_states for ground truth (set False to use camera)
camera_topic = (
    '/ur5/zed_node/point_cloud/cloud_registered'  # Point cloud topic for vision-based perception
)

# Automatic object spawning configuration
auto_spawn_objects = True  # Automatically spawn random objects at startup
num_objects_to_spawn = 5  # Number of objects to spawn
spawn_area_center = [0.5, 0.5]  # Center of spawning area [x, y]
spawn_area_size = [0.5, 0.5]  # Size of spawning area [width, depth]

# Allowed brick types for spawning (None = all types), for futher info see documentation
allowed_brick_types = None

# Gripper flag
gripper = True
soft_gripper = True  # Two-fingered soft gripper
robotiq_gripper = False

# Slow factor for visualization
SLOW_FACTOR = 1.0

# Aliases for compatibility with old config names
home_joint_config = q0  # Alias for motion planner

# Missing attributes from old config
table_initial = {'position': source_table_pos.tolist(), 'size': [0.6, 0.4, 0.02]}
table_final = {'position': target_table_pos.tolist(), 'size': [0.6, 0.4, 0.02]}
camera_frame = 'camera_link'
min_object_points = 50
segmentation_threshold = 0.02
