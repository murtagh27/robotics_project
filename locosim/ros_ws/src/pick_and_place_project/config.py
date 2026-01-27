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

# End-effector frame name
frame_name = 'tool0'

# Home joint configuration - hovering above table center, ready to pick
# End-effector at approximately (0.5, 0.45, 0.90) - just above table
# [base, shoulder, elbow, wrist1, wrist2, wrist3]
q0 = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, -1.63])  # Ready pose hovering above table

# Velocity and acceleration
qd0 = np.zeros(6)
qdd0 = np.zeros(6)

# Simulation parameters
dt = 0.001  # Control loop period (1000 Hz) - matches base params
exp_duration = 300.0  # Experiment duration in seconds

# Table heights (tavolo surface is at z=0.85)
table_height = 0.85  # Surface height of main tavolo
table_thickness = 0.04  # Table thickness (actual tavolo model)
target_table_height = 0.85  # Target table also at 0.85

# Source table center (tavolo - where objects start)
# Actual tavolo: center at (0.5, 0.4), size 1.0x0.8x0.04, pose at z=0.85 (which is center+half_thickness)
# So actual center is at z = 0.85 - 0.04/2 = 0.83
source_table_pos = np.array([0.5, 0.4, 0.83])  # Actual tavolo center
source_table_size = [1.0, 0.8, 0.04]  # Actual tavolo dimensions

# Target table center (where objects go) - LEFT side of tavolo (away from singularity)
# Robot at (0.5, 0.35), moving left means higher X values
target_table_pos = np.array([0.75, 0.55, 0.83])  # Left side of tavolo (X > robot X)
target_table_size = [0.3, 0.3, 0.04]  # Section of tavolo for placing

# Motion planning parameters
# NOTE: Ground truth returns object CENTER positions (z≈0.95)
# So grasp_height should be NEGATIVE to grasp below center
approach_height = 0.10  # Height above object center for approach
grasp_height = -0.03  # Height below object center for grasping (negative!)
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

# Automatic object spawning configuration
auto_spawn_objects = False  # Don't auto-spawn - use p.spawn_objects() or it spawns 3 on start_task
num_objects_to_spawn = 3  # Number of objects to spawn for sorting task
spawn_area_center = [0.75, 0.50]  # LEFT side of tavolo (higher X, away from singularity)
spawn_area_size = [0.2, 0.15]  # Size of spawning area [width, depth]
allowed_brick_types = ['X1-Y1-Z2', 'X1-Y2-Z2', 'X2-Y2-Z2']  # 3 different types for sorting demo

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
