# -*- coding: utf-8 -*-
"""
PAPA (Pick and Place Automation) - Configuration
All parameters for robot, objects, and motion planning
"""

import numpy as np

# Robot configuration
robot_name = 'ur5'
frame_name = 'tool0'

# Home joint configuration [base, shoulder, elbow, wrist1, wrist2, wrist3]
q0 = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, -1.63])
qd0 = np.zeros(6)
qdd0 = np.zeros(6)

# Simulation
dt = 0.001
exp_duration = 300.0

# Table geometry
table_height = 0.85
table_thickness = 0.04
target_table_height = 0.85

source_table_pos = np.array([0.5, 0.4, 0.83])
source_table_size = [1.0, 0.8, 0.04]

# Target placement area
target_table_pos = np.array([0.15, 0.55, 0.88])
target_table_size = [0.20, 0.20, 0.04]

# Motion planning
approach_height = 0.12
grasp_height = 0.06
lift_height = 0.12
place_height = 0.10
safe_transit_height = 1.10

# Movement timing
default_move_duration = 2.0
arc_move_duration = 2.0
approach_duration = 1.8
grasp_duration = 1.5

# Velocity limits
max_joint_velocity = 1.0
max_ee_velocity = 0.3

# Control gains
kp = np.array([300, 300, 300, 30, 30, 1])
kd = np.array([20, 20, 20, 5, 5, 0.5])

# Gripper (positive=open, negative=closed)
gripper_open_pos = 2.5
gripper_close_pos = -0.5
gripper_force = 10.0
gripper_empty_threshold = -0.5

# Control mode
use_torque_control = True

world_name = 'papa.world'
use_ground_truth = True
use_analytical_ik = False
seed = 40

# IK early exit thresholds
ik_early_exit_joint_dist = 40
ik_early_exit_pos_err = 0.5

# Object spawning
auto_spawn_objects = False
num_objects_to_spawn = 11
spawn_area_center = [0.80, 0.50]
spawn_area_size = [0.30, 0.30]

# Predefined spawn positions for each brick class (on source table - PICK area)
# Format: 'brick_type': [x, y, z] in world frame
# These are on the +X side of robot (x > 0.5)
SPAWN_POSITIONS = {
    'X1-Y1-Z2': [0.65, 0.15, 0.88],
    'X1-Y2-Z1': [0.80, 0.15, 0.88],
    'X1-Y2-Z2': [0.95, 0.15, 0.88],
    'X1-Y2-Z2-CHAMFER': [0.65, 0.30, 0.88],
    'X1-Y2-Z2-TWINFILLET': [0.80, 0.30, 0.88],
    'X1-Y3-Z2': [0.95, 0.30, 0.88],
    'X1-Y3-Z2-FILLET': [0.65, 0.45, 0.88],
    'X1-Y4-Z1': [0.80, 0.45, 0.88],
    'X1-Y4-Z2': [0.95, 0.45, 0.88],
    'X2-Y2-Z2': [0.72, 0.60, 0.88],
    'X2-Y2-Z2-FILLET': [0.88, 0.60, 0.88],
}

# Predefined target positions for each brick class (silhouette positions - PLACE area)
# Format: 'brick_type': [x, y, z] in world frame
# Place area is at higher Y values (y=0.70-1.00), same X range as pick area
TARGET_POSITIONS = {
    'X1-Y1-Z2': [0.55, 0.70, 0.89],      # Row 1, Col 1
    'X1-Y2-Z1': [0.70, 0.70, 0.89],      # Row 1, Col 2
    'X1-Y2-Z2': [0.85, 0.70, 0.89],      # Row 1, Col 3
    'X1-Y2-Z2-CHAMFER': [0.55, 0.80, 0.89],  # Row 2, Col 1
    'X1-Y2-Z2-TWINFILLET': [0.70, 0.80, 0.89],  # Row 2, Col 2
    'X1-Y3-Z2': [0.85, 0.80, 0.89],      # Row 2, Col 3
    'X1-Y3-Z2-FILLET': [0.55, 0.90, 0.89],  # Row 3, Col 1
    'X1-Y4-Z1': [0.70, 0.90, 0.89],      # Row 3, Col 2
    'X1-Y4-Z2': [0.85, 0.90, 0.89],      # Row 3, Col 3
    'X2-Y2-Z2': [0.60, 1.00, 0.89],      # Row 4, Col 1
    'X2-Y2-Z2-FILLET': [0.75, 1.00, 0.89],  # Row 4, Col 2
}

# Brick types that will be spawned - derived from TARGET_POSITIONS
allowed_brick_types = list(TARGET_POSITIONS.keys())

# Gripper type
gripper = True
soft_gripper = True
robotiq_gripper = False

SLOW_FACTOR = 1.0

# Alias for compatibility
home_joint_config = q0
