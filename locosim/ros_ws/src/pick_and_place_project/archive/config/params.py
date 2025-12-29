"""
Configuration parameters for pick and place project
"""

# Robot configuration
robot_name = 'ur5'  # Using UR5 manipulator
simulation = True  # Set to False when using real robot
use_ground_truth = True  # Set to False to use camera-based perception

# Gripper configuration
gripper_type = 'soft_gripper'  # Two-fingered soft gripper (options: 'soft_gripper' or '3finger_gripper')

# Table positions (in robot base frame)
table_initial = {
    'position': [0.5, -0.3, 0.0],  # x, y, z in meters
    'size': [0.6, 0.4, 0.02],  # length, width, height
}

table_final = {'position': [0.5, 0.3, 0.0], 'size': [0.6, 0.4, 0.02]}

# Object classes and their final positions
object_classes = {
    'cube_red': {
        'color': [1.0, 0.0, 0.0, 1.0],  # RGBA
        'size': [0.05, 0.05, 0.05],  # x, y, z dimensions
        'final_position': [0.4, 0.2, 0.02],  # Where to place on final table
    },
    'cube_blue': {
        'color': [0.0, 0.0, 1.0, 1.0],
        'size': [0.05, 0.05, 0.05],
        'final_position': [0.5, 0.2, 0.02],
    },
    'cube_green': {
        'color': [0.0, 1.0, 0.0, 1.0],
        'size': [0.05, 0.05, 0.05],
        'final_position': [0.6, 0.2, 0.02],
    },
    'cylinder_yellow': {
        'color': [1.0, 1.0, 0.0, 1.0],
        'radius': 0.025,
        'length': 0.08,
        'final_position': [0.4, 0.4, 0.02],
    },
}

# Motion planning parameters
approach_height = 0.15  # Height above object before grasping (meters)
grasp_height_offset = 0.02  # How much to lower from center to grasp
lift_height = 0.2  # Height to lift object after grasping

# Camera configuration (if using vision)
camera_topic = '/camera/depth/points'  # Point cloud topic
camera_frame = 'camera_link'

# Safety limits
max_velocity = 0.5  # m/s
max_acceleration = 1.0  # m/s^2

# Home position (joint angles in radians)
home_joint_config = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

# Perception parameters
min_object_points = 100  # Minimum points to consider valid object
segmentation_threshold = 0.02  # Distance threshold for plane segmentation (meters)
