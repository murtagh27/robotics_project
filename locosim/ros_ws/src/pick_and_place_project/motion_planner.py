"""
Simplified Motion Planner - Uses joint-space waypoints instead of IK
This version doesn't require Pinocchio to work
"""

import numpy as np
import rospy


class MotionPlanner:
    """
    Simplified motion planner using pre-defined joint configurations
    """

    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

        # Pre-defined joint configurations (no IK needed)
        self.home_joints = np.array(config.home_joint_config)

        rospy.loginfo("Motion planner initialized (simplified mode)")

    def move_to_joints(self, target_joints, controller, duration=3.0):
        """
        Move to target joint configuration using smooth interpolation

        Args:
            target_joints: Target joint angles [6]
            controller: Controller instance
            duration: Movement duration in seconds
        """
        rospy.loginfo(f"Moving to joint configuration over {duration}s")

        # Get current joint state
        current_joints, _ = controller.get_current_joint_state()

        # Pad target joints to 8 if only 6 provided (add gripper)
        if len(target_joints) == 6:
            target_joints_full = np.concatenate([target_joints, current_joints[6:8]])
        else:
            target_joints_full = target_joints

        # Simple linear interpolation
        rate = rospy.Rate(100)  # 100 Hz
        steps = int(duration * 100)

        for i in range(steps + 1):
            if rospy.is_shutdown():
                break

            alpha = float(i) / steps  # 0 to 1

            # Interpolate between current and target
            desired_joints = current_joints + alpha * (target_joints_full - current_joints)

            # Send command
            controller.send_joint_command(desired_joints)

            rate.sleep()

        rospy.loginfo("Movement complete")
        return True

    def pick_object(self, object_pos, controller):
        """
        Execute pick sequence using hardcoded waypoints

        Args:
            object_pos: Object position [x, y, z]
            controller: Controller instance
        """
        rospy.loginfo(f"Picking object at position: {object_pos}")

        # TODO: For now, use predefined waypoints
        # In a full implementation, you'd compute these from object_pos

        # Move to approach position (above object)
        rospy.loginfo("  Moving to approach position...")
        approach_joints = np.array([-0.5, -1.0, -2.0, -1.5, -1.57, 0.0])
        self.move_to_joints(approach_joints, controller, duration=2.0)

        # Move down to grasp
        rospy.loginfo("  Moving down to grasp...")
        grasp_joints = np.array([-0.5, -0.8, -2.2, -1.3, -1.57, 0.0])
        self.move_to_joints(grasp_joints, controller, duration=1.5)

        # Close gripper
        rospy.loginfo("  Closing gripper...")
        controller.send_gripper_command(0.0)  # Close
        rospy.sleep(1.5)

        # Lift object
        rospy.loginfo("  Lifting object...")
        self.move_to_joints(approach_joints, controller, duration=1.5)

        rospy.loginfo("Pick complete")
        return True

    def place_object(self, target_pos, controller):
        """
        Execute place sequence using hardcoded waypoints

        Args:
            target_pos: Target position [x, y, z]
            controller: Controller instance
        """
        rospy.loginfo(f"Placing object at position: {target_pos}")

        # Move to place approach
        rospy.loginfo("  Moving to place approach...")
        place_approach_joints = np.array([0.5, -1.0, -2.0, -1.5, -1.57, 0.0])
        self.move_to_joints(place_approach_joints, controller, duration=2.0)

        # Move down to place
        rospy.loginfo("  Moving down to place...")
        place_joints = np.array([0.5, -0.8, -2.2, -1.3, -1.57, 0.0])
        self.move_to_joints(place_joints, controller, duration=1.5)

        # Open gripper
        rospy.loginfo("  Opening gripper...")
        controller.send_gripper_command(0.085)  # Open
        rospy.sleep(1.5)

        # Lift up
        rospy.loginfo("  Lifting after place...")
        self.move_to_joints(place_approach_joints, controller, duration=1.5)

        rospy.loginfo("Place complete")
        return True

    def compute_ik(self, target_pose, initial_guess=None):
        """
        Stub for IK computation (not implemented in simple version)
        """
        rospy.logwarn("IK computation not available in simplified mode")
        return None

    def move_to_pose(self, target_pos, target_orient, controller, duration=3.0):
        """
        Stub for Cartesian motion (not implemented in simple version)
        """
        rospy.logwarn("Cartesian motion not available in simplified mode")
        return False
