"""
Motion Planner for UR5 - Full analytical inverse kinematics
Based on ur5Inverse.m from class (Prof. Luigi Palopoli)
"""

import numpy as np
import rospy


class MotionPlanner:
    """
    Motion planner using full analytical IK for UR5
    """

    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

        # Pre-defined joint configurations
        self.home_joints = np.array(config.home_joint_config)

        # UR5 DH parameters from ur5Inverse.m (meters)
        # A = link lengths, D = link offsets, Alpha = twist angles
        self.A = np.array([0, -0.425, -0.3922, 0, 0, 0])
        self.D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        self.Alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

        # Robot base position in world frame (from Gazebo - ur5 mounted on tavolo)
        self.robot_base_x = 0.5
        self.robot_base_y = 0.35
        self.robot_base_z = 1.75

        # UR5 joint limits (radians)
        self.joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_upper = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])

        rospy.loginfo("Motion planner initialized (full analytical IK mode)")

    def _dh_transform(self, theta, alpha, d, a):
        """
        Compute DH transformation matrix.

        Args:
            theta: Joint angle
            alpha: Link twist
            d: Link offset
            a: Link length

        Returns:
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def _almzero(self, x):
        """Check if value is approximately zero."""
        return np.abs(x) < 1e-7

    def ur5_inverse(self, p60, R60):
        """
        Full analytical inverse kinematics for UR5.
        Ported from ur5Inverse.m (Prof. Luigi Palopoli)

        Args:
            p60: End-effector position [x, y, z] in robot base frame
            R60: End-effector rotation matrix (3x3)

        Returns:
            6x8 matrix of joint angles (8 possible solutions), or None if unreachable
        """
        A = self.A
        D = self.D
        Alpha = self.Alpha

        # Build T60 (end-effector pose)
        T60 = np.eye(4)
        T60[:3, :3] = R60
        T60[:3, 3] = p60

        # Finding th1: compute wrist center p50
        p50_h = T60 @ np.array([0, 0, -D[5], 1])
        p50 = p50_h[:3]

        psi = np.arctan2(p50[1], p50[0])
        p50xy = np.hypot(p50[1], p50[0])

        if p50xy < D[3]:
            rospy.logwarn("Position in unreachable cylinder")
            return None

        phi1_1 = np.arccos(D[3] / p50xy)
        phi1_2 = -phi1_1

        th1_1 = psi + phi1_1 + np.pi/2
        th1_2 = psi + phi1_2 + np.pi/2

        # Finding th5
        p61z_1 = p60[0]*np.sin(th1_1) - p60[1]*np.cos(th1_1)
        p61z_2 = p60[0]*np.sin(th1_2) - p60[1]*np.cos(th1_2)

        # Check for valid acos arguments
        arg5_1 = (p61z_1 - D[3]) / D[5]
        arg5_2 = (p61z_2 - D[3]) / D[5]

        if np.abs(arg5_1) > 1 or np.abs(arg5_2) > 1:
            rospy.logwarn("th5 argument out of range")
            return None

        th5_1_1 = np.arccos(np.clip(arg5_1, -1, 1))
        th5_1_2 = -th5_1_1
        th5_2_1 = np.arccos(np.clip(arg5_2, -1, 1))
        th5_2_2 = -th5_2_1

        # Compute T10 for both th1 values
        T10_1 = self._dh_transform(th1_1, Alpha[0], D[0], A[0])
        T10_2 = self._dh_transform(th1_2, Alpha[0], D[0], A[0])

        T16_1 = np.linalg.inv(np.linalg.inv(T10_1) @ T60)
        T16_2 = np.linalg.inv(np.linalg.inv(T10_2) @ T60)

        # Finding th6
        def compute_th6(T16, th5):
            zy = T16[1, 2]
            zx = T16[0, 2]
            if self._almzero(np.sin(th5)) or (self._almzero(zy) and self._almzero(zx)):
                return 0  # Singular configuration
            return np.arctan2(-zy/np.sin(th5), zx/np.sin(th5))

        th6_1_1 = compute_th6(T16_1, th5_1_1)
        th6_1_2 = compute_th6(T16_1, th5_1_2)
        th6_2_1 = compute_th6(T16_2, th5_2_1)
        th6_2_2 = compute_th6(T16_2, th5_2_2)

        T61_1 = np.linalg.inv(T16_1)
        T61_2 = np.linalg.inv(T16_2)

        # Compute T54 and T65 for all combinations
        T54_1_1 = self._dh_transform(th5_1_1, Alpha[4], D[4], A[4])
        T54_1_2 = self._dh_transform(th5_1_2, Alpha[4], D[4], A[4])
        T54_2_1 = self._dh_transform(th5_2_1, Alpha[4], D[4], A[4])
        T54_2_2 = self._dh_transform(th5_2_2, Alpha[4], D[4], A[4])

        T65_1_1 = self._dh_transform(th6_1_1, Alpha[5], D[5], A[5])
        T65_1_2 = self._dh_transform(th6_1_2, Alpha[5], D[5], A[5])
        T65_2_1 = self._dh_transform(th6_2_1, Alpha[5], D[5], A[5])
        T65_2_2 = self._dh_transform(th6_2_2, Alpha[5], D[5], A[5])

        T41_1_1 = T61_1 @ np.linalg.inv(T54_1_1 @ T65_1_1)
        T41_1_2 = T61_1 @ np.linalg.inv(T54_1_2 @ T65_1_2)
        T41_2_1 = T61_2 @ np.linalg.inv(T54_2_1 @ T65_2_1)
        T41_2_2 = T61_2 @ np.linalg.inv(T54_2_2 @ T65_2_2)

        # Compute P31 for all combinations
        def compute_P31(T41):
            P = T41 @ np.array([0, -D[3], 0, 1])
            return P[:3]

        P31_1_1 = compute_P31(T41_1_1)
        P31_1_2 = compute_P31(T41_1_2)
        P31_2_1 = compute_P31(T41_2_1)
        P31_2_2 = compute_P31(T41_2_2)

        # Finding th3 for all combinations
        def compute_th3(P31):
            C = (np.linalg.norm(P31)**2 - A[1]**2 - A[2]**2) / (2*A[1]*A[2])
            if np.abs(C) > 1:
                return np.nan, np.nan
            th3_1 = np.arccos(C)
            th3_2 = -th3_1
            return th3_1, th3_2

        th3_1_1_1, th3_1_1_2 = compute_th3(P31_1_1)
        th3_1_2_1, th3_1_2_2 = compute_th3(P31_1_2)
        th3_2_1_1, th3_2_1_2 = compute_th3(P31_2_1)
        th3_2_2_1, th3_2_2_2 = compute_th3(P31_2_2)

        # Finding th2 for all combinations
        def compute_th2(P31, th3):
            if np.isnan(th3):
                return np.nan
            return -np.arctan2(P31[1], -P31[0]) + np.arcsin((A[2]*np.sin(th3))/np.linalg.norm(P31))

        th2_1_1_1 = compute_th2(P31_1_1, th3_1_1_1)
        th2_1_1_2 = compute_th2(P31_1_1, th3_1_1_2)
        th2_1_2_1 = compute_th2(P31_1_2, th3_1_2_1)
        th2_1_2_2 = compute_th2(P31_1_2, th3_1_2_2)
        th2_2_1_1 = compute_th2(P31_2_1, th3_2_1_1)
        th2_2_1_2 = compute_th2(P31_2_1, th3_2_1_2)
        th2_2_2_1 = compute_th2(P31_2_2, th3_2_2_1)
        th2_2_2_2 = compute_th2(P31_2_2, th3_2_2_2)

        # Finding th4 for all combinations
        def compute_th4(th2, th3, T41):
            if np.isnan(th2) or np.isnan(th3):
                return np.nan
            T21 = self._dh_transform(th2, Alpha[1], D[1], A[1])
            T32 = self._dh_transform(th3, Alpha[2], D[2], A[2])
            T43 = np.linalg.inv(T21 @ T32) @ T41
            return np.arctan2(T43[1, 0], T43[0, 0])

        th4_1_1_1 = compute_th4(th2_1_1_1, th3_1_1_1, T41_1_1)
        th4_1_1_2 = compute_th4(th2_1_1_2, th3_1_1_2, T41_1_1)
        th4_1_2_1 = compute_th4(th2_1_2_1, th3_1_2_1, T41_1_2)
        th4_1_2_2 = compute_th4(th2_1_2_2, th3_1_2_2, T41_1_2)
        th4_2_1_1 = compute_th4(th2_2_1_1, th3_2_1_1, T41_2_1)
        th4_2_1_2 = compute_th4(th2_2_1_2, th3_2_1_2, T41_2_1)
        th4_2_2_1 = compute_th4(th2_2_2_1, th3_2_2_1, T41_2_2)
        th4_2_2_2 = compute_th4(th2_2_2_2, th3_2_2_2, T41_2_2)

        # Build solution matrix (6 joints x 8 solutions)
        Th = np.array([
            [th1_1,     th1_1,     th1_1,     th1_1,     th1_2,     th1_2,     th1_2,     th1_2],
            [th2_1_1_1, th2_1_1_2, th2_1_2_1, th2_1_2_2, th2_2_1_1, th2_2_1_2, th2_2_2_1, th2_2_2_2],
            [th3_1_1_1, th3_1_1_2, th3_1_2_1, th3_1_2_2, th3_2_1_1, th3_2_1_2, th3_2_2_1, th3_2_2_2],
            [th4_1_1_1, th4_1_1_2, th4_1_2_1, th4_1_2_2, th4_2_1_1, th4_2_1_2, th4_2_2_1, th4_2_2_2],
            [th5_1_1,   th5_1_1,   th5_1_2,   th5_1_2,   th5_2_1,   th5_2_1,   th5_2_2,   th5_2_2],
            [th6_1_1,   th6_1_1,   th6_1_2,   th6_1_2,   th6_2_1,   th6_2_1,   th6_2_2,   th6_2_2]
        ])

        return Th

    def select_best_solution(self, solutions, reference=None):
        """
        Select the best IK solution from multiple possibilities.

        Args:
            solutions: 6x8 matrix of joint solutions
            reference: Reference joint configuration (default: home)

        Returns:
            Best 6-element joint array, or None if no valid solution
        """
        if solutions is None:
            return None

        if reference is None:
            reference = self.home_joints

        best_solution = None
        best_distance = float('inf')

        for i in range(solutions.shape[1]):
            sol = solutions[:, i]

            # Skip solutions with NaN
            if np.any(np.isnan(sol)):
                continue

            # Check joint limits
            if not self._check_joint_limits(sol):
                continue

            # Compute distance to reference (infinity norm)
            distance = np.max(np.abs(sol - reference))

            if distance < best_distance:
                best_distance = distance
                best_solution = sol.copy()

        return best_solution

    def gripper_down_rotation(self):
        """
        Get rotation matrix for gripper pointing straight down (-Z).
        The gripper's Z-axis points down, X-axis points forward in robot frame.
        """
        # Gripper pointing down: Z points to -Z_world, X points to +X_world
        R = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        return R

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
        Execute pick sequence using actual object position

        Args:
            object_pos: Object position [x, y, z]
            controller: Controller instance
        """
        rospy.loginfo(f"Picking object at position: {object_pos}")

        # Compute approach position (above object)
        approach_pos = np.array([
            object_pos[0],
            object_pos[1],
            object_pos[2] + self.config.approach_height
        ])
        
        # Compute grasp position (at object)
        grasp_pos = np.array([
            object_pos[0],
            object_pos[1],
            object_pos[2] + self.config.grasp_height
        ])

        # Move to approach position
        rospy.loginfo(f"  Moving to approach position: {approach_pos}")
        approach_joints = self.simple_ik(approach_pos, gripper_down=True)
        if approach_joints is None:
            rospy.logerr("Failed to compute approach IK")
            return False
        self.move_to_joints(approach_joints, controller, duration=2.0)

        # Open gripper
        rospy.loginfo("  Opening gripper...")
        controller.send_gripper_command(self.config.gripper_open_pos)
        rospy.sleep(1.0)

        # Move down to grasp
        rospy.loginfo(f"  Moving down to grasp: {grasp_pos}")
        grasp_joints = self.simple_ik(grasp_pos, gripper_down=True)
        if grasp_joints is None:
            rospy.logerr("Failed to compute grasp IK")
            return False
        self.move_to_joints(grasp_joints, controller, duration=1.5)

        # Close gripper
        rospy.loginfo("  Closing gripper...")
        controller.send_gripper_command(self.config.gripper_close_pos)
        rospy.sleep(1.5)

        # Lift object
        rospy.loginfo("  Lifting object...")
        self.move_to_joints(approach_joints, controller, duration=1.5)

        rospy.loginfo("Pick complete")
        return True

    def place_object(self, target_pos, controller):
        """
        Execute place sequence using actual target position

        Args:
            target_pos: Target position [x, y, z]
            controller: Controller instance
        """
        rospy.loginfo(f"Placing object at position: {target_pos}")

        # Compute place approach position (above target)
        place_approach_pos = np.array([
            target_pos[0],
            target_pos[1],
            target_pos[2] + self.config.approach_height
        ])
        
        # Compute place position (at target surface)
        place_pos = np.array([
            target_pos[0],
            target_pos[1],
            target_pos[2] + self.config.place_height
        ])

        # Move to place approach
        rospy.loginfo(f"  Moving to place approach: {place_approach_pos}")
        place_approach_joints = self.simple_ik(place_approach_pos, gripper_down=True)
        if place_approach_joints is None:
            rospy.logerr("Failed to compute place approach IK")
            return False
        self.move_to_joints(place_approach_joints, controller, duration=2.0)

        # Move down to place
        rospy.loginfo(f"  Moving down to place: {place_pos}")
        place_joints = self.simple_ik(place_pos, gripper_down=True)
        if place_joints is None:
            rospy.logerr("Failed to compute place IK")
            return False
        self.move_to_joints(place_joints, controller, duration=1.5)

        # Open gripper
        rospy.loginfo("  Opening gripper...")
        controller.send_gripper_command(self.config.gripper_open_pos)
        rospy.sleep(1.5)

        # Lift up
        rospy.loginfo("  Lifting after place...")
        self.move_to_joints(place_approach_joints, controller, duration=1.5)

        rospy.loginfo("Place complete")
        return True

    def simple_ik(self, target_pos, gripper_down=True):
        """
        Inverse kinematics for UR5 using full analytical solution.
        Uses ur5_inverse() and selects the best solution.

        Args:
            target_pos: Target XYZ position [x, y, z] in WORLD frame
            gripper_down: If True, orient gripper downward (for picking)

        Returns:
            Joint angles [6] or None if unreachable
        """
        x, y, z = target_pos

        # Transform from world frame to robot base frame
        # Robot base is at (0.5, 0.35, 1.75) in world frame
        x_world_rel = x - self.robot_base_x
        y_world_rel = y - self.robot_base_y
        z_robot = z - self.robot_base_z

        # Robot base frame is rotated -90° from world frame:
        # - Robot q1=0 points towards world +Y
        # - Robot q1=90° points towards world +X
        # So we swap: robot_x = world_y, robot_y = world_x
        p_robot = np.array([y_world_rel, x_world_rel, z_robot])

        rospy.loginfo(f"IK: target world=[{x:.3f}, {y:.3f}, {z:.3f}]")
        rospy.loginfo(f"IK: target robot frame=[{p_robot[0]:.3f}, {p_robot[1]:.3f}, {p_robot[2]:.3f}]")

        # Get rotation matrix for desired orientation
        if gripper_down:
            R = self.gripper_down_rotation()
        else:
            R = np.eye(3)

        # Compute all 8 IK solutions
        solutions = self.ur5_inverse(p_robot, R)

        if solutions is None:
            rospy.logwarn("IK: No solutions found (target unreachable)")
            return None

        # Count valid solutions
        valid_count = np.sum(~np.any(np.isnan(solutions), axis=0))
        rospy.loginfo(f"IK: Found {valid_count}/8 valid solutions")

        # Select the best solution (closest to home configuration)
        best = self.select_best_solution(solutions)

        if best is None:
            rospy.logwarn("IK: No valid solution within joint limits")
            return None

        rospy.loginfo(f"IK: Best solution (deg): [{np.degrees(best[0]):.1f}, {np.degrees(best[1]):.1f}, {np.degrees(best[2]):.1f}, {np.degrees(best[3]):.1f}, {np.degrees(best[4]):.1f}, {np.degrees(best[5]):.1f}]")

        return best

    def _check_joint_limits(self, joints):
        """Check if joint angles are within limits"""
        for i, (q, lower, upper) in enumerate(zip(joints, self.joint_limits_lower, self.joint_limits_upper)):
            if q < lower or q > upper:
                rospy.logwarn(f"Joint {i+1} out of limits: {np.degrees(q):.1f}° (limits: {np.degrees(lower):.1f}° to {np.degrees(upper):.1f}°)")
                return False
        return True

    def compute_ik(self, target_pose, initial_guess=None):
        """
        IK computation using simple geometric approach
        """
        return self.simple_ik(target_pose[:3])

    def move_to_pose(self, target_pos, target_orient, controller, duration=3.0):
        """
        Move to Cartesian pose using IK
        """
        joints = self.simple_ik(target_pos, gripper_down=True)
        if joints is None:
            return False
        return self.move_to_joints(joints, controller, duration)
