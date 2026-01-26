"""
@file motion_planner.py
@brief Motion Planner for UR5 - Full analytical inverse kinematics
@author PAPA Team (Lilla, Benjamin, David)
@date 2024

Based on ur5Inverse.m from class (Prof. Luigi Palopoli).
Implements complete analytical IK with 8 solutions, joint limit avoidance,
and smooth quintic trajectory interpolation.
"""

import numpy as np
import rospy


class MotionPlanner:
    """
    @brief Motion planner using full analytical IK for UR5 robot.

    Provides inverse kinematics, trajectory planning, and pick-and-place
    motion sequences for the UR5 manipulator arm.

    @note Robot is mounted INVERTED (hanging from ceiling), so DH frame
          +Z points DOWN in world frame.
    """

    def __init__(self, controller, config):
        """
        @brief Initialize the motion planner.

        @param controller Reference to the main controller instance
        @param config Configuration object with robot parameters
        """
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
        # Note: Joint 3 (elbow) has tighter limits [-π, π]
        self.joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_upper = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])

        # Precompute joint middle positions and ranges for joint limit avoidance
        self.joint_middle = (self.joint_limits_lower + self.joint_limits_upper) / 2.0
        self.joint_range = self.joint_limits_upper - self.joint_limits_lower

        # Track last commanded position (since get_current_joint_state returns zeros)
        self.last_commanded_joints = np.array(config.home_joint_config)

        rospy.loginfo("Motion planner initialized (full analytical IK mode)")

    def _dh_transform(self, theta, alpha, d, a):
        """
        @brief Compute standard DH (Denavit-Hartenberg) transformation matrix.

        @param theta Joint angle (rotation about Z)
        @param alpha Link twist (rotation about X)
        @param d Link offset (translation along Z)
        @param a Link length (translation along X)
        @return 4x4 homogeneous transformation matrix
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
        """
        @brief Check if value is approximately zero.

        @param x Value to check
        @return True if |x| < 1e-7
        """
        return np.abs(x) < 1e-7

    def ur5_inverse(self, p60, R60):
        """
        @brief Full analytical inverse kinematics for UR5.

        Computes all 8 possible joint configurations that achieve the
        desired end-effector pose. Ported from ur5Inverse.m (Prof. Luigi Palopoli).

        @param p60 End-effector position [x, y, z] in robot base (DH) frame
        @param R60 End-effector rotation matrix (3x3)
        @return 6x8 matrix of joint angles (8 possible solutions), or None if unreachable

        @note Returns None if position is in unreachable cylinder (XY distance < D[3])
        @warning Some solutions may contain NaN if geometrically impossible
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

    def _joint_limit_cost(self, joints):
        """
        @brief Compute cost function for joint limit avoidance (nullspace optimization).

        Implements the cost function:
        w(q) = (1/2n) * Σᵢ ((qᵢ - q̄ᵢ)/(qᵢₘ - qᵢₘ))²

        Where:
        - q̄ᵢ = middle of joint range
        - qᵢₘ, qᵢₘ = min/max joint limits
        - n = number of joints

        @param joints Array of 6 joint angles in radians
        @return Cost value (0 when all joints at middle, increases toward limits)

        @note This function is used in solution selection to prefer configurations
              that stay away from joint limits.
        """
        n = len(joints)
        cost = 0.0
        for i in range(n):
            if self.joint_range[i] > 0:  # Avoid division by zero
                normalized_deviation = (joints[i] - self.joint_middle[i]) / self.joint_range[i]
                cost += normalized_deviation ** 2
        return cost / (2.0 * n)

    def _angular_distance(self, angle1, angle2):
        """
        @brief Compute the shortest angular distance between two angles.

        @param angle1 First angle in radians
        @param angle2 Second angle in radians
        @return Absolute shortest distance in radians [0, π]

        @note Handles wrapping around 2π correctly
        """
        diff = angle1 - angle2
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return np.abs(diff)

    def _normalize_to_reference(self, solution, reference):
        """
        @brief Normalize solution angles to be close to reference configuration.

        Adds/subtracts 2π to each joint angle to minimize distance from reference,
        while respecting joint limits (especially joint 3 which has [-π, π] limit).

        @param solution Array of 6 joint angles from IK solver
        @param reference Reference joint configuration (typically home)
        @return Normalized solution with angles close to reference

        @note This ensures the robot takes the SHORT path when interpolating
              between configurations.
        """
        normalized = solution.copy()
        for i in range(6):
            # Try to get closest to reference by adding/subtracting 2π
            # But don't violate joint limits
            best_angle = normalized[i]
            best_distance = abs(normalized[i] - reference[i])

            # Try adding 2π
            candidate = normalized[i] + 2 * np.pi
            if self.joint_limits_lower[i] <= candidate <= self.joint_limits_upper[i]:
                dist = abs(candidate - reference[i])
                if dist < best_distance:
                    best_angle = candidate
                    best_distance = dist

            # Try subtracting 2π
            candidate = normalized[i] - 2 * np.pi
            if self.joint_limits_lower[i] <= candidate <= self.joint_limits_upper[i]:
                dist = abs(candidate - reference[i])
                if dist < best_distance:
                    best_angle = candidate
                    best_distance = dist

            normalized[i] = best_angle
        return normalized

    def select_best_solution(self, solutions, reference=None):
        """
        @brief Select the best IK solution from multiple possibilities.

        Uses multiple criteria to select the best solution:
        1. Joint limits check (reject invalid solutions)
        2. Q1 constraint (prevent arm reaching from wrong side) - 90 degrees
        3. Distance to reference configuration
        4. Joint limit avoidance cost (prefer configurations away from limits)

        @param solutions 6x8 matrix of joint solutions from ur5_inverse()
        @param reference Reference joint configuration (default: home)
        @return Best 6-element joint array, or None if no valid solution

        @note Solutions are normalized before comparison to ensure short interpolation paths
        """
        if solutions is None:
            return None

        # Always use HOME as reference for IK selection
        # This ensures consistent behavior and prevents wild swings
        if reference is None:
            reference = self.home_joints

        best_solution = None
        best_distance = float('inf')

        # Q1 constraint: 180 degrees from home (full front hemisphere)
        # Relaxed to allow reaching objects across the table
        max_q1_deviation = np.pi  # 180 degrees
        
        # Debug: track rejection reasons
        rejections = {'nan': 0, 'limits': 0, 'q1': 0, 'elbow': 0}

        for i in range(solutions.shape[1]):
            sol = solutions[:, i]

            # Skip solutions with NaN
            if np.any(np.isnan(sol)):
                rejections['nan'] += 1
                continue

            # Normalize solution to be close to reference
            sol_normalized = self._normalize_to_reference(sol, reference)

            # Check joint limits
            if not self._check_joint_limits(sol_normalized):
                rejections['limits'] += 1
                continue

            # Q1 constraint: reject if base rotation too far from home
            q1_distance = np.abs(sol_normalized[0] - reference[0])
            if q1_distance > max_q1_deviation:
                rejections['q1'] += 1
                continue
            
            # Elbow-up preference for inverted robot:
            # Q2 (shoulder_lift) should be negative (arm reaching forward/down in DH frame)
            # Q3 (elbow) should be negative (elbow bent inward)
            # Reject configurations where Q2 > 0.5 rad (~30 deg) which would swing elbow low
            if sol_normalized[1] > 0.5:  # Q2 > ~30 degrees means bad config
                rejections['elbow'] += 1
                continue

            # Compute distance to reference
            distances = [np.abs(sol_normalized[j] - reference[j]) for j in range(6)]
            ref_distance = np.sum(distances)

            # Joint limit avoidance cost
            limit_cost = self._joint_limit_cost(sol_normalized)

            # Combined score
            combined_score = ref_distance + 2.0 * limit_cost

            if combined_score < best_distance:
                best_distance = combined_score
                best_solution = sol_normalized.copy()

        # Log rejection reasons if no valid solution found
        if best_solution is None:
            total_rejected = sum(rejections.values())
            rospy.logwarn(f"IK: All {total_rejected} solutions rejected - NaN:{rejections['nan']}, Limits:{rejections['limits']}, Q1:{rejections['q1']}, Elbow:{rejections['elbow']}")
        
        return best_solution

    def gripper_down_rotation(self):
        """
        @brief Get rotation matrix for gripper pointing straight down.

        @return 3x3 identity rotation matrix

        @note In DH frame, +Z points down (robot is inverted).
              Identity matrix means end-effector Z aligns with base Z (down).
        """
        R = np.eye(3)
        return R

    def _quintic_interpolation(self, t, duration):
        """
        @brief Quintic polynomial interpolation for smooth motion profiles.

        Computes s(t) = 10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5

        This polynomial has the properties:
        - s(0) = 0, s(T) = 1
        - s'(0) = 0, s'(T) = 0 (zero velocity at start/end)
        - s''(0) = 0, s''(T) = 0 (zero acceleration at start/end)

        @param t Current time in seconds
        @param duration Total duration of motion in seconds
        @return Interpolation factor in [0, 1]
        """
        tau = t / duration  # Normalized time [0, 1]
        return 10 * tau**3 - 15 * tau**4 + 6 * tau**5

    def move_to_joints(self, target_joints, controller, duration=3.0):
        """
        @brief Move to target joint configuration.

        @param target_joints Target joint angles [6] in radians
        @param controller Controller instance for sending commands
        @param duration Movement duration in seconds (default: 3.0)
        @return True on success

        @note Sends target position repeatedly. The robot's PD controller
              handles smooth motion internally.
        """
        rospy.loginfo(f"Moving to joint configuration over {duration}s")
        rospy.loginfo(f"  Target (deg): [{', '.join([f'{np.degrees(j):.1f}' for j in target_joints[:6]])}]")

        # Ensure target is numpy array
        target = np.array(target_joints[:6])

        # Send target position repeatedly for the duration
        # The PD controller will move the robot smoothly
        rate = rospy.Rate(100)  # 100 Hz
        steps = int(duration * 100)

        for i in range(steps + 1):
            if rospy.is_shutdown():
                break
            controller.send_joint_command(target)
            rate.sleep()

        # Update tracked position
        self.last_commanded_joints = target.copy()

        rospy.loginfo("Movement complete")
        return True

    def pick_object(self, object_pos, controller, robot_relative=True):
        """
        @brief Execute complete pick sequence for an object.

        Performs the following steps:
        1. Move to safe waypoint (high above object)
        2. Move down to approach position
        3. Open gripper
        4. Move down to grasp position
        5. Close gripper
        6. Lift to safe height

        @param object_pos Object position [x, y, z] in robot-relative or world frame
        @param controller Controller instance for sending commands
        @param robot_relative If True, object_pos is relative to robot base (default)
        @return True on success, False on IK failure

        @warning Grasp Z is clamped to table surface to prevent collision
        @note Table surface is at Z=-0.90 in robot-relative frame
        """
        rospy.loginfo(f"Picking object at position: {object_pos} (robot_relative={robot_relative})")

        # Table plane constraint (robot-relative Z coordinate)
        # Robot base at Z=1.75, table surface at Z=0.85, so table is at -0.9 in robot frame
        # Add small margin to never touch table
        min_z_robot_relative = -0.89  # 1cm above table surface

        # Safe height - use absolute Z in robot frame to guarantee clearance
        # Robot is at Z=1.75, table at Z=0.85 (0.9m below robot base)
        # Safe waypoint should be at least 0.5m below robot base (0.4m above table)
        safe_z_robot_relative = -0.45  # About 40cm above table

        # Compute safe waypoint position (high above object - use absolute safe Z)
        safe_pos = np.array([
            object_pos[0],
            object_pos[1],
            safe_z_robot_relative  # Use absolute safe height, not relative to object
        ])

        # Compute approach position (above object, but ensure minimum clearance from table)
        approach_z = object_pos[2] + self.config.approach_height
        min_approach_z = -0.75  # At least 15cm above table (table is at -0.9)
        if robot_relative and approach_z < min_approach_z:
            approach_z = min_approach_z
            
        approach_pos = np.array([
            object_pos[0],
            object_pos[1],
            approach_z
        ])

        # Compute grasp position (at object) - but clamp to not go below table!
        grasp_z = object_pos[2] + self.config.grasp_height
        if robot_relative and grasp_z < min_z_robot_relative:
            rospy.logwarn(f"Grasp Z={grasp_z:.3f} below table! Clamping to {min_z_robot_relative}")
            grasp_z = min_z_robot_relative

        grasp_pos = np.array([
            object_pos[0],
            object_pos[1],
            grasp_z
        ])

        # Step 1: Move to safe waypoint (high above object, avoids table)
        rospy.loginfo(f"  Moving to safe waypoint: {safe_pos}")
        safe_joints = self.simple_ik(safe_pos, gripper_down=True, robot_relative=robot_relative)
        if safe_joints is None:
            rospy.logerr("Failed to compute safe waypoint IK")
            return False
        self.move_to_joints(safe_joints, controller, duration=2.0)

        # Step 2: Move down to approach position
        rospy.loginfo(f"  Moving to approach position: {approach_pos}")
        approach_joints = self.simple_ik(approach_pos, gripper_down=True, robot_relative=robot_relative)
        if approach_joints is None:
            rospy.logerr("Failed to compute approach IK")
            return False
        self.move_to_joints(approach_joints, controller, duration=1.5)

        # Open gripper
        rospy.loginfo("  Opening gripper...")
        controller.send_gripper_command(self.config.gripper_open_pos)
        rospy.sleep(1.0)

        # Move down to grasp
        rospy.loginfo(f"  Moving down to grasp: {grasp_pos}")
        grasp_joints = self.simple_ik(grasp_pos, gripper_down=True, robot_relative=robot_relative)
        if grasp_joints is None:
            rospy.logerr("Failed to compute grasp IK")
            return False
        self.move_to_joints(grasp_joints, controller, duration=1.5)

        # Close gripper
        rospy.loginfo("  Closing gripper...")
        controller.send_gripper_command(self.config.gripper_close_pos)
        rospy.sleep(1.5)

        # Lift object to safe height (avoids collisions when moving to place)
        rospy.loginfo("  Lifting object to safe height...")
        self.move_to_joints(safe_joints, controller, duration=1.5)

        rospy.loginfo("Pick complete")
        return True

    def place_object(self, target_pos, controller, robot_relative=True):
        """
        @brief Execute complete place sequence at target position.

        Performs the following steps:
        1. Move to safe waypoint (high above target)
        2. Move down to place approach position
        3. Move down to place position
        4. Open gripper
        5. Lift up

        @param target_pos Target position [x, y, z] in robot-relative or world frame
        @param controller Controller instance for sending commands
        @param robot_relative If True, target_pos is relative to robot base (default)
        @return True on success, False on IK failure
        """
        rospy.loginfo(f"Placing object at position: {target_pos} (robot_relative={robot_relative})")

        # Safe height for waypoint
        safe_height = 0.30  # 30cm above target

        # Compute safe waypoint position
        safe_pos = np.array([
            target_pos[0],
            target_pos[1],
            target_pos[2] + safe_height
        ])

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

        # Step 1: Move to safe waypoint above target
        rospy.loginfo(f"  Moving to safe waypoint: {safe_pos}")
        safe_joints = self.simple_ik(safe_pos, gripper_down=True, robot_relative=robot_relative)
        if safe_joints is None:
            rospy.logerr("Failed to compute safe waypoint IK")
            return False
        self.move_to_joints(safe_joints, controller, duration=2.0)

        # Step 2: Move down to place approach
        rospy.loginfo(f"  Moving to place approach: {place_approach_pos}")
        place_approach_joints = self.simple_ik(place_approach_pos, gripper_down=True, robot_relative=robot_relative)
        if place_approach_joints is None:
            rospy.logerr("Failed to compute place approach IK")
            return False
        self.move_to_joints(place_approach_joints, controller, duration=1.5)

        # Move down to place
        rospy.loginfo(f"  Moving down to place: {place_pos}")
        place_joints = self.simple_ik(place_pos, gripper_down=True, robot_relative=robot_relative)
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

    def simple_ik(self, target_pos, gripper_down=True, robot_relative=False):
        """
        @brief High-level inverse kinematics interface.

        Computes joint angles for a target Cartesian position using the full
        analytical IK solver, then selects the best solution.

        @param target_pos Target XYZ position [x, y, z]
        @param gripper_down If True, orient gripper downward (default: True)
        @param robot_relative If True, target_pos is relative to robot base
                              If False, target_pos is in world frame (default)
        @return Joint angles [6] in radians, or None if unreachable

        @note Automatically applies coordinate transform for inverted robot mounting
        @see ur5_inverse(), select_best_solution()
        """
        x, y, z = target_pos

        if robot_relative:
            # Input is already relative to robot base (from perception module)
            x_rel = x
            y_rel = y
            z_rel = z
        else:
            # Transform from world frame to robot base frame
            # Robot base is at (0.5, 0.35, 1.75) in world frame
            x_rel = x - self.robot_base_x
            y_rel = y - self.robot_base_y
            z_rel = z - self.robot_base_z

        # Coordinate transform from world-relative to DH frame:
        # - Robot DH +X axis points in world +Y direction (towards table)
        # - Robot DH +Y axis points in world -X direction (to the left)
        # - Robot DH +Z axis points DOWN (robot is inverted)
        # So: DH_x = world_y, DH_y = -world_x, DH_z = -world_z
        p_robot = np.array([y_rel, -x_rel, -z_rel])

        if robot_relative:
            rospy.loginfo(f"IK: target (robot-relative)=[{x:.3f}, {y:.3f}, {z:.3f}]")
        else:
            rospy.loginfo(f"IK: target (world)=[{x:.3f}, {y:.3f}, {z:.3f}]")
        rospy.loginfo(f"IK: robot DH frame=[{p_robot[0]:.3f}, {p_robot[1]:.3f}, {p_robot[2]:.3f}]")

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
        """
        @brief Check if joint angles are within physical limits.

        @param joints Array of 6 joint angles in radians
        @return True if all joints within limits, False otherwise

        @note Logs warning message indicating which joint is out of limits
        """
        for i, (q, lower, upper) in enumerate(zip(joints, self.joint_limits_lower, self.joint_limits_upper)):
            if q < lower or q > upper:
                rospy.logwarn(f"Joint {i+1} out of limits: {np.degrees(q):.1f}° (limits: {np.degrees(lower):.1f}° to {np.degrees(upper):.1f}°)")
                return False
        return True

    def compute_ik(self, target_pose, initial_guess=None):
        """
        @brief Legacy IK interface for compatibility.

        @param target_pose Target pose (only position [0:3] is used)
        @param initial_guess Ignored (for interface compatibility)
        @return Joint angles [6] or None if unreachable

        @deprecated Use simple_ik() instead
        """
        return self.simple_ik(target_pose[:3])

    def move_to_pose(self, target_pos, target_orient, controller, duration=3.0):
        """
        @brief Move to Cartesian pose using IK.

        @param target_pos Target XYZ position [x, y, z]
        @param target_orient Target orientation (currently ignored, uses gripper_down)
        @param controller Controller instance
        @param duration Movement duration in seconds
        @return True on success, False on IK failure
        """
        joints = self.simple_ik(target_pos, gripper_down=True)
        if joints is None:
            return False
        return self.move_to_joints(joints, controller, duration)
