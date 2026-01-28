"""
Motion planning for UR5 robot
Provides analytical IK and trajectory generation
"""

import numpy as np
import rospy


class CollisionChecker:
    """
    Basic collision checking for UR5 workspace
    Checks for:
    - Workspace boundaries
    - Table collisions
    - Ground plane
    """

    def __init__(self, config):
        self.config = config

        # Workspace limits (in world frame)
        self.workspace_limits = {'x': [0.0, 1.0], 'y': [0.0, 1.5], 'z': [0.0, 2.0]}

        # Table geometries
        self.source_table = {
            'center': np.array(config.source_table_pos),
            'size': np.array(getattr(config, 'source_table_size', [0.6, 0.4, 0.02])),
        }

        self.target_table = {
            'center': np.array(config.target_table_pos),
            'size': np.array(getattr(config, 'target_table_size', [0.6, 0.4, 0.02])),
        }

        # Safety margins
        self.margin = 0.05  # 5cm safety margin

    def check_workspace_bounds(self, position):
        """Check if position is within workspace"""
        x, y, z = position

        if not (self.workspace_limits['x'][0] <= x <= self.workspace_limits['x'][1]):
            return False, f"X position {x:.3f} outside workspace"
        if not (self.workspace_limits['y'][0] <= y <= self.workspace_limits['y'][1]):
            return False, f"Y position {y:.3f} outside workspace"
        if not (self.workspace_limits['z'][0] <= z <= self.workspace_limits['z'][1]):
            return False, f"Z position {z:.3f} outside workspace"

        return True, "Within workspace"

    def check_table_collision(self, position):
        """
        Check if position collides with tables.
        Only checks if going THROUGH table from below, not if above table surface.
        Target table area is not checked as collision (objects can be placed there).
        """
        x, y, z = position

        # Check source table only (not target - we place objects there!)
        table = self.source_table
        center = table['center']
        half_size = table['size'] / 2.0

        # Target area is at y >= 0.5, don't check collisions there
        if y >= 0.5:
            return True, "In target placement area"

        # Only check collision if position is BELOW the table surface
        table_top = center[2] + half_size[2]
        if z < table_top:
            # Check if point is inside table volume
            if (
                abs(x - center[0]) < half_size[0] + self.margin
                and abs(y - center[1]) < half_size[1] + self.margin
                and z > center[2] - half_size[2] - self.margin
            ):
                return False, f"Collision with source table"

        return True, "No table collision"

    def check_ground_collision(self, position):
        """Check if position is above ground"""
        if position[2] < self.margin:
            return False, f"Below ground plane (z={position[2]:.3f})"
        return True, "Above ground"

    def check_position(self, position):
        """Complete collision check for a position"""
        # Check workspace
        ok, msg = self.check_workspace_bounds(position)
        if not ok:
            return False, msg

        # Check tables (allow positions ON tables, just not inside)
        ok, msg = self.check_table_collision(position)
        if not ok:
            return False, msg

        # Check ground
        ok, msg = self.check_ground_collision(position)
        if not ok:
            return False, msg

        return True, "Position safe"

    def check_trajectory(self, waypoints):
        """Check if trajectory is collision-free"""
        for i, wp in enumerate(waypoints):
            ok, msg = self.check_position(wp)
            if not ok:
                return False, f"Waypoint {i}: {msg}"

        return True, "Trajectory safe"


class MotionPlanner:
    """
    Motion planner using full analytical IK for UR5
    """

    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

        # Pre-defined joint configurations
        self.home_joints = np.array(config.home_joint_config)

        # Robot base position in world frame
        # From Gazebo model_states: ur5 is at [0.5, 0.35, 1.75]
        # TF base_link is at origin [0,0,0] in its own frame
        # To convert Gazebo world coords to base_link: subtract robot position
        self.robot_base_x = 0.5
        self.robot_base_y = 0.35
        self.robot_base_z = 1.75  # Actual position from Gazebo, not 1.8!

        # UR5 joint limits (radians) - actual UR5 limits are ±360° for most joints
        # But we use continuous rotation limits since IK may return wrapped angles
        # Joint 3 has limited range (-180° to 180°)
        self.joint_limits_lower = np.array(
            [-2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi]
        )
        self.joint_limits_upper = np.array(
            [2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi]
        )

        # UR5 DH parameters (Standard DH from UR5 URDF)
        # From urdf: a = [0, -0.425, -0.39225, 0, 0, 0]
        #            d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        #            alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        self.A = np.array([0, -0.425, -0.39225, 0, 0, 0])
        self.D = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
        self.Alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])

        # URDF link lengths (different from DH - these are the actual offsets from URDF joints)
        # Used for urdf_fk which matches Gazebo exactly
        self.urdf_d1 = 0.089159  # shoulder_pan_joint: Z offset
        self.urdf_a2 = 0.13585  # shoulder_lift_joint: Y offset
        self.urdf_d3 = 0.425  # elbow_joint: Z offset (upper arm length)
        self.urdf_a3 = 0.1197  # elbow_joint: -Y offset
        self.urdf_d4 = 0.39225  # wrist_1_joint: Z offset (forearm length)
        self.urdf_a5 = 0.093  # wrist_2_joint: Y offset
        self.urdf_d6 = 0.09465  # wrist_3_joint: Z offset
        self.urdf_a7 = 0.0823  # tool0: Y offset

        # Collision checking
        self.collision_checker = CollisionChecker(config)
        self.enable_collision_checking = False  # Disabled - only checks targets, not paths

        rospy.loginfo("Motion planner initialized (analytical IK mode)")

    def urdf_fk(self, q):
        """
        Forward kinematics that matches Gazebo URDF exactly.
        Returns only position. Use urdf_fk_full() for full transform.

        Args:
            q: Joint angles [q1, q2, q3, q4, q5, q6] in radians

        Returns:
            np.array: End-effector (tool0) position [x, y, z] in base_link frame
        """
        T = self.urdf_fk_full(q)
        return T[:3, 3]

    def urdf_fk_full(self, q):
        """
        Forward kinematics that matches Gazebo URDF exactly.
        Returns full 4x4 transformation matrix.

        CRITICAL: There's a Rx(π) rotation between base_link and base_link_inertia!

        Full chain from base_link to tool0:
        - base_link -> base_link_inertia: rpy=(π, 0, 0)
        - shoulder_pan_joint:  xyz=(0, 0, 0.1625), axis Z
        - shoulder_lift_joint: rpy=(π/2, 0, 0), axis Z
        - elbow_joint:         xyz=(-0.425, 0, 0), axis Z
        - wrist_1_joint:       xyz=(-0.3922, 0, 0.1333), axis Z
        - wrist_2_joint:       xyz=(0, -0.0997, 0), rpy=(π/2, 0, 0), axis Z
        - wrist_3_joint:       xyz=(0, 0.0996, 0), rpy=(π/2, π, π), axis Z
        - wrist_3-flange:      rpy=(0, -π/2, -π/2)
        - flange-tool0:        rpy=(π/2, 0, π/2)  -> to tool0_without_gripper
        - gripper mount:       rpy=(0, 0, π/2)    -> to gripper_base
        - fixed_ee_gripper:    xyz=(0, 0, 0.12)   -> to tool0

        Args:
            q: Joint angles [q1, q2, q3, q4, q5, q6] in radians

        Returns:
            np.array: 4x4 transformation matrix (tool0 in base_link frame)
        """
        import math

        q1, q2, q3, q4, q5, q6 = q[:6]

        # Helper function for 4x4 transforms
        def Rz(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        def Ry(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

        def Rx(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

        def Txyz(x, y, z):
            return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

        # Transform chain from URDF (base_link to tool0)
        # CRITICAL: base_link -> base_link_inertia has Rx(π) rotation
        T_base = Rx(math.pi)

        # Joint 1: shoulder_pan_joint - origin xyz=(0, 0, 0.1625), axis Z
        T01 = Txyz(0, 0, 0.1625) @ Rz(q1)

        # Joint 2: shoulder_lift_joint - rpy=(π/2, 0, 0), axis Z
        T12 = Rx(math.pi / 2) @ Rz(q2)

        # Joint 3: elbow_joint - origin xyz=(-0.425, 0, 0), axis Z
        T23 = Txyz(-0.425, 0, 0) @ Rz(q3)

        # Joint 4: wrist_1_joint - origin xyz=(-0.3922, 0, 0.1333), axis Z
        T34 = Txyz(-0.3922, 0, 0.1333) @ Rz(q4)

        # Joint 5: wrist_2_joint - origin xyz=(0, -0.0997, 0), rpy=(π/2, 0, 0), axis Z
        T45 = Txyz(0, -0.0997, 0) @ Rx(math.pi / 2) @ Rz(q5)

        # Joint 6: wrist_3_joint - origin xyz=(0, 0.0996, 0), rpy=(π/2, π, π), axis Z
        T56 = Txyz(0, 0.0996, 0) @ Rz(math.pi) @ Ry(math.pi) @ Rx(math.pi / 2) @ Rz(q6)

        # Fixed joints from wrist_3_link to tool0 (including gripper)
        # wrist_3-flange: rpy=(0, -π/2, -π/2)
        T6F = Rz(-math.pi / 2) @ Ry(-math.pi / 2)

        # flange-tool0: rpy=(π/2, 0, π/2) -> tool0_without_gripper
        TF_t0wg = Rz(math.pi / 2) @ Rx(math.pi / 2)

        # tool0_without_gripper -> gripper_base: rpy=(0, 0, π/2)
        T_grip = Rz(math.pi / 2)

        # gripper_base -> tool0 (fixed_ee_gripper): xyz=(0, 0, 0.12)
        T_ee = Txyz(0, 0, 0.12)

        # Full transform: base_link -> base_link_inertia -> joints -> tool0
        T = T_base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6F @ TF_t0wg @ T_grip @ T_ee

        return T

    def _dh_transform_modified(self, theta, alpha, d, a):
        """
        Compute MODIFIED DH transformation matrix (as used by Universal Robots).

        Modified DH: T_i = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)

        Args:
            theta: Joint angle
            alpha: Link twist (alpha_{i-1})
            d: Link offset
            a: Link length (a_{i-1})

        Returns:
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array(
            [
                [ct, -st, 0, a],
                [st * ca, ct * ca, -sa, -sa * d],
                [st * sa, ct * sa, ca, ca * d],
                [0, 0, 0, 1],
            ]
        )

    def _dh_transform(self, theta, alpha, d, a):
        """
        Compute STANDARD DH transformation matrix.
        T = Rz(θ) * Tz(d) * Tx(a) * Rx(α)

        NOTE: This is kept for the IK solver which uses standard DH.
        For FK, use _dh_transform_modified instead.

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

        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ]
        )

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

        rospy.logdebug(f"ur5_inverse: p60={p60}, p50={p50}, p50xy={p50xy:.4f}, D[3]={D[3]:.4f}")

        if p50xy < D[3]:
            rospy.logwarn(
                f"Position in unreachable cylinder: p50xy={p50xy:.4f}m < D[3]={D[3]:.4f}m"
            )
            rospy.logwarn(f"  End-effector: {p60}")
            rospy.logwarn(f"  Wrist center: {p50}")
            return None

        phi1_1 = np.arccos(D[3] / p50xy)
        phi1_2 = -phi1_1

        th1_1 = psi + phi1_1 + np.pi / 2
        th1_2 = psi + phi1_2 + np.pi / 2

        # Finding th5
        p61z_1 = p60[0] * np.sin(th1_1) - p60[1] * np.cos(th1_1)
        p61z_2 = p60[0] * np.sin(th1_2) - p60[1] * np.cos(th1_2)

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
            return np.arctan2(-zy / np.sin(th5), zx / np.sin(th5))

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
            C = (np.linalg.norm(P31) ** 2 - A[1] ** 2 - A[2] ** 2) / (2 * A[1] * A[2])
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
            return -np.arctan2(P31[1], -P31[0]) + np.arcsin(
                (A[2] * np.sin(th3)) / np.linalg.norm(P31)
            )

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
        Th = np.array(
            [
                [th1_1, th1_1, th1_1, th1_1, th1_2, th1_2, th1_2, th1_2],
                [
                    th2_1_1_1,
                    th2_1_1_2,
                    th2_1_2_1,
                    th2_1_2_2,
                    th2_2_1_1,
                    th2_2_1_2,
                    th2_2_2_1,
                    th2_2_2_2,
                ],
                [
                    th3_1_1_1,
                    th3_1_1_2,
                    th3_1_2_1,
                    th3_1_2_2,
                    th3_2_1_1,
                    th3_2_1_2,
                    th3_2_2_1,
                    th3_2_2_2,
                ],
                [
                    th4_1_1_1,
                    th4_1_1_2,
                    th4_1_2_1,
                    th4_1_2_2,
                    th4_2_1_1,
                    th4_2_1_2,
                    th4_2_2_1,
                    th4_2_2_2,
                ],
                [th5_1_1, th5_1_1, th5_1_2, th5_1_2, th5_2_1, th5_2_1, th5_2_2, th5_2_2],
                [th6_1_1, th6_1_1, th6_1_2, th6_1_2, th6_2_1, th6_2_1, th6_2_2, th6_2_2],
            ]
        )

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
        MAX_SINGLE_JOINT_MOVE = np.radians(
            170
        )  # Increased limit - impedance controller may still reject large moves

        for i in range(solutions.shape[1]):
            sol = solutions[:, i]

            # Skip solutions with NaN
            if np.any(np.isnan(sol)):
                rospy.loginfo(f"  Sol {i}: REJECTED - contains NaN")
                rospy.loginfo(f"    Joints: {sol}")
                continue

            # NOTE: Joint limit checking disabled - IK solutions are already valid
            # The issue is that IK returns angles like 4.5 rad (258°) which are equivalent
            # to -1.78 rad (-102°) after wrapping, both physically valid
            # The impedance controller will handle wrapping automatically

            # Check if any single joint needs to move too far (impedance controller may reject)
            # Account for angle wrapping: find shortest path
            joint_deltas = np.abs(sol - reference)
            # Normalize to [-pi, pi] to find shortest angular distance
            joint_deltas = np.minimum(joint_deltas, 2 * np.pi - joint_deltas)
            max_single_joint = np.max(joint_deltas)

            if max_single_joint > MAX_SINGLE_JOINT_MOVE:
                rospy.loginfo(
                    f"  Sol {i}: REJECTED - requires {np.degrees(max_single_joint):.1f}° on single joint (max {np.degrees(MAX_SINGLE_JOINT_MOVE):.1f}°)"
                )
                continue

            # Compute distance to reference (L2 norm for smoother movements)
            # Use wrapped distances
            distance = np.linalg.norm(joint_deltas)

            rospy.loginfo(
                f"  Sol {i}: ACCEPTED - dist={distance:.2f}, max_joint={np.degrees(max_single_joint):.1f}°"
            )

            if distance < best_distance:
                best_distance = distance
                best_solution = sol.copy()

        return best_solution

    def gripper_down_rotation(self):
        """
        Get rotation matrix for gripper pointing straight down.

        The analytical IK solver uses DH convention, but our URDF has extra
        gripper transforms. To get the URDF gripper Z-axis pointing down,
        we need to pass this specific rotation to the DH-based IK.

        Computed as: R_ik = R_desired @ inv(R_gripper_chain)
        where R_gripper_chain = T6F @ TF_t0wg @ T_grip (rotation parts)
        """
        # This rotation, when passed to DH-based IK, results in
        # URDF gripper Z-axis pointing DOWN ([0, 0, -1] in base frame)
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        return R

    def generate_waypoints(self, start_joints, target_joints, max_step_deg=45.0):
        """
        Generate intermediate waypoints for large movements.
        Breaks movement into steps where no joint moves more than max_step_deg.
        """
        start = np.array(start_joints).flatten()[:6]
        target = np.array(target_joints).flatten()[:6]

        # Calculate movement for each joint with angle wrapping
        joint_deltas = target - start
        # Wrap to [-pi, pi]
        joint_deltas = np.arctan2(np.sin(joint_deltas), np.cos(joint_deltas))

        # Find maximum movement
        max_movement_deg = np.max(np.abs(np.rad2deg(joint_deltas)))

        if max_movement_deg <= max_step_deg:
            # Movement is small enough, no waypoints needed
            return [target]

        # Calculate number of steps needed
        num_steps = int(np.ceil(max_movement_deg / max_step_deg))

        # Generate waypoints
        waypoints = []
        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            waypoint = start + alpha * joint_deltas
            waypoints.append(waypoint)

        rospy.loginfo(
            f"Generated {len(waypoints)} waypoints (max step: {max_step_deg}°, total: {max_movement_deg:.1f}°)"
        )
        return waypoints

    def move_to_joints_direct(self, target_joints, controller, duration=1.0, target_world_pos=None):
        """
        Move directly to target joint configuration (use only for small movements).

        Args:
            target_joints: Target joint angles [6]
            controller: Controller instance
            duration: Movement duration in seconds
            target_world_pos: Optional target position in world frame (for verification)
        """
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"STARTING MOVEMENT (duration={duration}s)")
        rospy.loginfo(f"{'='*60}")

        # Get current joint state
        current_joints, _ = controller.get_current_joint_state()

        def wrap_to_pi(angle):
            """Wrap angle to [-π, π]"""
            return ((angle + np.pi) % (2 * np.pi)) - np.pi

        # CRITICAL: First wrap target to [-π, π], then find shortest path from current
        target_wrapped = np.zeros(6)
        for i in range(6):
            # Wrap target to [-π, π] first
            target_in_pi = wrap_to_pi(target_joints[i])
            # Then find shortest path from current (which should also be in reasonable range)
            current_wrapped = wrap_to_pi(current_joints[i])

            # Find the shortest angular difference
            diff = wrap_to_pi(target_in_pi - current_wrapped)
            target_wrapped[i] = current_wrapped + diff

            # Final clamp to [-π, π] to be safe
            target_wrapped[i] = wrap_to_pi(target_wrapped[i])

        rospy.loginfo(
            f"Current joints (rad): {np.array2string(current_joints[:6], precision=3, suppress_small=True)}"
        )
        rospy.loginfo(
            f"Current joints (deg): {np.array2string(np.degrees(current_joints[:6]), precision=1, suppress_small=True)}"
        )
        rospy.loginfo(
            f"Target joints (raw, rad):  {np.array2string(target_joints[:6] if len(target_joints) >= 6 else target_joints, precision=3, suppress_small=True)}"
        )
        rospy.loginfo(
            f"Target joints (wrapped, rad):  {np.array2string(target_wrapped, precision=3, suppress_small=True)}"
        )
        rospy.loginfo(
            f"Target joints (wrapped, deg):  {np.array2string(np.degrees(target_wrapped), precision=1, suppress_small=True)}"
        )

        # Use wrapped target
        target_joints = target_wrapped

        # Get current EE position from TF (ground truth)
        current_ee_pos = self._get_actual_ee_position()
        if current_ee_pos is None:
            rospy.logwarn("Could not get current EE from TF, using (0,0,0)")
            current_ee_pos = np.zeros(3)

        # World frame positions
        current_ee_world = current_ee_pos + np.array(
            [self.robot_base_x, self.robot_base_y, self.robot_base_z]
        )

        # Target position: use provided world target, or estimate distance from joints
        if target_world_pos is not None:
            target_ee_world = np.array(target_world_pos)
            target_ee_pos = target_ee_world - np.array(
                [self.robot_base_x, self.robot_base_y, self.robot_base_z]
            )
        else:
            # Estimate target position (won't be accurate but shows movement intent)
            target_ee_pos = current_ee_pos  # Will be updated after motion
            target_ee_world = current_ee_world

        rospy.loginfo(
            f"\nCurrent EE (base_link): {np.array2string(current_ee_pos, precision=3, suppress_small=True)}"
        )
        rospy.loginfo(
            f"Current EE (world):     {np.array2string(current_ee_world, precision=3, suppress_small=True)}"
        )
        if target_world_pos is not None:
            rospy.loginfo(
                f"Target EE (base_link):  {np.array2string(target_ee_pos, precision=3, suppress_small=True)}"
            )
            rospy.loginfo(
                f"Target EE (world):      {np.array2string(target_ee_world, precision=3, suppress_small=True)}"
            )
            ee_distance = np.linalg.norm(target_ee_pos - current_ee_pos)
            rospy.loginfo(f"EE distance to travel: {ee_distance:.4f} m ({ee_distance*1000:.1f} mm)")
        rospy.loginfo(f"{'='*60}\n")

        # Pad target joints to 8 if only 6 provided (add gripper)
        if len(target_joints) == 6:
            target_joints_full = np.concatenate([target_joints, current_joints[6:8]])
        else:
            target_joints_full = target_joints

        # Store initial position for verification
        initial_joints = current_joints.copy()
        initial_ee_pos = current_ee_pos.copy()
        last_check_joints = current_joints.copy()
        check_interval = 1000  # Check every 2 seconds (1000 steps at 500Hz)
        motion_threshold = 0.01  # Minimum movement in radians to consider "moving"

        # Publish initial TF
        self._publish_ee_tf(controller, current_ee_pos, "current_ee")
        self._publish_ee_tf(controller, target_ee_pos, "target_ee")

        # Check velocity limits and auto-adjust duration if needed
        joint_delta = target_joints_full[:6] - current_joints[:6]
        max_velocity = np.max(np.abs(joint_delta / duration))
        SAFE_VELOCITY_LIMIT = 2.0  # rad/s - fast movement

        if max_velocity > SAFE_VELOCITY_LIMIT:
            # Auto-increase duration to stay within velocity limits
            required_duration = np.max(np.abs(joint_delta)) / SAFE_VELOCITY_LIMIT
            old_duration = duration
            duration = max(required_duration * 1.2, old_duration)  # Add 20% margin
            rospy.logwarn(f"⚠️  Auto-adjusting duration: {old_duration:.1f}s → {duration:.1f}s")
            rospy.logwarn(
                f"   Original velocity: {max_velocity:.3f} rad/s (limit: {SAFE_VELOCITY_LIMIT} rad/s)"
            )
            max_velocity = np.max(np.abs(joint_delta / duration))
            rospy.loginfo(f"   New velocity: {max_velocity:.3f} rad/s")
        else:
            rospy.loginfo(
                f"✓ Velocity OK: {max_velocity:.3f} rad/s (limit: {SAFE_VELOCITY_LIMIT} rad/s)"
            )

        # Simple linear interpolation
        # CRITICAL: Must run at high rate to match impedance controller (1000 Hz)
        # Lower rates cause jerky motion and poor tracking
        CONTROL_RATE = 500  # Hz - compromise between CPU load and smoothness
        rate = rospy.Rate(CONTROL_RATE)
        steps = int(duration * CONTROL_RATE)

        rospy.loginfo(f"Control loop: {steps} steps at {CONTROL_RATE} Hz for {duration}s duration")
        commands_sent = 0

        for i in range(steps + 1):
            if rospy.is_shutdown():
                rospy.logerr("ROS shutdown detected in control loop!")
                break

            alpha = float(i) / steps  # 0 to 1

            # Interpolate between current and target
            desired_joints = current_joints + alpha * (target_joints_full - current_joints)

            # Compute desired velocity (derivative of position trajectory)
            # Use trapezoidal velocity profile for smoother motion
            # Ramp up for first 20%, constant for middle 60%, ramp down for last 20%
            if alpha < 0.2:
                # Ramp up
                velocity_scale = alpha / 0.2
            elif alpha > 0.8:
                # Ramp down
                velocity_scale = (1.0 - alpha) / 0.2
            else:
                # Constant velocity
                velocity_scale = 1.0

            # Velocity = (target - start) / duration * scale
            desired_velocity = (target_joints_full - current_joints) / duration * velocity_scale

            # Send command with velocity feedforward
            controller.send_joint_command(desired_joints, velocities=desired_velocity)
            commands_sent += 1

            # Periodically check if robot is actually moving
            if i > 0 and i % check_interval == 0:
                actual_joints, _ = controller.get_current_joint_state()
                actual_ee_pos_fk = self._compute_fk(
                    actual_joints[:6], controller
                )  # Computed from joints

                joint_movement = np.linalg.norm(actual_joints[:6] - last_check_joints[:6])
                ee_movement = np.linalg.norm(actual_ee_pos_fk - initial_ee_pos)
                expected_joint_movement = np.linalg.norm(desired_joints[:6] - last_check_joints[:6])

                progress = i / steps * 100

                rospy.loginfo(
                    f"Progress: {progress:.0f}% | Joint Δ: {joint_movement:.4f} rad | EE Δ: {ee_movement*1000:.1f} mm"
                )
                rospy.loginfo(
                    f"  Computed EE (FK): {np.array2string(actual_ee_pos_fk, precision=3, suppress_small=True)}"
                )

                if joint_movement < motion_threshold and expected_joint_movement > motion_threshold:
                    rospy.logerr(f"⚠️  ROBOT NOT MOVING! Step {i}/{steps} ({progress:.0f}%)")
                    rospy.logerr(
                        f"   Joint movement: {joint_movement:.4f} rad (expected {expected_joint_movement:.4f})"
                    )
                    rospy.logerr(f"   EE movement: {ee_movement*1000:.1f} mm")
                    rospy.logerr(f"   Desired joints: {desired_joints[:6]}")
                    rospy.logerr(f"   Actual joints:  {actual_joints[:6]}")
                    rospy.logerr(f"   Commands may not be reaching robot!")

                # Publish current EE position as TF (use FK computed position)
                if actual_ee_pos_fk is not None:
                    self._publish_ee_tf(controller, actual_ee_pos_fk, "current_ee")

                last_check_joints = actual_joints.copy()

            rate.sleep()

        rospy.loginfo(f"✓ Control loop completed: {commands_sent} commands sent")

        # Final verification using TF (ground truth)
        final_joints, _ = controller.get_current_joint_state()
        final_ee_pos = self._get_actual_ee_position()
        if final_ee_pos is None:
            rospy.logwarn("Could not get final EE from TF")
            final_ee_pos = np.zeros(3)

        total_joint_movement = np.linalg.norm(final_joints[:6] - initial_joints[:6])
        total_ee_movement = np.linalg.norm(final_ee_pos - initial_ee_pos)
        expected_joint_total = np.linalg.norm(target_joints_full[:6] - initial_joints[:6])
        final_joint_error = np.linalg.norm(final_joints[:6] - target_joints_full[:6])

        # Calculate EE error if we have a target position
        if target_world_pos is not None:
            target_base_link = np.array(target_world_pos) - np.array(
                [self.robot_base_x, self.robot_base_y, self.robot_base_z]
            )
            final_ee_error = np.linalg.norm(final_ee_pos - target_base_link)
            ee_distance = np.linalg.norm(target_base_link - initial_ee_pos)
        else:
            final_ee_error = 0.0
            ee_distance = total_ee_movement

        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo("MOVEMENT COMPLETE - Final Status:")
        rospy.loginfo(f"{'='*60}")
        rospy.loginfo(f"Joint space:")
        rospy.loginfo(
            f"  Total movement: {total_joint_movement:.4f} rad (expected {expected_joint_total:.4f})"
        )
        rospy.loginfo(
            f"  Final error: {final_joint_error:.4f} rad ({np.degrees(final_joint_error):.1f} deg)"
        )
        rospy.loginfo(f"Cartesian space (TF ground truth):")
        rospy.loginfo(f"  Total EE movement: {total_ee_movement*1000:.1f} mm")
        rospy.loginfo(
            f"  Final EE position (base_link): {np.array2string(final_ee_pos, precision=3, suppress_small=True)}"
        )
        if target_world_pos is not None:
            rospy.loginfo(
                f"  Target position (base_link):   {np.array2string(target_base_link, precision=3, suppress_small=True)}"
            )
            rospy.loginfo(f"  Final EE error: {final_ee_error*1000:.1f} mm")

        # Publish final TF
        self._publish_ee_tf(controller, final_ee_pos, "current_ee")

        if total_joint_movement < motion_threshold and expected_joint_total > motion_threshold:
            rospy.logerr("❌ MOVEMENT FAILED - Robot did not move!")
            rospy.logerr(f"   Check if /command topic is connected")
            rospy.logerr(f"   Check if Gazebo controllers are running")
            rospy.loginfo(f"{'='*60}\n")
            return False
        elif target_world_pos is not None and final_ee_error > 0.05:  # 50mm error
            rospy.logerr(f"❌ LARGE EE POSITION ERROR: {final_ee_error*1000:.1f} mm")
            rospy.logerr("   End-effector did not reach target!")
            rospy.logerr("   Possible IK solution error or tracking issue")
            rospy.loginfo(f"{'='*60}\n")
            return False
        elif final_joint_error > 0.1:  # 0.1 rad = ~5.7 degrees
            rospy.logwarn(
                f"⚠️  Large joint error: {final_joint_error:.4f} rad ({np.degrees(final_joint_error):.1f} deg)"
            )
            if target_world_pos is not None:
                rospy.logwarn(f"   EE error: {final_ee_error*1000:.1f} mm")
            rospy.loginfo(f"{'='*60}\n")
        else:
            rospy.loginfo("✓ Movement successful - Target reached!")
            rospy.loginfo(f"{'='*60}\n")

        return True

    def move_to_joints(self, target_joints, controller, duration=1.5, max_step_deg=45.0):
        """
        Move to target joint configuration with automatic waypoint generation.
        For large movements, breaks into smaller steps to avoid controller issues.

        Args:
            target_joints: Target joint angles [6]
            controller: Controller instance
            duration: Duration per waypoint segment (not total duration)
            max_step_deg: Maximum joint movement per segment (degrees)
        """
        current_joints, _ = controller.get_current_joint_state()

        # Generate waypoints
        waypoints = self.generate_waypoints(current_joints, target_joints, max_step_deg)

        rospy.loginfo(f"\n{'='*70}")
        rospy.loginfo(f"MULTI-STEP MOVEMENT: {len(waypoints)} waypoint(s)")
        rospy.loginfo(f"  Duration per step: {duration}s")
        rospy.loginfo(f"  Max step size: {max_step_deg}°")
        rospy.loginfo(f"{'='*70}")

        # Execute each segment
        for i, waypoint in enumerate(waypoints):
            rospy.loginfo(f"\n>>> Executing waypoint {i+1}/{len(waypoints)} <<<")
            success = self.move_to_joints_direct(waypoint, controller, duration)
            if not success:
                rospy.logerr(f"Failed at waypoint {i+1}/{len(waypoints)}")
                return False
            if i < len(waypoints) - 1:
                rospy.loginfo("Pausing 0.1s between waypoints...")
                rospy.sleep(0.1)

        rospy.loginfo(f"\n{'='*70}")
        rospy.loginfo(f"MULTI-STEP MOVEMENT COMPLETE")
        rospy.loginfo(f"{'='*70}\n")
        return True

    def move_to_joints_with_target(
        self, target_joints, controller, target_world_pos, duration=1.5, max_step_deg=45.0
    ):
        """
        Move to target joint configuration with automatic waypoint generation.
        Includes target world position for final verification.

        Args:
            target_joints: Target joint angles [6]
            controller: Controller instance
            target_world_pos: Target position in world frame [x, y, z] for verification
            duration: Duration per waypoint segment (not total duration)
            max_step_deg: Maximum joint movement per segment (degrees)
        """
        current_joints, _ = controller.get_current_joint_state()

        # Generate waypoints
        waypoints = self.generate_waypoints(current_joints, target_joints, max_step_deg)

        rospy.loginfo(f"\n{'='*70}")
        rospy.loginfo(f"MULTI-STEP MOVEMENT: {len(waypoints)} waypoint(s)")
        rospy.loginfo(
            f"  Target world position: [{target_world_pos[0]:.3f}, {target_world_pos[1]:.3f}, {target_world_pos[2]:.3f}]"
        )
        rospy.loginfo(f"  Duration per step: {duration}s")
        rospy.loginfo(f"  Max step size: {max_step_deg}°")
        rospy.loginfo(f"{'='*70}")

        # Execute each segment, only pass target_world_pos for the LAST waypoint
        for i, waypoint in enumerate(waypoints):
            rospy.loginfo(f"\n>>> Executing waypoint {i+1}/{len(waypoints)} <<<")
            is_final = i == len(waypoints) - 1
            wp_target = target_world_pos if is_final else None
            success = self.move_to_joints_direct(
                waypoint, controller, duration, target_world_pos=wp_target
            )
            if not success:
                rospy.logerr(f"Failed at waypoint {i+1}/{len(waypoints)}")
                return False
            if i < len(waypoints) - 1:
                rospy.loginfo("Pausing 0.1s between waypoints...")
                rospy.sleep(0.1)

        rospy.loginfo(f"\n{'='*70}")
        rospy.loginfo(f"MULTI-STEP MOVEMENT COMPLETE")
        rospy.loginfo(f"{'='*70}\n")
        return True

    def _get_actual_ee_position(self):
        """Get actual current end-effector position from TF (ground truth) in base_link frame"""
        try:
            import tf

            if not hasattr(self, '_tf_listener'):
                self._tf_listener = tf.TransformListener()
                rospy.sleep(0.2)

            try:
                # Use base_link frame to match DH FK
                (trans, rot) = self._tf_listener.lookupTransform(
                    '/base_link', '/tool0', rospy.Time(0)
                )
                return np.array(trans)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logdebug(f"TF lookup failed: {e}")
                return None
        except Exception as e:
            rospy.logdebug(f"Could not get actual EE from TF: {e}")
            return None

    def _compute_fk(self, joint_angles, controller=None):
        """
        Compute forward kinematics to get end-effector position.
        Uses TF lookup from Gazebo (ground truth) if available, falls back to DH.

        Args:
            joint_angles: Array of 6 joint angles in radians (used for fallback only)
            controller: Controller instance (used for TF lookup)

        Returns:
            np.array: End-effector position [x, y, z] in base frame
        """
        # Try TF lookup first (ground truth from Gazebo)
        tf_pos = self._get_actual_ee_position()
        if tf_pos is not None:
            return tf_pos

        # Fallback to DH-based FK (may not be perfectly accurate)
        # Using standard DH which matches the IK solver
        a = self.A
        d = self.D
        alpha = self.Alpha

        T = np.eye(4)
        for i in range(6):
            T_i = self._dh_transform(joint_angles[i], alpha[i], d[i], a[i])
            T = T @ T_i

        rospy.logwarn("FK: Using DH fallback (TF unavailable)")
        return T[:3, 3]

    def _publish_ee_tf(self, controller, position, frame_name):
        """Publish end-effector position as TF for visualization.

        Args:
            position: Position in BASE_LINK frame (from FK)
            frame_name: Name of the TF frame to publish
        """
        try:
            from geometry_msgs.msg import TransformStamped
            import tf2_ros

            if hasattr(controller, 'tf_broadcaster'):
                # CRITICAL: Convert from base_link frame to world frame for visualization
                # FK returns positions in base_link frame, but TF should be in world frame
                world_x = position[0] + self.robot_base_x
                world_y = position[1] + self.robot_base_y
                world_z = position[2] + self.robot_base_z

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "world"
                t.child_frame_id = frame_name
                t.transform.translation.x = world_x
                t.transform.translation.y = world_y
                t.transform.translation.z = world_z
                t.transform.rotation.w = 1.0

                controller.tf_broadcaster.sendTransform(t)
        except Exception as e:
            rospy.logdebug(f"Failed to publish TF: {e}")

    def pick_object(self, object_pos, controller):
        """
        Execute pick sequence using actual object position
        Uses safe transit height to avoid collisions with other objects.

        Args:
            object_pos: Object position [x, y, z]
            controller: Controller instance
        """
        rospy.loginfo(f"\n{'#'*60}")
        rospy.loginfo(f"PICK SEQUENCE STARTING")
        rospy.loginfo(f"{'#'*60}")
        rospy.loginfo(
            f"Object position: {np.array2string(object_pos, precision=3, suppress_small=True)}"
        )

        # Collision check
        if self.enable_collision_checking:
            ok, msg = self.collision_checker.check_position(object_pos)
            if not ok:
                rospy.logerr(f"Collision detected: {msg}")
                return False

        # FIRST THING: Open gripper BEFORE any movement to avoid hitting objects
        rospy.loginfo("  Opening gripper FIRST (before any movement)...")
        controller.send_gripper_command(self.config.gripper_open_pos)
        rospy.sleep(1.5)  # Wait for gripper to fully open

        # Get safe transit height from config (defaults to 1.05m if not set)
        safe_z = getattr(self.config, 'safe_transit_height', 1.05)

        # Compute safe transit position (high above object XY)
        safe_transit_pos = np.array([object_pos[0], object_pos[1], safe_z])

        # Compute approach position (above object)
        approach_pos = np.array(
            [object_pos[0], object_pos[1], object_pos[2] + self.config.approach_height]
        )

        # Compute grasp position (at object)
        grasp_pos = np.array(
            [object_pos[0], object_pos[1], object_pos[2] + self.config.grasp_height]
        )

        rospy.loginfo(
            f"Safe transit: {np.array2string(safe_transit_pos, precision=3, suppress_small=True)} (z={safe_z})"
        )
        rospy.loginfo(
            f"Approach position: {np.array2string(approach_pos, precision=3, suppress_small=True)} (object + {self.config.approach_height:.3f}m)"
        )
        rospy.loginfo(
            f"Grasp position: {np.array2string(grasp_pos, precision=3, suppress_small=True)} (object + {self.config.grasp_height:.3f}m)"
        )

        # Publish TF markers for visualization
        self._publish_ee_tf(controller, object_pos, "object_to_pick")
        self._publish_ee_tf(controller, approach_pos, "pick_approach")
        self._publish_ee_tf(controller, grasp_pos, "pick_grasp")

        # STEP 1: Move to safe transit height above object (avoids collisions)
        rospy.loginfo(f"  Step 1: Moving to safe transit height: {safe_transit_pos}")
        safe_joints = self.simple_ik(safe_transit_pos, gripper_down=True)
        if safe_joints is None:
            rospy.logerr("Failed to compute safe transit IK")
            return False
        self.move_to_joints(safe_joints, controller, duration=1.5)

        # STEP 2: Move down to approach position
        rospy.loginfo(f"  Step 2: Moving to approach position: {approach_pos}")
        approach_joints = self.simple_ik(approach_pos, gripper_down=True)
        if approach_joints is None:
            rospy.logerr("Failed to compute approach IK")
            return False
        self.move_to_joints(approach_joints, controller, duration=1.2)

        # STEP 3: Move down to grasp
        rospy.loginfo(f"  Step 3: Moving down to grasp: {grasp_pos}")
        grasp_joints = self.simple_ik(grasp_pos, gripper_down=True)
        if grasp_joints is None:
            rospy.logerr("Failed to compute grasp IK")
            return False
        self.move_to_joints(grasp_joints, controller, duration=1.2)

        # Small pause to let robot settle before closing gripper
        rospy.sleep(0.3)

        # Close gripper
        rospy.loginfo("  Closing gripper...")
        controller.send_gripper_command(self.config.gripper_close_pos)
        rospy.sleep(2.5)  # Wait longer for gripper to fully close and grip object

        # GRASP VERIFICATION: Check if gripper actually grabbed something
        # Read actual gripper joint positions from controller
        rospy.sleep(0.3)  # Extra wait for state update

        gripper_pos_1 = controller.q[6] if len(controller.q) >= 7 else -999
        gripper_pos_2 = controller.q[7] if len(controller.q) >= 8 else -999
        gripper_avg = (gripper_pos_1 + gripper_pos_2) / 2.0

        # Also check what we commanded vs what we got
        commanded_close = self.config.gripper_close_pos

        rospy.loginfo(f"  GRASP CHECK:")
        rospy.loginfo(f"    Commanded close pos: {commanded_close:.3f}")
        rospy.loginfo(f"    Actual gripper joints: [{gripper_pos_1:.3f}, {gripper_pos_2:.3f}]")
        rospy.loginfo(f"    Average gripper pos: {gripper_avg:.3f}")

        # Gripper closed fully if it reached close to commanded position (nothing blocking)
        # If object is gripped, gripper will stop before reaching full close
        # commanded_close = -0.8, if gripper reaches < -0.5, it's probably empty
        grasp_margin = 0.25  # More strict: if within 0.25 of commanded, probably empty

        grasp_failed = False
        if abs(gripper_avg - commanded_close) < grasp_margin:
            rospy.logwarn(f"  ⚠ GRASP FAILED: Gripper closed too far (nothing blocking)")
            rospy.logwarn(f"    gripper_avg={gripper_avg:.3f} ≈ commanded={commanded_close:.3f}")
            grasp_failed = True
        elif gripper_avg < -0.4:
            # Also fail if gripper is more closed than -0.4 (should have object by then)
            rospy.logwarn(f"  ⚠ GRASP FAILED: Gripper too closed ({gripper_avg:.3f} < -0.4)")
            grasp_failed = True

        if grasp_failed:
            rospy.logwarn(f"  Object was likely missed or pushed away - RETURNING FALSE")
            # Open gripper and return to safe height
            controller.send_gripper_command(self.config.gripper_open_pos)
            rospy.sleep(1.0)
            self.move_to_joints(safe_joints, controller, duration=1.5)
            return False
        else:
            rospy.loginfo(f"  ✓ GRASP SUCCESS: Gripper holding object")
            rospy.loginfo(f"    gripper_avg={gripper_avg:.3f} (object blocking closure)")

        # STEP 4: Lift object to safe transit height
        rospy.loginfo(f"  Step 4: Lifting to safe transit height...")
        self.move_to_joints(safe_joints, controller, duration=1.5)

        rospy.loginfo("Pick complete")
        return True

    def place_object(self, target_pos, controller):
        """
        Execute place sequence using actual target position
        Uses safe transit height to avoid collisions with other objects.

        Args:
            target_pos: Target position [x, y, z]
            controller: Controller instance
        """
        rospy.loginfo(f"\n{'#'*60}")
        rospy.loginfo(f"PLACE SEQUENCE STARTING")
        rospy.loginfo(f"{'#'*60}")
        rospy.loginfo(
            f"Target position: {np.array2string(target_pos, precision=3, suppress_small=True)}"
        )

        # Collision check
        if self.enable_collision_checking:
            ok, msg = self.collision_checker.check_position(target_pos)
            if not ok:
                rospy.logerr(f"Collision detected: {msg}")
                return False

        # Get safe transit height from config (defaults to 1.05m if not set)
        safe_z = getattr(self.config, 'safe_transit_height', 1.05)

        # Compute safe transit position (high above target XY)
        safe_transit_pos = np.array([target_pos[0], target_pos[1], safe_z])

        # Compute place approach position (above target)
        place_approach_pos = np.array(
            [target_pos[0], target_pos[1], target_pos[2] + self.config.approach_height]
        )

        # Compute place position (at target surface)
        place_pos = np.array(
            [target_pos[0], target_pos[1], target_pos[2] + self.config.place_height]
        )

        rospy.loginfo(
            f"Safe transit: {np.array2string(safe_transit_pos, precision=3, suppress_small=True)} (z={safe_z})"
        )
        rospy.loginfo(
            f"Place approach position: {np.array2string(place_approach_pos, precision=3, suppress_small=True)} (target + {self.config.approach_height:.3f}m)"
        )
        rospy.loginfo(
            f"Place position: {np.array2string(place_pos, precision=3, suppress_small=True)} (target + {self.config.place_height:.3f}m)"
        )

        # Publish TF markers for visualization
        self._publish_ee_tf(controller, target_pos, "place_target")
        self._publish_ee_tf(controller, place_approach_pos, "place_approach")
        self._publish_ee_tf(controller, place_pos, "place_position")

        # STEP 1: Move to safe transit height above target (avoids collisions)
        rospy.loginfo(f"  Step 1: Moving to safe transit height: {safe_transit_pos}")
        safe_joints = self.simple_ik(safe_transit_pos, gripper_down=True)
        if safe_joints is None:
            rospy.logerr("Failed to compute safe transit IK")
            return False
        self.move_to_joints(safe_joints, controller, duration=1.5)

        # STEP 2: Move down to place approach
        rospy.loginfo(f"  Step 2: Moving to place approach: {place_approach_pos}")
        place_approach_joints = self.simple_ik(place_approach_pos, gripper_down=True)
        if place_approach_joints is None:
            rospy.logerr("Failed to compute place approach IK")
            return False
        self.move_to_joints(place_approach_joints, controller, duration=1.2)

        # STEP 3: Move down to place
        rospy.loginfo(f"  Step 3: Moving down to place: {place_pos}")
        place_joints = self.simple_ik(place_pos, gripper_down=True)
        if place_joints is None:
            rospy.logerr("Failed to compute place IK")
            return False
        self.move_to_joints(place_joints, controller, duration=1.2)

        # Open gripper to release object
        rospy.loginfo("  Opening gripper to release object...")
        controller.send_gripper_command(self.config.gripper_open_pos)
        rospy.sleep(1.5)  # Wait for gripper to fully open and release

        # STEP 4: Lift to safe transit height
        rospy.loginfo(f"  Step 4: Lifting to safe transit height...")
        self.move_to_joints(safe_joints, controller, duration=1.5)

        rospy.loginfo("Place complete")
        return True

    def simple_ik(self, target_pos, gripper_down=True):
        """
        Inverse kinematics using analytical solver.

        Args:
            target_pos: Target XYZ position [x, y, z] in WORLD frame
            gripper_down: If True, orient gripper downward (for picking)

        Returns:
            Joint angles [6] or None if unreachable
        """
        x, y, z = target_pos
        rospy.loginfo(f"IK: target world=[{x:.3f}, {y:.3f}, {z:.3f}]")

        # Convert from world frame to base_link frame
        x_base = x - self.robot_base_x
        y_base = y - self.robot_base_y
        z_base = z - self.robot_base_z
        rospy.loginfo(f"IK: target base_link=[{x_base:.3f}, {y_base:.3f}, {z_base:.3f}]")

        target_base = np.array([x_base, y_base, z_base])

        # Check XY distance from base (UR5 has a hole in workspace directly below)
        xy_distance = np.sqrt(x_base**2 + y_base**2)
        min_xy = 0.12  # Minimum XY distance (D[3] + margin)
        if xy_distance < min_xy:
            rospy.logerr(
                f"IK: Target too close to robot base in XY! XY distance={xy_distance:.3f}m < min={min_xy:.2f}m"
            )
            rospy.logerr(
                f"  Robot base (world): ({self.robot_base_x}, {self.robot_base_y}, {self.robot_base_z})"
            )
            rospy.logerr(f"  Target (world): ({x}, {y}, {z})")
            rospy.logerr(f"  Move target further from X={self.robot_base_x}, Y={self.robot_base_y}")
            return None

        # Basic workspace validation
        # UR5 reach is ~0.85m but with gripper can extend to ~0.95m for near-vertical poses
        max_reach = 0.95
        distance_from_base = np.sqrt(x_base**2 + y_base**2 + z_base**2)
        rospy.loginfo(
            f"IK: distance from base: {distance_from_base:.3f}m (max reach: {max_reach:.2f}m)"
        )

        if distance_from_base > max_reach:
            rospy.logerr(
                f"IK: Target is outside workspace! Distance {distance_from_base:.3f}m > max reach {max_reach:.2f}m"
            )
            return None

        # Get current joints for solution selection
        current_joints = None
        if self.controller is not None:
            try:
                current_joints, _ = self.controller.get_current_joint_state()
            except:
                pass

        # Helper to wrap angles
        def wrap_to_pi(angle):
            return ((angle + np.pi) % (2 * np.pi)) - np.pi

        # Use NUMERICAL IK ONLY - analytical IK has frame convention mismatch
        # The TWO-PHASE IK handles orientation separately, so we can start from current joints
        best_result = None
        best_error = float('inf')

        # ALWAYS use current joints as starting config
        # The two-phase IK will:
        #   1. First converge position (works well from any start)
        #   2. Then adjust wrist joints for orientation
        # This is faster and more reliable than starting far away
        if current_joints is not None:
            starting_configs = [np.array(current_joints[:6])]
            rospy.loginfo(f"IK: Using current joints as starting config")
        else:
            # Fallback: pick-ready config with gripper pointing down
            pick_ready_joints = np.array(
                [
                    np.radians(0),  # q1: base rotation
                    np.radians(-90),  # q2: shoulder
                    np.radians(90),  # q3: elbow
                    np.radians(-90),  # q4: wrist 1
                    np.radians(90),  # q5: wrist 2 - makes gripper point down
                    np.radians(0),  # q6: wrist 3
                ]
            )
            starting_configs = [pick_ready_joints]
            rospy.loginfo(f"IK: No current joints, using pick-ready config")

        rospy.loginfo(f"IK: Trying {len(starting_configs)} starting configuration(s)")

        for i, start_config in enumerate(starting_configs):
            result = self._numerical_ik_urdf(target_base, start_config, gripper_down=gripper_down)

            if result is not None:
                T = self.urdf_fk_full(result)
                pos = T[:3, 3]
                pos_err = np.linalg.norm(target_base - pos)

                if pos_err < best_error:
                    best_error = pos_err
                    best_result = result
                    rospy.loginfo(f"IK: Found solution with error {pos_err*1000:.1f}mm")

                if pos_err < 0.02:  # 20mm good enough, stop searching
                    break

        if best_result is not None:
            result_wrapped = np.array([wrap_to_pi(a) for a in best_result])
            T = self.urdf_fk_full(result_wrapped)
            pos = T[:3, 3]
            z_axis = T[:3, 2]
            pos_err = np.linalg.norm(target_base - pos) * 1000
            rospy.loginfo(
                f"IK solution: [{', '.join([f'{np.degrees(a):.1f}' for a in result_wrapped])}]°"
            )
            rospy.loginfo(
                f"IK FK: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], Z=[{z_axis[0]:.2f}, {z_axis[1]:.2f}, {z_axis[2]:.2f}]"
            )
            rospy.loginfo(f"IK error: {pos_err:.1f}mm")
            return result_wrapped

        rospy.logerr(f"IK FAILED: Cannot reach target [{x_base:.3f}, {y_base:.3f}, {z_base:.3f}]")
        return None

    def _numerical_ik_urdf(self, target_base, current_joints=None, gripper_down=True):
        """
        Numerical IK using URDF-based FK (matches Gazebo exactly).
        TWO-PHASE approach:
          Phase 1: Position-only IK (fast convergence)
          Phase 2: Adjust wrist joints for orientation (if gripper_down=True)

        Args:
            target_base: Target position in base_link frame [x, y, z]
            current_joints: Current joint angles (used as initial guess)
            gripper_down: Whether gripper should point down with proper finger orientation

        Returns:
            Joint angles [6] or None if failed
        """
        rospy.loginfo(f"Using numerical IK with URDF FK (gripper_down={gripper_down})")

        # UR5 joint limits (radians)
        joint_limits_lower = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        joint_limits_upper = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

        def wrap_to_pi(angle):
            """Wrap angle to [-π, π]"""
            return ((angle + np.pi) % (2 * np.pi)) - np.pi

        def clamp_joints(q):
            """Wrap and clamp joints to [-π, π]"""
            q_clamped = np.array([wrap_to_pi(a) for a in q])
            return np.clip(q_clamped, joint_limits_lower, joint_limits_upper)

        # Initial guess
        if current_joints is not None:
            q = np.array(current_joints[:6]).copy()
            q = np.array([wrap_to_pi(a) for a in q])
            rospy.loginfo(
                f"  Using PROVIDED initial guess: [{', '.join([f'{np.degrees(a):.1f}' for a in q])}]°"
            )
        else:
            q = self.home_joints.copy()
            rospy.loginfo(f"  Using HOME as initial guess")

        # Verify initial guess orientation
        T_init = self.urdf_fk_full(q)
        z_init = T_init[:3, 2]
        x_init = T_init[:3, 0]
        rospy.loginfo(f"  Initial Tool Z-axis: [{z_init[0]:.3f}, {z_init[1]:.3f}, {z_init[2]:.3f}]")
        rospy.loginfo(f"  Initial Tool X-axis: [{x_init[0]:.3f}, {x_init[1]:.3f}, {x_init[2]:.3f}]")

        # ==================== PHASE 1: POSITION + Y-AXIS HORIZONTAL IK ====================
        # Converge on position AND enforce gripper Y-axis is horizontal (Y_z = 0)
        # This is a 4DOF constraint: 3 for position + 1 for Y_z = 0
        rospy.loginfo(f"  PHASE 1: Position + Y-axis horizontal IK")

        max_iter_phase1 = 150
        pos_epsilon = 2e-3  # 2mm position tolerance
        orient_epsilon = 0.05  # Y_z tolerance (close to 0)
        lambda_dls = 0.05  # Lower damping for faster convergence
        delta = 0.001

        for iteration in range(max_iter_phase1):
            T_current = self.urdf_fk_full(q)
            current_pos = T_current[:3, 3]
            current_Y = T_current[:3, 1]  # Y-axis of gripper frame

            # Position error (3D)
            pos_error = target_base - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # Orientation error: Y_z should be 0 (Y axis parallel to XY plane)
            orient_error = -current_Y[2]  # We want Y_z = 0, so error = 0 - Y_z = -Y_z

            if iteration == 0:
                rospy.loginfo(
                    f"  Initial FK pos: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]"
                )
                rospy.loginfo(
                    f"  Initial Y-axis: [{current_Y[0]:.3f}, {current_Y[1]:.3f}, {current_Y[2]:.3f}]"
                )
                rospy.loginfo(
                    f"  Initial pos error: {pos_error_norm*1000:.1f} mm, Y_z error: {abs(current_Y[2]):.3f}"
                )

            # Check convergence
            if pos_error_norm < pos_epsilon and abs(current_Y[2]) < orient_epsilon:
                rospy.loginfo(
                    f"  Phase 1 converged in {iteration} iterations (pos error: {pos_error_norm*1000:.1f}mm, Y_z: {current_Y[2]:.3f})"
                )
                break

            # Combined error vector [pos_x, pos_y, pos_z, orient]
            error_vec = np.array([pos_error[0], pos_error[1], pos_error[2], orient_error])

            # Numerical Jacobian (4x6) - 3 for position + 1 for Y_z
            J = np.zeros((4, 6))
            for j in range(6):
                q_plus = q.copy()
                q_plus[j] += delta
                T_plus = self.urdf_fk_full(q_plus)
                pos_plus = T_plus[:3, 3]
                Y_plus = T_plus[:3, 1]

                J[0:3, j] = (pos_plus - current_pos) / delta
                J[3, j] = (Y_plus[2] - current_Y[2]) / delta  # Jacobian for Y_z

            # Damped least squares
            JT = J.T
            JJT = J @ JT
            dq = JT @ np.linalg.solve(JJT + lambda_dls**2 * np.eye(4), error_vec)

            # Limit step size
            max_step = 0.2  # radians per iteration
            dq_norm = np.linalg.norm(dq)
            if dq_norm > max_step:
                dq = dq * (max_step / dq_norm)

            # Update with line search
            alpha = 1.0
            for _ in range(8):
                q_new = clamp_joints(q + alpha * dq)
                T_new = self.urdf_fk_full(q_new)
                pos_new = T_new[:3, 3]
                Y_new = T_new[:3, 1]
                error_new_pos = np.linalg.norm(target_base - pos_new)
                error_new_orient = abs(Y_new[2])
                # Weighted error for line search
                error_new = error_new_pos + 0.1 * error_new_orient
                error_old = pos_error_norm + 0.1 * abs(current_Y[2])
                if error_new < error_old:
                    q = q_new
                    break
                alpha *= 0.5
            else:
                q = clamp_joints(q + 0.1 * dq)

        # Check Phase 1 result
        T_phase1 = self.urdf_fk_full(q)
        pos_phase1 = T_phase1[:3, 3]
        pos_error_phase1 = np.linalg.norm(target_base - pos_phase1)
        z_phase1 = T_phase1[:3, 2]
        y_phase1 = T_phase1[:3, 1]

        rospy.loginfo(f"  Phase 1 result: pos error={pos_error_phase1*1000:.1f}mm")
        rospy.loginfo(f"    Z-axis: [{z_phase1[0]:.2f}, {z_phase1[1]:.2f}, {z_phase1[2]:.2f}]")
        rospy.loginfo(
            f"    Y-axis: [{y_phase1[0]:.2f}, {y_phase1[1]:.2f}, {y_phase1[2]:.2f}] (Y_z should be ~0)"
        )

        if pos_error_phase1 > 0.05:  # 50mm - Phase 1 failed
            rospy.logerr(f"  Phase 1 FAILED: position error {pos_error_phase1*1000:.1f}mm > 50mm")
            return None

        # ==================== PHASE 2: ORIENTATION VALIDATION ====================
        # Check that gripper has reasonable orientation:
        # 1. Position is accurate (< 30mm error)
        # 2. Gripper Y-axis is horizontal (Y_z close to 0) - THIS IS THE KEY CONSTRAINT
        # 3. Gripper X-axis (finger opening) is roughly horizontal (X_z close to 0)
        # 4. Gripper Z (approach direction) is pointing somewhat downward (Z_z < 0)

        if not gripper_down:
            # No orientation constraint needed
            rospy.loginfo(f"  Skipping orientation check (gripper_down=False)")
            return clamp_joints(q)

        # Check orientation after Phase 1
        T_check = self.urdf_fk_full(q)
        z_after_phase1 = T_check[:3, 2]
        y_after_phase1 = T_check[:3, 1]  # Y-axis - should be horizontal
        x_after_phase1 = T_check[:3, 0]  # Finger opening direction
        final_pos = T_check[:3, 3]
        final_pos_error = np.linalg.norm(target_base - final_pos)

        # Check if Y-axis is horizontal (Y_z close to 0) - PRIMARY CONSTRAINT
        y_horizontal = abs(y_after_phase1[2]) < 0.15  # Y_z should be close to 0
        # Check if fingers are roughly horizontal (X_z close to 0)
        fingers_horizontal = abs(x_after_phase1[2]) < 0.5  # Allow up to ~30° tilt
        gripper_pointing_down = (
            z_after_phase1[2] < -0.5
        )  # Z has negative component (pointing down-ish)

        q = clamp_joints(q)
        rospy.loginfo(f"  Final joints (deg): [{', '.join([f'{np.degrees(a):.1f}' for a in q])}]")
        rospy.loginfo(
            f"  Final tool Z-axis: [{z_after_phase1[0]:.3f}, {z_after_phase1[1]:.3f}, {z_after_phase1[2]:.3f}]"
        )
        rospy.loginfo(
            f"  Final tool Y-axis: [{y_after_phase1[0]:.3f}, {y_after_phase1[1]:.3f}, {y_after_phase1[2]:.3f}] (Y_z={y_after_phase1[2]:.3f} should be ~0)"
        )
        rospy.loginfo(
            f"  Final tool X-axis: [{x_after_phase1[0]:.3f}, {x_after_phase1[1]:.3f}, {x_after_phase1[2]:.3f}]"
        )
        rospy.loginfo(f"  Final pos error: {final_pos_error*1000:.1f}mm")
        rospy.loginfo(f"  Fingers horizontal: {fingers_horizontal} (X_z={x_after_phase1[2]:.3f})")
        rospy.loginfo(
            f"  Gripper pointing down: {gripper_pointing_down} (Z_z={z_after_phase1[2]:.3f})"
        )
        rospy.loginfo(f"  Y-axis horizontal: {y_horizontal} (Y_z={y_after_phase1[2]:.3f})")

        # Accept if position is good AND Y-axis is horizontal AND gripper points down
        if final_pos_error < 0.03 and y_horizontal and gripper_pointing_down:
            rospy.loginfo(f"  ✓ IK SUCCESS (position + Y horizontal + gripper down)")
            return q
        elif final_pos_error < 0.03 and y_horizontal:
            # Position OK, Y horizontal, gripper not fully down - still acceptable
            rospy.logwarn(
                f"  ⚠ IK WARNING: Gripper not fully down (Z_z={z_after_phase1[2]:.3f}), but Y horizontal"
            )
            return q
        elif final_pos_error < 0.03 and gripper_pointing_down:
            # Position OK, gripper pointing down, but Y not horizontal - warn but accept
            rospy.logwarn(
                f"  ⚠ IK WARNING: Y not horizontal (Y_z={y_after_phase1[2]:.3f}), but gripper down"
            )
            return q
        elif final_pos_error < 0.03 and z_after_phase1[2] < -0.3:
            # Position OK, gripper pointing somewhat down (Z_z < -0.3) - acceptable
            rospy.logwarn(
                f"  ⚠ IK WARNING: Gripper not fully down (Z_z={z_after_phase1[2]:.3f}), but acceptable"
            )
            return q
        elif final_pos_error < 0.03:
            # Position OK but gripper pointing sideways or up - REJECT
            rospy.logerr(
                f"  ✗ IK FAILED: Gripper pointing sideways/up (Z_z={z_after_phase1[2]:.3f} > -0.3)"
            )
            return None
        else:
            rospy.logerr(f"  ✗ IK FAILED: pos={final_pos_error*1000:.1f}mm error")
            return None

        # PHASE 2 REMOVED - was causing more problems than it solved
        # The flexible orientation approach above is more robust

    def _analytical_ik(self, target_pos, gripper_down=True, current_joints=None):
        """
        Analytical IK using full UR5 inverse kinematics solver.

        Args:
            target_pos: Target position in BASE_LINK frame [x, y, z]
            gripper_down: Orient gripper downward

        Returns:
            Joint angles [6] or None if unreachable
        """
        # Get rotation matrix for desired orientation
        if gripper_down:
            R = self.gripper_down_rotation()
        else:
            R = np.eye(3)

        # Call analytical IK solver directly with target position
        solutions = self.ur5_inverse(target_pos, R)

        if solutions is None:
            rospy.logwarn("IK: No analytical solution found")
            return None

        # Use current joints as reference if available (better than home)
        reference = current_joints[:6] if current_joints is not None else None

        rospy.loginfo(f"IK: Found {solutions.shape[1]} solutions")
        rospy.loginfo(
            f"IK: Current joints (deg): {np.degrees(reference) if reference is not None else 'None'}"
        )

        # Log all valid solutions (just check for NaN, skip joint limit check)
        for i in range(solutions.shape[1]):
            sol = solutions[:, i]
            if not np.any(np.isnan(sol)):
                distance = np.max(
                    np.abs(sol - (reference if reference is not None else self.home_joints))
                )
                rospy.loginfo(f"  Sol {i}: {np.degrees(sol[:3])}, dist={distance:.2f}")

        # Select best solution (closest to current position or home)
        best_sol = self.select_best_solution(solutions, reference=reference)

        if best_sol is None:
            rospy.logwarn("IK: All solutions violate joint limits or contain NaN")
            return None

        # NOTE: We cannot verify IK with FK before execution because:
        # 1. DH-based FK doesn't match Gazebo URDF exactly
        # 2. TF-based FK only works for the CURRENT robot position, not hypothetical
        # Instead, we verify AFTER execution by comparing actual TF position with target

        rospy.loginfo(
            f"IK: Found solution (deg): [{np.degrees(best_sol[0]):.1f}, {np.degrees(best_sol[1]):.1f}, {np.degrees(best_sol[2]):.1f}, {np.degrees(best_sol[3]):.1f}, {np.degrees(best_sol[4]):.1f}, {np.degrees(best_sol[5]):.1f}]"
        )

        # Log FK check (informational only - don't reject based on it since DH/URDF mismatch)
        T_check = self.urdf_fk_full(best_sol)
        pos_check = T_check[:3, 3]
        z_check = T_check[:3, 2]
        pos_error = np.linalg.norm(pos_check - target_pos) * 1000  # mm

        rospy.loginfo(
            f"  FK check: pos=[{pos_check[0]:.3f}, {pos_check[1]:.3f}, {pos_check[2]:.3f}], Z=[{z_check[0]:.2f}, {z_check[1]:.2f}, {z_check[2]:.2f}]"
        )
        rospy.loginfo(f"  FK error: {pos_error:.1f}mm (DH/URDF mismatch expected)")

        return best_sol

    def _geometric_ik(self, target_pos, gripper_down=True):
        """
        Fallback geometric IK using simplified 2-link kinematics.

        Args:
            target_pos: Target position in BASE_LINK frame [x, y, z]
            gripper_down: Orient gripper downward
        """
        x, y, z = target_pos

        # Target is already in base_link frame (relative to robot base)
        # Calculate angle to target in XY plane (base rotation)
        q1 = np.arctan2(y, x)  # Base rotation to point at target

        # Distance to target in XY plane
        r = np.sqrt(x**2 + y**2)

        # Height (z is already relative to base)
        dz = z

        # Simple 2-link IK for shoulder and elbow (simplified UR5 kinematics)
        # UR5 actual link lengths
        L1 = 0.425  # Upper arm (shoulder to elbow)
        L2 = 0.392  # Forearm (elbow to wrist)

        # Total reach (approximate, considering wrist can extend further)
        L_max = 0.82  # UR5 max reach in meters

        # Distance to target
        d = np.sqrt(r**2 + dz**2)

        if d > L_max:  # Unreachable
            rospy.logwarn(f"IK: Target too far (d={d:.3f}m, max≈{L_max:.3f}m)")
            return None

        if d < 0.1:  # Too close (minimum reach)
            rospy.logwarn(f"IK: Target too close (d={d:.3f}m, min≈0.1m)")
            return None

        # For 2-link planar IK, we solve for the arm position
        # The wrist will extend from there
        d_arm = min(d * 0.9, L1 + L2 - 0.05)  # Target for 2-link solver

        # Elbow down configuration (typical for pick-and-place)
        # Using cosine law for 2-link planar arm
        cos_q3 = (d_arm**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_q3 = np.clip(cos_q3, -1, 1)
        q3 = np.arccos(cos_q3)  # Elbow angle (positive = bent)

        # Shoulder angle
        alpha = np.arctan2(-dz, r)
        beta = np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
        q2 = alpha - beta - np.pi / 2  # Shoulder angle (offset for UR5 frame)

        # Wrist angles to keep gripper pointing straight down
        # For downward pointing: the sum (q2 + q3 + q4) should equal π/2
        # This keeps the tool pointing down
        q4 = np.pi / 2 - (q2 + q3)  # Wrist 1 compensates to achieve downward orientation
        q5 = -np.pi / 2  # Wrist 2 rotates tool frame
        q6 = 0.0  # Wrist 3 (tool rotation around approach axis)

        joints = np.array([q1, q2, q3, q4, q5, q6])

        # Check joint limits
        if not np.all(joints >= self.joint_limits_lower) or not np.all(
            joints <= self.joint_limits_upper
        ):
            rospy.logwarn("IK: Solution violates joint limits")
            return None

        rospy.loginfo(
            f"IK: Found solution (deg): [{np.degrees(joints[0]):.1f}, {np.degrees(joints[1]):.1f}, {np.degrees(joints[2]):.1f}, {np.degrees(joints[3]):.1f}, {np.degrees(joints[4]):.1f}, {np.degrees(joints[5]):.1f}]"
        )
        return joints

    def _check_joint_limits(self, joints):
        """Check if joint angles are within limits (with angle wrapping)"""
        for i, (q, lower, upper) in enumerate(
            zip(joints, self.joint_limits_lower, self.joint_limits_upper)
        ):
            # Normalize angle to [-pi, pi] for comparison
            q_wrapped = np.arctan2(np.sin(q), np.cos(q))

            # Check if wrapped angle is within limits
            if q_wrapped < lower or q_wrapped > upper:
                rospy.logwarn(
                    f"Joint {i+1} out of limits: {np.degrees(q):.1f}° (wrapped: {np.degrees(q_wrapped):.1f}°, limits: {np.degrees(lower):.1f}° to {np.degrees(upper):.1f}°)"
                )
                return False
        return True

    def compute_ik(self, target_pose, initial_guess=None):
        """
        IK computation using simple geometric approach
        """
        return self.simple_ik(target_pose[:3])

    def move_to_pose(self, target_pos, target_orient, controller, duration=1.5):
        """
        Move to Cartesian pose using IK
        """
        joints = self.simple_ik(target_pos, gripper_down=True)
        if joints is None:
            return False
        return self.move_to_joints(joints, controller, duration)
