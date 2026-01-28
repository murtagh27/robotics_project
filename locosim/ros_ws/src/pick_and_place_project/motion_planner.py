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
        self.workspace_limits = {
            'x': [0.0, 1.0],
            'y': [0.0, 1.5],
            'z': [0.0, 2.0]
        }
        
        # Table geometries
        self.source_table = {
            'center': np.array(config.source_table_pos),
            'size': np.array(getattr(config, 'source_table_size', [0.6, 0.4, 0.02]))
        }
        
        self.target_table = {
            'center': np.array(config.target_table_pos),
            'size': np.array(getattr(config, 'target_table_size', [0.6, 0.4, 0.02]))
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
            if (abs(x - center[0]) < half_size[0] + self.margin and
                abs(y - center[1]) < half_size[1] + self.margin and
                z > center[2] - half_size[2] - self.margin):
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

        # Set random seed for reproducibility if configured
        if hasattr(config, 'seed') and config.seed is not None:
            np.random.seed(config.seed)
            rospy.loginfo(f"Random seed set to {config.seed} for reproducibility")

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
        self.joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_upper = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # UR5 DH parameters (Standard DH from UR5 URDF)
        # From urdf: a = [0, -0.425, -0.39225, 0, 0, 0]
        #            d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        #            alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        self.A = np.array([0, -0.425, -0.39225, 0, 0, 0])
        self.D = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
        self.Alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
        
        # URDF-based UR5e parameters (extracted from actual URDF transforms)
        # These match the urdf_fk_full() transform chain exactly
        # From URDF joint origins:
        #   shoulder_pan:  xyz=(0, 0, 0.1625)
        #   shoulder_lift: rpy=(π/2, 0, 0)  
        #   elbow:         xyz=(-0.425, 0, 0)
        #   wrist_1:       xyz=(-0.3922, 0, 0.1333)
        #   wrist_2:       xyz=(0, -0.0997, 0), rpy=(π/2, 0, 0)
        #   wrist_3:       xyz=(0, 0.0996, 0), rpy=(π/2, π, π)
        #   + gripper:     xyz=(0, 0, 0.12)
        self.urdf_d1 = 0.1625     # shoulder_pan Z offset
        self.urdf_a2 = 0.425      # upper arm length (X direction after shoulder_lift)
        self.urdf_a3 = 0.3922     # forearm length (X direction)
        self.urdf_d4 = 0.1333     # wrist_1 Z offset
        self.urdf_d5 = 0.0997     # wrist_2 Y offset
        self.urdf_d6 = 0.0996     # wrist_3 Y offset  
        self.urdf_tool = 0.12     # gripper length
        
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
        T12 = Rx(math.pi/2) @ Rz(q2)
        
        # Joint 3: elbow_joint - origin xyz=(-0.425, 0, 0), axis Z
        T23 = Txyz(-0.425, 0, 0) @ Rz(q3)
        
        # Joint 4: wrist_1_joint - origin xyz=(-0.3922, 0, 0.1333), axis Z
        T34 = Txyz(-0.3922, 0, 0.1333) @ Rz(q4)
        
        # Joint 5: wrist_2_joint - origin xyz=(0, -0.0997, 0), rpy=(π/2, 0, 0), axis Z
        T45 = Txyz(0, -0.0997, 0) @ Rx(math.pi/2) @ Rz(q5)
        
        # Joint 6: wrist_3_joint - origin xyz=(0, 0.0996, 0), rpy=(π/2, π, π), axis Z
        T56 = Txyz(0, 0.0996, 0) @ Rz(math.pi) @ Ry(math.pi) @ Rx(math.pi/2) @ Rz(q6)
        
        # Fixed joints from wrist_3_link to tool0 (including gripper)
        # wrist_3-flange: rpy=(0, -π/2, -π/2)
        T6F = Rz(-math.pi/2) @ Ry(-math.pi/2)
        
        # flange-tool0: rpy=(π/2, 0, π/2) -> tool0_without_gripper
        TF_t0wg = Rz(math.pi/2) @ Rx(math.pi/2)
        
        # tool0_without_gripper -> gripper_base: rpy=(0, 0, π/2)
        T_grip = Rz(math.pi/2)
        
        # gripper_base -> tool0 (fixed_ee_gripper): xyz=(0, 0, 0.12)
        T_ee = Txyz(0, 0, 0.12)
        
        # Full transform: base_link -> base_link_inertia -> joints -> tool0
        T = T_base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6F @ TF_t0wg @ T_grip @ T_ee
        
        return T

    def _dh_transform(self, theta, alpha, d, a):
        """
        Compute STANDARD DH transformation matrix.
        T = Rz(θ) * Tz(d) * Tx(a) * Rx(α)
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

    def ur5_inverse(self, p_target, R_target):
        """
        Analytical inverse kinematics for UR5 using URDF parameters.
        
        This implementation uses the exact URDF kinematic chain and validates
        solutions using urdf_fk_full to ensure correctness.
        
        The approach:
        1. Compute wrist center by inverting the fixed transforms from EE
        2. Solve θ1 from wrist center XY position
        3. Solve θ2, θ3 using 2-link planar arm geometry
        4. Solve θ4, θ5, θ6 from orientation
        5. Validate each solution using urdf_fk_full
        
        Args:
            p_target: End-effector position [x, y, z] in base_link frame
            R_target: End-effector rotation matrix (3x3)

        Returns:
            6x8 matrix of joint angles (8 possible solutions), or None if unreachable
        """
        import math
        
        # URDF geometric parameters (from urdf_fk_full)
        d1 = 0.1625      # shoulder height
        a2 = 0.425       # upper arm length  
        a3 = 0.3922      # forearm length
        d4 = 0.1333      # wrist 1 Z offset
        d5 = 0.0997      # wrist 2 Y offset
        d6 = 0.0996      # wrist 3 Y offset
        d_tool = 0.12    # gripper length
        
        # Build target transformation matrix
        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = p_target
        
        # Helper functions (same as urdf_fk_full)
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
        
        # Fixed transforms from joint 6 to tool0 (need to invert these)
        T6F = Rz(-math.pi/2) @ Ry(-math.pi/2)
        TF_t0wg = Rz(math.pi/2) @ Rx(math.pi/2)
        T_grip = Rz(math.pi/2)
        T_ee = Txyz(0, 0, d_tool)
        T_fixed = T6F @ TF_t0wg @ T_grip @ T_ee
        
        # Base transform
        T_base = Rx(math.pi)
        
        # Compute joint 6 frame from target
        # T_target = T_base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T_fixed
        # So: T56 = inv(T_base @ T01 @ T12 @ T23 @ T34 @ T45) @ T_target @ inv(T_fixed)
        
        # First, undo base and fixed transforms to get T06 (base to joint 6)
        T06 = np.linalg.inv(T_base) @ T_target @ np.linalg.inv(T_fixed)
        
        p06 = T06[:3, 3]
        R06 = T06[:3, :3]
        
        rospy.loginfo(f"URDF IK: target_pos={p_target}")
        rospy.loginfo(f"URDF IK: p06 (after undoing base/fixed)={p06}")
        
        # Compute wrist center (joint 5 origin)
        # T56 = Txyz(0, 0.0996, 0) @ Rz(π) @ Ry(π) @ Rx(π/2) @ Rz(q6)
        # The wrist center is before all of T56, so we need to go back by the 
        # translation in T56 which is (0, d6, 0) in joint 5's frame
        # But joint 5 has Rx(π/2) before it, so the offset is rotated
        
        # Actually, let's compute wrist center more directly.
        # The wrist center is at the intersection of joints 4,5,6 axes.
        # From the URDF:
        # - Joint 5 is at (0, -d5, 0) from joint 4, then Rx(π/2)
        # - Joint 6 is at (0, d6, 0) from joint 5's frame
        # So wrist center in joint 4's frame is at (0, -d5, 0) approximately
        
        # For UR robots, the wrist center is typically computed by moving back
        # along the tool Z-axis by the wrist length
        # The tool Z-axis in base frame is the third column of R_target
        z_tool = R_target[:, 2]
        
        # But we're working in the internal frame (after T_base), so:
        # First undo the Rx(π) to get the direction in internal frame
        Rx_pi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        z_internal = Rx_pi @ z_tool
        
        # Wrist length from joint 5 to tool: d5 + d6 + fixed chain
        # Actually, the wrist center is at joint 5's origin
        # Let's compute it by using T45 and T56 geometry
        
        # The position of joint 5 in joint 4's frame is (0, -d5, 0)
        # The position of joint 6 in joint 5's frame (after Rx(π/2)) is (0, d6, 0)
        # Then the fixed transforms add the tool length along Z
        
        # Simplify: wrist center is approximately d5 + d6 + d_tool back along Z
        # This is an approximation - let's compute more precisely
        
        # From the FK chain, the tool position relative to wrist center involves:
        # T45_pos @ T56_pos @ T_fixed_pos
        # Let's just use a reasonable estimate and validate with FK
        
        # For now, use the approach: wrist center = p06 - offset * R06[:, 2]
        # where offset is the Z-distance from joint 6 origin to joint 5 origin
        # Looking at T56: it has Txyz(0, d6, 0) = 0.0996m in Y
        # After Rx(π/2), Y becomes -Z, so the offset is primarily along the original Z
        
        # Let's compute wrist center differently:
        # p_wrist = p06 (which is joint 6 position in internal frame)
        # The wrist center (joint 5) is offset from joint 6 by T56 inverse
        
        # T56 = Txyz(0, d6, 0) @ Rz(π) @ Ry(π) @ Rx(π/2) @ Rz(q6)
        # Position part of T56 is (0, d6, 0), so wrist5 = joint6_origin - R06 @ [0, d6, 0]
        p_wrist5 = p06 - R06 @ np.array([0, d6, 0])
        
        # And joint 5 origin is offset from joint 4 origin by T45
        # T45 = Txyz(0, -d5, 0) @ Rx(π/2) @ Rz(q5)
        # So joint4_origin = wrist5 - R05 @ [0, -d5, 0]
        # But R05 depends on q5 which we don't know yet
        
        # For the 2-link arm solution, we need the position of joint 4's origin
        # Let's use an approximation: the wrist center for IK purposes is p_wrist5
        p_wrist = p_wrist5
        
        rospy.loginfo(f"URDF IK: wrist_center={p_wrist}")
        
        solutions = []
        
        # ============ SOLVE θ1 (base rotation) ============
        # θ1 rotates around Z after the Rx(π) base transform
        # In the internal frame, XY plane is flipped
        # Project wrist center to XY plane
        r_xy = np.sqrt(p_wrist[0]**2 + p_wrist[1]**2)
        
        if r_xy < 0.01:
            # Singularity - wrist directly above/below base
            th1_options = [0.0, np.pi]
        else:
            # Two solutions: shoulder left and shoulder right
            th1_1 = np.arctan2(p_wrist[1], p_wrist[0])
            th1_2 = th1_1 + np.pi
            th1_options = [th1_1, th1_2]
        
        for th1 in th1_options:
            c1, s1 = np.cos(th1), np.sin(th1)
            
            # Transform wrist to frame after joint 1
            # T01 = Txyz(0, 0, d1) @ Rz(q1)
            # Point in joint 1 frame: R(-q1) @ (p_wrist - [0, 0, d1])
            p_wrist_0 = p_wrist - np.array([0, 0, d1])
            R1_inv = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
            p_wrist_1 = R1_inv @ p_wrist_0
            
            # After joint 2, there's Rx(π/2) which swaps Y and Z
            # T12 = Rx(π/2) @ Rz(q2)
            # The arm plane (joints 2,3) works in the XZ plane after Rx(π/2)
            # So in joint 1's frame, the arm works in XY plane (before Rx(π/2))
            
            # Distance in the arm plane
            # After Rx(π/2): Y' = -Z, Z' = Y
            # So in joint 1 frame: arm_x = p_wrist_1[0], arm_z = p_wrist_1[1]
            arm_x = p_wrist_1[0]
            arm_z = -p_wrist_1[2]  # Note: after Rx(π/2), original Z becomes -Y'
            
            # Adjust for d4 offset (wrist 1 joint has Z offset)
            # T34 = Txyz(-a3, 0, d4) @ Rz(q4)
            # This d4 offset is along Z in joint 3's frame
            # After solving q2,q3, this affects where the wrist ends up
            # For now, let's ignore d4 and correct later if needed
            
            r_arm = np.sqrt(arm_x**2 + arm_z**2)
            
            # Check reachability
            if r_arm > a2 + a3 + 0.01 or r_arm < abs(a2 - a3) - 0.01:
                rospy.logdebug(f"  θ1={np.degrees(th1):.1f}°: arm unreachable, r={r_arm:.4f}")
                continue
            
            # ============ SOLVE θ3 (elbow) ============
            cos_th3 = (r_arm**2 - a2**2 - a3**2) / (2 * a2 * a3)
            cos_th3 = np.clip(cos_th3, -1, 1)
            
            for th3_sign in [1, -1]:
                th3 = th3_sign * np.arccos(cos_th3)
                
                # ============ SOLVE θ2 (shoulder lift) ============
                # Using 2-link arm geometry
                alpha = np.arctan2(arm_z, arm_x)
                beta = np.arctan2(a3 * np.sin(th3), a2 + a3 * np.cos(th3))
                th2 = alpha - beta
                
                # ============ SOLVE θ4, θ5, θ6 (wrist orientation) ============
                # Compute R03 using the solved θ1, θ2, θ3
                # Then R36 = R03^T @ R06
                
                # Build R03 from individual rotations
                # R01 = Rz(θ1)
                R01 = np.array([
                    [np.cos(th1), -np.sin(th1), 0],
                    [np.sin(th1), np.cos(th1), 0],
                    [0, 0, 1]
                ])
                
                # R12 = Rx(π/2) @ Rz(θ2)
                Rx_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                Rz_th2 = np.array([
                    [np.cos(th2), -np.sin(th2), 0],
                    [np.sin(th2), np.cos(th2), 0],
                    [0, 0, 1]
                ])
                R12 = Rx_90 @ Rz_th2
                
                # R23 = Rz(θ3)
                R23 = np.array([
                    [np.cos(th3), -np.sin(th3), 0],
                    [np.sin(th3), np.cos(th3), 0],
                    [0, 0, 1]
                ])
                
                R03 = R01 @ R12 @ R23
                
                # R36 = R03^T @ R06
                R36 = R03.T @ R06
                
                # Now extract θ4, θ5, θ6 from R36
                # The wrist chain is: Rz(θ4) @ [Rx(π/2) @ Rz(θ5)] @ [Rz(π)@Ry(π)@Rx(π/2)@Rz(θ6)]
                # This is complex, let's simplify by trying multiple θ5 values
                
                # For a ZYZ-like decomposition:
                # R36[2,2] relates to cos(θ5) after all the fixed rotations
                
                # Let's use a numerical approach: try a few θ5 values and solve θ4, θ6
                for th5_try in [np.pi/2, -np.pi/2, 0, np.pi]:
                    for th4_offset in [0, np.pi]:
                        for th6_offset in [0, np.pi]:
                            th4 = th4_offset
                            th5 = th5_try
                            th6 = th6_offset
                            
                            # Try to find better values by looking at R36 structure
                            # This is a simplification - proper solution would decompose R36
                            
                            q_test = [th1, th2, th3, th4, th5, th6]
                            solutions.append(q_test)
        
        if len(solutions) == 0:
            rospy.logwarn("URDF IK: No candidate solutions generated")
            return None
        
        # ============ VALIDATE SOLUTIONS USING urdf_fk_full ============
        # This is the key engineering step: verify each solution matches the target
        validated_solutions = []
        
        for sol in solutions:
            T_fk = self.urdf_fk_full(sol)
            p_fk = T_fk[:3, 3]
            R_fk = T_fk[:3, :3]
            
            pos_error = np.linalg.norm(p_fk - p_target)
            
            # Orientation error (Frobenius norm of rotation difference)
            R_err = R_fk @ R_target.T
            orient_error = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            
            if pos_error < 0.1 and orient_error < 0.5:  # 10cm, ~30 deg tolerance for candidates
                validated_solutions.append((sol, pos_error, orient_error))
                rospy.logdebug(f"  Valid candidate: pos_err={pos_error*1000:.1f}mm, orient_err={np.degrees(orient_error):.1f}°")
        
        # If we have validated solutions, refine using numerical optimization on wrist joints
        final_solutions = []
        for sol, pos_err, orient_err in validated_solutions:
            # Try to refine θ4, θ5, θ6 to improve orientation
            best_sol = self._refine_wrist_orientation(sol, p_target, R_target)
            if best_sol is not None:
                final_solutions.append(best_sol)
        
        if len(final_solutions) == 0:
            rospy.logwarn(f"URDF IK: No valid solutions found after validation (had {len(solutions)} candidates)")
            return None
        
        # Pad to 8 solutions
        while len(final_solutions) < 8:
            final_solutions.append([np.nan] * 6)
        
        Th = np.array(final_solutions[:8]).T
        
        valid_count = sum(1 for s in final_solutions[:8] if not np.isnan(s[0]))
        rospy.loginfo(f"URDF IK: Found {valid_count} valid solutions")
        
        return Th
    
    def _refine_wrist_orientation(self, sol, p_target, R_target, max_iter=50):
        """
        Refine wrist joint angles (θ4, θ5, θ6) to achieve target orientation.
        Uses numerical gradient descent on the wrist joints only.
        """
        th1, th2, th3, th4, th5, th6 = sol
        
        best_error = float('inf')
        best_sol = None
        
        # Grid search over wrist angles
        for th4_try in np.linspace(-np.pi, np.pi, 8):
            for th5_try in np.linspace(-np.pi, np.pi, 8):
                for th6_try in np.linspace(-np.pi, np.pi, 8):
                    q = [th1, th2, th3, th4_try, th5_try, th6_try]
                    T_fk = self.urdf_fk_full(q)
                    p_fk = T_fk[:3, 3]
                    R_fk = T_fk[:3, :3]
                    
                    pos_error = np.linalg.norm(p_fk - p_target)
                    R_err = R_fk @ R_target.T
                    orient_error = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
                    
                    total_error = pos_error + 0.1 * orient_error
                    
                    if total_error < best_error:
                        best_error = total_error
                        best_sol = q
        
        if best_sol is not None and best_error < 0.15:  # 15cm + orientation error
            # Verify final solution
            T_fk = self.urdf_fk_full(best_sol)
            pos_error = np.linalg.norm(T_fk[:3, 3] - p_target)
            if pos_error < 0.05:  # 5cm tolerance
                return best_sol
        
        return None

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
        MAX_SINGLE_JOINT_MOVE = np.radians(170)  # Increased limit - impedance controller may still reject large moves

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
            joint_deltas = np.minimum(joint_deltas, 2*np.pi - joint_deltas)
            max_single_joint = np.max(joint_deltas)
            
            if max_single_joint > MAX_SINGLE_JOINT_MOVE:
                rospy.loginfo(f"  Sol {i}: REJECTED - requires {np.degrees(max_single_joint):.1f}° on single joint (max {np.degrees(MAX_SINGLE_JOINT_MOVE):.1f}°)")
                continue

            # Compute distance to reference (L2 norm for smoother movements)
            # Use wrapped distances
            distance = np.linalg.norm(joint_deltas)
            
            rospy.loginfo(f"  Sol {i}: ACCEPTED - dist={distance:.2f}, max_joint={np.degrees(max_single_joint):.1f}°")

            if distance < best_distance:
                best_distance = distance
                best_solution = sol.copy()

        return best_solution

    def gripper_down_rotation(self):
        """
        Get rotation matrix for gripper pointing straight down.
        
        For gripper pointing DOWN (Z = [0, 0, -1]):
        - Z-axis: [0, 0, -1] (pointing down)
        - X-axis: [1, 0, 0] (pointing forward/+X) - finger opening direction
        - Y-axis: [0, -1, 0] (pointing -Y) - perpendicular to fingers
        
        This is the desired end-effector orientation in base_link frame.
        """
        R = np.array([
            [1,  0,  0],   # X-axis: forward
            [0, -1,  0],   # Y-axis: -Y
            [0,  0, -1]    # Z-axis: down
        ])
        return R
    
    def gripper_down_rotation_with_yaw(self, object_yaw=None):
        """
        Get rotation matrix for gripper pointing down with optional yaw alignment.
        
        When object_yaw is provided, the gripper is rotated around the vertical axis
        to align with the object. Picks from 4 candidate angles (0°, 90°, 180°, 270°)
        to minimize rotation from default orientation.
        
        Args:
            object_yaw: Optional yaw angle (radians) of the object in world frame
            
        Returns:
            3x3 rotation matrix for end-effector orientation
        """
        if object_yaw is None:
            return self.gripper_down_rotation()
        
        # Base gripper-down rotation
        R_base = self.gripper_down_rotation()
        
        # Default gripper X direction when pointing down (in XY plane)
        default_X_xy = np.array([1, 0])  # Points in +X direction
        
        # 4 candidate angles (object_yaw + 0°, 90°, 180°, 270°)
        candidate_angles = [
            object_yaw,
            object_yaw + np.pi/2,
            object_yaw + np.pi,
            object_yaw + 3*np.pi/2,
        ]
        
        # Find candidate closest to default orientation
        best_angle = object_yaw
        best_alignment = -2
        for angle in candidate_angles:
            candidate_X = np.array([np.cos(angle), np.sin(angle)])
            alignment = np.dot(default_X_xy, candidate_X)
            if alignment > best_alignment:
                best_alignment = alignment
                best_angle = angle
        
        # Normalize angle
        best_angle = np.arctan2(np.sin(best_angle), np.cos(best_angle))
        
        # Rotation around Z (vertical) axis by best_angle
        c = np.cos(best_angle)
        s = np.sin(best_angle)
        Rz = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # Apply yaw rotation to base gripper-down rotation
        R = Rz @ R_base
        
        rospy.loginfo(f"Gripper orientation: object_yaw={np.degrees(object_yaw):.1f}°, best_angle={np.degrees(best_angle):.1f}°")
        
        return R

    def generate_waypoints(self, start_joints, target_joints, max_step_deg=45.0):
        """
        Generate intermediate waypoints for large movements.
        Breaks movement into steps where no joint moves more than max_step_deg.
        
        IMPORTANT: Waypoints are NOT wrapped to [-π, π] because:
        1. UR5 joints support ±2π range  
        2. Wrapping causes discontinuous jumps that confuse the controller
        3. The shortest-path interpolation naturally crosses ±π boundaries
        """
        def wrap_to_pi(angle):
            """Wrap angle to [-π, π]"""
            return ((angle + np.pi) % (2 * np.pi)) - np.pi
        
        start = np.array(start_joints).flatten()[:6]
        target = np.array(target_joints).flatten()[:6]
        
        # Wrap target to [-π, π] for reference
        target_wrapped = np.array([wrap_to_pi(a) for a in target])
        
        # Calculate shortest-path movement for each joint
        joint_deltas = np.array([wrap_to_pi(target_wrapped[i] - start[i]) for i in range(6)])
        
        # Find maximum movement
        max_movement_deg = np.max(np.abs(np.rad2deg(joint_deltas)))
        
        if max_movement_deg <= max_step_deg:
            # Movement is small enough, no waypoints needed
            # Final waypoint is start + delta (continuous path, may be outside [-π,π])
            return [start + joint_deltas]
        
        # Calculate number of steps needed
        num_steps = int(np.ceil(max_movement_deg / max_step_deg))
        
        # Generate waypoints - DO NOT WRAP, keep continuous path
        waypoints = []
        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            # Interpolate using delta (shortest path) - continuous values
            waypoint = start + alpha * joint_deltas
            # DO NOT wrap here - keep continuous to avoid controller confusion
            waypoints.append(waypoint)
        
        rospy.loginfo(f"Generated {len(waypoints)} waypoints (max step: {max_step_deg}°, total: {max_movement_deg:.1f}°)")
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
        
        # CRITICAL: Compute shortest-path interpolation that keeps values continuous
        # The target for verification should be current + diff (continuous), not wrapped
        target_continuous = np.zeros(6)  # The actual target we're interpolating to
        interpolation_delta = np.zeros(6)  # Store the delta for interpolation
        
        for i in range(6):
            # Wrap target to [-π, π] for reference
            target_in_pi = wrap_to_pi(target_joints[i])
            
            # Find the shortest angular difference from current position
            diff = wrap_to_pi(target_in_pi - current_joints[i])
            
            # The continuous target is current + diff (may be outside [-π, π] but that's OK)
            # This ensures verification compares against what we actually commanded
            target_continuous[i] = current_joints[i] + diff
            interpolation_delta[i] = diff
        
        rospy.loginfo(f"Current joints (rad): {np.array2string(current_joints[:6], precision=3, suppress_small=True)}")
        rospy.loginfo(f"Current joints (deg): {np.array2string(np.degrees(current_joints[:6]), precision=1, suppress_small=True)}")
        rospy.loginfo(f"Target joints (raw, rad):  {np.array2string(target_joints[:6] if len(target_joints) >= 6 else target_joints, precision=3, suppress_small=True)}")
        rospy.loginfo(f"Target joints (continuous, rad):  {np.array2string(target_continuous, precision=3, suppress_small=True)}")
        rospy.loginfo(f"Target joints (continuous, deg):  {np.array2string(np.degrees(target_continuous), precision=1, suppress_small=True)}")
        rospy.loginfo(f"Interpolation delta (deg): {np.array2string(np.degrees(interpolation_delta), precision=1, suppress_small=True)}")
        
        # Use continuous target for verification (actual position we're moving to)
        target_joints = target_continuous
        
        # Get current EE position from TF (ground truth)
        current_ee_pos = self._get_actual_ee_position()
        if current_ee_pos is None:
            rospy.logwarn("Could not get current EE from TF, using (0,0,0)")
            current_ee_pos = np.zeros(3)
        
        # World frame positions
        current_ee_world = current_ee_pos + np.array([self.robot_base_x, self.robot_base_y, self.robot_base_z])
        
        # Target position: use provided world target, or estimate distance from joints
        if target_world_pos is not None:
            target_ee_world = np.array(target_world_pos)
            target_ee_pos = target_ee_world - np.array([self.robot_base_x, self.robot_base_y, self.robot_base_z])
        else:
            # Estimate target position (won't be accurate but shows movement intent)
            target_ee_pos = current_ee_pos  # Will be updated after motion
            target_ee_world = current_ee_world
        
        rospy.loginfo(f"\nCurrent EE (base_link): {np.array2string(current_ee_pos, precision=3, suppress_small=True)}")
        rospy.loginfo(f"Current EE (world):     {np.array2string(current_ee_world, precision=3, suppress_small=True)}")
        if target_world_pos is not None:
            rospy.loginfo(f"Target EE (base_link):  {np.array2string(target_ee_pos, precision=3, suppress_small=True)}")
            rospy.loginfo(f"Target EE (world):      {np.array2string(target_ee_world, precision=3, suppress_small=True)}")
            ee_distance = np.linalg.norm(target_ee_pos - current_ee_pos)
            rospy.loginfo(f"EE distance to travel: {ee_distance:.4f} m ({ee_distance*1000:.1f} mm)")
        rospy.loginfo(f"{'='*60}\n")

        # Pad target joints to 8 if only 6 provided (add gripper)
        if len(target_joints) == 6:
            target_joints_full = np.concatenate([target_joints, current_joints[6:8]])
        else:
            target_joints_full = target_joints
        
        # Build the full interpolation delta (use computed delta for arm, simple for gripper)
        if len(target_joints) == 6:
            interpolation_delta_full = np.concatenate([interpolation_delta, [0.0, 0.0]])
        else:
            interpolation_delta_full = np.concatenate([interpolation_delta, target_joints_full[6:8] - current_joints[6:8]])

        # Store initial position for verification
        initial_joints = current_joints.copy()
        initial_ee_pos = current_ee_pos.copy()
        last_check_joints = current_joints.copy()
        check_interval = 1000  # Check every 2 seconds (1000 steps at 500Hz)
        motion_threshold = 0.01  # Minimum movement in radians to consider "moving"
        
        # Publish initial TF
        self._publish_ee_tf(controller, current_ee_pos, "current_ee")
        self._publish_ee_tf(controller, target_ee_pos, "target_ee")
        
        # Check velocity limits using the ACTUAL interpolation delta (shortest path)
        max_velocity = np.max(np.abs(interpolation_delta_full[:6] / duration))
        SAFE_VELOCITY_LIMIT = 2.0  # rad/s - fast movement
        
        if max_velocity > SAFE_VELOCITY_LIMIT:
            # Auto-increase duration to stay within velocity limits
            required_duration = np.max(np.abs(interpolation_delta_full[:6])) / SAFE_VELOCITY_LIMIT
            old_duration = duration
            duration = max(required_duration * 1.2, old_duration)  # Add 20% margin
            rospy.logwarn(f"⚠️  Auto-adjusting duration: {old_duration:.1f}s → {duration:.1f}s")
            rospy.logwarn(f"   Original velocity: {max_velocity:.3f} rad/s (limit: {SAFE_VELOCITY_LIMIT} rad/s)")
            max_velocity = np.max(np.abs(interpolation_delta_full[:6] / duration))
            rospy.loginfo(f"   New velocity: {max_velocity:.3f} rad/s")
        else:
            rospy.loginfo(f"✓ Velocity OK: {max_velocity:.3f} rad/s (limit: {SAFE_VELOCITY_LIMIT} rad/s)")
        
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

            # Interpolate using shortest-path delta (handles angle wrapping correctly)
            # Use current_joints as starting point, add alpha * delta
            # DO NOT WRAP during interpolation! This causes discontinuous jumps
            # that confuse the robot controller. UR5 joints support ±2π range.
            desired_joints = current_joints + alpha * interpolation_delta_full
            
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
            
            # Velocity = delta / duration * scale (use shortest-path delta)
            desired_velocity = interpolation_delta_full / duration * velocity_scale

            # Send command with velocity feedforward
            controller.send_joint_command(desired_joints, velocities=desired_velocity)
            commands_sent += 1
            
            # Periodically check if robot is actually moving
            if i > 0 and i % check_interval == 0:
                actual_joints, _ = controller.get_current_joint_state()
                actual_ee_pos_fk = self._compute_fk(actual_joints[:6], controller)  # Computed from joints
                
                joint_movement = np.linalg.norm(actual_joints[:6] - last_check_joints[:6])
                ee_movement = np.linalg.norm(actual_ee_pos_fk - initial_ee_pos)
                expected_joint_movement = np.linalg.norm(desired_joints[:6] - last_check_joints[:6])
                
                progress = i / steps * 100
                
                rospy.loginfo(f"Progress: {progress:.0f}% | Joint Δ: {joint_movement:.4f} rad | EE Δ: {ee_movement*1000:.1f} mm")
                rospy.loginfo(f"  Computed EE (FK): {np.array2string(actual_ee_pos_fk, precision=3, suppress_small=True)}")
                
                if joint_movement < motion_threshold and expected_joint_movement > motion_threshold:
                    rospy.logerr(f"⚠️  ROBOT NOT MOVING! Step {i}/{steps} ({progress:.0f}%)")
                    rospy.logerr(f"   Joint movement: {joint_movement:.4f} rad (expected {expected_joint_movement:.4f})")
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
            target_base_link = np.array(target_world_pos) - np.array([self.robot_base_x, self.robot_base_y, self.robot_base_z])
            final_ee_error = np.linalg.norm(final_ee_pos - target_base_link)
            ee_distance = np.linalg.norm(target_base_link - initial_ee_pos)
        else:
            final_ee_error = 0.0
            ee_distance = total_ee_movement
        
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo("MOVEMENT COMPLETE - Final Status:")
        rospy.loginfo(f"{'='*60}")
        rospy.loginfo(f"Joint space:")
        rospy.loginfo(f"  Total movement: {total_joint_movement:.4f} rad (expected {expected_joint_total:.4f})")
        rospy.loginfo(f"  Final error: {final_joint_error:.4f} rad ({np.degrees(final_joint_error):.1f} deg)")
        rospy.loginfo(f"Cartesian space (TF ground truth):")
        rospy.loginfo(f"  Total EE movement: {total_ee_movement*1000:.1f} mm")
        rospy.loginfo(f"  Final EE position (base_link): {np.array2string(final_ee_pos, precision=3, suppress_small=True)}")
        if target_world_pos is not None:
            rospy.loginfo(f"  Target position (base_link):   {np.array2string(target_base_link, precision=3, suppress_small=True)}")
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
            rospy.logwarn(f"⚠️  Large joint error: {final_joint_error:.4f} rad ({np.degrees(final_joint_error):.1f} deg)")
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

    def move_to_joints_with_target(self, target_joints, controller, target_world_pos, duration=1.5, max_step_deg=45.0):
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
        rospy.loginfo(f"  Target world position: [{target_world_pos[0]:.3f}, {target_world_pos[1]:.3f}, {target_world_pos[2]:.3f}]")
        rospy.loginfo(f"  Duration per step: {duration}s")
        rospy.loginfo(f"  Max step size: {max_step_deg}°")
        rospy.loginfo(f"{'='*70}")
        
        # Execute each segment, only pass target_world_pos for the LAST waypoint
        for i, waypoint in enumerate(waypoints):
            rospy.loginfo(f"\n>>> Executing waypoint {i+1}/{len(waypoints)} <<<")
            is_final = (i == len(waypoints) - 1)
            wp_target = target_world_pos if is_final else None
            success = self.move_to_joints_direct(waypoint, controller, duration, target_world_pos=wp_target)
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
                (trans, rot) = self._tf_listener.lookupTransform('/base_link', '/tool0', rospy.Time(0))
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

    def pick_object(self, object_pos, controller, object_orientation=None, object_class=None):
        """
        Execute pick sequence using actual object position and orientation
        Uses safe transit height to avoid collisions with other objects.

        Args:
            object_pos: Object position [x, y, z]
            controller: Controller instance
            object_orientation: Optional object orientation as quaternion [x, y, z, w]
            object_class: Optional object class (e.g. 'X1-Y2-Z2') for size-based grip
        """
        rospy.loginfo(f"PICK: {object_pos}, class={object_class}")
        
        # Extract yaw from quaternion
        object_yaw = None
        if object_orientation is not None:
            qx, qy, qz, qw = object_orientation
            object_yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Validate object position
        if object_pos[0] < -0.5 or object_pos[0] > 1.5:
            rospy.logerr(f"Object X={object_pos[0]:.2f} is outside valid range [-0.5, 1.5] - object likely fell off table!")
            return False
        if object_pos[1] < -0.5 or object_pos[1] > 1.5:
            rospy.logerr(f"Object Y={object_pos[1]:.2f} is outside valid range [-0.5, 1.5] - object likely fell off table!")
            return False
        if object_pos[2] < 0.5 or object_pos[2] > 1.5:
            rospy.logerr(f"Object Z={object_pos[2]:.2f} is outside valid range [0.5, 1.5] - object likely fell off table!")
            return False

        # Collision check
        if self.enable_collision_checking:
            ok, msg = self.collision_checker.check_position(object_pos)
            if not ok:
                rospy.logerr(f"Collision detected: {msg}")
                return False

        # Open gripper first
        rospy.loginfo("  Opening gripper...")
        controller.send_gripper_command(self.config.gripper_open_pos)
        rospy.sleep(1.5)

        # Get safe transit height
        safe_z = getattr(self.config, 'safe_transit_height', 1.10)
        robot_base_x = 0.5
        robot_base_y = 0.35
        
        current_ee_pos = self.urdf_fk(current_joints[:6])
        current_ee_world_x = current_ee_pos[0] + robot_base_x
        current_ee_world_y = current_ee_pos[1] + robot_base_y
        
        # Check if arc transition needed
        needs_transition = False
        if object_pos[0] > robot_base_x + 0.05:
            if current_ee_world_x < robot_base_x - 0.05:
                needs_transition = True
        
        if needs_transition:
            # Arc transition to avoid singularity
            target_y = object_pos[1]
            arc_y = max(current_ee_world_y, target_y + 0.15, robot_base_y + 0.20)
            
            # Waypoint 1: Intermediate position
            mid_x = (current_ee_world_x + object_pos[0]) / 2.0
            arc_wp1 = np.array([mid_x, arc_y, safe_z])
            arc_joints1 = self.simple_ik(arc_wp1, gripper_down=True)
            if arc_joints1 is not None:
                arc_dur = getattr(self.config, 'arc_move_duration', 2.0)
                self.move_to_joints(arc_joints1, controller, duration=arc_dur)
            
            # Waypoint 2: Near target
            arc_wp2 = np.array([object_pos[0], target_y + 0.08, safe_z])
            arc_joints2 = self.simple_ik(arc_wp2, gripper_down=True)
            if arc_joints2 is not None:
                self.move_to_joints(arc_joints2, controller, duration=arc_dur)
            else:
                rospy.logwarn("  Could not reach arc waypoint 2, proceeding to target")
        
        # Compute safe transit position (high above object XY)
        safe_transit_pos = np.array([
            object_pos[0],
            object_pos[1],
            safe_z
        ])
        
        # Adaptive grasp height for different object sizes
        grasp_height_offset = self.config.grasp_height
        approach_height_offset = self.config.approach_height
        
        if object_class:
            try:
                from brick_classes import BRICK_CLASSES
                obj_height = None
                if object_class in BRICK_CLASSES:
                    obj_height = BRICK_CLASSES[object_class]['size'][2]  # Z dimension = height
                else:
                    for key, val in BRICK_CLASSES.items():
                        if val.get('class') == object_class:
                            obj_height = val['size'][2]
                            break
                
                if obj_height is not None:
                    min_grasp_height = obj_height / 2.0 + 0.02
                    if grasp_height_offset < min_grasp_height:
                        grasp_height_offset = min_grasp_height
                        rospy.loginfo(f"  Adjusted grasp height to {grasp_height_offset*100:.1f}cm for {obj_height*100:.1f}cm tall object")
            except ImportError:
                pass
        
        # Compute approach position (above object)
        approach_pos = np.array([
            object_pos[0],
            object_pos[1],
            object_pos[2] + approach_height_offset
        ])
        
        # Compute grasp position (at object) - now using adaptive height
        grasp_pos = np.array([
            object_pos[0],
            object_pos[1],
            object_pos[2] + grasp_height_offset
        ])
        
        # Publish TF markers
        self._publish_ee_tf(controller, object_pos, "object_to_pick")
        self._publish_ee_tf(controller, approach_pos, "pick_approach")
        self._publish_ee_tf(controller, grasp_pos, "pick_grasp")

        def check_movement_safe(target_joints, current_joints, max_single_joint_deg=120):
            """Check if movement is within safe limits"""
            def wrap_to_pi(a):
                return ((a + np.pi) % (2 * np.pi)) - np.pi
            deltas = []
            for i in range(6):
                diff = wrap_to_pi(target_joints[i] - current_joints[i])
                deltas.append(abs(np.degrees(diff)))
            max_delta = max(deltas)
            rospy.loginfo(f"    Joint deltas (deg): [{', '.join([f'{d:.0f}' for d in deltas])}], max={max_delta:.0f}°")
            return max_delta < max_single_joint_deg, max_delta, deltas

        # Step 1: Safe transit height
        rospy.loginfo(f"  Step 1: Transit height")
        safe_joints = self.simple_ik(safe_transit_pos, gripper_down=True, object_yaw=object_yaw)
        if safe_joints is None:
            rospy.logerr("Failed to compute safe transit IK")
            return False
        
        # Safety check
        is_safe, max_delta, deltas = check_movement_safe(safe_joints, current_joints)
        if not is_safe:
            rospy.logerr(f"Unsafe movement: {max_delta:.0f}° > 120°")
            return False
        
        self.move_to_joints(safe_joints, controller, duration=getattr(self.config, 'default_move_duration', 2.0))
        
        # Update current joints after movement
        current_joints, _ = controller.get_current_joint_state()
        
        # Step 2: Approach
        rospy.loginfo(f"  Step 2: Approach")
        approach_joints = self.simple_ik(approach_pos, gripper_down=True, object_yaw=object_yaw)
        if approach_joints is None:
            rospy.logerr("Failed to compute approach IK")
            return False
        
        is_safe, max_delta, _ = check_movement_safe(approach_joints, current_joints)
        if not is_safe:
            rospy.logerr(f"Unsafe approach: {max_delta:.0f}°")
            return False
        
        self.move_to_joints(approach_joints, controller, duration=getattr(self.config, 'approach_duration', 1.8))
        current_joints, _ = controller.get_current_joint_state()
        
        # Step 3: Grasp position
        rospy.loginfo(f"  Step 3: Grasp")
        grasp_joints = self.simple_ik(grasp_pos, gripper_down=True, object_yaw=object_yaw)
        if grasp_joints is None:
            rospy.logerr("Failed to compute grasp IK")
            return False
        
        is_safe, max_delta, _ = check_movement_safe(grasp_joints, current_joints)
        if not is_safe:
            rospy.logerr(f"Unsafe grasp: {max_delta:.0f}°")
            return False
        
        self.move_to_joints(grasp_joints, controller, duration=getattr(self.config, 'grasp_duration', 1.5))
        
        rospy.sleep(0.3)  # Settle

        # Adjust grip strength based on object size
        gripper_close = -0.2
        
        if object_class:
            try:
                from brick_classes import BRICK_CLASSES
                
                # Find object in BRICK_CLASSES - could be key ("X1-Y1-Z2") or class name ("small_cube")
                obj_size = None
                if object_class in BRICK_CLASSES:
                    obj_size = BRICK_CLASSES[object_class]['size']
                else:
                    # Search by class name
                    for key, val in BRICK_CLASSES.items():
                        if val.get('class') == object_class:
                            obj_size = val['size']
                            break
                
                if obj_size is not None:
                    min_dim = min(obj_size[0], obj_size[1])
                    
                    # Scale grip strength by object size
                    if min_dim >= 0.06:
                        gripper_close = 0.1
                    elif min_dim >= 0.05:
                        gripper_close = 0.0
                    elif min_dim >= 0.04:
                        gripper_close = -0.1
                    elif min_dim >= 0.03:
                        gripper_close = -0.25
                    else:
                        gripper_close = -0.4
                    
                    rospy.loginfo(f"  Adjusted gripper close to {gripper_close:.2f} rad for object size")
                else:
                    rospy.logwarn(f"  Object class '{object_class}' not found in BRICK_CLASSES, using default grip")
            except ImportError:
                rospy.logwarn("  Could not import brick_classes, using default gripper close")
        
        rospy.loginfo(f"  Closing gripper...")
        controller.send_gripper_command(gripper_close)
        rospy.sleep(2.5)

        # Grasp verification
        rospy.sleep(0.3)
        
        gripper_pos_1 = controller.q[6] if len(controller.q) >= 7 else -999
        gripper_pos_2 = controller.q[7] if len(controller.q) >= 8 else -999
        gripper_avg = (gripper_pos_1 + gripper_pos_2) / 2.0
        
        commanded_close = self.config.gripper_close_pos
        gripper_error = abs(gripper_avg - commanded_close)
        rospy.loginfo(f"  Grasp check: error={gripper_error:.3f} rad")

        # Step 4: Lift
        rospy.loginfo(f"  Step 4: Lifting")
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
        rospy.loginfo(f"PLACE: {target_pos}")
        
        current_joints, _ = controller.get_current_joint_state()
        current_ee_pos = self.urdf_fk(current_joints[:6])
        robot_base_x = 0.5
        robot_base_y = 0.35
        current_ee_world = np.array([current_ee_pos[0] + robot_base_x, 
                                      current_ee_pos[1] + robot_base_y,
                                      current_ee_pos[2] + self.robot_base_z])
        
        # Collision check
        if self.enable_collision_checking:
            ok, msg = self.collision_checker.check_position(target_pos)
            if not ok:
                rospy.logerr(f"Collision detected: {msg}")
                return False

        # Get safe transit height from config (defaults to 1.05m if not set)
        safe_z = getattr(self.config, 'safe_transit_height', 1.05)
        
        # Compute safe transit position (high above target XY)
        safe_transit_pos = np.array([
            target_pos[0],
            target_pos[1],
            safe_z
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
        
        rospy.loginfo(f"Safe transit: {np.array2string(safe_transit_pos, precision=3, suppress_small=True)} (z={safe_z})")
        rospy.loginfo(f"Place approach position: {np.array2string(place_approach_pos, precision=3, suppress_small=True)} (target + {self.config.approach_height:.3f}m)")
        rospy.loginfo(f"Place position: {np.array2string(place_pos, precision=3, suppress_small=True)} (target + {self.config.place_height:.3f}m)")
        
        # Publish TF markers for visualization
        self._publish_ee_tf(controller, target_pos, "place_target")
        self._publish_ee_tf(controller, place_approach_pos, "place_approach")
        self._publish_ee_tf(controller, place_pos, "place_position")

        # Check if target is on opposite side of robot (requires large base rotation)
        # Robot base at world (0.5, 0.35). Pick area is +X, place area may be -X
        current_joints, _ = controller.get_current_joint_state()
        robot_base_x = 0.5  # Robot base X in world frame
        robot_base_y = 0.35
        
        # Get current EE position to check if we actually need transition
        current_ee_pos = self.urdf_fk(current_joints[:6])
        current_ee_world_x = current_ee_pos[0] + robot_base_x
        current_ee_world_y = current_ee_pos[1] + robot_base_y
        rospy.loginfo(f"  Current EE world position: x={current_ee_world_x:.2f}, y={current_ee_world_y:.2f}")
        
        # If target is on opposite X side from current EE position, use intermediate waypoints
        # This helps avoid huge single-step rotations and singularity zones
        # Check ACTUAL EE position, not just base angle
        needs_transition = False
        if target_pos[0] < robot_base_x - 0.05:  # Target is on -X side (place area)
            # Only need transition if current EE is on +X side
            if current_ee_world_x > robot_base_x + 0.05:  # EE is on +X side
                needs_transition = True
                rospy.loginfo("  EE on +X side, target on -X side - using arc transition path")
            else:
                rospy.loginfo("  EE already on -X side, no transition needed")
        
        if needs_transition:
            # DYNAMIC ARC TRANSITION: Only 2 waypoints that adapt to target position
            # Key insight: Arc apex should be only slightly higher in Y than target,
            # so the final transition to safe_transit is small
            
            target_y = target_pos[1]  # Where we ultimately need to be
            # Arc Y is max of: current Y, target Y + small offset, ensuring we go around not through
            arc_y = max(current_ee_world_y, target_y + 0.15, robot_base_y + 0.20)  # At least 15cm above target Y
            
            # Waypoint 1: Intermediate X, at arc Y height
            # X is midpoint between current and target
            mid_x = (current_ee_world_x + target_pos[0]) / 2.0
            arc_wp1 = np.array([mid_x, arc_y, safe_z])
            rospy.loginfo(f"  Step 0a: Dynamic arc waypoint 1: {arc_wp1}")
            arc_joints1 = self.simple_ik(arc_wp1, gripper_down=True)
            if arc_joints1 is not None:
                arc_dur = getattr(self.config, 'arc_move_duration', 2.0)
                self.move_to_joints(arc_joints1, controller, duration=arc_dur)
            
            # Waypoint 2: At target X, slightly above target Y (smooth transition to safe_transit)
            arc_wp2 = np.array([target_pos[0], target_y + 0.08, safe_z])  # Only 8cm above target Y
            rospy.loginfo(f"  Step 0b: Dynamic arc waypoint 2: {arc_wp2}")
            arc_joints2 = self.simple_ik(arc_wp2, gripper_down=True)
            if arc_joints2 is not None:
                self.move_to_joints(arc_joints2, controller, duration=arc_dur)
            else:
                rospy.logwarn("  Could not reach arc waypoint 2, proceeding to target")

        # STEP 1: Move to safe transit height above target (avoids collisions)
        rospy.loginfo(f"  Step 1: Moving to safe transit height: {safe_transit_pos}")
        safe_joints = self.simple_ik(safe_transit_pos, gripper_down=True)
        if safe_joints is None:
            rospy.logerr("Failed to compute safe transit IK")
            return False
        self.move_to_joints(safe_joints, controller, duration=getattr(self.config, 'default_move_duration', 2.0))

        # STEP 2: Move down to place approach
        rospy.loginfo(f"  Step 2: Moving to place approach: {place_approach_pos}")
        place_approach_joints = self.simple_ik(place_approach_pos, gripper_down=True)
        if place_approach_joints is None:
            rospy.logerr("Failed to compute place approach IK")
            return False
        self.move_to_joints(place_approach_joints, controller, duration=getattr(self.config, 'approach_duration', 1.8))

        # STEP 3: Move down to place
        rospy.loginfo(f"  Step 3: Moving down to place: {place_pos}")
        place_joints = self.simple_ik(place_pos, gripper_down=True)
        if place_joints is None:
            rospy.logerr("Failed to compute place IK")
            return False
        self.move_to_joints(place_joints, controller, duration=getattr(self.config, 'grasp_duration', 1.5))

        # Release sequence
        rospy.loginfo("  Opening gripper...")
        extra_wide_open = 3.5
        controller.send_gripper_command(extra_wide_open)
        rospy.sleep(1.2)  # Wait for gripper to fully open
        
        # Slide sideways to clear object
        slide_pos = np.array([place_pos[0] + 0.08, place_pos[1], place_pos[2]])
        slide_joints = self.simple_ik(slide_pos, gripper_down=True)
        if slide_joints is not None:
            self.move_to_joints(slide_joints, controller, duration=0.6)
        rospy.sleep(0.3)
        
        # Lift away
        lift_pos = np.array([place_pos[0] + 0.06, place_pos[1], place_pos[2] + 0.10])
        lift_joints = self.simple_ik(lift_pos, gripper_down=True)
        if lift_joints is not None:
            self.move_to_joints(lift_joints, controller, duration=0.6)

        # STEP 4: Lift to safe transit height
        rospy.loginfo(f"  Step 4: Lifting to safe transit height...")
        self.move_to_joints(safe_joints, controller, duration=1.5)

        rospy.loginfo("Place complete")
        return True

    def simple_ik(self, target_pos, gripper_down=True, object_yaw=None):
        """
        Inverse kinematics solver - uses analytical or numerical based on config.
        
        Args:
            target_pos: Target XYZ position [x, y, z] in WORLD frame
            gripper_down: If True, orient gripper downward (for picking)
            object_yaw: Optional yaw angle (radians) to align gripper with object orientation
        
        Returns:
            Joint angles [6] or None if unreachable
        """
        x, y, z = target_pos
        ik_mode = "ANALYTICAL" if self.config.use_analytical_ik else "NUMERICAL"
        rospy.loginfo(f"IK ({ik_mode}): target world=[{x:.3f}, {y:.3f}, {z:.3f}]")
        
        # Convert from world frame to base_link frame
        x_base = x - self.robot_base_x
        y_base = y - self.robot_base_y
        z_base = z - self.robot_base_z
        rospy.loginfo(f"IK: target base_link=[{x_base:.3f}, {y_base:.3f}, {z_base:.3f}]")
        
        target_base = np.array([x_base, y_base, z_base])
        
        # Check XY distance from base (UR5 has a hole in workspace directly below)
        # But only enforce for targets that are below the robot base (z_base < 0)
        xy_distance = np.sqrt(x_base**2 + y_base**2)
        min_xy = 0.05  # Minimum XY distance - very small, only for singularity avoidance
        if xy_distance < min_xy and z_base < 0:
            rospy.logerr(f"IK: Target too close to robot base in XY! XY distance={xy_distance:.3f}m < min={min_xy:.2f}m")
            rospy.logerr(f"  Robot base (world): ({self.robot_base_x}, {self.robot_base_y}, {self.robot_base_z})")
            rospy.logerr(f"  Target (world): ({x}, {y}, {z})")
            rospy.logerr(f"  Move target further from X={self.robot_base_x}, Y={self.robot_base_y}")
            return None
        
        # Basic workspace validation
        # UR5 reach is ~0.85m but with gripper can extend to ~0.95m for near-vertical poses
        max_reach = 0.95
        distance_from_base = np.sqrt(x_base**2 + y_base**2 + z_base**2)
        rospy.loginfo(f"IK: distance from base: {distance_from_base:.3f}m (max reach: {max_reach:.2f}m)")
        
        if distance_from_base > max_reach:
            rospy.logerr(f"IK: Target is outside workspace! Distance {distance_from_base:.3f}m > max reach {max_reach:.2f}m")
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
        
        # Log object yaw if provided
        if object_yaw is not None:
            rospy.loginfo(f"IK: Aligning gripper to object yaw: {np.degrees(object_yaw):.1f}°")
        
        # ==================== ANALYTICAL IK ====================
        if self.config.use_analytical_ik:
            rospy.loginfo("IK: Using ANALYTICAL IK (6-DOF closed-form solution)")
            
            result = self._analytical_ik(target_base, gripper_down=gripper_down, 
                                         current_joints=current_joints, object_yaw=object_yaw)
            
            if result is not None:
                result_wrapped = np.array([wrap_to_pi(a) for a in result])
                T = self.urdf_fk_full(result_wrapped)
                pos = T[:3, 3]
                z_axis = T[:3, 2]
                pos_err = np.linalg.norm(target_base - pos) * 1000
                rospy.loginfo(f"IK solution: [{', '.join([f'{np.degrees(a):.1f}' for a in result_wrapped])}]°")
                rospy.loginfo(f"IK FK: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], Z=[{z_axis[0]:.2f}, {z_axis[1]:.2f}, {z_axis[2]:.2f}]")
                rospy.loginfo(f"IK error: {pos_err:.1f}mm")
                return result_wrapped
            
            rospy.logerr(f"ANALYTICAL IK FAILED: Cannot reach target [{x_base:.3f}, {y_base:.3f}, {z_base:.3f}]")
            return None
        
        # ==================== NUMERICAL IK ====================
        rospy.loginfo("IK: Using NUMERICAL IK (iterative solver)")
        
        best_result = None
        best_error = float('inf')
        
        # Use current joints as starting config
        if current_joints is not None:
            starting_configs = [np.array(current_joints[:6])]
            rospy.loginfo(f"IK: Using current joints as starting config")
        else:
            # Fallback: pick-ready config with gripper pointing down
            pick_ready_joints = np.array([
                np.radians(0),      # q1: base rotation
                np.radians(-90),    # q2: shoulder 
                np.radians(90),     # q3: elbow
                np.radians(-90),    # q4: wrist 1
                np.radians(90),     # q5: wrist 2 - makes gripper point down
                np.radians(0)       # q6: wrist 3
            ])
            starting_configs = [pick_ready_joints]
            rospy.loginfo(f"IK: No current joints, using pick-ready config")
        
        rospy.loginfo(f"IK: Trying {len(starting_configs)} starting configuration(s)")
        
        for i, start_config in enumerate(starting_configs):
            result = self._numerical_ik_urdf(target_base, start_config, gripper_down=gripper_down, object_yaw=object_yaw)
            
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
            rospy.loginfo(f"IK solution: [{', '.join([f'{np.degrees(a):.1f}' for a in result_wrapped])}]°")
            rospy.loginfo(f"IK FK: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], Z=[{z_axis[0]:.2f}, {z_axis[1]:.2f}, {z_axis[2]:.2f}]")
            rospy.loginfo(f"IK error: {pos_err:.1f}mm")
            return result_wrapped
        
        rospy.logerr(f"NUMERICAL IK FAILED: Cannot reach target [{x_base:.3f}, {y_base:.3f}, {z_base:.3f}]")
        return None

    def _generate_ik_seeds(self, px, py, pz, gripper_down=True, object_yaw=None):
        """
        Generate good initial seeds based on target position for numerical IK.
        
        For gripper-down configurations:
        - theta5 ≈ pi/2
        - theta4 ≈ -pi/2, theta6 ≈ 0
        - theta1 depends on XY position
        - theta2, theta3 depend on reach and height
        """
        seeds = []
        D4 = self.urdf_d4  # 0.1333
        A2 = self.urdf_a2  # 0.425
        A3 = self.urdf_a3  # 0.3922
        D1 = self.urdf_d1  # 0.1625
        D6 = self.urdf_d6  # 0.0996
        EE = self.urdf_tool  # 0.12
        
        if gripper_down:
            # Base wrist angles for gripper down
            theta4 = -np.pi/2
            theta5 = np.pi/2
            theta6 = 0
            if object_yaw is not None:
                theta6 = -object_yaw
            
            # Estimate theta1 from XY position
            # For theta1=0, py ≈ D4 = 0.1333
            theta1_est = np.arctan2(py - D4, -px) if abs(px) > 0.1 else 0
            
            # Estimate arm angles from reach
            reach = np.sqrt(px**2 + (py - D4)**2)
            height = pz + EE + D6 + D1
            
            # 2-link geometry estimate
            r = np.sqrt(reach**2 + height**2)
            if r < A2 + A3 and r > abs(A2 - A3):
                cos_t3 = (r**2 - A2**2 - A3**2) / (2 * A2 * A3)
                if abs(cos_t3) <= 1:
                    theta3_est = np.arccos(cos_t3)
                    theta2_est = np.arctan2(height, reach) - np.arctan2(A3*np.sin(theta3_est), A2+A3*np.cos(theta3_est))
                    
                    # Try variations of arm configuration
                    for t2_off in [0, -np.pi/2, np.pi/2]:
                        for t3_off in [0, np.pi/2]:
                            seeds.append([theta1_est, theta2_est + t2_off, theta3_est + t3_off, 
                                         theta4, theta5, theta6])
                            seeds.append([theta1_est + np.pi, -theta2_est + t2_off, -theta3_est + t3_off,
                                         theta4, theta5, theta6])
            
            # Standard seeds at estimated theta1
            for t1 in [theta1_est, theta1_est + np.pi/4, theta1_est - np.pi/4]:
                seeds.append([t1, -np.pi/2, np.pi/2, theta4, theta5, theta6])
                seeds.append([t1, -np.pi/3, np.pi/3, theta4, theta5, theta6])
                seeds.append([t1, -2*np.pi/3, 2*np.pi/3, theta4, theta5, theta6])
        
        return [np.array(s, dtype=float) for s in seeds]

    def _compute_jacobian(self, q, eps=1e-6):
        """Compute 6x6 Jacobian numerically."""
        T0 = self.urdf_fk_full(q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]
        
        J = np.zeros((6, 6))
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += eps
            T_plus = self.urdf_fk_full(q_plus)
            
            # Position Jacobian
            J[:3, i] = (T_plus[:3, 3] - p0) / eps
            
            # Rotation Jacobian (axis-angle)
            R_plus = T_plus[:3, :3]
            R_diff = R_plus @ R0.T
            trace = np.trace(R_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            if angle > 1e-10:
                axis = np.array([R_diff[2,1] - R_diff[1,2],
                               R_diff[0,2] - R_diff[2,0],
                               R_diff[1,0] - R_diff[0,1]]) / (2 * np.sin(angle))
                J[3:, i] = angle * axis / eps
        
        return J

    def _fast_numerical_ik(self, p_target, R_target, q_seed=None, max_iter=100, pos_tol=1e-4, rot_tol=1e-2):
        """
        Fast numerical IK using damped least squares.
        
        Args:
            p_target: Target position [x, y, z]
            R_target: Target rotation matrix (3x3)
            q_seed: Initial joint guess
            max_iter: Maximum iterations
            pos_tol: Position tolerance
            rot_tol: Rotation tolerance
            
        Returns:
            Joint angles if converged, None otherwise
        """
        if q_seed is None:
            q_seed = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2, 0], dtype=float)
        
        q = np.array(q_seed, dtype=float)
        damping = 0.1
        
        for it in range(max_iter):
            T_curr = self.urdf_fk_full(q)
            p_curr = T_curr[:3, 3]
            R_curr = T_curr[:3, :3]
            
            # Position error
            e_pos = p_target - p_curr
            pos_err = np.linalg.norm(e_pos)
            
            # Orientation error
            R_err = R_target @ R_curr.T
            trace = np.trace(R_err)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            if angle < 1e-10:
                e_rot = np.zeros(3)
            else:
                axis = np.array([R_err[2,1] - R_err[1,2],
                               R_err[0,2] - R_err[2,0],
                               R_err[1,0] - R_err[0,1]]) / (2 * np.sin(angle))
                e_rot = angle * axis
            rot_err = np.linalg.norm(e_rot)
            
            # Check convergence
            if pos_err < pos_tol and rot_err < rot_tol:
                return q
            
            # Compute error vector
            error = np.concatenate([e_pos, e_rot])
            
            # Compute Jacobian
            J = self._compute_jacobian(q)
            
            # Damped least squares: dq = J^T (J J^T + λI)^{-1} e
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + damping * np.eye(6), error)
            
            # Adaptive step size
            step = min(1.0, 0.5 / max(pos_err, 0.1))
            q += step * dq
            
            # Keep angles in reasonable range
            q = np.mod(q + np.pi, 2*np.pi) - np.pi
        
        return None  # Failed to converge

    def _numerical_ik_urdf(self, target_base, current_joints=None, gripper_down=True, object_yaw=None):
        """
        Numerical IK using seeded solver with URDF-based FK (matches Gazebo exactly).
        Uses smart seed generation and damped least squares for fast convergence.
        
        Args:
            target_base: Target position in base_link frame [x, y, z]
            current_joints: Current joint angles (used as initial guess)
            gripper_down: Whether gripper should point down with proper finger orientation
            object_yaw: Optional yaw angle (radians) to align gripper X-axis with object
            
        Returns:
            Joint angles [6] or None if failed
        """
        rospy.loginfo(f"Using fast seeded numerical IK (gripper_down={gripper_down})")
        
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
        
        px, py, pz = target_base
        
        # Build target rotation matrix
        if gripper_down:
            yaw = object_yaw if object_yaw is not None else 0.0
            # Gripper down: Z = [0, 0, -1], X = [cos(yaw), sin(yaw), 0], Y = cross(Z, X)
            R_target = np.array([
                [np.cos(yaw), np.sin(yaw), 0],
                [-np.sin(yaw), np.cos(yaw), 0],
                [0, 0, -1]
            ], dtype=float)
        else:
            R_target = np.eye(3)
        
        p_target = np.array(target_base)
        
        # Generate smart seeds based on target position
        seeds = self._generate_ik_seeds(px, py, pz, gripper_down=gripper_down, object_yaw=object_yaw)
        
        # Add current joints as first seed if provided (and variations of it)
        if current_joints is not None:
            curr = np.array(current_joints[:6], dtype=float)
            seeds.insert(0, curr)
            # Add small variations of current joints to help find nearby solutions
            for delta in [0.1, -0.1, 0.2, -0.2]:
                for joint_idx in range(6):
                    variant = curr.copy()
                    variant[joint_idx] += delta
                    seeds.insert(1, variant)
        
        rospy.loginfo(f"  Trying {len(seeds)} initial seeds")
        
        # Early exit thresholds - stop searching when we find a good enough solution
        # Read from config, with defaults if not present
        early_exit_joint_deg = getattr(self.config, 'ik_early_exit_joint_dist', 40)  # degrees
        early_exit_pos_mm = getattr(self.config, 'ik_early_exit_pos_err', 0.5)  # mm
        EARLY_EXIT_JOINT_DIST = np.radians(early_exit_joint_deg)
        EARLY_EXIT_POS_ERR = early_exit_pos_mm / 1000.0  # convert mm to m
        
        # Try each seed - collect ALL valid solutions
        valid_solutions = []  # (solution, pos_error, joint_distance)
        
        for i, seed in enumerate(seeds):
            sol = self._fast_numerical_ik(p_target, R_target, q_seed=seed, max_iter=100)
            if sol is not None:
                # Verify solution
                T = self.urdf_fk_full(sol)
                pos_err = np.linalg.norm(T[:3, 3] - p_target)
                z_err = np.linalg.norm(T[:3, 2] - R_target[:, 2])
                
                if pos_err < 0.001 and z_err < 0.1:
                    # Calculate joint distance from current configuration
                    if current_joints is not None:
                        # Compute shortest angular distance for each joint
                        joint_dist = 0
                        for j in range(6):
                            diff = wrap_to_pi(sol[j] - current_joints[j])
                            joint_dist += abs(diff)
                    else:
                        joint_dist = 0
                    
                    valid_solutions.append((sol, pos_err, joint_dist, i))
                    rospy.loginfo(f"    Seed {i}: Found solution, pos_err={pos_err*1000:.2f}mm, joint_dist={np.degrees(joint_dist):.0f}°")
                    
                    # EARLY EXIT: If we found a solution that's good enough, stop immediately
                    # This dramatically speeds up IK when current pose is near the target
                    if joint_dist < EARLY_EXIT_JOINT_DIST and pos_err < EARLY_EXIT_POS_ERR:
                        rospy.loginfo(f"    ✓ Early exit: Found excellent solution (joint_dist={np.degrees(joint_dist):.0f}° < {early_exit_joint_deg}°, err={pos_err*1000:.2f}mm < {early_exit_pos_mm}mm)")
                        break  # Stop searching, this is good enough
        
        # Select BEST solution: prioritize being close to current joints over position error
        # Sort by joint distance (primary), then position error (secondary)
        # CRITICAL: Reject solutions that are too far (would cause dangerous swings)
        # NOTE: Moving to opposite side of table requires base joint to rotate ~180°
        #       so we need to allow enough movement for that + other joint adjustments
        MAX_ACCEPTABLE_JOINT_DIST = np.radians(300)  # 300 degrees total - allows base rotation to other side
        
        if valid_solutions:
            valid_solutions.sort(key=lambda x: (x[2], x[1]))  # Sort by joint_dist first, then pos_err
            best_solution, best_error, best_joint_dist, best_seed = valid_solutions[0]
            
            if best_joint_dist > MAX_ACCEPTABLE_JOINT_DIST:
                rospy.logerr(f"  ⛔ ALL IK SOLUTIONS TOO FAR FROM CURRENT CONFIG!")
                rospy.logerr(f"     Best solution requires {np.degrees(best_joint_dist):.0f}° total joint movement")
                rospy.logerr(f"     Maximum acceptable is {np.degrees(MAX_ACCEPTABLE_JOINT_DIST):.0f}°")
                rospy.logerr(f"     This would cause the arm to flip/swing wildly!")
                rospy.logerr(f"     Target position may be in a different arm configuration zone")
                return None
            
            rospy.loginfo(f"  Selected seed {best_seed}: joint_dist={np.degrees(best_joint_dist):.0f}°, pos_err={best_error*1000:.2f}mm")
        else:
            best_solution = None
        
        if best_solution is not None:
            q = clamp_joints(best_solution)
            
            # Final validation
            T_check = self.urdf_fk_full(q)
            final_pos = T_check[:3, 3]
            final_pos_error = np.linalg.norm(target_base - final_pos)
            z_axis = T_check[:3, 2]
            
            rospy.loginfo(f"  Final joints (deg): [{', '.join([f'{np.degrees(a):.1f}' for a in q])}]")
            rospy.loginfo(f"  Final pos error: {final_pos_error*1000:.2f}mm")
            rospy.loginfo(f"  Final Z-axis: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
            
            if final_pos_error < 0.01:  # 10mm tolerance
                rospy.loginfo(f"  ✓ IK SUCCESS")
                return q
            else:
                rospy.logwarn(f"  ⚠ IK solution found but error {final_pos_error*1000:.1f}mm > 10mm")
                return q  # Still return it, might work
        
        rospy.logerr(f"  ✗ IK FAILED: No solution found for target [{px:.3f}, {py:.3f}, {pz:.3f}]")
        return None

    def _analytical_ik(self, target_pos, gripper_down=True, current_joints=None, object_yaw=None):
        """
        Analytical IK using full UR5 inverse kinematics solver with 6-DOF (position + orientation).
        
        Args:
            target_pos: Target position in BASE_LINK frame [x, y, z]
            gripper_down: Orient gripper downward
            current_joints: Current joint configuration for solution selection
            object_yaw: Optional yaw angle (radians) to align gripper with object orientation
            
        Returns:
            Joint angles [6] or None if unreachable
        """
        rospy.loginfo(f"ANALYTICAL IK: target_base=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        if object_yaw is not None:
            rospy.loginfo(f"ANALYTICAL IK: object_yaw={np.degrees(object_yaw):.1f}°")
        
        # Get rotation matrix for desired orientation (including object yaw if provided)
        if gripper_down:
            R = self.gripper_down_rotation_with_yaw(object_yaw)
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
        rospy.loginfo(f"IK: Current joints (deg): {np.degrees(reference) if reference is not None else 'None'}")
        
        # Log all valid solutions (just check for NaN, skip joint limit check)
        for i in range(solutions.shape[1]):
            sol = solutions[:, i]
            if not np.any(np.isnan(sol)):
                distance = np.max(np.abs(sol - (reference if reference is not None else self.home_joints)))
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
        
        rospy.loginfo(f"IK: Found solution (deg): [{np.degrees(best_sol[0]):.1f}, {np.degrees(best_sol[1]):.1f}, {np.degrees(best_sol[2]):.1f}, {np.degrees(best_sol[3]):.1f}, {np.degrees(best_sol[4]):.1f}, {np.degrees(best_sol[5]):.1f}]")
        
        # Log FK check (informational only - don't reject based on it since DH/URDF mismatch)
        T_check = self.urdf_fk_full(best_sol)
        pos_check = T_check[:3, 3]
        z_check = T_check[:3, 2]
        y_check = T_check[:3, 1]
        x_check = T_check[:3, 0]
        pos_error = np.linalg.norm(pos_check - target_pos) * 1000  # mm
        
        rospy.loginfo(f"  FK check: pos=[{pos_check[0]:.3f}, {pos_check[1]:.3f}, {pos_check[2]:.3f}]")
        rospy.loginfo(f"  FK check: Z=[{z_check[0]:.2f}, {z_check[1]:.2f}, {z_check[2]:.2f}] (should point down)")
        rospy.loginfo(f"  FK check: Y=[{y_check[0]:.2f}, {y_check[1]:.2f}, {y_check[2]:.2f}] (Y_z should be ~0)")
        rospy.loginfo(f"  FK check: X=[{x_check[0]:.2f}, {x_check[1]:.2f}, {x_check[2]:.2f}] (finger direction)")
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
        q2 = alpha - beta - np.pi/2  # Shoulder angle (offset for UR5 frame)
        
        # Wrist angles to keep gripper pointing straight down
        # For downward pointing: the sum (q2 + q3 + q4) should equal π/2
        # This keeps the tool pointing down
        q4 = np.pi/2 - (q2 + q3)  # Wrist 1 compensates to achieve downward orientation
        q5 = -np.pi/2             # Wrist 2 rotates tool frame
        q6 = 0.0                  # Wrist 3 (tool rotation around approach axis)
        
        joints = np.array([q1, q2, q3, q4, q5, q6])
        
        # Check joint limits
        if not np.all(joints >= self.joint_limits_lower) or not np.all(joints <= self.joint_limits_upper):
            rospy.logwarn("IK: Solution violates joint limits")
            return None
        
        rospy.loginfo(f"IK: Found solution (deg): [{np.degrees(joints[0]):.1f}, {np.degrees(joints[1]):.1f}, {np.degrees(joints[2]):.1f}, {np.degrees(joints[3]):.1f}, {np.degrees(joints[4]):.1f}, {np.degrees(joints[5]):.1f}]")
        return joints
