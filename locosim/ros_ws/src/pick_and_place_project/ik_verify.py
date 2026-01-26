#!/usr/bin/env python3
"""
IK Verification Script - Systematic testing of FK/IK consistency.
"""

import numpy as np

def dh_transform(theta, alpha, d, a):
    """Standard DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])

def forward_kinematics(joints):
    """Compute FK for UR5."""
    A = np.array([0, -0.425, -0.3922, 0, 0, 0])
    D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
    Alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(joints[i], Alpha[i], D[i], A[i])
    return T

def verify_fk_ik_roundtrip(controller):
    """
    Complete verification of FK/IK roundtrip at home position.
    """
    print("\n" + "="*70)
    print("COMPLETE FK/IK VERIFICATION")
    print("="*70)

    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])

    # Step 1: FK at home
    print("\n[1] FORWARD KINEMATICS AT HOME")
    T_home = forward_kinematics(home)
    pos_home = T_home[:3, 3]
    R_home = T_home[:3, :3]

    print(f"Home joints (deg): {np.degrees(home)}")
    print(f"FK position: [{pos_home[0]:.4f}, {pos_home[1]:.4f}, {pos_home[2]:.4f}]")
    print(f"FK rotation:")
    for row in R_home:
        print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")

    # Step 2: IK for FK home pose
    print("\n[2] INVERSE KINEMATICS FOR HOME POSE")
    solutions = controller.motion_planner.ur5_inverse(pos_home, R_home)

    if solutions is None:
        print("ERROR: IK returned None!")
        return

    print(f"Got {solutions.shape[1]} solutions")

    # Check each solution
    print("\nChecking each solution against home:")
    for i in range(8):
        sol = solutions[:, i]
        if np.any(np.isnan(sol)):
            print(f"  Sol {i+1}: NaN")
            continue

        # Normalize angles to [-pi, pi] for comparison
        sol_norm = sol.copy()
        home_norm = home.copy()
        for j in range(6):
            while sol_norm[j] > np.pi: sol_norm[j] -= 2*np.pi
            while sol_norm[j] < -np.pi: sol_norm[j] += 2*np.pi
            while home_norm[j] > np.pi: home_norm[j] -= 2*np.pi
            while home_norm[j] < -np.pi: home_norm[j] += 2*np.pi

        diff = np.abs(sol_norm - home_norm)
        max_diff = np.max(diff)

        print(f"  Sol {i+1}: max_diff = {np.degrees(max_diff):6.2f}°", end="")
        if max_diff < 0.01:
            print(" *** MATCH! ***")
        else:
            print()

    # Step 3: Verify FK of IK solution
    print("\n[3] VERIFY FK OF BEST IK SOLUTION")
    best = controller.motion_planner.select_best_solution(solutions, home)

    if best is not None:
        T_verify = forward_kinematics(best)
        pos_verify = T_verify[:3, 3]
        R_verify = T_verify[:3, :3]

        pos_error = np.linalg.norm(pos_verify - pos_home)
        R_error = np.linalg.norm(R_verify - R_home)

        print(f"Best solution joints (deg): {np.degrees(best)}")
        print(f"FK of solution: [{pos_verify[0]:.4f}, {pos_verify[1]:.4f}, {pos_verify[2]:.4f}]")
        print(f"Position error: {pos_error:.6f} m")
        print(f"Rotation error (Frobenius): {R_error:.6f}")

    # Step 4: Test at brick position
    print("\n[4] TEST AT BRICK POSITION")
    objects = controller.perception.get_ground_truth_objects()
    if objects:
        brick_pos = objects[0]['position']
        print(f"Brick robot-relative: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}]")

        # A2 transform
        p_dh = np.array([brick_pos[0], brick_pos[1], -brick_pos[2] - 0.15])
        print(f"DH frame (approach): [{p_dh[0]:.4f}, {p_dh[1]:.4f}, {p_dh[2]:.4f}]")

        # IK with identity rotation
        R_identity = np.eye(3)
        sol_brick = controller.motion_planner.ur5_inverse(p_dh, R_identity)

        if sol_brick is not None:
            best_brick = controller.motion_planner.select_best_solution(sol_brick, home)
            if best_brick is not None:
                print(f"IK solution (deg): {np.degrees(best_brick)}")

                # Verify FK
                T_brick = forward_kinematics(best_brick)
                pos_brick_fk = T_brick[:3, 3]
                R_brick_fk = T_brick[:3, :3]

                pos_err = np.linalg.norm(pos_brick_fk - p_dh)
                print(f"FK position: [{pos_brick_fk[0]:.4f}, {pos_brick_fk[1]:.4f}, {pos_brick_fk[2]:.4f}]")
                print(f"Position error: {pos_err:.6f} m")
                print(f"FK rotation Z-column (gripper direction): [{R_brick_fk[0,2]:.3f}, {R_brick_fk[1,2]:.3f}, {R_brick_fk[2,2]:.3f}]")
                print("  (Should be [0, 0, 1] for gripper pointing in +Z DH direction)")

def analyze_gripper_direction(controller):
    """
    Analyze what rotation matrix gives gripper pointing down.
    """
    print("\n" + "="*70)
    print("GRIPPER DIRECTION ANALYSIS")
    print("="*70)

    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])
    T_home = forward_kinematics(home)
    R_home = T_home[:3, :3]

    print("\nAt HOME position (gripper visually points DOWN):")
    print(f"End-effector Z-axis (R[:,2]): [{R_home[0,2]:.3f}, {R_home[1,2]:.3f}, {R_home[2,2]:.3f}]")
    print(f"End-effector X-axis (R[:,0]): [{R_home[0,0]:.3f}, {R_home[1,0]:.3f}, {R_home[2,0]:.3f}]")
    print(f"End-effector Y-axis (R[:,1]): [{R_home[0,1]:.3f}, {R_home[1,1]:.3f}, {R_home[2,1]:.3f}]")

    print("\nInterpretation:")
    print(f"  Z-axis ≈ [0, 0, 1] means gripper Z points in DH +Z direction")
    print(f"  At home, Z-axis = [{R_home[0,2]:.3f}, {R_home[1,2]:.3f}, {R_home[2,2]:.3f}]")

    # The Z component of Z-axis tells us if gripper points up or down in DH frame
    if R_home[2,2] > 0.9:
        print(f"  -> Gripper Z strongly aligned with DH +Z (good for 'down' if DH +Z = world down)")
    elif R_home[2,2] < -0.9:
        print(f"  -> Gripper Z strongly aligned with DH -Z")
    else:
        print(f"  -> Gripper Z is tilted, not perfectly vertical")

if __name__ == "__main__":
    print("IK Verification module loaded.")
    print("Functions: verify_fk_ik_roundtrip(p), analyze_gripper_direction(p)")
