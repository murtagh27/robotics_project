#!/usr/bin/env python3
"""
Forward Kinematics Diagnostic Script.
Verifies if our DH parameters match the actual robot.

Run from interactive session:
>>> from fk_diagnostic import *
>>> diagnose_fk(p)
"""

import numpy as np


def dh_transform(theta, alpha, d, a):
    """Standard DH transformation matrix."""
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


def forward_kinematics(joints):
    """
    Compute forward kinematics for UR5.

    Args:
        joints: 6-element array of joint angles [q1, q2, q3, q4, q5, q6]

    Returns:
        T06: 4x4 homogeneous transformation matrix (end-effector pose)
    """
    # UR5 DH parameters (from ur5Inverse.m)
    A = np.array([0, -0.425, -0.3922, 0, 0, 0])
    D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
    Alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

    T = np.eye(4)
    for i in range(6):
        Ti = dh_transform(joints[i], Alpha[i], D[i], A[i])
        T = T @ Ti

    return T


def get_actual_ee_position(controller):
    """Get actual end-effector position from simulation via TF or calculation."""
    # Get current joint state
    joints, _ = controller.get_current_joint_state()
    return joints[:6]


def diagnose_fk(controller):
    """
    Diagnose FK/IK by comparing computed vs actual positions.
    """
    print("\n" + "="*70)
    print("FORWARD KINEMATICS DIAGNOSTIC")
    print("="*70)

    # Get current joints
    current_joints, _ = controller.get_current_joint_state()
    q = current_joints[:6]

    print(f"\nCurrent joint angles (rad):")
    print(f"  q1={q[0]:.4f}, q2={q[1]:.4f}, q3={q[2]:.4f}")
    print(f"  q4={q[3]:.4f}, q5={q[4]:.4f}, q6={q[5]:.4f}")
    print(f"\nCurrent joint angles (deg):")
    print(f"  q1={np.degrees(q[0]):.1f}, q2={np.degrees(q[1]):.1f}, q3={np.degrees(q[2]):.1f}")
    print(f"  q4={np.degrees(q[3]):.1f}, q5={np.degrees(q[4]):.1f}, q6={np.degrees(q[5]):.1f}")

    # Compute FK
    T06 = forward_kinematics(q)
    computed_pos = T06[:3, 3]
    computed_rot = T06[:3, :3]

    print(f"\n--- Computed FK (DH base frame) ---")
    print(f"Position: [{computed_pos[0]:.4f}, {computed_pos[1]:.4f}, {computed_pos[2]:.4f}]")
    print(f"Rotation matrix:")
    print(f"  [{computed_rot[0,0]:.3f}, {computed_rot[0,1]:.3f}, {computed_rot[0,2]:.3f}]")
    print(f"  [{computed_rot[1,0]:.3f}, {computed_rot[1,1]:.3f}, {computed_rot[1,2]:.3f}]")
    print(f"  [{computed_rot[2,0]:.3f}, {computed_rot[2,1]:.3f}, {computed_rot[2,2]:.3f}]")

    # Get robot base position
    robot_base = np.array([0.5, 0.35, 1.75])
    print(f"\n--- Robot base in world frame ---")
    print(f"Position: [{robot_base[0]}, {robot_base[1]}, {robot_base[2]}]")

    # Now let's try to figure out the actual mapping
    # by looking at where the gripper visually appears to be
    print("\n" + "="*70)
    print("VISUAL VERIFICATION NEEDED")
    print("="*70)
    print("\nPlease look at the simulation and note where the gripper is.")
    print("Compare with the computed FK position above.")
    print("\nIf the gripper is at home position (above bricks, pointing down):")
    print("  - The FK Z should be negative (below robot base)")
    print("  - The FK X,Y should be small (near the robot's vertical axis)")

    return T06


def test_home_fk(controller):
    """Test FK at home position."""
    print("\n" + "="*70)
    print("HOME POSITION FK TEST")
    print("="*70)

    # Move to home first
    print("Moving to home position...")
    controller.go_home()

    import rospy
    rospy.sleep(2.0)

    # Now diagnose
    T06 = diagnose_fk(controller)

    # Get first brick position for reference
    objects = controller.perception.get_ground_truth_objects()
    if objects:
        brick_pos = objects[0]['position']
        print(f"\n--- First brick (robot-relative) ---")
        print(f"Position: [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}]")

    return T06


def test_ik_roundtrip(controller):
    """
    Test IK by doing a roundtrip:
    1. Get current position via FK
    2. Compute IK for that position
    3. Compare joint solutions
    """
    print("\n" + "="*70)
    print("IK ROUNDTRIP TEST")
    print("="*70)

    # Get current joints
    current_joints, _ = controller.get_current_joint_state()
    q_current = current_joints[:6]

    print(f"\nCurrent joints (deg):")
    print(f"  [{np.degrees(q_current[0]):.1f}, {np.degrees(q_current[1]):.1f}, "
          f"{np.degrees(q_current[2]):.1f}, {np.degrees(q_current[3]):.1f}, "
          f"{np.degrees(q_current[4]):.1f}, {np.degrees(q_current[5]):.1f}]")

    # Compute FK to get current pose
    T06 = forward_kinematics(q_current)
    pos = T06[:3, 3]
    rot = T06[:3, :3]

    print(f"\nFK computed pose (DH frame):")
    print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # Now compute IK for this exact position
    print(f"\nComputing IK for this position...")
    solutions = controller.motion_planner.ur5_inverse(pos, rot)

    if solutions is None:
        print("  ERROR: IK returned None!")
        return

    # Find how many valid solutions
    valid_count = np.sum(~np.any(np.isnan(solutions), axis=0))
    print(f"  Found {valid_count}/8 valid solutions")

    # Check if any solution matches current joints
    print(f"\nComparing solutions with current joints:")
    for i in range(8):
        sol = solutions[:, i]
        if np.any(np.isnan(sol)):
            continue

        diff = np.abs(sol - q_current)
        max_diff = np.max(diff)
        print(f"  Solution {i+1}: max diff = {np.degrees(max_diff):.1f} deg")

        if max_diff < 0.1:  # Less than ~6 degrees
            print(f"    ^ This solution matches current joints!")


def analyze_dh_base_frame(controller):
    """
    Analyze DH base frame orientation by testing single joint movements.
    """
    print("\n" + "="*70)
    print("DH BASE FRAME ANALYSIS")
    print("="*70)
    print("\nThis test moves only q1 and computes FK to understand")
    print("the relationship between DH frame and world frame.\n")

    # Test positions for q1
    test_angles = [0, np.pi/2, np.pi, -np.pi/2]

    # Use a simple arm configuration
    base_joints = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0])

    print("With joints q2=-90°, q3=0°, q4=-90°, q5=0°, q6=0°:")
    print("(This extends the arm roughly horizontally)\n")

    for q1 in test_angles:
        joints = base_joints.copy()
        joints[0] = q1

        T06 = forward_kinematics(joints)
        pos = T06[:3, 3]

        print(f"q1 = {np.degrees(q1):6.1f}° -> FK position: "
              f"[{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")

    print("\n" + "-"*70)
    print("Interpretation:")
    print("  - q1=0°:   Arm points in DH +X direction")
    print("  - q1=90°:  Arm points in DH +Y direction")
    print("  - q1=180°: Arm points in DH -X direction")
    print("  - q1=-90°: Arm points in DH -Y direction")
    print("\nCompare with simulation to determine world-to-DH mapping.")


if __name__ == "__main__":
    print("FK Diagnostic module loaded!")
    print("\nAvailable functions:")
    print("  diagnose_fk(p)           - Show FK for current position")
    print("  test_home_fk(p)          - Move home and compute FK")
    print("  test_ik_roundtrip(p)     - Test if IK can recover current joints")
    print("  analyze_dh_base_frame(p) - Analyze DH frame orientation")
