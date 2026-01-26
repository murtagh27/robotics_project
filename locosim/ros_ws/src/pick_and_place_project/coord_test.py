#!/usr/bin/env python3
"""
Coordinate transformation test script.
Tests all 8 possible coordinate mappings to find the correct one.

Run from interactive controller session:
>>> from coord_test import *
>>> test_all_transforms(p)
"""

import numpy as np


# All coordinate transformations - including Z-flip variants
TRANSFORMS = {
    # Without Z-flip
    'A': lambda x, y, z: (x, y, z),           # No change
    'B': lambda x, y, z: (y, x, z),           # Swap X/Y
    'C': lambda x, y, z: (-x, y, z),          # Negate X
    'D': lambda x, y, z: (x, -y, z),          # Negate Y
    'E': lambda x, y, z: (-x, -y, z),         # Negate both
    'F': lambda x, y, z: (-y, x, z),          # -Y, X (90 deg rotation)
    'G': lambda x, y, z: (y, -x, z),          # Y, -X (90 deg rotation other way)
    'H': lambda x, y, z: (-y, -x, z),         # Current experimental
    # WITH Z-flip (DH frame Z points opposite to world Z)
    'A2': lambda x, y, z: (x, y, -z),         # No XY change, flip Z
    'B2': lambda x, y, z: (y, x, -z),         # Swap X/Y, flip Z
    'C2': lambda x, y, z: (-x, y, -z),        # Negate X, flip Z
    'D2': lambda x, y, z: (x, -y, -z),        # Negate Y, flip Z
    'E2': lambda x, y, z: (-x, -y, -z),       # Negate X,Y,Z
    'F2': lambda x, y, z: (-y, x, -z),        # 90 deg CW, flip Z
    'G2': lambda x, y, z: (y, -x, -z),        # 90 deg CCW, flip Z
    'H2': lambda x, y, z: (-y, -x, -z),       # Current + flip Z
}

TRANSFORM_DESCRIPTIONS = {
    'A': '(x, y, z)      - No change',
    'B': '(y, x, z)      - Swap X/Y',
    'C': '(-x, y, z)     - Negate X',
    'D': '(x, -y, z)     - Negate Y',
    'E': '(-x, -y, z)    - Negate both X,Y',
    'F': '(-y, x, z)     - 90 deg CW rotation',
    'G': '(y, -x, z)     - 90 deg CCW rotation',
    'H': '(-y, -x, z)    - Negate and swap',
    'A2': '(x, y, -z)    - Z-flip only (LIKELY!)',
    'B2': '(y, x, -z)    - Swap X/Y + Z-flip',
    'C2': '(-x, y, -z)   - Negate X + Z-flip',
    'D2': '(x, -y, -z)   - Negate Y + Z-flip',
    'E2': '(-x, -y, -z)  - Negate all',
    'F2': '(-y, x, -z)   - 90 CW + Z-flip',
    'G2': '(y, -x, -z)   - 90 CCW + Z-flip',
    'H2': '(-y, -x, -z)  - Neg+swap + Z-flip',
}


def test_single_transform(controller, transform_key, object_pos):
    """
    Test a single coordinate transformation.

    Args:
        controller: PapaController instance
        transform_key: Key from TRANSFORMS dict ('A' through 'H' or with '2' suffix)
        object_pos: Object position [x, y, z] in robot-relative coordinates

    Returns:
        Joint solution or None
    """
    x_rel, y_rel, z_rel = object_pos

    # Apply transformation
    transform_func = TRANSFORMS[transform_key]
    px, py, pz = transform_func(x_rel, y_rel, z_rel)
    p_robot = np.array([px, py, pz])

    # Add approach height - direction depends on Z-flip
    if '2' in transform_key:
        p_robot[2] -= 0.15  # Z-flip: subtract for approach (smaller Z = higher in world)
    else:
        p_robot[2] += 0.15  # Normal: add for approach

    print(f"\n--- Transform {transform_key}: {TRANSFORM_DESCRIPTIONS[transform_key]} ---")
    print(f"  Input (robot-rel): [{x_rel:.3f}, {y_rel:.3f}, {z_rel:.3f}]")
    print(f"  DH frame input:    [{p_robot[0]:.3f}, {p_robot[1]:.3f}, {p_robot[2]:.3f}]")

    # Get rotation matrix
    R = controller.motion_planner.gripper_down_rotation()

    # Compute IK
    solutions = controller.motion_planner.ur5_inverse(p_robot, R)

    if solutions is None:
        print(f"  Result: UNREACHABLE")
        return None

    # Count valid solutions
    valid_count = np.sum(~np.any(np.isnan(solutions), axis=0))
    print(f"  Valid solutions: {valid_count}/8")

    # Select best solution
    best = controller.motion_planner.select_best_solution(solutions)

    if best is None:
        print(f"  Result: No valid solution within limits")
        return None

    print(f"  Best q1: {np.degrees(best[0]):.1f} deg")

    return best


def test_all_transforms(controller):
    """
    Test all 8 coordinate transformations and show which ones produce valid IK.

    Args:
        controller: PapaController instance (typically 'p')
    """
    print("\n" + "="*70)
    print("COORDINATE TRANSFORMATION TEST")
    print("="*70)

    # Get first object position
    objects = controller.perception.get_ground_truth_objects()

    if not objects:
        print("ERROR: No objects found!")
        return

    obj = objects[0]
    pos = obj['position']

    print(f"\nTest object: {obj['name']}")
    print(f"Robot-relative position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"\nTesting all 8 coordinate transformations...")

    results = {}
    # Test Z-flip transforms first (more likely based on FK analysis)
    for key in ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        result = test_single_transform(controller, key, pos)
        results[key] = result

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Valid transformations:")
    print("="*70)

    valid_transforms = []
    for key, result in results.items():
        if result is not None:
            valid_transforms.append(key)
            q1_deg = np.degrees(result[0])
            print(f"  {key}: {TRANSFORM_DESCRIPTIONS[key]}")
            print(f"     q1 = {q1_deg:.1f} deg")

    if not valid_transforms:
        print("  No valid transformations found!")

    print("\n" + "="*70)
    print("NEXT STEP: Test each valid transform by moving the robot")
    print("="*70)
    print("\nUse: move_with_transform(p, 'X')  where X is A, B, C, etc.")
    print("Watch where the robot moves and compare to object position.")

    return results


def move_with_transform(controller, transform_key, duration=3.0):
    """
    Move robot using a specific coordinate transformation.

    Args:
        controller: PapaController instance
        transform_key: 'A' through 'H' or with '2' suffix for Z-flip
        duration: Movement duration
    """
    print(f"\n" + "="*70)
    print(f"MOVING WITH TRANSFORM {transform_key}: {TRANSFORM_DESCRIPTIONS[transform_key]}")
    print("="*70)

    # Get first object
    objects = controller.perception.get_ground_truth_objects()
    if not objects:
        print("ERROR: No objects found!")
        return False

    obj = objects[0]
    pos = obj['position']
    x_rel, y_rel, z_rel = pos

    # Apply transformation
    transform_func = TRANSFORMS[transform_key]
    px, py, pz = transform_func(x_rel, y_rel, z_rel)

    # For Z-flip transforms, approach height needs to be SUBTRACTED
    # (because DH Z points down, so smaller Z = higher in world)
    if '2' in transform_key:
        p_robot = np.array([px, py, pz - 0.15])  # Z-flip: subtract for approach
    else:
        p_robot = np.array([px, py, pz + 0.15])  # Normal: add for approach

    print(f"Object: {obj['name']}")
    print(f"Robot-relative: [{x_rel:.3f}, {y_rel:.3f}, {z_rel:.3f}]")
    print(f"DH frame (with approach): [{p_robot[0]:.3f}, {p_robot[1]:.3f}, {p_robot[2]:.3f}]")

    # Compute IK
    R = controller.motion_planner.gripper_down_rotation()
    solutions = controller.motion_planner.ur5_inverse(p_robot, R)

    if solutions is None:
        print("ERROR: Position unreachable!")
        return False

    best = controller.motion_planner.select_best_solution(solutions)

    if best is None:
        print("ERROR: No valid solution!")
        return False

    print(f"Joint solution (deg): [{np.degrees(best[0]):.1f}, {np.degrees(best[1]):.1f}, "
          f"{np.degrees(best[2]):.1f}, {np.degrees(best[3]):.1f}, "
          f"{np.degrees(best[4]):.1f}, {np.degrees(best[5]):.1f}]")

    print(f"\nMoving robot... Watch where it goes!")
    print("The gripper should be 15cm ABOVE the first brick.")

    controller.motion_planner.move_to_joints(best, controller, duration=duration)

    print("\nDone! Is the gripper above the brick?")
    print("  - If YES: Transform", transform_key, "is correct!")
    print("  - If NO: Try another transform with move_with_transform(p, 'X')")

    return True


def quick_test_q1_directions(controller):
    """
    Quick test to understand q1 direction mapping.
    Tests q1 = 0, 90, 180, -90 degrees from home.
    """
    import rospy

    print("\n" + "="*70)
    print("Q1 DIRECTION TEST")
    print("="*70)
    print("This test rotates only q1 to understand the direction mapping.")
    print("Watch which world direction the arm points for each angle.\n")

    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])

    tests = [
        (0, "q1 = 0 deg"),
        (np.pi/2, "q1 = 90 deg"),
        (np.pi, "q1 = 180 deg"),
        (-np.pi/2, "q1 = -90 deg"),
    ]

    for q1_val, description in tests:
        print(f"\n--- {description} ---")
        test_joints = home.copy()
        test_joints[0] = q1_val

        input(f"Press Enter to move to {description}...")
        controller.motion_planner.move_to_joints(test_joints, controller, duration=2.0)
        rospy.sleep(1.0)

        direction = input("Which world direction does the arm point? (e.g., +X, -Y, etc.): ")
        print(f"  {description} -> {direction}")

    print("\n" + "="*70)
    print("Test complete! Use these observations to determine correct transform.")
    print("="*70)


def test_ik_to_home_fk(controller):
    """
    Critical test: Use IK to reach the FK-computed home position.
    If this doesn't return to home, the rotation matrix is wrong.
    """
    from fk_diagnostic import forward_kinematics

    print("\n" + "="*70)
    print("IK-TO-HOME-FK TEST")
    print("="*70)
    print("This test uses IK to reach the FK-computed home position.")
    print("If successful, robot should return to approximately home position.\n")

    # Home joints
    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])

    # Compute FK at home
    T_home = forward_kinematics(home)
    pos_home = T_home[:3, 3]
    rot_home = T_home[:3, :3]

    print(f"Home joints (deg): [{np.degrees(home[0]):.1f}, {np.degrees(home[1]):.1f}, "
          f"{np.degrees(home[2]):.1f}, {np.degrees(home[3]):.1f}, "
          f"{np.degrees(home[4]):.1f}, {np.degrees(home[5]):.1f}]")
    print(f"\nFK at home:")
    print(f"  Position: [{pos_home[0]:.4f}, {pos_home[1]:.4f}, {pos_home[2]:.4f}]")
    print(f"  Rotation:")
    print(f"    [{rot_home[0,0]:.3f}, {rot_home[0,1]:.3f}, {rot_home[0,2]:.3f}]")
    print(f"    [{rot_home[1,0]:.3f}, {rot_home[1,1]:.3f}, {rot_home[1,2]:.3f}]")
    print(f"    [{rot_home[2,0]:.3f}, {rot_home[2,1]:.3f}, {rot_home[2,2]:.3f}]")

    # Test 1: Use the ACTUAL rotation from FK
    print("\n--- Test 1: IK with FK rotation matrix ---")
    solutions = controller.motion_planner.ur5_inverse(pos_home, rot_home)
    if solutions is not None:
        valid = np.sum(~np.any(np.isnan(solutions), axis=0))
        print(f"  Valid solutions: {valid}/8")

        # Check ALL 8 solutions
        print(f"\n  ALL 8 SOLUTIONS vs HOME:")
        print(f"  Home: [{np.degrees(home[0]):7.1f}, {np.degrees(home[1]):7.1f}, {np.degrees(home[2]):7.1f}, "
              f"{np.degrees(home[3]):7.1f}, {np.degrees(home[4]):7.1f}, {np.degrees(home[5]):7.1f}]")

        closest_diff = float('inf')
        closest_idx = -1
        for i in range(8):
            sol = solutions[:, i]
            if np.any(np.isnan(sol)):
                print(f"  Sol {i+1}: NaN")
                continue
            diff = np.max(np.abs(sol - home))
            if diff < closest_diff:
                closest_diff = diff
                closest_idx = i
            print(f"  Sol {i+1}: [{np.degrees(sol[0]):7.1f}, {np.degrees(sol[1]):7.1f}, {np.degrees(sol[2]):7.1f}, "
                  f"{np.degrees(sol[3]):7.1f}, {np.degrees(sol[4]):7.1f}, {np.degrees(sol[5]):7.1f}] "
                  f"diff={np.degrees(diff):6.1f}°")

        print(f"\n  Closest solution: #{closest_idx+1} with diff={np.degrees(closest_diff):.1f}°")
        if closest_diff < 0.1:
            print("  SUCCESS: A solution matches home!")
        else:
            print("  PROBLEM: NO solution matches home - IK algorithm is broken!")
    else:
        print("  No solution found!")

    # Test 2: Use gripper_down rotation
    print("\n--- Test 2: IK with gripper_down_rotation() ---")
    R_down = controller.motion_planner.gripper_down_rotation()
    print(f"  gripper_down_rotation:")
    print(f"    [{R_down[0,0]:.3f}, {R_down[0,1]:.3f}, {R_down[0,2]:.3f}]")
    print(f"    [{R_down[1,0]:.3f}, {R_down[1,1]:.3f}, {R_down[1,2]:.3f}]")
    print(f"    [{R_down[2,0]:.3f}, {R_down[2,1]:.3f}, {R_down[2,2]:.3f}]")

    solutions2 = controller.motion_planner.ur5_inverse(pos_home, R_down)
    if solutions2 is not None:
        valid = np.sum(~np.any(np.isnan(solutions2), axis=0))
        print(f"  Valid solutions: {valid}/8")
        best2 = controller.motion_planner.select_best_solution(solutions2)
        if best2 is not None:
            print(f"  Best solution (deg): [{np.degrees(best2[0]):.1f}, {np.degrees(best2[1]):.1f}, "
                  f"{np.degrees(best2[2]):.1f}, {np.degrees(best2[3]):.1f}, "
                  f"{np.degrees(best2[4]):.1f}, {np.degrees(best2[5]):.1f}]")
    else:
        print("  No solution found!")

    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("- If Test 1 matches home -> IK works, gripper_down_rotation is wrong")
    print("- If Test 1 doesn't match -> Deeper IK/FK mismatch")
    print("="*70)

    return pos_home, rot_home


def test_ik_without_pi2_offset(controller):
    """
    Test if removing the +pi/2 offset in th1 fixes the IK.
    The MATLAB code has: th1 = psi + phi + pi/2
    This offset might be specific to a different robot configuration.
    """
    from fk_diagnostic import forward_kinematics

    print("\n" + "="*70)
    print("TEST: IK WITHOUT +PI/2 OFFSET")
    print("="*70)

    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])
    T_home = forward_kinematics(home)
    pos = T_home[:3, 3]
    R = T_home[:3, :3]

    # Manually compute IK without the +pi/2 offset
    A = np.array([0, -0.425, -0.3922, 0, 0, 0])
    D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])

    # Build T60
    T60 = np.eye(4)
    T60[:3, :3] = R
    T60[:3, 3] = pos

    # Compute wrist center p50
    p50_h = T60 @ np.array([0, 0, -D[5], 1])
    p50 = p50_h[:3]

    psi = np.arctan2(p50[1], p50[0])
    p50xy = np.hypot(p50[1], p50[0])
    phi1 = np.arccos(D[3] / p50xy)

    # Standard th1 (WITH +pi/2)
    th1_with_offset = psi + phi1 + np.pi/2
    th1_with_offset_alt = psi - phi1 + np.pi/2

    # th1 WITHOUT +pi/2
    th1_no_offset = psi + phi1
    th1_no_offset_alt = psi - phi1

    print(f"Home q1: {np.degrees(home[0]):.1f}°")
    print(f"\npsi (atan2 of wrist center): {np.degrees(psi):.1f}°")
    print(f"phi1 (acos offset): {np.degrees(phi1):.1f}°")
    print(f"\nth1 WITH +pi/2 offset:")
    print(f"  psi + phi + 90° = {np.degrees(th1_with_offset):.1f}°")
    print(f"  psi - phi + 90° = {np.degrees(th1_with_offset_alt):.1f}°")
    print(f"\nth1 WITHOUT +pi/2 offset:")
    print(f"  psi + phi = {np.degrees(th1_no_offset):.1f}°")
    print(f"  psi - phi = {np.degrees(th1_no_offset_alt):.1f}°")

    # Check which one is closest to home q1
    candidates = [
        ("psi + phi + 90°", th1_with_offset),
        ("psi - phi + 90°", th1_with_offset_alt),
        ("psi + phi", th1_no_offset),
        ("psi - phi", th1_no_offset_alt),
    ]

    print(f"\nDifference from home q1 ({np.degrees(home[0]):.1f}°):")
    for name, val in candidates:
        # Normalize to [-pi, pi]
        diff = val - home[0]
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        print(f"  {name}: diff = {np.degrees(diff):.1f}°")


def move_to_fk_home(controller):
    """Move to FK home position using IK with actual FK rotation."""
    from fk_diagnostic import forward_kinematics

    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])
    T_home = forward_kinematics(home)
    pos_home = T_home[:3, 3]
    rot_home = T_home[:3, :3]

    print(f"Computing IK for FK home position: {pos_home}")
    solutions = controller.motion_planner.ur5_inverse(pos_home, rot_home)

    if solutions is None:
        print("ERROR: No IK solution!")
        return False

    best = controller.motion_planner.select_best_solution(solutions)
    if best is None:
        print("ERROR: No valid solution!")
        return False

    print(f"Moving to IK solution...")
    controller.motion_planner.move_to_joints(best, controller, duration=3.0)
    print("Done! Robot should be at home position now.")
    return True


def test_rotation_matrices(controller):
    """
    Test different rotation matrices to find which gives gripper pointing down.
    """
    from fk_diagnostic import forward_kinematics

    print("\n" + "="*70)
    print("ROTATION MATRIX TEST")
    print("="*70)

    # Get brick position with A2 transform
    objects = controller.perception.get_ground_truth_objects()
    if not objects:
        print("ERROR: No objects!")
        return

    pos = objects[0]['position']
    x, y, z = pos
    # A2 transform
    p_dh = np.array([x, y, -z - 0.15])  # approach position

    print(f"Target DH position: [{p_dh[0]:.3f}, {p_dh[1]:.3f}, {p_dh[2]:.3f}]")

    # Different rotation matrices to try
    rotations = {
        'Identity': np.eye(3),
        'Rz_90': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        'Rz_-90': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        'Rz_180': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        'Rx_180': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        'Ry_180': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        'Z_down': np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # flip Z
        'FK_home': forward_kinematics(np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49]))[:3, :3],
    }

    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])

    print("\nTesting rotations (comparing q5 to home q5=-90°):")
    print("Home has q5=-90° which gives gripper pointing down.\n")

    for name, R in rotations.items():
        solutions = controller.motion_planner.ur5_inverse(p_dh, R)
        if solutions is None:
            print(f"{name:12s}: No solution")
            continue

        best = controller.motion_planner.select_best_solution(solutions, home)
        if best is None:
            print(f"{name:12s}: No valid solution")
            continue

        print(f"{name:12s}: q5={np.degrees(best[4]):7.1f}°  "
              f"q1={np.degrees(best[0]):7.1f}°  "
              f"q6={np.degrees(best[5]):7.1f}°")


def move_with_rotation(controller, rotation_name):
    """Move using A2 transform with a specific rotation matrix."""
    from fk_diagnostic import forward_kinematics

    objects = controller.perception.get_ground_truth_objects()
    if not objects:
        print("ERROR: No objects!")
        return False

    pos = objects[0]['position']
    x, y, z = pos
    p_dh = np.array([x, y, -z - 0.15])

    rotations = {
        'identity': np.eye(3),
        'rz90': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        'rz-90': np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        'rx180': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        'fk_home': forward_kinematics(np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49]))[:3, :3],
    }

    if rotation_name not in rotations:
        print(f"Unknown rotation. Options: {list(rotations.keys())}")
        return False

    R = rotations[rotation_name]
    print(f"\nUsing rotation '{rotation_name}'")
    print(f"Target: [{p_dh[0]:.3f}, {p_dh[1]:.3f}, {p_dh[2]:.3f}]")

    solutions = controller.motion_planner.ur5_inverse(p_dh, R)
    if solutions is None:
        print("No IK solution!")
        return False

    home = np.array([-0.32, -0.78, -2.56, -1.63, -1.57, 3.49])
    best = controller.motion_planner.select_best_solution(solutions, home)
    if best is None:
        print("No valid solution!")
        return False

    print(f"Moving with q5={np.degrees(best[4]):.1f}°...")
    controller.motion_planner.move_to_joints(best, controller, duration=3.0)
    return True


if __name__ == "__main__":
    print("Coordinate test module loaded!")
    print("\nAvailable functions:")
    print("  test_all_transforms(p)           - Test all 8 transforms, show which have valid IK")
    print("  move_with_transform(p, 'X')      - Move robot using transform X (A-H)")
    print("  quick_test_q1_directions(p)      - Interactive test of q1 directions")
    print("\nExample usage:")
    print("  >>> from coord_test import *")
    print("  >>> test_all_transforms(p)")
    print("  >>> move_with_transform(p, 'F')  # Try transform F")
