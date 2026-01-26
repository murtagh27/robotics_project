#!/usr/bin/env python3
"""
Test script to verify coordinate frame transformations.
"""

import numpy as np

def test_coordinate_frames(controller):
    """
    Test different coordinate transformations to find the correct one.
    """
    print("\n" + "="*60)
    print("COORDINATE FRAME TEST")
    print("="*60)
    
    # Get brick position
    objects = controller.perception.get_ground_truth_objects()
    if not objects:
        print("ERROR: No objects found!")
        return
    
    brick = objects[0]
    pos = brick['position']
    
    print(f"\nBrick: {brick['name']}")
    print(f"Position (robot-relative world axes): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Test point above the brick
    approach_z = pos[2] + 0.15
    
    print(f"\nTesting 4 different transformations:")
    print(f"Target approach Z = {approach_z:.3f}")
    
    # Transformation 1: No swap (x→x, y→y)
    p1 = np.array([pos[0], pos[1], approach_z])
    print(f"\n1. No swap:     [{p1[0]:7.3f}, {p1[1]:7.3f}, {p1[2]:7.3f}]")
    
    # Transformation 2: Swap (x→y, y→x)  
    p2 = np.array([pos[1], pos[0], approach_z])
    print(f"2. Swap:        [{p2[0]:7.3f}, {p2[1]:7.3f}, {p2[2]:7.3f}]")
    
    # Transformation 3: Swap and negate X (x→y, y→-x)
    p3 = np.array([pos[1], -pos[0], approach_z])
    print(f"3. Swap+negate: [{p3[0]:7.3f}, {p3[1]:7.3f}, {p3[2]:7.3f}]")
    
    # Transformation 4: Negate both (x→-x, y→-y)
    p4 = np.array([-pos[0], -pos[1], approach_z])
    print(f"4. Negate both: [{p4[0]:7.3f}, {p4[1]:7.3f}, {p4[2]:7.3f}]")
    
    # Now test each one
    R = controller.motion_planner.gripper_down_rotation()
    
    transformations = [
        ("No swap", p1),
        ("Swap", p2),
        ("Swap+negate", p3),
        ("Negate both", p4)
    ]
    
    print("\n" + "-"*60)
    print("Testing IK for each transformation:")
    print("-"*60)
    
    for name, p_test in transformations:
        solutions = controller.motion_planner.ur5_inverse(p_test, R)
        if solutions is not None:
            best = controller.motion_planner.select_best_solution(solutions)
            if best is not None:
                print(f"\n{name}: IK SUCCESS")
                print(f"  Joint 1: {np.degrees(best[0]):6.1f}°")
                
                # Move to this position
                response = input(f"  Move to this position? (y/n): ")
                if response.lower() == 'y':
                    controller.motion_planner.move_to_joints(best, controller, duration=2.0)
                    check = input(f"  Is gripper above brick? (y/n): ")
                    if check.lower() == 'y':
                        print(f"\n{'='*60}")
                        print(f"SOLUTION FOUND: {name}")
                        print(f"Transformation: {name}")
                        print(f"{'='*60}")
                        return name
                    else:
                        print(f"  Not correct, continuing...")
                        # Go back home
                        controller.go_home()
                        import rospy
                        rospy.sleep(2)
            else:
                print(f"\n{name}: IK failed (no valid solution within limits)")
        else:
            print(f"\n{name}: IK failed (unreachable)")
    
    print("\n" + "="*60)
    print("Test complete. None of the transformations worked correctly.")
    print("="*60)

if __name__ == "__main__":
    print("Coordinate test module loaded!")
    print("\nUsage:")
    print("  >>> from coordinate_test import *")
    print("  >>> test_coordinate_frames(p)")
