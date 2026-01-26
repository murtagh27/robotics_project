#!/usr/bin/env python3
"""
Motion test script for testing IK and motion to object positions.
Run this from the interactive controller session.
"""

def test_approach_first_object(controller):
    """
    Test approaching the first detected object.
    
    Args:
        controller: The PapaController instance (typically 'p' in interactive mode)
    """
    print("\n" + "="*60)
    print("MOTION TEST: Approaching First Object")
    print("="*60)
    
    # Get ground truth objects
    objects = controller.perception.get_ground_truth_objects()
    
    if not objects:
        print("ERROR: No objects found!")
        return False
    
    print(f"\nFound {len(objects)} objects:")
    for obj in objects:
        print(f"  {obj['name']}: pos={obj['position']}")
    
    # Get first object position
    pos = objects[0]['position']
    print(f"\nTesting with first object: {objects[0]['name']}")
    print(f"Object position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Compute approach position (15cm above object)
    approach_pos = [pos[0], pos[1], pos[2] + 0.15]
    print(f"Approach position: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}, {approach_pos[2]:.3f}]")
    
    # Compute IK
    print("\nComputing IK...")
    joints = controller.motion_planner.simple_ik(approach_pos, robot_relative=True)
    
    if joints is None:
        print("ERROR: IK failed!")
        return False
    
    print(f"IK solution found!")
    
    # Move to position
    print("\nMoving to approach position...")
    success = controller.motion_planner.move_to_joints(joints, controller)
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: Robot reached approach position!")
        print("="*60)
    else:
        print("\nERROR: Motion failed!")
    
    return success


def test_all_objects(controller):
    """
    Test approaching all detected objects one by one.
    
    Args:
        controller: The PapaController instance
    """
    print("\n" + "="*60)
    print("MOTION TEST: Approaching All Objects")
    print("="*60)
    
    objects = controller.perception.get_ground_truth_objects()
    
    if not objects:
        print("ERROR: No objects found!")
        return
    
    print(f"\nFound {len(objects)} objects to test")
    
    for i, obj in enumerate(objects):
        print(f"\n--- Object {i+1}/{len(objects)}: {obj['name']} ---")
        pos = obj['position']
        approach_pos = [pos[0], pos[1], pos[2] + 0.15]
        
        joints = controller.motion_planner.simple_ik(approach_pos, robot_relative=True)
        
        if joints is None:
            print(f"  SKIP: IK failed for {obj['name']}")
            continue
        
        print(f"  Moving to approach position...")
        controller.motion_planner.move_to_joints(joints, controller, duration=2.0)
        
        import rospy
        rospy.sleep(1.0)
    
    print("\n" + "="*60)
    print("All objects tested!")
    print("="*60)


def test_pick_sequence(controller):
    """
    Test full pick sequence on first object.
    
    Args:
        controller: The PapaController instance
    """
    print("\n" + "="*60)
    print("MOTION TEST: Full Pick Sequence")
    print("="*60)
    
    objects = controller.perception.get_ground_truth_objects()
    
    if not objects:
        print("ERROR: No objects found!")
        return False
    
    obj = objects[0]
    print(f"\nTesting pick on: {obj['name']}")
    print(f"Position: {obj['position']}")
    
    success = controller.motion_planner.pick_object(obj['position'], controller, robot_relative=True)
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: Pick sequence completed!")
        print("="*60)
    else:
        print("\nERROR: Pick sequence failed!")
    
    return success


# If running directly in interactive mode, provide helper function
if __name__ == "__main__":
    print("Motion test module loaded!")
    print("\nAvailable functions:")
    print("  test_approach_first_object(p)  - Test approaching first object")
    print("  test_all_objects(p)            - Test approaching all objects")
    print("  test_pick_sequence(p)          - Test full pick sequence")
    print("\nExample usage:")
    print("  >>> from motion_test import *")
    print("  >>> test_approach_first_object(p)")
