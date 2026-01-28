"""
High-level task scheduler for pick and place operations
Coordinates perception, planning, and execution
"""

import rospy
import numpy as np
from enum import Enum
import math
from collections import defaultdict

# Retry configuration
MAX_PICK_RETRIES = 3
MAX_PLACE_RETRIES = 2
RETRY_OFFSET = 0.01  # meters - offset for retry attempts


class SlotManager:
    """Manages placement grid slots organized by brick class"""
    
    def __init__(self, config):
        self.config = config
        
        # Placement area bounds (world frame)
        self.x_min = 0.20
        self.x_max = 0.40
        self.y_min = 0.35
        self.y_max = 0.65
        self.z_height = 0.89
        
        # Slot tracking
        self.slots_per_class = defaultdict(int)
        
        # Grid config
        self.slots_per_row = 3
        self.brick_spacing_x = (self.x_max - self.x_min) / max(self.slots_per_row - 1, 1)
        self.brick_spacing_y = 0.10
        
    def get_slot_for_class(self, brick_class):
        """Get next available slot position for this brick class"""
        slot_idx = self.slots_per_class[brick_class]
        self.slots_per_class[brick_class] += 1
        
        # Each class gets its own row
        class_list = sorted(set(self.slots_per_class.keys()))
        try:
            row = class_list.index(brick_class)
        except ValueError:
            row = 0
        
        col = slot_idx % self.slots_per_row
        
        x = self.x_min + col * self.brick_spacing_x
        y = self.y_min + row * self.brick_spacing_y
        z = self.z_height
        
        # Clamp to bounds
        x = max(self.x_min, min(x, self.x_max))
        y = max(self.y_min, min(y, self.y_max))
        
        return np.array([x, y, z])
    
    def reset(self):
        """Reset all slot counters"""
        self.slots_per_class.clear()


class TaskState(Enum):
    """States for pick and place task"""

    IDLE = 0
    DETECTING_OBJECTS = 1
    PLANNING_SEQUENCE = 2
    MOVING_TO_OBJECT = 3
    PICKING = 4
    MOVING_TO_TARGET = 5
    PLACING = 6
    RETURNING_HOME = 7
    COMPLETED = 8
    ERROR = 9
    RETRYING = 10


class TaskScheduler:
    """
    High-level planner for pick and place task
    Implements state machine for task execution
    """

    def __init__(self, perception, motion_planner, robot_interface, config):
        self.perception = perception
        self.motion_planner = motion_planner
        self.robot_interface = robot_interface
        self.config = config

        self.state = TaskState.IDLE
        self.task_sequence = []
        self.current_task_index = 0
        
        self.slot_manager = SlotManager(config)
        
        # Statistics
        self.total_attempts = 0
        self.successful_picks = 0
        self.successful_places = 0
        self.failed_picks = 0
        self.failed_places = 0
        self.retries_used = 0

    def detect_and_plan(self):
        """
        Detect objects and plan pick-place sequence
        Returns: List of (object, target_position) tuples
        """
        rospy.loginfo("Detecting objects...")

        # If using ground truth, wait a moment for the existing subscriber to receive data
        if self.config.use_ground_truth:
            rospy.loginfo("Waiting for perception to receive Gazebo data...")

            # Wait up to 5 seconds for the controller's existing subscriber to populate perception
            wait_time = 0
            while len(self.perception.get_detected_objects()) == 0 and wait_time < 5.0:
                rospy.sleep(0.5)
                wait_time += 0.5
                rospy.loginfo(f"Waiting... ({wait_time}s)")

            objects_found = len(self.perception.get_detected_objects())
            if objects_found > 0:
                rospy.loginfo(f"Perception ready with {objects_found} objects")
            else:
                rospy.logwarn("Still no objects detected after waiting")

        # Get detected objects from perception module
        objects = self.perception.get_detected_objects()

        if not objects:
            rospy.logwarn("No objects detected!")
            return []

        rospy.loginfo(f"Detected {len(objects)} objects")

        # Create task sequence: move all objects to target table
        task_sequence = []
        detected_positions = []
        target_positions = []

        for i, obj in enumerate(objects):
            obj_class = obj.get('class', 'unknown')

            # Use SlotManager to get target position for this brick class
            target_pos = self.slot_manager.get_slot_for_class(obj_class)

            task_sequence.append({'object': obj, 'target': target_pos, 'class': obj_class})
            
            # Collect positions for visualization
            detected_positions.append(obj['position'])
            target_positions.append(target_pos)

        # Visualize detected objects and target positions in RViz
        if hasattr(self.robot_interface, 'visualize_objects_and_targets'):
            self.robot_interface.visualize_objects_and_targets(detected_positions, target_positions)

        # Sort by position (process objects left to right, front to back)
        task_sequence.sort(key=lambda x: (x['object']['position'][1], x['object']['position'][0]))

        return task_sequence

    def execute_task_sequence(self):
        """
        Execute the planned pick-place sequence
        Main state machine loop
        """
        self.state = TaskState.DETECTING_OBJECTS

        # Detect objects and plan
        self.task_sequence = self.detect_and_plan()

        if not self.task_sequence:
            rospy.logerr("No valid tasks to execute")
            self.state = TaskState.ERROR
            return False

        rospy.loginfo(f"Planning to move {len(self.task_sequence)} objects")
        self.state = TaskState.PLANNING_SEQUENCE

        # Move to home position first
        rospy.loginfo("Moving to home position...")
        self.motion_planner.move_to_joints(self.config.home_joint_config, self.robot_interface)

        # Execute each pick-place operation
        successful_tasks = 0
        failed_tasks = 0
        
        for idx, task in enumerate(self.task_sequence):
            self.current_task_index = idx

            rospy.loginfo(
                f"Task {idx+1}/{len(self.task_sequence)}: " f"Moving {task['class']} object"
            )

            # Pick phase
            self.state = TaskState.MOVING_TO_OBJECT
            obj_pos = task['object']['position']

            rospy.loginfo(f"  Picking from position: {obj_pos}")
            if not self._attempt_pick_with_retry(task['object']):
                rospy.logwarn(f"Failed to pick object at task {idx+1} - SKIPPING to next object")
                failed_tasks += 1
                # Don't place if we didn't grab anything - continue to next object
                continue

            # VERIFY PICK SUCCESS: Check if object is now lifted (attached to gripper)
            obj_name = task['object']['name']
            original_z = obj_pos[2]  # Original object Z position on table
            if not self._verify_object_lifted(obj_name, original_z):
                rospy.logerr(f"⛔ PICK VERIFICATION FAILED: {obj_name} is still on table!")
                rospy.logerr(f"   Object was not picked up - SKIPPING to next object")
                failed_tasks += 1
                # Open gripper and continue to next object
                self.robot_interface.send_gripper_command(self.config.gripper_open_pos)
                rospy.sleep(0.5)
                continue
            
            rospy.loginfo(f"✓ Pick verified: {obj_name} is now lifted")

            self.state = TaskState.PICKING

            # Place phase - only if pick succeeded
            self.state = TaskState.MOVING_TO_TARGET
            target_pos = task['target']

            rospy.loginfo(f"  Placing at position: {target_pos}")
            if not self._attempt_place_with_retry(target_pos):
                rospy.logerr(f"Failed to place object at task {idx+1}")
                failed_tasks += 1
                # Open gripper to drop whatever we're holding
                self.robot_interface.send_gripper_command(self.config.gripper_open_pos)
                continue

            self.state = TaskState.PLACING
            successful_tasks += 1

            rospy.loginfo(f"Task {idx+1} completed successfully")
            
            # Return to safe height before next pick
            if idx < len(self.task_sequence) - 1:
                rospy.loginfo("  Returning to safe height...")
                safe_z = getattr(self.config, 'safe_transit_height', 1.10)
                safe_pos = np.array([0.75, 0.40, safe_z])
                safe_joints = self.motion_planner.simple_ik(safe_pos, gripper_down=True)
                if safe_joints is not None:
                    self.motion_planner.move_to_joints(safe_joints, self.robot_interface)

        # Return to home
        self.state = TaskState.RETURNING_HOME
        rospy.loginfo("Returning to home position...")
        self.motion_planner.move_to_joints(self.config.home_joint_config, self.robot_interface)

        self.state = TaskState.COMPLETED
        total_tasks = len(self.task_sequence)
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"TASK SUMMARY")
        rospy.loginfo(f"{'='*60}")
        rospy.loginfo(f"  Total objects: {total_tasks}")
        rospy.loginfo(f"  Successful: {successful_tasks}")
        rospy.loginfo(f"  Failed: {failed_tasks}")
        rospy.loginfo(f"  Success rate: {100*successful_tasks/total_tasks:.1f}%")
        rospy.loginfo(f"{'='*60}")
        
        return successful_tasks > 0  # Return True if at least one succeeded

    def get_current_state(self):
        """Return current state of task execution"""
        return self.state

    def get_progress(self):
        """Return task progress (0.0 to 1.0)"""
        if not self.task_sequence:
            return 0.0
        return self.current_task_index / len(self.task_sequence)

    def emergency_stop(self):
        """Emergency stop - halt all motion"""
        rospy.logwarn("EMERGENCY STOP activated")
        self.robot_interface.stop()
        self.state = TaskState.ERROR

    def reset(self):
        """Reset scheduler to initial state"""
        self.state = TaskState.IDLE
        self.task_sequence = []
        self.current_task_index = 0
    
    def _attempt_pick_with_retry(self, obj_data, max_retries=MAX_PICK_RETRIES):
        """Attempt to pick object with retries using different approach strategies"""
        obj_name = obj_data['name']
        
        # Retry offset strategies
        retry_offsets = [
            np.array([0, 0, 0]),
            np.array([0.01, 0, 0]),
            np.array([-0.01, 0, 0]),
            np.array([0, 0.01, 0]),
            np.array([0, 0, -0.01]),
        ]
        
        for attempt in range(max_retries):
            self.total_attempts += 1
            
            # Refresh object position from perception
            current_pos, current_orient = self._get_current_object_position(obj_name)
            if current_pos is not None:
                rospy.loginfo(f"  Refreshed position for {obj_name}: {current_pos}")
                obj_data['position'] = current_pos
                if current_orient is not None:
                    obj_data['orientation'] = current_orient
                    rospy.loginfo(f"  Refreshed orientation for {obj_name}: {current_orient}")
            else:
                rospy.logwarn(f"  Could not refresh position for {obj_name}, using stored position")
            
            # Apply retry offset
            offset = retry_offsets[min(attempt, len(retry_offsets)-1)]
            pick_pos = np.array(obj_data['position']) + offset
            
            # Get orientation (may be None if not available)
            pick_orient = obj_data.get('orientation', None)
            
            if attempt > 0:
                self.retries_used += 1
                rospy.logwarn(f"Retry attempt {attempt+1}/{max_retries} for picking {obj_name}")
                rospy.logwarn(f"  Using offset: {offset} -> pick pos: {pick_pos}")
            
            success = self.motion_planner.pick_object(pick_pos, self.robot_interface, 
                                                        object_orientation=pick_orient,
                                                        object_class=obj_data.get('class', None))
            
            if success:
                self.successful_picks += 1
                rospy.loginfo(f"Successfully picked {obj_name} on attempt {attempt+1}")
                return True
            else:
                self.failed_picks += 1
                rospy.logwarn(f"Pick attempt {attempt+1} failed for {obj_name}")
        
        rospy.logerr(f"Failed to pick {obj_name} after {max_retries} attempts")
        return False
    
    def _get_current_object_position(self, obj_name):
        """Get current position and orientation of object from perception"""
        # Get fresh object list from perception
        objects = self.perception.get_detected_objects()
        for obj in objects:
            if obj['name'] == obj_name:
                position = np.array(obj['position'])
                orientation = obj.get('orientation', None)
                if orientation is not None:
                    orientation = np.array(orientation)
                return position, orientation
        return None, None
    
    def _verify_object_lifted(self, obj_name, original_z, min_lift_height=0.05):
        """Check if object Z is above original position by at least min_lift_height"""
        rospy.sleep(0.3)
        
        current_pos, _ = self._get_current_object_position(obj_name)
        if current_pos is None:
            rospy.logwarn(f"  Could not find {obj_name} in perception - assuming pick failed")
            return False
        
        current_z = current_pos[2]
        height_diff = current_z - original_z
        
        rospy.loginfo(f"  PICK VERIFICATION for {obj_name}:")
        rospy.loginfo(f"    Original Z: {original_z:.3f}m")
        rospy.loginfo(f"    Current Z:  {current_z:.3f}m")
        rospy.loginfo(f"    Height diff: {height_diff:.3f}m (need > {min_lift_height:.3f}m)")
        
        if height_diff > min_lift_height:
            return True
        else:
            rospy.logwarn(f"    Object NOT lifted: still at original height!")
            return False
    
    def _attempt_place_with_retry(self, target_pos, max_retries=MAX_PLACE_RETRIES):
        """Attempt to place object with retries"""
        for attempt in range(max_retries):
            self.total_attempts += 1
            if attempt > 0:
                self.retries_used += 1
                rospy.logwarn(f"Retry attempt {attempt+1}/{max_retries} for placing")
                
                # Add small offset for retry
                offset = np.array([0, RETRY_OFFSET * attempt, 0])
                target_pos = target_pos + offset
            
            success = self.motion_planner.place_object(target_pos, self.robot_interface)
            
            if success:
                self.successful_places += 1
                rospy.loginfo(f"Successfully placed object on attempt {attempt+1}")
                return True
            else:
                self.failed_places += 1
                rospy.logwarn(f"Place attempt {attempt+1} failed")
        
        rospy.logerr(f"Failed to place object after {max_retries} attempts")
        return False
    
    def get_stats(self):
        """Get execution statistics"""
        return {
            'total_attempts': self.total_attempts,
            'successful_picks': self.successful_picks,
            'successful_places': self.successful_places,
            'failed_picks': self.failed_picks,
            'failed_places': self.failed_places,
            'retries_used': self.retries_used
        }
