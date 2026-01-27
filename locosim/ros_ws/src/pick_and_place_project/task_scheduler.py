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
    """Manages target positions for placing bricks by class"""
    
    def __init__(self, config):
        self.config = config
        self.target_center = np.array(config.target_table_pos)
        self.table_size = getattr(config, 'target_table_size', [0.5, 0.35])
        
        # Track which slots are used per class
        self.slots_per_class = defaultdict(int)  # class_name -> next_slot_index
        
        # Grid configuration
        self.slots_per_row = 3  # 3 bricks per row
        self.brick_spacing = 0.08  # 8cm between brick centers
        
    def get_slot_for_class(self, brick_class):
        """Get next available slot for this brick class"""
        slot_idx = self.slots_per_class[brick_class]
        self.slots_per_class[brick_class] += 1
        
        # Calculate position: each class gets its own row
        class_list = sorted(set(self.slots_per_class.keys()))
        try:
            row = class_list.index(brick_class)
        except ValueError:
            row = 0
        
        col = slot_idx % self.slots_per_row
        
        # Calculate offset from table center
        x_offset = (col - (self.slots_per_row - 1) / 2.0) * self.brick_spacing
        y_offset = row * self.brick_spacing
        
        position = self.target_center.copy()
        position[0] += x_offset
        position[1] += y_offset
        
        return position
    
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
            
            # Return to safe height before next pick to avoid collisions
            # This prevents the robot from moving horizontally at low height and hitting objects
            if idx < len(self.task_sequence) - 1:
                rospy.loginfo("  Returning to safe height before next pick...")
                self.motion_planner.move_to_joints(self.config.home_joint_config, self.robot_interface)

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
        original_pos = np.array(obj_data['position']).copy()
        
        # Different retry strategies: try different small offsets
        retry_offsets = [
            np.array([0, 0, 0]),           # First try: exact position
            np.array([0.01, 0, 0]),         # Retry 1: slight X offset
            np.array([-0.01, 0, 0]),        # Retry 2: opposite X offset
            np.array([0, 0.01, 0]),         # Retry 3: slight Y offset
            np.array([0, 0, -0.01]),        # Retry 4: slightly lower
        ]
        
        for attempt in range(max_retries):
            self.total_attempts += 1
            
            # Apply retry offset
            offset = retry_offsets[min(attempt, len(retry_offsets)-1)]
            obj_data['position'] = original_pos + offset
            
            if attempt > 0:
                self.retries_used += 1
                rospy.logwarn(f"Retry attempt {attempt+1}/{max_retries} for picking {obj_data['name']}")
                rospy.logwarn(f"  Using offset: {offset} -> new pos: {obj_data['position']}")
            
            success = self.motion_planner.pick_object(obj_data['position'], self.robot_interface)
            
            if success:
                self.successful_picks += 1
                rospy.loginfo(f"Successfully picked {obj_data['name']} on attempt {attempt+1}")
                return True
            else:
                self.failed_picks += 1
                rospy.logwarn(f"Pick attempt {attempt+1} failed for {obj_data['name']}")
        
        # Restore original position
        obj_data['position'] = original_pos
        rospy.logerr(f"Failed to pick {obj_data['name']} after {max_retries} attempts")
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
