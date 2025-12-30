"""
High-level task scheduler for pick and place operations
Coordinates perception, planning, and execution
"""

import rospy
import numpy as np
from enum import Enum


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
        target_base = self.config.target_table_pos

        for i, obj in enumerate(objects):
            obj_class = obj.get('class', 'unknown')

            # Calculate target position (spread objects across target table)
            # Arrange in a grid pattern on target table
            spacing = 0.08  # 8cm spacing between objects
            row = i // 3  # 3 objects per row
            col = i % 3
            offset_x = (col - 1) * spacing  # Center around target
            offset_y = (row - 1) * spacing

            target_pos = np.array(
                [
                    target_base[0] + offset_x,
                    target_base[1] + offset_y,
                    target_base[2],  # Same height as target table
                ]
            )

            task_sequence.append({'object': obj, 'target': target_pos, 'class': obj_class})

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
        for idx, task in enumerate(self.task_sequence):
            self.current_task_index = idx

            rospy.loginfo(
                f"Task {idx+1}/{len(self.task_sequence)}: " f"Moving {task['class']} object"
            )

            # Pick phase
            self.state = TaskState.MOVING_TO_OBJECT
            obj_pos = task['object']['position']

            rospy.loginfo(f"  Picking from position: {obj_pos}")
            if not self.motion_planner.pick_object(obj_pos, self.robot_interface):
                rospy.logerr(f"Failed to pick object at task {idx}")
                self.state = TaskState.ERROR
                return False

            self.state = TaskState.PICKING

            # Place phase
            self.state = TaskState.MOVING_TO_TARGET
            target_pos = task['target']

            rospy.loginfo(f"  Placing at position: {target_pos}")
            if not self.motion_planner.place_object(target_pos, self.robot_interface):
                rospy.logerr(f"Failed to place object at task {idx}")
                self.state = TaskState.ERROR
                return False

            self.state = TaskState.PLACING

            rospy.loginfo(f"Task {idx+1} completed successfully")

        # Return to home
        self.state = TaskState.RETURNING_HOME
        rospy.loginfo("Returning to home position...")
        self.motion_planner.move_to_joints(self.config.home_joint_config, self.robot_interface)

        self.state = TaskState.COMPLETED
        rospy.loginfo("All tasks completed successfully!")
        return True

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
