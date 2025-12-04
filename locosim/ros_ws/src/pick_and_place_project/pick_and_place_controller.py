#!/usr/bin/env python3
"""
Main controller for Pick and Place Project
Integrates perception, motion planning, and task scheduling

Usage:
    python3 -i pick_and_place_controller.py

Controls:
    p.start_task()  - Start the pick and place task
    p.stop()        - Emergency stop
    p.reset()       - Reset to initial state
"""

import sys
import os
import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# Import locosim base controller
sys.path.insert(0, os.path.join(os.environ['LOCOSIM_DIR'], 'robot_control/base_controllers'))
from base_controller import BaseController

# Import project modules
from config import params
from perception_module import PerceptionModule
from motion_planner import MotionPlanner
from task_scheduler import TaskScheduler


class PickAndPlaceController(BaseController):
    """
    Main controller for pick and place project
    """
    
    def __init__(self):
        # Initialize base controller with UR5
        super().__init__('ur5')
        
        rospy.loginfo("Initializing Pick and Place Controller...")
        
        # Load configuration
        self.config = params
        
        # Initialize modules
        self.perception = PerceptionModule(self.config)
        self.motion_planner = MotionPlanner(self.robot, self.config)
        self.task_scheduler = TaskScheduler(
            self.perception,
            self.motion_planner,
            self,
            self.config
        )
        
        # Subscribe to Gazebo model states for ground truth
        if self.config.use_ground_truth:
            self.model_states_sub = rospy.Subscriber(
                '/gazebo/model_states',
                ModelStates,
                self.model_states_callback
            )
        
        self.task_running = False
        
        rospy.loginfo("Pick and Place Controller initialized!")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Commands:")
        rospy.loginfo("  p.start_task()  - Start pick and place task")
        rospy.loginfo("  p.stop()        - Emergency stop")
        rospy.loginfo("  p.reset()       - Reset controller")
        rospy.loginfo("  p.go_home()     - Move to home position")
        rospy.loginfo("=" * 60)
    
    def model_states_callback(self, msg):
        """Callback for Gazebo model states (ground truth)"""
        if self.config.use_ground_truth:
            self.perception.use_ground_truth_positions(msg)
    
    def start_task(self):
        """Start the pick and place task"""
        if self.task_running:
            rospy.logwarn("Task is already running!")
            return
        
        rospy.loginfo("Starting pick and place task...")
        self.task_running = True
        
        try:
            success = self.task_scheduler.execute_task_sequence()
            if success:
                rospy.loginfo("Task completed successfully!")
            else:
                rospy.logerr("Task failed!")
        except Exception as e:
            rospy.logerr(f"Error during task execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.task_running = False
    
    def stop(self):
        """Emergency stop"""
        rospy.logwarn("Emergency stop activated!")
        self.task_scheduler.emergency_stop()
        self.task_running = False
    
    def reset(self):
        """Reset controller to initial state"""
        rospy.loginfo("Resetting controller...")
        self.task_scheduler.reset()
        self.task_running = False
        self.go_home()
    
    def go_home(self):
        """Move robot to home position"""
        rospy.loginfo("Moving to home position...")
        self.motion_planner.move_to_joints(
            np.array(self.config.home_joint_config),
            self
        )
    
    def send_joint_command(self, joints, velocities=None):
        """
        Send joint position commands to robot
        Interface for motion planner
        """
        # Convert to proper format for base controller
        for i, (joint_name, pos) in enumerate(zip(self.joint_names, joints)):
            self.q_des[i] = pos
            if velocities is not None:
                self.qd_des[i] = velocities[i]
            else:
                self.qd_des[i] = 0.0
        
        # Send commands through base controller
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)
    
    def send_gripper_command(self, width):
        """
        Send gripper command
        width: desired gripper opening (meters)
        """
        # Implement gripper control
        # This depends on your gripper type
        rospy.loginfo(f"Gripper command: {width}m")
        # TODO: Implement actual gripper control
    
    def get_current_joint_state(self):
        """Get current joint positions and velocities"""
        return self.q.copy(), self.qd.copy()
    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(1.0 / self.dt)
        
        while not rospy.is_shutdown():
            # Update base controller
            self.updateKinematics()
            
            # Task scheduler handles high-level control
            # Low-level control is handled by send_joint_command
            
            rate.sleep()


def main():
    """Main function"""
    rospy.init_node('pick_and_place_controller', anonymous=False)
    
    # Create controller
    controller = PickAndPlaceController()
    
    rospy.loginfo("Controller ready. Waiting for commands...")
    rospy.loginfo("Type 'p.start_task()' to begin")
    
    # Return controller for interactive use
    return controller


if __name__ == '__main__':
    # Run in interactive mode
    p = main()
    
    # Keep alive for interactive commands
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
