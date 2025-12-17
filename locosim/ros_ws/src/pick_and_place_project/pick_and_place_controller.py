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
sys.path.append(os.path.join(project_dir, 'config'))  # Add config directory

# Import locosim base controller
sys.path.insert(0, os.path.join(os.environ['LOCOSIM_DIR'], 'robot_control/base_controllers'))
from base_controller_fixed import BaseControllerFixed  # Use BaseControllerFixed instead

# Import project modules
from config.params import (  # Import from config subdirectory
    robot_name,
    use_ground_truth,
    gripper_type,
    table_initial,
    table_final,
    object_classes,
    approach_height,
    grasp_height_offset,
    lift_height,
    camera_topic,
    camera_frame,
    max_velocity,
    max_acceleration,
    home_joint_config,
    min_object_points,
    segmentation_threshold,
)


# Create a simple config object
class Config:
    def __init__(self):
        self.robot_name = robot_name
        self.use_ground_truth = use_ground_truth
        self.gripper_type = gripper_type
        self.table_initial = table_initial
        self.table_final = table_final
        self.object_classes = object_classes
        self.approach_height = approach_height
        self.grasp_height_offset = grasp_height_offset
        self.lift_height = lift_height
        self.camera_topic = camera_topic
        self.camera_frame = camera_frame
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.home_joint_config = home_joint_config
        self.min_object_points = min_object_points
        self.segmentation_threshold = segmentation_threshold


from perception_module import PerceptionModule
from motion_planner_simple import MotionPlanner  # Using simplified version without IK
from task_scheduler import TaskScheduler


class PickAndPlaceController(BaseControllerFixed):  # Inherit from BaseControllerFixed
    """
    Main controller for pick and place project
    """

    def __init__(self):
        # Ensure ROS is initialized first
        if not rospy.core.is_initialized():
            rospy.init_node('pick_and_place_controller', anonymous=False)
            rospy.loginfo("ROS node initialized")

        # Initialize base controller with UR5
        super().__init__('ur5')

        rospy.loginfo("Initializing Pick and Place Controller...")

        # Load configuration
        self.config = Config()

        # Initialize modules
        self.perception = PerceptionModule(self.config)
        # Pass controller instance to motion planner instead of self.robot
        self.motion_planner = MotionPlanner(self, self.config)
        self.task_scheduler = TaskScheduler(self.perception, self.motion_planner, self, self.config)

        self.task_running = False
        self.model_states_sub = None  # Will be created after Gazebo starts

        rospy.loginfo("Pick and Place Controller initialized!")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Commands:")
        rospy.loginfo("  p.start_task()  - Start pick and place task")
        rospy.loginfo("  p.stop()        - Emergency stop")
        rospy.loginfo("  p.reset()       - Reset controller")
        rospy.loginfo("  p.go_home()     - Move to home position")
        rospy.loginfo("=" * 60)

    def startSimulator(self):
        """Start the Gazebo simulation with pick and place world"""
        # Import config values
        import pick_and_place_conf as conf

        additional_args = [
            f'gripper:={str(conf.gripper).lower()}',
            f'soft_gripper:={str(conf.soft_gripper).lower()}',
            f'robotiq_gripper:={str(conf.robotiq_gripper).lower()}',
        ]
        rospy.loginfo(f"Starting Gazebo with world: {conf.world_name}")
        super().startSimulator(world_name=conf.world_name, additional_args=additional_args)

    def initVars(self):
        """Initialize variables after simulator starts"""
        import pick_and_place_conf as conf

        # Don't call super().initVars() since we don't have robot model loaded
        # Manually initialize what we need (8 joints: 6 arm + 2 gripper)
        self.q_des = np.concatenate([conf.q0, np.zeros(2)])  # Add gripper joints
        self.qd_des = np.zeros(8)  # 6 arm + 2 gripper

        # Initialize state variables that base controller expects
        self.q = np.zeros(8)
        self.qd = np.zeros(8)
        self.tau_ffwd = np.zeros(8)

        # Track gripper state
        self.gripper_pos = 0.0

        # Create publisher for desired joint states (Gazebo listens on /command)
        from sensor_msgs.msg import JointState

        self.pub_des_jstate = rospy.Publisher("/command", JointState, queue_size=1)
        rospy.sleep(0.5)  # Wait for publisher to connect

        # Now that Gazebo is running, subscribe to model states for ground truth
        if self.config.use_ground_truth:
            rospy.loginfo("Creating model_states subscriber...")
            self.model_states_sub = rospy.Subscriber(
                '/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1
            )
            rospy.loginfo("Subscriber created, waiting for first message...")
            # Give the subscriber a moment to connect
            rospy.sleep(0.5)

        rospy.loginfo("Variables initialized")

    def model_states_callback(self, msg):
        """Callback for Gazebo model states (ground truth)"""
        rospy.logdebug(f"model_states_callback triggered with {len(msg.name)} models")
        if self.config.use_ground_truth:
            rospy.logdebug(f"Models in callback: {msg.name}")
            self.perception.update_ground_truth(msg)
            rospy.loginfo_once(f"Ground truth callback working! Found {len(msg.name)} models")

    def test_model_states(self):
        """Diagnostic: Test if model_states topic is working"""
        rospy.loginfo("Testing /gazebo/model_states topic...")
        try:
            from gazebo_msgs.msg import ModelStates

            msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=2.0)
            rospy.loginfo(f"SUCCESS! Received message with {len(msg.name)} models:")
            for name in msg.name:
                rospy.loginfo(f"  - {name}")
            rospy.loginfo(f"Manually updating perception...")
            self.perception.update_ground_truth(msg)
            rospy.loginfo(
                f"Perception now has {len(self.perception.get_detected_objects())} objects"
            )
        except Exception as e:
            rospy.logerr(f"FAILED to receive model_states: {e}")

    def start_task(self):
        """Start the pick and place task"""
        if self.task_running:
            rospy.logwarn("Task is already running!")
            return

        rospy.loginfo(f"Ground truth mode: {self.config.use_ground_truth}")
        rospy.loginfo(
            f"Perception has detected {len(self.perception.get_detected_objects())} objects"
        )

        rospy.loginfo("=" * 60)
        rospy.loginfo("Starting pick and place task...")
        rospy.loginfo("=" * 60)
        self.task_running = True

        try:
            # Check if perception has detected objects
            detected = self.perception.get_detected_objects()
            rospy.loginfo(f"Perception has detected {len(detected)} objects")

            success = self.task_scheduler.execute_task_sequence()
            if success:
                rospy.loginfo("=" * 60)
                rospy.loginfo("Task completed successfully!")
                rospy.loginfo("=" * 60)
            else:
                rospy.logerr("=" * 60)
                rospy.logerr("Task failed!")
                rospy.logerr("=" * 60)
        except Exception as e:
            rospy.logerr("=" * 60)
            rospy.logerr(f"Error during task execution: {e}")
            rospy.logerr("=" * 60)
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
        self.motion_planner.move_to_joints(np.array(self.config.home_joint_config), self)

    def send_des_jstate(self, q_des, qd_des, tau_ffwd):
        """
        Override base controller's send_des_jstate to use position controller
        """
        from std_msgs.msg import Float64MultiArray

        # Update internal state
        self.q_des = q_des.copy()
        self.qd_des = qd_des.copy()
        self.tau_ffwd = tau_ffwd.copy()

        # Publish JointState to /command (where Gazebo listens)
        from sensor_msgs.msg import JointState

        msg = JointState()
        msg.position = q_des.tolist()
        msg.velocity = qd_des.tolist()
        msg.effort = tau_ffwd.tolist()
        self.pub_des_jstate.publish(msg)

    def send_joint_command(self, joints, velocities=None):
        """
        Send joint position commands to robot
        Interface for motion planner
        """
        # Pad to 8 joints if only 6 provided (add gripper)
        if len(joints) == 6:
            joints_full = np.concatenate([joints, [self.gripper_pos, self.gripper_pos]])
        else:
            joints_full = joints.copy()

        # Update desired joint states
        self.q_des = joints_full
        if velocities is not None:
            if len(velocities) == 6:
                self.qd_des = np.concatenate([velocities, [0.0, 0.0]])
            else:
                self.qd_des = velocities.copy()
        else:
            self.qd_des = np.zeros(8)

        # Send commands through base controller
        self.send_des_jstate(self.q_des, self.qd_des, self.tau_ffwd)

    def send_gripper_command(self, width):
        """
        Send gripper command
        width: desired gripper opening (meters)
        For soft gripper: 0.0 = closed, 0.085 = fully open
        """
        from std_msgs.msg import Float64

        # Update gripper position tracking
        self.gripper_pos = width / 2.0  # Divide by 2 for each finger

        # Create publisher if it doesn't exist
        if not hasattr(self, 'gripper_pub'):
            self.gripper_pub = rospy.Publisher('/gripper_controller/command', Float64, queue_size=1)
            rospy.sleep(0.5)  # Wait for publisher to be ready

        # Publish command
        self.gripper_pub.publish(Float64(width))
        rospy.loginfo(f"Gripper command: {width*1000:.1f}mm")
        rospy.sleep(1.5)  # Wait for gripper to actuate

    def get_current_joint_state(self):
        """Get current joint positions and velocities"""
        return self.q.copy(), self.qd.copy()

    def run(self):
        """Main control loop"""
        rate = rospy.Rate(1.0 / self.dt)

        while not rospy.is_shutdown():
            # Update base controller kinematics
            self.updateKinematics()

            # Task scheduler handles high-level control
            # Low-level control is handled by send_joint_command

            rate.sleep()


def main():
    """Main function"""
    import rospkg
    import base_controllers.params as base_conf

    # Create controller (it will initialize ROS node internally)
    p = PickAndPlaceController()

    try:
        # Start Gazebo simulator
        rospy.loginfo("Starting Gazebo simulator...")
        p.startSimulator()
        rospy.sleep(5.0)  # Wait for Gazebo to fully start

        # NOTE: Skipping loadModelAndPublishers() because Pinocchio URDF loading fails
        # Our simplified motion planner doesn't need the Pinocchio robot model
        rospy.logwarn("Skipping Pinocchio model loading (using simplified controller)")

        # Initialize variables (this also creates the model_states subscriber)
        rospy.loginfo("Initializing variables...")
        p.tau_ffwd = np.zeros(6)
        p.initVars()

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("  Pick and Place Controller Ready!")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Commands:")
        rospy.loginfo("  p.start_task()  - Start pick and place task")
        rospy.loginfo("  p.stop()        - Emergency stop")
        rospy.loginfo("  p.reset()       - Reset controller")
        rospy.loginfo("  p.go_home()     - Move to home position")
        rospy.loginfo("=" * 60 + "\n")

        # Return controller for interactive use
        return p

    except (rospy.ROSInterruptException, rospy.service.ServiceException):
        rospy.signal_shutdown("killed")
        p.deregister_node()
        raise


if __name__ == '__main__':
    import threading
    import base_controllers.params as base_conf

    # Run in interactive mode
    p = main()

    # Start control loop in background thread
    def control_loop():
        rate = rospy.Rate(1 / base_conf.robot_params[p.robot_name]['dt'])
        while not rospy.is_shutdown():
            if hasattr(p, 'updateKinematics'):
                p.updateKinematics()
            rate.sleep()

    # Start control thread
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    rospy.loginfo("\n>>> Python interactive mode active. Type commands here.")
    rospy.loginfo(">>> Example: p.start_task()\n")

    # Note: The script continues and drops into Python interactive mode
    # because it was launched with 'python3 -i'
