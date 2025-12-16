"""
Motion planning and control for UR5 manipulator
Handles inverse kinematics, trajectory generation, and gripper control
"""

import numpy as np
import rospy
from geometry_msgs.msg import Pose
import pinocchio as pin


class MotionPlanner:
    """
    Handles arm motion planning and execution
    """

    def __init__(self, controller, config):
        self.controller = controller  # Controller object
        self.config = config
        self.current_joint_pos = np.zeros(6)

    def compute_ik(self, target_pose, initial_guess=None):
        """
        Compute inverse kinematics for target end-effector pose

        Args:
            target_pose: Desired end-effector pose [x, y, z, roll, pitch, yaw]
            initial_guess: Initial joint configuration

        Returns:
            Joint configuration or None if IK fails
        """
        if initial_guess is None:
            # Use current joint configuration from controller
            initial_guess, _ = self.controller.get_current_joint_state()

        # Convert target pose to SE3
        x, y, z = target_pose[:3]

        # Create target SE3 transformation
        target_SE3 = pin.SE3.Identity()
        target_SE3.translation = np.array([x, y, z])

        if len(target_pose) > 3:
            # Apply orientation if provided
            roll, pitch, yaw = target_pose[3:6]
            R = pin.rpy.rpyToMatrix(roll, pitch, yaw)
            target_SE3.rotation = R

        # Solve IK using Pinocchio (accessing through controller's robot model)
        # The controller has self.robot_instance which contains the Pinocchio model
        robot = self.controller
        q = initial_guess.copy()
        eps = 1e-4
        max_iter = 1000
        dt = 0.1

        for i in range(max_iter):
            # Forward kinematics
            pin.forwardKinematics(robot.robot.model, robot.robot.data, q)
            pin.updateFramePlacements(robot.robot.data.model, robot.robot.data)

            # Get current end-effector pose
            ee_frame_id = robot.robot.model.getFrameId("tool0")
            current_SE3 = robot.robot.data.oMf[ee_frame_id]

            # Compute error
            error = pin.log(current_SE3.inverse() * target_SE3).vector

            if np.linalg.norm(error) < eps:
                return q

            # Compute Jacobian
            J = pin.computeFrameJacobian(
                robot.robot.model, robot.robot.data, q, ee_frame_id, pin.ReferenceFrame.LOCAL
            )

            # Compute joint velocity
            v = pin.utils.zero(robot.robot.model.nv)
            v = np.linalg.pinv(J) @ error

            # Update joint positions
            q = pin.integrate(robot.robot.model, q, v * dt)
            """
        Generate smooth trajectory between joint configurations
        
        Args:
            start_joints: Starting joint configuration
            end_joints: Ending joint configuration
            duration: Trajectory duration in seconds
            
        Returns:
            List of (time, joint_pos, joint_vel) tuples
        """
        trajectory = []
        num_points = int(duration * 100)  # 100 Hz

        for i in range(num_points):
            t = i / num_points
            # Simple linear interpolation (can use splines for smoother motion)
            joints = start_joints + t * (end_joints - start_joints)

            # Compute velocity
            if i > 0:
                vel = (joints - trajectory[-1][1]) / 0.01
            else:
                vel = np.zeros_like(joints)

            trajectory.append((t * duration, joints, vel))

        return trajectory

    def execute_trajectory(self, trajectory, robot_interface):
        """
        Execute planned trajectory on robot

        Args:
            trajectory: List of (time, joints, vel) tuples
            robot_interface: Robot control interface
        """
        rate = rospy.Rate(100)  # 100 Hz

        for t, joints, vel in trajectory:
            # Send joint commands
            robot_interface.send_joint_command(joints, vel)
            rate.sleep()

    def move_to_pose(self, target_pose, robot_interface):
        """
        Move end-effector to target pose

        Args:
            target_pose: [x, y, z, roll, pitch, yaw]
            robot_interface: Robot control interface

        Returns:
            True if successful, False otherwise
        """
        # Compute IK
        target_joints = self.compute_ik(target_pose)

        if target_joints is None:
            rospy.logerr("IK failed for target pose")
            return False

        # Plan trajectory
        traj = self.plan_trajectory(self.current_joint_pos, target_joints)

        # Execute
        self.execute_trajectory(traj, robot_interface)

        self.current_joint_pos = target_joints
        return True

    def move_to_joints(self, target_joints, robot_interface):
        """
        Move to target joint configuration
        """
        traj = self.plan_trajectory(self.current_joint_pos, target_joints)
        self.execute_trajectory(traj, robot_interface)
        self.current_joint_pos = target_joints

    def open_gripper(self, robot_interface):
        """Open gripper"""
        robot_interface.send_gripper_command(0.08)  # Open width in meters
        rospy.sleep(0.5)

    def close_gripper(self, robot_interface):
        """Close gripper to grasp object"""
        robot_interface.send_gripper_command(0.0)  # Close
        rospy.sleep(0.5)

    def pick_object(self, object_pose, robot_interface):
        """
        Execute pick motion for object

        Args:
            object_pose: [x, y, z] position of object
            robot_interface: Robot control interface

        Returns:
            True if successful
        """
        x, y, z = object_pose[:3]

        # 1. Move to approach position (above object)
        approach_pose = [x, y, z + self.config.approach_height, 0, np.pi, 0]
        if not self.move_to_pose(approach_pose, robot_interface):
            return False

        # 2. Open gripper
        self.open_gripper(robot_interface)

        # 3. Move down to grasp height
        grasp_pose = [x, y, z + self.config.grasp_height_offset, 0, np.pi, 0]
        if not self.move_to_pose(grasp_pose, robot_interface):
            return False

        # 4. Close gripper
        self.close_gripper(robot_interface)

        # 5. Lift object
        lift_pose = [x, y, z + self.config.lift_height, 0, np.pi, 0]
        if not self.move_to_pose(lift_pose, robot_interface):
            return False

        return True

    def place_object(self, target_pose, robot_interface):
        """
        Execute place motion for object

        Args:
            target_pose: [x, y, z] position to place object
            robot_interface: Robot control interface

        Returns:
            True if successful
        """
        x, y, z = target_pose[:3]

        # 1. Move to approach position above target
        approach_pose = [x, y, z + self.config.approach_height, 0, np.pi, 0]
        if not self.move_to_pose(approach_pose, robot_interface):
            return False

        # 2. Move down to place height
        place_pose = [x, y, z + self.config.grasp_height_offset, 0, np.pi, 0]
        if not self.move_to_pose(place_pose, robot_interface):
            return False

        # 3. Open gripper
        self.open_gripper(robot_interface)

        # 4. Retract
        retract_pose = [x, y, z + self.config.approach_height, 0, np.pi, 0]
        if not self.move_to_pose(retract_pose, robot_interface):
            return False

        return True
