# -*- coding: utf-8 -*-
"""
Pick and Place Controller using BaseControllerFixed
This version integrates with the Gazebo simulation properly

Run with: python3 -i pick_and_place_gazebo.py

Then use:
  p.start_task()  - Start pick and place
  p.stop()        - Stop
  p.reset()       - Reset
"""

from __future__ import print_function
import os
import sys
import numpy as np
import math
import time as tm
import threading
import rospkg

# Add paths
roscontrol_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, roscontrol_path)

from base_controllers.base_controller_fixed import BaseControllerFixed
from base_controllers.utils.common_functions import plotJoint
import base_controllers.params as base_conf
import pick_and_place_conf as conf
import rospy as ros
from gazebo_msgs.msg import ModelStates
import pinocchio as pin


class PickAndPlaceController(BaseControllerFixed):
    """Pick and Place Controller extending the base controller."""

    # State machine states
    IDLE = 0
    MOVE_HOME = 1
    MOVE_TO_APPROACH = 2
    MOVE_TO_GRASP = 3
    GRASP = 4
    LIFT = 5
    MOVE_TO_PLACE_APPROACH = 6
    MOVE_TO_PLACE = 7
    RELEASE = 8
    LIFT_AFTER_PLACE = 9
    DONE = 10

    def __init__(self):
        super().__init__(robot_name="ur5")
        self.use_torque_control = conf.use_torque_control

        # Task state
        self.state = self.IDLE
        self.current_object_idx = 0
        self.object_names = sorted(conf.objects.keys(), key=lambda x: conf.objects[x]['priority'])
        self.grasping = False

        # Trajectory
        self.traj_start_time = 0.0
        self.traj_duration = 2.0
        self.traj_start_pos = np.zeros(3)
        self.traj_end_pos = np.zeros(3)

        # Ground truth object poses
        self.object_poses = {}

        print("PickAndPlaceController initialized")

    def initVars(self):
        """Initialize variables."""
        super().initVars()
        self.q_des = conf.q0.copy()
        self.qd_des = np.zeros(self.robot.na)

        # Get EE frame
        self.frame_ee = self.robot.model.getFrameId(conf.frame_name)

        # Initial EE position
        self.p0 = self.robot.framePlacement(conf.q0, self.frame_ee, True).translation.copy()
        self.p_des = self.p0.copy()

    def startSimulator(self):
        """Start the Gazebo simulation."""
        additional_args = [
            f'gripper:={str(conf.gripper).lower()}',
            f'soft_gripper:={str(conf.soft_gripper).lower()}',
            f'robotiq_gripper:={str(conf.robotiq_gripper).lower()}',
        ]
        super().startSimulator(world_name=conf.world_name, additional_args=additional_args)

    def loadModelAndPublishers(self):
        """Load model and set up subscribers."""
        super().loadModelAndPublishers()

        # Subscribe to Gazebo model states for ground truth
        self.sub_model_states = ros.Subscriber(
            "/gazebo/model_states", ModelStates, callback=self._receive_model_states, queue_size=1
        )

    def _receive_model_states(self, msg):
        """Callback for ground truth object positions."""
        for i, name in enumerate(msg.name):
            if name in conf.objects:
                pos = msg.pose[i].position
                self.object_poses[name] = np.array([pos.x, pos.y, pos.z])

    def start_trajectory(self, target_pos, duration=2.0):
        """Start a trajectory to target position."""
        self.traj_start_time = self.time
        self.traj_duration = duration
        self.traj_start_pos = self.robot.framePlacement(
            self.q, self.frame_ee, True
        ).translation.copy()
        self.traj_end_pos = target_pos.copy()
        print(f"  -> Trajectory to [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

    def trajectory_complete(self):
        """Check if trajectory is complete."""
        return (self.time - self.traj_start_time) >= self.traj_duration

    def interpolate_position(self):
        """Get interpolated position along trajectory."""
        t = self.time
        t0 = self.traj_start_time
        duration = self.traj_duration

        s = (t - t0) / duration
        s = np.clip(s, 0, 1)
        # Quintic polynomial for smooth motion
        s_smooth = 10 * s**3 - 15 * s**4 + 6 * s**5
        return self.traj_start_pos + s_smooth * (self.traj_end_pos - self.traj_start_pos)

    def get_current_object(self):
        """Get current object info."""
        if self.current_object_idx < len(self.object_names):
            name = self.object_names[self.current_object_idx]
            return name, conf.objects[name]
        return None, None

    def start_task(self):
        """Start pick and place task."""
        print("\n" + "=" * 50)
        print("  Starting Pick and Place Task")
        print("  Objects:", self.object_names)
        print("=" * 50)
        self.state = self.MOVE_HOME
        self.current_object_idx = 0
        self.grasping = False
        self.start_trajectory(self.p0, 2.0)

    def stop(self):
        """Stop task."""
        self.state = self.IDLE
        print("Task stopped")

    def reset(self):
        """Reset controller."""
        self.state = self.IDLE
        self.current_object_idx = 0
        self.grasping = False
        self.q_des = conf.q0.copy()
        print("Reset complete")

    def state_machine(self):
        """Execute state machine."""
        obj_name, obj = self.get_current_object()

        if self.state == self.IDLE:
            pass

        elif self.state == self.MOVE_HOME:
            if self.trajectory_complete():
                if obj is None:
                    self.state = self.DONE
                    print("\n*** All objects placed! Task complete! ***\n")
                else:
                    # Get object position (ground truth or config)
                    if obj_name in self.object_poses:
                        obj_pos = self.object_poses[obj_name]
                    else:
                        obj_pos = obj['initial_pos']

                    approach_pos = obj_pos.copy()
                    approach_pos[2] += conf.approach_height
                    self.start_trajectory(approach_pos, 2.0)
                    self.state = self.MOVE_TO_APPROACH
                    print(
                        f"\n[{self.current_object_idx+1}/{len(self.object_names)}] Picking {obj_name}"
                    )

        elif self.state == self.MOVE_TO_APPROACH:
            if self.trajectory_complete():
                if obj_name in self.object_poses:
                    obj_pos = self.object_poses[obj_name]
                else:
                    obj_pos = obj['initial_pos']

                grasp_pos = obj_pos.copy()
                grasp_pos[2] += conf.grasp_height
                self.start_trajectory(grasp_pos, 1.5)
                self.state = self.MOVE_TO_GRASP
                print(f"  Moving to grasp")

        elif self.state == self.MOVE_TO_GRASP:
            if self.trajectory_complete():
                self.grasping = True
                self.state = self.GRASP
                self.traj_start_time = self.time  # Reset for wait
                self.traj_duration = 0.5  # Wait time
                print(f"  Grasping...")

        elif self.state == self.GRASP:
            if self.trajectory_complete():
                if obj_name in self.object_poses:
                    obj_pos = self.object_poses[obj_name]
                else:
                    obj_pos = obj['initial_pos']

                lift_pos = obj_pos.copy()
                lift_pos[2] += conf.lift_height
                self.start_trajectory(lift_pos, 1.5)
                self.state = self.LIFT
                print(f"  Lifting")

        elif self.state == self.LIFT:
            if self.trajectory_complete():
                place_approach = obj['target_pos'].copy()
                place_approach[2] += conf.approach_height
                self.start_trajectory(place_approach, 2.5)
                self.state = self.MOVE_TO_PLACE_APPROACH
                print(f"  Moving to target")

        elif self.state == self.MOVE_TO_PLACE_APPROACH:
            if self.trajectory_complete():
                place_pos = obj['target_pos'].copy()
                place_pos[2] += conf.place_height
                self.start_trajectory(place_pos, 1.5)
                self.state = self.MOVE_TO_PLACE
                print(f"  Lowering to place")

        elif self.state == self.MOVE_TO_PLACE:
            if self.trajectory_complete():
                self.grasping = False
                self.state = self.RELEASE
                self.traj_start_time = self.time
                self.traj_duration = 0.5
                print(f"  Releasing")

        elif self.state == self.RELEASE:
            if self.trajectory_complete():
                lift_pos = obj['target_pos'].copy()
                lift_pos[2] += conf.lift_height
                self.start_trajectory(lift_pos, 1.5)
                self.state = self.LIFT_AFTER_PLACE

        elif self.state == self.LIFT_AFTER_PLACE:
            if self.trajectory_complete():
                print(f"  ✓ {obj_name} placed successfully!")
                self.current_object_idx += 1
                self.start_trajectory(self.p0, 2.0)
                self.state = self.MOVE_HOME

    def controlLoop(self):
        """Main control loop."""
        # State machine
        self.state_machine()

        # Generate reference
        if self.state not in [self.IDLE, self.DONE]:
            self.p_des = self.interpolate_position()

        # Get current EE state
        p = self.robot.framePlacement(self.q, self.frame_ee).translation
        J6 = self.robot.frameJacobian(
            self.q, self.frame_ee, False, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J6[:3, :]
        pd = J.dot(self.qd)

        # Compute dynamics
        M = self.robot.mass(self.q, False)
        h = self.robot.nle(self.q, self.qd, False)
        g = self.robot.gravity(self.q)
        M_inv = np.linalg.inv(M)

        # Damped pseudoinverse
        rho = 0.01
        JTpinv = np.linalg.inv(J.dot(J.T) + rho * np.eye(3)).dot(J)

        # Null space projector
        N = np.eye(self.robot.na) - J.T.dot(JTpinv)

        # Cartesian PD control
        Kp = np.diag([300, 300, 300])
        Kd = np.diag([50, 50, 50])
        F_des = Kp.dot(self.p_des - p) + Kd.dot(-pd)

        # Null space postural task
        tau_null = N.dot(100 * (conf.q0 - self.q) - 20 * self.qd)

        # Total torque with gravity compensation
        tau = J.T.dot(F_des) + g + tau_null

        # Send command
        self.send_des_jstate(self.q_des, self.qd_des, tau)

        # Visualization
        self.ros_pub.add_marker(p, color="green" if self.grasping else "red")
        self.ros_pub.add_marker(self.p_des, color="blue")


def main():
    """Main function."""
    p = PickAndPlaceController()

    try:
        # Start simulation
        p.startSimulator()
        ros.sleep(1.0)

        # Load model and publishers
        xacro_path = (
            rospkg.RosPack().get_path('ur_description') + '/urdf/' + p.robot_name + '.urdf.xacro'
        )
        p.loadModelAndPublishers(xacro_path)
        p.initVars()
        p.startupProcedure()

        # Print instructions
        print("\n" + "=" * 60)
        print("  Pick and Place Controller Ready!")
        print("=" * 60)
        print("Commands:")
        print("  p.start_task()  - Start pick and place")
        print("  p.stop()        - Stop")
        print("  p.reset()       - Reset")
        print("=" * 60 + "\n")

        # Control loop
        rate = ros.Rate(1 / base_conf.robot_params[p.robot_name]['dt'])

        while not ros.is_shutdown():
            p.controlLoop()
            p.logData()
            rate.sleep()
            p.time = np.round(p.time + base_conf.robot_params[p.robot_name]['dt'], 4)

        return p

    except (ros.ROSInterruptException, ros.service.ServiceException):
        ros.signal_shutdown("killed")
        p.deregister_node()
        raise


if __name__ == "__main__":
    p = main()
