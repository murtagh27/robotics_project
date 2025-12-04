# -*- coding: utf-8 -*-
"""
Pick and Place Controller for UR5 Robot
Based on locosim lab exercise structure

Run with: python3 -i pick_and_place_main.py
Then call: start_task() to begin pick and place sequence
"""

from __future__ import print_function
import os
import sys
import numpy as np
import math
import time as tm

# Add paths
roscontrol_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, roscontrol_path)

from base_controllers.utils.common_functions import *
from base_controllers.utils.ros_publish import RosPub
from base_controllers.utils.math_tools import Math
import pick_and_place_conf as conf

# Kill previous instances and initialize
os.system("killall rosmaster rviz gzserver gzclient")
tm.sleep(1.0)

# Initialize ROS publisher and robot model
ros_pub = RosPub("ur5")
robot = getRobotModel("ur5")

math_utils = Math()

# State machine states
class State:
    IDLE = 0
    MOVE_TO_APPROACH = 1
    MOVE_TO_GRASP = 2
    GRASP = 3
    LIFT = 4
    MOVE_TO_PLACE_APPROACH = 5
    MOVE_TO_PLACE = 6
    RELEASE = 7
    LIFT_AFTER_PLACE = 8
    MOVE_HOME = 9
    DONE = 10

# Initialize variables
zero = np.zeros(6)
zero_cart = np.zeros(3)
time_sim = 0.0

# Joint state
q = conf.q0.copy()
qd = conf.qd0.copy()
qdd = conf.qdd0.copy()

# Desired joint state  
q_des = conf.q0.copy()
qd_des = zero.copy()
qdd_des = zero.copy()

# Get end-effector frame ID
assert(robot.model.existFrame(conf.frame_name))
frame_ee = robot.model.getFrameId(conf.frame_name)

# Compute initial end-effector position
p0 = robot.framePlacement(conf.q0, frame_ee, True).translation.copy()
p_des = p0.copy()

# Task state
current_state = State.IDLE
current_object_idx = 0
object_names = sorted(conf.objects.keys(), key=lambda x: conf.objects[x]['priority'])
grasping = False

# Trajectory variables
traj_start_time = 0.0
traj_duration = 2.0
traj_start_pos = p0.copy()
traj_end_pos = p0.copy()

# Logging
buffer_size = int(math.floor(conf.exp_duration/conf.dt))
log_counter = 0
p_log = np.empty((3, buffer_size)) * np.nan
p_des_log = np.empty((3, buffer_size)) * np.nan
time_log = np.empty(buffer_size) * np.nan
state_log = np.empty(buffer_size) * np.nan

def compute_ik(target_pos, target_orient=None, q_init=None):
    """Compute inverse kinematics for target position."""
    if q_init is None:
        q_init = q.copy()
    
    # Simple iterative IK
    q_ik = q_init.copy()
    max_iter = 100
    eps = 1e-4
    
    for i in range(max_iter):
        # Current EE position
        p_curr = robot.framePlacement(q_ik, frame_ee, True).translation
        
        # Error
        error = target_pos - p_curr
        
        if np.linalg.norm(error) < eps:
            break
        
        # Jacobian
        J6 = robot.frameJacobian(q_ik, frame_ee, False, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J6[:3, :]  # Position only
        
        # Pseudoinverse
        J_pinv = np.linalg.pinv(J)
        
        # Update
        dq = J_pinv.dot(error) * 0.5
        q_ik = q_ik + dq
        
    return q_ik

def interpolate_position(t, t0, duration, start, end):
    """Smooth interpolation between positions using quintic polynomial."""
    s = (t - t0) / duration
    s = np.clip(s, 0, 1)
    # Quintic polynomial for smooth start/stop
    s_smooth = 10*s**3 - 15*s**4 + 6*s**5
    return start + s_smooth * (end - start)

def start_trajectory(target_pos, duration=2.0):
    """Start a new trajectory to target position."""
    global traj_start_time, traj_duration, traj_start_pos, traj_end_pos
    traj_start_time = time_sim
    traj_duration = duration
    traj_start_pos = robot.framePlacement(q, frame_ee, True).translation.copy()
    traj_end_pos = target_pos.copy()
    print(f"Starting trajectory to {target_pos}")

def trajectory_complete():
    """Check if current trajectory is complete."""
    return (time_sim - traj_start_time) >= traj_duration

def get_current_object():
    """Get current object being manipulated."""
    if current_object_idx < len(object_names):
        name = object_names[current_object_idx]
        return name, conf.objects[name]
    return None, None

def start_task():
    """Start the pick and place task."""
    global current_state, current_object_idx, grasping
    print("\n" + "="*50)
    print("Starting Pick and Place Task")
    print("Objects to move:", object_names)
    print("="*50 + "\n")
    current_state = State.MOVE_HOME
    current_object_idx = 0
    grasping = False

def stop_task():
    """Stop the current task."""
    global current_state
    print("Task stopped!")
    current_state = State.IDLE

def reset():
    """Reset to initial state."""
    global current_state, current_object_idx, q, qd, grasping
    current_state = State.IDLE
    current_object_idx = 0
    q = conf.q0.copy()
    qd = conf.qd0.copy()
    grasping = False
    print("Reset complete")

def state_machine():
    """Execute state machine logic."""
    global current_state, current_object_idx, grasping
    
    obj_name, obj = get_current_object()
    
    if current_state == State.IDLE:
        pass
        
    elif current_state == State.MOVE_HOME:
        if trajectory_complete():
            if obj is None:
                current_state = State.DONE
                print("All objects placed! Task complete.")
            else:
                # Move to approach position above object
                approach_pos = obj['initial_pos'].copy()
                approach_pos[2] += conf.approach_height
                start_trajectory(approach_pos, 2.0)
                current_state = State.MOVE_TO_APPROACH
                print(f"Moving to approach {obj_name}")
        else:
            start_trajectory(p0, 2.0)
            
    elif current_state == State.MOVE_TO_APPROACH:
        if trajectory_complete():
            # Move down to grasp
            grasp_pos = obj['initial_pos'].copy()
            grasp_pos[2] += conf.grasp_height
            start_trajectory(grasp_pos, 1.5)
            current_state = State.MOVE_TO_GRASP
            print(f"Moving to grasp {obj_name}")
            
    elif current_state == State.MOVE_TO_GRASP:
        if trajectory_complete():
            # Grasp object
            grasping = True
            current_state = State.GRASP
            print(f"Grasping {obj_name}")
            
    elif current_state == State.GRASP:
        # Wait a bit for grasp
        if trajectory_complete():
            # Lift object
            lift_pos = obj['initial_pos'].copy()
            lift_pos[2] += conf.lift_height
            start_trajectory(lift_pos, 1.5)
            current_state = State.LIFT
            print(f"Lifting {obj_name}")
            
    elif current_state == State.LIFT:
        if trajectory_complete():
            # Move to place approach
            place_approach = obj['target_pos'].copy()
            place_approach[2] += conf.approach_height
            start_trajectory(place_approach, 2.5)
            current_state = State.MOVE_TO_PLACE_APPROACH
            print(f"Moving to place position for {obj_name}")
            
    elif current_state == State.MOVE_TO_PLACE_APPROACH:
        if trajectory_complete():
            # Move down to place
            place_pos = obj['target_pos'].copy()
            place_pos[2] += conf.place_height
            start_trajectory(place_pos, 1.5)
            current_state = State.MOVE_TO_PLACE
            print(f"Lowering {obj_name}")
            
    elif current_state == State.MOVE_TO_PLACE:
        if trajectory_complete():
            # Release object
            grasping = False
            current_state = State.RELEASE
            print(f"Releasing {obj_name}")
            
    elif current_state == State.RELEASE:
        if trajectory_complete():
            # Lift after place
            lift_pos = obj['target_pos'].copy()
            lift_pos[2] += conf.lift_height
            start_trajectory(lift_pos, 1.5)
            current_state = State.LIFT_AFTER_PLACE
            print(f"Lifting after placing {obj_name}")
            
    elif current_state == State.LIFT_AFTER_PLACE:
        if trajectory_complete():
            print(f"Successfully placed {obj_name}!")
            current_object_idx += 1
            # Move to home before next object
            start_trajectory(p0, 2.0)
            current_state = State.MOVE_HOME
            
    elif current_state == State.DONE:
        pass

# Print instructions
print("\n" + "="*60)
print("Pick and Place Controller Ready")
print("="*60)
print("Commands:")
print("  start_task()  - Start pick and place sequence")
print("  stop_task()   - Stop current task")
print("  reset()       - Reset to initial state")
print("="*60 + "\n")

# MAIN CONTROL LOOP
print("Starting control loop...")
while True:
    
    # Execute state machine
    state_machine()
    
    # Generate trajectory reference
    if current_state != State.IDLE and current_state != State.DONE:
        p_des = interpolate_position(time_sim, traj_start_time, traj_duration, 
                                     traj_start_pos, traj_end_pos)
    
    # Check experiment duration
    if time_sim >= conf.exp_duration:
        print("Experiment duration reached")
        break
    
    # Compute robot dynamics
    robot.computeAllTerms(q, qd)
    M = robot.mass(q, False)
    h = robot.nle(q, qd, False)
    g = robot.gravity(q)
    
    # Compute Jacobian
    J6 = robot.frameJacobian(q, frame_ee, False, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J = J6[:3, :]
    
    # Current end-effector position and velocity
    p = robot.framePlacement(q, frame_ee).translation
    pd = J.dot(qd)
    
    # Inverse dynamics matrices
    M_inv = np.linalg.inv(M)
    JTpinv = np.linalg.inv(J.dot(J.T) + 0.001*np.eye(3)).dot(J)
    lambda_ = np.linalg.inv(J.dot(M_inv).dot(J.T) + 0.001*np.eye(3))
    
    # Null space projector
    N = np.eye(6) - J.T.dot(JTpinv)
    
    # Cartesian PD control with gravity compensation
    F_des = conf.kp[:3].reshape(3,1) * (p_des - p).reshape(3,1) - conf.kd[:3].reshape(3,1) * pd.reshape(3,1)
    F_des = F_des.flatten()
    
    # Null space postural task
    tau_null = N.dot(50*(conf.q0 - q) - 10*qd)
    
    # Total torque
    tau = J.T.dot(F_des) + g + tau_null
    
    # Forward dynamics simulation
    qdd = M_inv.dot(tau - h)
    
    # Euler integration
    qd = qd + qdd * conf.dt
    q = q + qd * conf.dt + 0.5 * conf.dt * conf.dt * qdd
    
    # Log data
    if log_counter < buffer_size:
        time_log[log_counter] = time_sim
        p_log[:, log_counter] = p
        p_des_log[:, log_counter] = p_des
        state_log[log_counter] = current_state
        log_counter += 1
    
    # Update time
    time_sim += conf.dt
    
    # Visualization
    ros_pub.add_marker(p, color="green" if grasping else "red")
    ros_pub.add_marker(p_des, color="blue")
    ros_pub.publish(robot, q, qd, tau)
    
    # Real-time factor
    tm.sleep(conf.dt * 1.0)
    
    # Check for shutdown
    if ros_pub.isShuttingDown():
        print("Shutting down")
        break

ros_pub.deregister_node()
print("Control loop ended")
