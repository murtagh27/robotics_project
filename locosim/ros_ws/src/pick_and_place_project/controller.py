#!/usr/bin/env python3
"""
Pick and Place Automation (PAPA) - Main Controller
Integrates perception, motion planning, and task scheduling

Usage:
    python3 -i controller.py

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
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, 'config'))  # Add config directory

# Import locosim base controller
sys.path.insert(0, os.path.join(os.environ['LOCOSIM_DIR'], 'robot_control/base_controllers'))
from base_controller_fixed import BaseControllerFixed  # Use BaseControllerFixed instead


from perception_module import PerceptionModule
from motion_planner import MotionPlanner
from task_scheduler import TaskScheduler
from object_spawner import ObjectSpawner
try:
    from world_model import WorldModel
except ImportError:
    WorldModel = None


class PapaController(BaseControllerFixed):
    """
    PAPA (Pick and Place Automation) - Main Controller

    Orchestrates perception, motion planning, and task scheduling for
    autonomous pick and place operations using a UR5 robot with soft gripper.

    Inherits from BaseControllerFixed for low-level robot control capabilities.

    Attributes:
        config: Configuration module with robot and task parameters
        perception: PerceptionModule for object detection
        world_model: WorldModel for maintaining brick state (optional)
        motion_planner: MotionPlanner for trajectory generation
        task_scheduler: TaskScheduler for high-level task coordination
        task_running: Flag indicating if task is currently executing
        use_world_model: Flag to enable/disable world model integration
    """

    def __init__(self, use_world_model=False):
        # Ensure ROS is initialized before anything else
        if not rospy.core.is_initialized():
            rospy.init_node('papa_controller', anonymous=False)
        
        # Initialize base controller with UR5
        super().__init__('ur5')

        rospy.loginfo("Initializing PAPA Controller...")

        # Load configuration
        import config

        self.config = config
        
        # Initialize modules
        self.perception = PerceptionModule(self.config)
        
        # Optional world model
        self.use_world_model = use_world_model and WorldModel is not None
        if self.use_world_model:
            self.world_model = WorldModel()
            rospy.loginfo("World Model integration enabled")
        else:
            self.world_model = None
            if use_world_model and WorldModel is None:
                rospy.logwarn("World Model requested but not available")
        
        self.motion_planner = MotionPlanner(self, self.config)
        self.task_scheduler = TaskScheduler(self.perception, self.motion_planner, self, self.config)

        # Initialize object spawner
        self.object_spawner = ObjectSpawner(
            table_height=self.config.table_height,
            spawn_area_center=self.config.spawn_area_center,
            spawn_area_size=self.config.spawn_area_size,
        )

        self.task_running = False
        self.model_states_sub = None  # Will be created after Gazebo starts

        # RViz visualization publishers
        self.detected_objects_pub = rospy.Publisher('/papa/detected_objects', MarkerArray, queue_size=1)
        self.target_positions_pub = rospy.Publisher('/papa/target_positions', MarkerArray, queue_size=1)
        
        # TF broadcaster for visualization
        import tf
        import tf2_ros
        from geometry_msgs.msg import TransformStamped
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_broadcaster_old = tf.TransformBroadcaster()  # For world→base_link transform
        rospy.loginfo("TF broadcaster initialized for visualization")

        rospy.loginfo("Pick and Place Controller initialized!")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Commands:")
        rospy.loginfo("  p.start_task()  - Start pick and place task")
        rospy.loginfo("  p.start_test()  - Run simple test (move down + gripper open/close)")
        rospy.loginfo("  p.check_controllers() - Check if Gazebo controllers are running")
        rospy.loginfo("  p.tune_impedance_gains(p, d, i) - Tune PID gains (default: 800, 50, 0.1)")
        rospy.loginfo("  p.stop()        - Emergency stop")
        rospy.loginfo("  p.reset()       - Reset controller")
        rospy.loginfo("  p.go_home()     - Move to home position")
        rospy.loginfo("  p.move_to_position(x, y, z)  - Test: move end-effector to position")
        rospy.loginfo("  p.test_targets()  - Interactive target testing mode")
        rospy.loginfo("  p.check_fk_vs_tf()  - Diagnostic: compare FK with TF")
        rospy.loginfo("  p.visualize_objects_and_targets(objects, targets)  - Show markers in RViz")
        if self.use_world_model:
            rospy.loginfo("  p.world_model.print_status()  - Show world model status")
        rospy.loginfo("=" * 60)

    def startSimulator(self):
        """Start the Gazebo simulation with PAPA world"""
        # Import config values
        import config as conf

        # Suppress Gazebo model database warnings (harmless internet connection attempts)
        os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''

        additional_args = [
            f'gripper:={str(conf.gripper).lower()}',
            f'soft_gripper:={str(conf.soft_gripper).lower()}',
            f'robotiq_gripper:={str(conf.robotiq_gripper).lower()}',
        ]
        rospy.loginfo(f"Starting Gazebo with world: {conf.world_name}")
        super().startSimulator(world_name=conf.world_name, additional_args=additional_args)

    def initVars(self):
        """
        Initialize robot state variables and ROS communication.

        Sets up joint state arrays (8 joints: 6 arm + 2 gripper), creates
        publishers for joint commands, and subscribes to ground truth if enabled.
        Called after Gazebo simulation has started.
        """
        import config as conf

        # Initialize joint state arrays for 8 joints (6 arm + 2 gripper)
        self.q_des = np.concatenate([conf.q0, np.zeros(2)])  # desired positions

        # Initialize state variables that base controller expects
        self.qd_des = np.zeros(8)  # desired velocities
        self.q = np.concatenate([conf.q0, np.zeros(2)])  # current positions - start from home, not zeros!
        self.qd = np.zeros(8)  # current velocities
        self.tau_ffwd = np.zeros(8)  # torques

        # Track gripper state
        self.gripper_pos = 0.0

        # Create publisher for desired joint states (Gazebo listens on /command)
        from sensor_msgs.msg import JointState

        # CRITICAL: Use larger queue_size to avoid dropping commands
        # The impedance controller runs at 1000Hz, so we need to publish fast
        self.pub_des_jstate = rospy.Publisher("/command", JointState, queue_size=10)
        rospy.sleep(0.5)  # Wait for publisher to connect

        # Subscribe to actual joint states from Gazebo to update self.q
        rospy.loginfo("Creating joint_states subscriber...")
        self.joint_states_sub = rospy.Subscriber(
            '/ur5/joint_states', JointState, self.joint_states_callback, queue_size=1
        )
        rospy.sleep(0.5)  # Wait for subscriber to connect
        
        # Verify we're receiving joint states
        rospy.loginfo("Waiting for joint state feedback...")
        try:
            msg = rospy.wait_for_message('/ur5/joint_states', JointState, timeout=3.0)
            rospy.loginfo(f"✓ Receiving joint states: {len(msg.position)} joints")
        except rospy.ROSException:
            rospy.logerr("❌ NOT receiving joint states from /ur5/joint_states!")
            rospy.logerr("   Robot position feedback unavailable")
            rospy.logerr("   Motion verification will not work")

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

    def joint_states_callback(self, msg):
        """
        Update current joint state from robot feedback.
        
        CRITICAL: Gazebo sends joints in ALPHABETICAL order, but we need KINEMATIC CHAIN order!
        
        Args:
            msg (sensor_msgs/JointState): Joint state message from Gazebo
        """
        # Correct kinematic chain order for UR5
        CORRECT_ORDER = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
            'hand_1_joint', 'hand_2_joint'
        ]
        
        # Create mapping from received names to positions
        positions = {}
        velocities = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                positions[name] = msg.position[i]
            if i < len(msg.velocity):
                velocities[name] = msg.velocity[i]
        
        # Reorder to kinematic chain
        self.q = np.zeros(8)
        self.qd = np.zeros(8)
        for i, name in enumerate(CORRECT_ORDER):
            if name in positions:
                self.q[i] = positions[name]
            if name in velocities:
                self.qd[i] = velocities[name]
        
        rospy.logdebug(f"Joint state updated (reordered): {self.q}")

    def model_states_callback(self, msg):
        """
        Process Gazebo model states for ground truth perception.

        Args:
            msg (gazebo_msgs/ModelStates): Gazebo model states message containing
                positions and orientations of all models in simulation.
        """
        rospy.logdebug(f"model_states_callback triggered with {len(msg.name)} models")
        if self.config.use_ground_truth:
            rospy.logdebug(f"Models in callback: {msg.name}")
            self.perception.update_ground_truth(msg)
            rospy.loginfo_once(f"Ground truth callback working! Found {len(msg.name)} models")
            
            # Update world model if enabled
            if self.world_model is not None:
                detections = []
                for obj in self.perception.ground_truth_objects:
                    detections.append({
                        'name': obj['name'],
                        'class': obj['class'],
                        'position': obj['position'],
                        'timestamp': rospy.Time.now()
                    })
                self.world_model.update_from_perception(detections)

    def test_model_states(self):
        """
        Diagnostic tool to verify Gazebo model_states communication.

        Attempts to receive a single message from /gazebo/model_states topic
        and update perception module. Useful for debugging perception issues.

        Raises:
            rospy.ROSException: If unable to receive message within timeout.
        """
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

    def spawn_objects(self, num_objects=None, brick_types=None):
        """
        Spawn objects using the object spawner.

        Args:
            num_objects (int, optional): Number of objects to spawn. Uses config default if None.
            brick_types (list, optional): List of specific brick types to spawn. Uses config default if None.
                                          If provided, spawns one of each type from this list.
        
        Returns:
            list: List of spawned object info dicts
        """
        rospy.loginfo("=" * 60)
        rospy.loginfo("SPAWNING OBJECTS")
        rospy.loginfo("=" * 60)

        # Get brick types to spawn
        if brick_types is None:
            brick_types = self.config.allowed_brick_types
        
        # Get number of objects to spawn
        if num_objects is None:
            num_objects = getattr(self.config, 'num_objects_to_spawn', 3)

        if brick_types is not None and len(brick_types) > 0:
            # Spawn one of each specified type
            rospy.loginfo(f"Spawning one of each specified type: {brick_types}")
            spawned_objects = self.object_spawner.spawn_one_of_each(
                allowed_types=brick_types,
            )
        else:
            # Spawn random objects
            rospy.loginfo(f"Spawning {num_objects} random objects")
            spawned_objects = self.object_spawner.spawn_random_objects(
                num_objects=num_objects,
            )

        rospy.loginfo(f"Successfully spawned {len(spawned_objects)} objects")
        for obj in spawned_objects:
            rospy.loginfo(f"  - {obj['name']}: {obj['class']} at {obj['position']}")

        rospy.loginfo("=" * 60)

        # Wait a moment for objects to settle in Gazebo
        rospy.sleep(2.0)

        return spawned_objects

    def start_task(self):
        """
        Execute the complete pick and place task sequence.
        
        Automatically spawns 3 objects if none exist, then sorts them by type
        to the back of the table.

        Coordinates perception, planning, and execution phases. Handles errors
        gracefully and ensures task_running flag is reset on completion.
        """
        if self.task_running:
            rospy.logwarn("Task is already running!")
            return

        rospy.loginfo("=" * 60)
        rospy.loginfo("Starting pick and place task (sort by type)...")
        rospy.loginfo("=" * 60)
        
        # Move to home position first
        rospy.loginfo("Moving to home position before starting task...")
        self.go_home()
        rospy.sleep(1.0)
        
        # Check if objects exist - spawn them if not
        detected = self.perception.get_detected_objects()
        if len(detected) == 0:
            rospy.loginfo("No objects detected - spawning 3 objects for sorting task...")
            self.spawn_objects()
            # Wait longer for perception to detect newly spawned objects
            rospy.loginfo("Waiting for perception to detect spawned objects...")
            rospy.sleep(3.0)

        rospy.loginfo(f"Ground truth mode: {self.config.use_ground_truth}")

        self.task_running = True

        try:
            # Check if perception has detected objects
            detected = self.perception.get_detected_objects()
            rospy.loginfo(f"Perception has detected {len(detected)} objects")
            
            if len(detected) == 0:
                rospy.logerr("Still no objects detected after spawning!")
                rospy.logerr("Checking ground truth objects directly...")
                # Try to access ground truth objects directly
                if hasattr(self.perception, 'ground_truth_objects'):
                    rospy.logerr(f"Ground truth has {len(self.perception.ground_truth_objects)} objects")
                    if len(self.perception.ground_truth_objects) > 0:
                        rospy.logerr("Ground truth sees objects but get_detected_objects() returns 0!")
                        rospy.logerr("This is a perception module issue.")
                rospy.logerr("Make sure /gazebo/model_states topic is publishing and perception is subscribing.")
                self.task_running = False
                return

            success = self.task_scheduler.execute_task_sequence()
            if success:
                rospy.loginfo("=" * 60)
                rospy.loginfo("Task completed successfully! Objects sorted by type.")
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

    def start_test(self):
        """
        Execute a simple test sequence: downward movement + gripper open/close.
        
        This is useful for testing basic robot functionality without full perception
        and planning. Tests joint control and gripper commands.
        """
        if self.task_running:
            rospy.logwarn("Task is already running!")
            return

        rospy.loginfo("=" * 60)
        rospy.loginfo("Starting test sequence...")
        rospy.loginfo("=" * 60)
        
        self.task_running = True
        
        try:
            # Step 1: Move to home position first
            rospy.loginfo("Step 1: Moving to home position...")
            self.go_home()
            rospy.sleep(1.0)
            
            # Step 2: Move downward (lower the arm smoothly)
            rospy.loginfo("Step 2: Moving downward...")
            # Get home joints and modify to create a lowered position
            home_joints = np.array(self.config.home_joint_config)
            down_joints = home_joints.copy()
            down_joints[1] += 0.4  # Shoulder: move down (increase angle)
            down_joints[2] -= 0.3  # Elbow: adjust to keep orientation
            
            # Use motion planner for smooth movement (SLOW to avoid velocity limits)
            self.motion_planner.move_to_joints(down_joints, self, duration=10.0)
            rospy.sleep(0.5)
            
            # Step 3: Open gripper
            rospy.loginfo("Step 3: Opening gripper...")
            self.send_gripper_command(self.config.gripper_open_pos)
            rospy.sleep(1.5)
            
            # Step 4: Close gripper
            rospy.loginfo("Step 4: Closing gripper...")
            self.send_gripper_command(self.config.gripper_close_pos)
            rospy.sleep(1.5)
            
            # Step 5: Open gripper again
            rospy.loginfo("Step 5: Opening gripper again...")
            self.send_gripper_command(self.config.gripper_open_pos)
            rospy.sleep(1.5)
            
            # Step 6: Return to home
            rospy.loginfo("Step 6: Returning to home position...")
            self.go_home()
            rospy.sleep(1.0)
            
            rospy.loginfo("=" * 60)
            rospy.loginfo("Test sequence completed successfully!")
            rospy.loginfo("=" * 60)
            
        except Exception as e:
            rospy.logerr("=" * 60)
            rospy.logerr(f"Error during test sequence: {e}")
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

    def check_controllers(self):
        """Check if Gazebo controllers are running"""
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("CHECKING CONTROLLER STATUS")
        rospy.loginfo("="*60)
        
        # Check published topics
        topics = rospy.get_published_topics()
        
        rospy.loginfo("Checking key topics:")
        key_topics = ['/command', '/ur5/joint_states', '/gazebo/model_states', '/gripper_controller/command']
        for topic_name in key_topics:
            found = any(topic_name in t[0] for t in topics)
            status = "✓ FOUND" if found else "✗ MISSING"
            rospy.loginfo(f"  {topic_name}: {status}")
            if found:
                matching = [t for t in topics if topic_name in t[0]]
                for t in matching:
                    rospy.loginfo(f"    -> {t[0]} ({t[1]})")
        
        # Check /command subscribers
        num_subs = self.pub_des_jstate.get_num_connections()
        rospy.loginfo(f"\n/command topic has {num_subs} subscriber(s)")
        if num_subs == 0:
            rospy.logerr("  ✗ NO SUBSCRIBERS - Commands will not reach Gazebo!")
            rospy.logerr("  Check if Gazebo position controllers are loaded")
        else:
            rospy.loginfo("  ✓ Subscribers connected")
        
        rospy.loginfo("="*60 + "\n")
    
    def tune_impedance_gains(self, p_gain=3000.0, d_gain=100.0, i_gain=10.0):
        """
        Tune impedance controller PID gains for better position tracking.
        
        UR5 requires HIGH gains due to:
        - Heavy arm links (especially shoulder and elbow)
        - Gravity compensation via PID (no feedforward torque)
        - Impedance controller design
        
        Args:
            p_gain: Proportional gain (default 3000, must be high for UR5)
            d_gain: Derivative gain (default 100, damping)
            i_gain: Integral gain (default 10, eliminate steady-state error from gravity)
        """
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo(f"TUNING IMPEDANCE CONTROLLER GAINS")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Setting PID gains: P={p_gain}, I={i_gain}, D={d_gain}")
        
        # Joint names with different gain requirements
        # Shoulder and elbow need VERY high gains (heavy, fight gravity)
        heavy_joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint']
        # Wrist joints can use lower gains (lighter)
        light_joints = ['wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        gripper_joints = ['hand_1_joint', 'hand_2_joint']
        
        # Set VERY high gains for heavy joints (shoulder + elbow)
        for joint in heavy_joints:
            param_base = f"/ur5/ros_impedance_controller/gains/{joint}"
            rospy.set_param(f"{param_base}/p", p_gain)
            rospy.set_param(f"{param_base}/i", i_gain)
            rospy.set_param(f"{param_base}/d", d_gain)
            rospy.loginfo(f"  {joint}: P={p_gain}, I={i_gain}, D={d_gain} (heavy)")
        
        # Set moderate gains for wrist joints
        wrist_p = p_gain * 0.5  # Lighter joints need less gain
        wrist_d = d_gain * 0.5
        wrist_i = i_gain * 0.5
        for joint in light_joints:
            param_base = f"/ur5/ros_impedance_controller/gains/{joint}"
            rospy.set_param(f"{param_base}/p", wrist_p)
            rospy.set_param(f"{param_base}/i", wrist_i)
            rospy.set_param(f"{param_base}/d", wrist_d)
            rospy.loginfo(f"  {joint}: P={wrist_p}, I={wrist_i}, D={wrist_d} (light)")
        
        # Set lower gains for gripper (more compliant)
        for joint in gripper_joints:
            param_base = f"/ur5/ros_impedance_controller/gains/{joint}"
            rospy.set_param(f"{param_base}/p", 50.0)  # Much lower for compliant gripper
            rospy.set_param(f"{param_base}/i", 0.01)
            rospy.set_param(f"{param_base}/d", 5.0)
            rospy.loginfo(f"  {joint}: P=50.0, I=0.01, D=5.0 (compliant)")
        
        rospy.loginfo("\n✓ Impedance controller gains updated")
        rospy.loginfo("  Higher P gain = stiffer, better tracking")
        rospy.loginfo("  Higher D gain = more damping, less oscillation")
        rospy.loginfo("  Small I gain = eliminates steady-state error")
        rospy.loginfo("="*60 + "\n")
        
        rospy.sleep(0.5)  # Give params time to propagate
    
    def go_home(self):
        """Move robot to home position with proper tracking"""
        rospy.loginfo("Moving to home position...")
        rospy.loginfo("Home joints: " + str(np.degrees(self.config.home_joint_config)))
        # Use direct movement for speed
        self.motion_planner.move_to_joints_direct(np.array(self.config.home_joint_config), self, 
                                           duration=3.0)
        # Verify we reached home
        q_current, _ = self.get_current_joint_state()
        error = np.linalg.norm(q_current[:6] - np.array(self.config.home_joint_config))
        if error > 0.1:
            rospy.logwarn(f"Home position not fully reached. Error: {np.degrees(error):.1f}°")
        else:
            rospy.loginfo("✓ Reached home position")
    
    def move_to_position(self, x, y, z, duration=2.0):
        """
        Test function: Move end-effector to a specific XYZ position.
        
        Args:
            x, y, z: Target position in world frame
            duration: Movement duration in seconds (default 2s)
            
        Returns:
            True if successful, False otherwise
        """
        rospy.loginfo(f"Moving to position [{x:.3f}, {y:.3f}, {z:.3f}]")
        
        target_world = [x, y, z]
        
        # Compute IK
        target_joints = self.motion_planner.simple_ik(target_world, gripper_down=True)
        
        if target_joints is None:
            rospy.logerr("Failed to compute IK for target position")
            return False
        
        # Execute motion DIRECTLY without waypoints for speed
        success = self.motion_planner.move_to_joints_direct(
            target_joints, self, duration=duration, target_world_pos=target_world
        )
        rospy.loginfo("Movement complete")
        return success

    def visualize_objects_and_targets(self, detected_objects=None, target_positions=None):
        """
        Publish detected objects and target positions as RViz markers.
        
        Args:
            detected_objects: List of detected object positions [(x, y, z), ...]
            target_positions: List of target positions [(x, y, z), ...]
        """
        # Publish detected objects (green spheres)
        if detected_objects is not None:
            marker_array = MarkerArray()
            for i, pos in enumerate(detected_objects):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "detected_objects"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.pose.position.z = pos[2]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                marker_array.markers.append(marker)
            self.detected_objects_pub.publish(marker_array)
        
        # Publish target positions (red cubes)
        if target_positions is not None:
            marker_array = MarkerArray()
            for i, pos in enumerate(target_positions):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "target_positions"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.pose.position.z = pos[2]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.04
                marker.scale.y = 0.04
                marker.scale.z = 0.04
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                marker_array.markers.append(marker)
            self.target_positions_pub.publish(marker_array)

    def test_targets(self):
        """
        Interactive function to test robot movement to various positions.
        Prompts user for target coordinates and moves robot there.
        """
        rospy.loginfo("=== Test Target Movement ===")
        rospy.loginfo("Enter target position (or 'q' to quit)")
        rospy.loginfo("Format: x y z (in meters)")
        rospy.loginfo("Example: 0.5 0.0 0.8")
        rospy.loginfo("")
        
        while True:
            try:
                user_input = input("Target (x y z): ").strip()
                if user_input.lower() == 'q':
                    break
                
                coords = [float(x) for x in user_input.split()]
                if len(coords) != 3:
                    rospy.logwarn("Please provide exactly 3 coordinates")
                    continue
                
                # Visualize the target
                self.visualize_objects_and_targets(target_positions=[coords])
                
                # Move to target
                success = self.move_to_position(coords[0], coords[1], coords[2])
                if success:
                    rospy.loginfo("✓ Reached target")
                else:
                    rospy.logerr("✗ Failed to reach target")
                    
            except ValueError:
                rospy.logwarn("Invalid input. Please enter three numbers.")
            except KeyboardInterrupt:
                break
        
        rospy.loginfo("Test mode exited")

    def send_des_jstate(self, q_des, qd_des, tau_ffwd):
        """
        Send desired joint states to Gazebo via /command topic.

        Overrides base controller method to publish directly to Gazebo's
        position controller instead of using torque control.

        Args:
            q_des (np.ndarray): Desired joint positions (8 elements)
            qd_des (np.ndarray): Desired joint velocities (8 elements)
            tau_ffwd (np.ndarray): Feedforward torques (8 elements)
        """
        # Update internal state
        self.q_des = q_des.copy()
        self.qd_des = qd_des.copy()
        self.tau_ffwd = tau_ffwd.copy()

        # Check if publisher has subscribers
        if not hasattr(self, '_connection_warned'):
            self._connection_warned = False
            
        if self.pub_des_jstate.get_num_connections() == 0:
            if not self._connection_warned:
                rospy.logerr("❌ /command topic has NO subscribers!")
                rospy.logerr("   Gazebo controllers may not be running")
                rospy.logerr("   Commands will not reach the robot")
                self._connection_warned = True
        elif self._connection_warned:
            rospy.loginfo("✓ /command topic now has subscribers")
            self._connection_warned = False

        # Publish JointState to /command (where Gazebo listens)
        from sensor_msgs.msg import JointState

        msg = JointState()
        # CRITICAL: Must include joint names so Gazebo knows which value goes to which joint!
        # Commands are in kinematic chain order, same as how we store them
        msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
                    'hand_1_joint', 'hand_2_joint']
        msg.position = q_des.tolist()
        msg.velocity = qd_des.tolist()
        msg.effort = tau_ffwd.tolist()
        self.pub_des_jstate.publish(msg)
        
        rospy.logdebug(f"Sent command: pos={q_des[:3]}, subscribers={self.pub_des_jstate.get_num_connections()}")

    def send_joint_command(self, joints, velocities=None):
        """
        Send joint position commands to robot.

        Primary interface for motion planner to command robot motion.
        Automatically pads 6-joint commands to 8 joints by adding gripper state.

        Args:
            joints (np.ndarray): Target joint positions (6 or 8 elements)
            velocities (np.ndarray, optional): Target joint velocities. Defaults to zero.
            
        Note:
            The ros_impedance_controller computes: effort = feedforward_effort + PID(error)
            For pure position control, we send zero feedforward effort and let PID handle tracking.
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

        # Send commands with zero feedforward effort (pure PID control)
        # The C++ ros_impedance_controller will handle all the control
        zero_effort = np.zeros(8)
        self.send_des_jstate(self.q_des, self.qd_des, zero_effort)

    def send_gripper_command(self, position):
        """
        Command gripper position.

        Args:
            position (float): Desired gripper joint angle in radians.
                For soft gripper: 1.0 = fully open, -1.0 = fully closed
                Each finger gets the same joint angle.

        Note:
            The gripper is controlled via the main /command topic (same as arm).
            Gripper uses joints 6 and 7 (hand_1_joint, hand_2_joint).
        """
        rospy.loginfo(f"\n{'='*50}")
        rospy.loginfo(f"GRIPPER COMMAND: {position:.2f} rad ({'OPEN' if position > 0 else 'CLOSED'})")
        rospy.loginfo(f"{'='*50}")
        
        # Update gripper position tracking
        old_gripper_pos = self.gripper_pos
        self.gripper_pos = position  # Joint angle (not width)
        
        rospy.loginfo(f"Gripper position: {old_gripper_pos:.4f} -> {self.gripper_pos:.4f} rad")
        
        # Update q_des with new gripper position (keep arm position same)
        self.q_des[6] = self.gripper_pos
        self.q_des[7] = self.gripper_pos
        
        # Send command via main /command topic (not separate gripper topic!)
        # This is controlled by ros_impedance_controller along with arm
        rospy.loginfo(f"Sending gripper command via /command topic...")
        self.send_des_jstate(self.q_des, np.zeros(8), np.zeros(8))
        
        # Wait for gripper to move
        rospy.sleep(1.0)
        rospy.loginfo(f"Gripper command complete\n{'='*50}\n")

    def get_current_joint_state(self):
        """
        Retrieve current robot joint state.

        Returns:
            tuple: (positions, velocities) where each is an 8-element np.ndarray
        """
        return self.q.copy(), self.qd.copy()

    def check_fk_vs_tf(self):
        """
        Diagnostic: Show current EE position and orientation from TF (ground truth from Gazebo).
        """
        import tf2_ros
        from tf.transformations import quaternion_matrix
        
        rospy.loginfo("\n" + "="*70)
        rospy.loginfo("EE POSITION & ORIENTATION DIAGNOSTIC (TF = Ground Truth)")
        rospy.loginfo("="*70)
        
        # Get current joint state
        q_current = self.q[:6].copy()
        rospy.loginfo(f"\nCurrent joints (rad): {np.array2string(q_current, precision=4)}")
        rospy.loginfo(f"Current joints (deg): {np.array2string(np.degrees(q_current), precision=2)}")
        
        # Get TF tool0 position (ground truth)
        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        rospy.sleep(0.5)  # Wait for TF data
        
        try:
            # Get tool0 relative to base_link
            trans = tf_buffer.lookup_transform('base_link', 'tool0', rospy.Time(0), rospy.Duration(1.0))
            tf_position = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            rospy.loginfo(f"\nTF tool0 (base_link frame): {np.array2string(tf_position, precision=5)}")
            
            # Also show in world frame
            world_pos = tf_position + np.array([0.5, 0.35, 1.75])
            rospy.loginfo(f"TF tool0 (world frame): {np.array2string(world_pos, precision=5)}")
            
            # Get orientation from TF
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, 
                    trans.transform.rotation.z, trans.transform.rotation.w]
            R_tf = quaternion_matrix(quat)[:3, :3]
            tf_z_axis = R_tf[:, 2]  # Tool Z-axis direction
            rospy.loginfo(f"\nTF tool0 Z-axis (gripper direction): [{tf_z_axis[0]:.3f}, {tf_z_axis[1]:.3f}, {tf_z_axis[2]:.3f}]")
            if tf_z_axis[2] < -0.9:
                rospy.loginfo("  ✓ Gripper is pointing DOWN (Z-axis ≈ [0,0,-1])")
            elif tf_z_axis[2] > 0.9:
                rospy.loginfo("  ⚠ Gripper is pointing UP (Z-axis ≈ [0,0,+1])")
            else:
                rospy.logwarn(f"  ⚠ Gripper is NOT pointing down! Z-component: {tf_z_axis[2]:.3f}")
            
            # Compare with URDF FK
            T_urdf = self.motion_planner.urdf_fk_full(q_current)
            urdf_pos = T_urdf[:3, 3]
            urdf_z_axis = T_urdf[:3, 2]
            rospy.loginfo(f"\nURDF FK (base_link frame): {np.array2string(urdf_pos, precision=5)}")
            rospy.loginfo(f"URDF FK Z-axis: [{urdf_z_axis[0]:.3f}, {urdf_z_axis[1]:.3f}, {urdf_z_axis[2]:.3f}]")
            
            fk_error = np.linalg.norm(urdf_pos - tf_position) * 1000
            rospy.loginfo(f"\nFK vs TF position error: {fk_error:.2f} mm")
            z_axis_error = np.linalg.norm(urdf_z_axis - tf_z_axis)
            rospy.loginfo(f"FK vs TF Z-axis error: {z_axis_error:.4f}")
            
        except Exception as e:
            rospy.logerr(f"TF base_link->tool0 lookup failed: {e}")
            try:
                trans = tf_buffer.lookup_transform('world', 'tool0', rospy.Time(0), rospy.Duration(1.0))
                tf_world = np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z
                ])
                rospy.loginfo(f"\nTF tool0 (world frame): {np.array2string(tf_world, precision=5)}")
                
                # Convert to base_link
                base_link_pos = tf_world - np.array([0.5, 0.35, 1.75])
                rospy.loginfo(f"TF tool0 (base_link frame): {np.array2string(base_link_pos, precision=5)}")
            except Exception as e2:
                rospy.logerr(f"TF world->tool0 lookup also failed: {e2}")
        
        # Also check joint states directly from /ur5/joint_states topic
        rospy.loginfo("\nChecking raw joint states from topic...")
        try:
            from sensor_msgs.msg import JointState
            msg = rospy.wait_for_message('/ur5/joint_states', JointState, timeout=2.0)
            rospy.loginfo(f"Raw joint names: {msg.name}")
            rospy.loginfo(f"Raw joint positions: {[f'{p:.4f}' for p in msg.position]}")
            rospy.loginfo(f"Our q (reordered): {[f'{p:.4f}' for p in self.q]}")
        except Exception as e:
            rospy.logerr(f"Failed to get raw joint states: {e}")
        
        rospy.loginfo("="*70 + "\n")


def main():
    """
    Initialize and start PAPA controller.

    Creates controller instance, starts Gazebo simulation, and initializes
    all subsystems. Returns controller object for interactive use.

    Returns:
        PapaController: Initialized controller instance

    Raises:
        rospy.ROSInterruptException: If ROS shutdown is requested
    """
    # Create controller (it will initialize ROS node internally)
    p = PapaController()

    try:
        # Start Gazebo simulator
        rospy.loginfo("Starting Gazebo simulator...")
        p.startSimulator()
        rospy.sleep(5.0)  # Wait for Gazebo to fully start

        # Initialize controller variables
        rospy.loginfo("Initializing controller variables...")
        p.initVars()
        rospy.loginfo("Controller initialized successfully")
        
        # Tune impedance controller gains for better position tracking
        rospy.loginfo("Tuning impedance controller PID gains...")
        p.tune_impedance_gains()

        # Spawn random objects if enabled in config
        rospy.sleep(2.0)  # Give Gazebo time to stabilize
        p.spawn_objects()
        
        # Wait a bit more before moving
        rospy.sleep(1.0)
        
        # Move to home position
        rospy.loginfo("Moving to home position...")
        p.go_home()
        rospy.sleep(1.0)  # Wait for motion to complete

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("  PAPA (Pick and Place Automation) Ready!")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Commands:")
        rospy.loginfo("  p.start_task()    - Start automation task")
        rospy.loginfo("  p.start_test()    - Test basic movement")
        rospy.loginfo("  p.stop()          - Emergency stop")
        rospy.loginfo("  p.reset()         - Reset controller")
        rospy.loginfo("  p.go_home()       - Move to home position")
        rospy.loginfo("=" * 60 + "\n")

        # Return controller for interactive use
        return p

    except (rospy.ROSInterruptException, rospy.service.ServiceException):
        rospy.signal_shutdown("killed")
        p.deregister_node()
        raise


if __name__ == '__main__':
    # Run in interactive mode
    p = main()

    rospy.loginfo("\n>>> Python interactive mode active. Type commands here.")
    rospy.loginfo(">>> Example: p.start_test()\n")

    # Note: The script continues and drops into Python interactive mode
    # because it was launched with 'python3 -i'
