"""
World Model - Filtering and state management for detected objects
Bridges perception and task planning by tracking object states and providing filtering
"""

import rospy
import numpy as np
from collections import deque
from enum import Enum
from geometry_msgs.msg import PoseArray, Pose
from std_srvs.srv import Trigger, TriggerResponse
from papa_msgs.srv import ClaimBrick, ClaimBrickResponse


class BrickState(Enum):
    """State machine for brick lifecycle"""
    AVAILABLE = "available"  # Detected and can be claimed
    CLAIMED = "claimed"      # Claimed by task planner but not yet picked
    IN_TRANSIT = "in_transit"  # Currently being manipulated
    PLACED = "placed"        # Successfully placed at target
    LOST = "lost"            # No longer detected (stale)


class TrackedBrick:
    """
    Single brick with state tracking and filtering
    """
    
    def __init__(self, name, brick_class, position, timestamp):
        self.name = name
        self.brick_class = brick_class
        self.state = BrickState.AVAILABLE
        
        # Position filtering - keep history for median/average
        self.position_history = deque(maxlen=10)
        self.position_history.append(position)
        
        # Timestamps
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.last_update = timestamp
        
        # Claiming
        self.claimed_by = None
        self.claim_time = None
        
        # Dimensions (from perception if available)
        self.dimensions = [0.05, 0.05, 0.05]  # Default 5cm cube
        
    def update_position(self, new_pos, timestamp):
        """Update with new detection"""
        self.position_history.append(new_pos)
        self.last_seen = timestamp
        self.last_update = timestamp
        
    def get_filtered_position(self):
        """Get median-filtered position from history"""
        if len(self.position_history) == 0:
            return None
        
        # Use median filtering to reject outliers
        positions = np.array(self.position_history)
        return np.median(positions, axis=0)
    
    def get_bounding_box(self):
        """Get axis-aligned bounding box for collision checking"""
        pos = self.get_filtered_position()
        if pos is None:
            return None
            
        # Return [xmin, ymin, zmin, xmax, ymax, zmax]
        half_dims = np.array(self.dimensions) / 2.0
        return [
            pos[0] - half_dims[0],
            pos[1] - half_dims[1],
            pos[2] - half_dims[2],
            pos[0] + half_dims[0],
            pos[1] + half_dims[1],
            pos[2] + half_dims[2]
        ]
    
    def age(self, current_time):
        """Time since last detection"""
        return current_time - self.last_seen
    
    def is_fresh(self, current_time, threshold=5.0):
        """Check if brick was seen recently"""
        return self.age(current_time) < threshold
    
    def claim(self, requester):
        """Claim this brick"""
        if self.state != BrickState.AVAILABLE:
            return False
        self.state = BrickState.CLAIMED
        self.claimed_by = requester
        self.claim_time = rospy.Time.now()
        return True
    
    def release(self):
        """Release claim on this brick"""
        if self.state == BrickState.CLAIMED:
            self.state = BrickState.AVAILABLE
        self.claimed_by = None
        self.claim_time = None


class WorldModel:
    """
    World Model Node - Tracks all objects with state and filtering
    
    Responsibilities:
    - Maintain object registry with states (available, claimed, placed, lost)
    - Filter noisy detections (median filtering over time)
    - Provide claim/release services for task coordination
    - Publish filtered /bricks topic for visualization
    - Handle object matching and tracking
    """
    
    def __init__(self):
        rospy.init_node('world_model', anonymous=False)
        
        # Tracked bricks: name -> TrackedBrick
        self.bricks = {}
        
        # Parameters
        self.freshness_threshold = rospy.get_param('~freshness_threshold', 5.0)  # seconds
        self.position_match_threshold = rospy.get_param('~position_threshold', 0.1)  # meters
        self.publish_rate = rospy.get_param('~publish_rate', 10.0)  # Hz
        
        # ROS Publishers
        self.bricks_pub = rospy.Publisher('/bricks/poses', PoseArray, queue_size=10)
        
        # ROS Services
        self.claim_service = rospy.Service('/bricks/claim', ClaimBrick, self.handle_claim)
        self.release_all_service = rospy.Service('/bricks/release_all', Trigger, self.handle_release_all)
        self.refresh_service = rospy.Service('/bricks/refresh', Trigger, self.handle_refresh)
        
        # Publish timer
        self.publish_timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_bricks)
        
        rospy.loginfo("World Model initialized")
        rospy.loginfo(f"  Freshness threshold: {self.freshness_threshold}s")
        rospy.loginfo(f"  Position matching: {self.position_match_threshold}m")
        rospy.loginfo(f"  Publish rate: {self.publish_rate}Hz")
    
    def update_from_perception(self, detections):
        """
        Update world state from perception module
        
        Args:
            detections: List of dicts with 'name', 'class', 'position', 'timestamp'
        """
        current_time = rospy.Time.now()
        
        # Match detections to existing bricks
        matched = set()
        
        for det in detections:
            name = det['name']
            brick_class = det['class']
            position = det['position']
            timestamp = det.get('timestamp', current_time)
            
            # Try to match by name first
            if name in self.bricks:
                self.bricks[name].update_position(position, timestamp)
                matched.add(name)
            else:
                # Try to match by position (object might have been renamed)
                matched_brick = self._find_brick_by_position(position)
                if matched_brick:
                    matched_brick.update_position(position, timestamp)
                    matched.add(matched_brick.name)
                else:
                    # New brick
                    new_brick = TrackedBrick(name, brick_class, position, timestamp)
                    self.bricks[name] = new_brick
                    matched.add(name)
                    rospy.loginfo(f"New brick tracked: {name} ({brick_class}) at {position}")
        
        # Mark unmatched bricks as stale if not fresh
        for name, brick in self.bricks.items():
            if name not in matched and not brick.is_fresh(current_time, self.freshness_threshold):
                if brick.state == BrickState.AVAILABLE:
                    brick.state = BrickState.LOST
                    rospy.logwarn(f"Brick {name} marked as LOST (not seen for {brick.age(current_time):.1f}s)")
    
    def _find_brick_by_position(self, position, threshold=None):
        """Find existing brick near this position"""
        if threshold is None:
            threshold = self.position_match_threshold
        
        for brick in self.bricks.values():
            filtered_pos = brick.get_filtered_position()
            if filtered_pos is not None:
                dist = np.linalg.norm(filtered_pos - position)
                if dist < threshold:
                    return brick
        return None
    
    def get_available_bricks(self, brick_class=None):
        """
        Get all available (unclaimed) bricks, optionally filtered by class
        
        Args:
            brick_class: Optional class filter (e.g., 'X1-Y1-Z2')
            
        Returns:
            List of dicts with 'name', 'class', 'position', 'dimensions'
        """
        current_time = rospy.Time.now()
        available = []
        
        for brick in self.bricks.values():
            # Check if available and fresh
            if brick.state != BrickState.AVAILABLE:
                continue
            if not brick.is_fresh(current_time, self.freshness_threshold):
                continue
            
            # Class filter
            if brick_class is not None and brick.brick_class != brick_class:
                continue
            
            position = brick.get_filtered_position()
            if position is not None:
                available.append({
                    'name': brick.name,
                    'class': brick.brick_class,
                    'position': position,
                    'dimensions': brick.dimensions
                })
        
        return available
    
    def claim_brick(self, brick_name, requester="task_planner"):
        """
        Claim a brick for manipulation
        
        Args:
            brick_name: Name of brick to claim
            requester: Who is claiming (for tracking)
            
        Returns:
            (success, message)
        """
        if brick_name not in self.bricks:
            return False, f"Brick '{brick_name}' not found"
        
        brick = self.bricks[brick_name]
        
        # Check if fresh
        current_time = rospy.Time.now()
        if not brick.is_fresh(current_time, self.freshness_threshold):
            return False, f"Brick '{brick_name}' is stale (last seen {brick.age(current_time):.1f}s ago)"
        
        # Attempt claim
        if brick.claim(requester):
            rospy.loginfo(f"Brick '{brick_name}' claimed by {requester}")
            return True, f"Claimed {brick_name}"
        else:
            return False, f"Brick '{brick_name}' already claimed by {brick.claimed_by}"
    
    def release_brick(self, brick_name):
        """Release a claimed brick"""
        if brick_name in self.bricks:
            self.bricks[brick_name].release()
            rospy.loginfo(f"Brick '{brick_name}' released")
            return True
        return False
    
    def release_all_bricks(self):
        """Release all claimed bricks"""
        count = 0
        for brick in self.bricks.values():
            if brick.state == BrickState.CLAIMED:
                brick.release()
                count += 1
        rospy.loginfo(f"Released {count} claimed bricks")
        return count
    
    def mark_brick_placed(self, brick_name):
        """Mark brick as successfully placed"""
        if brick_name in self.bricks:
            self.bricks[brick_name].state = BrickState.PLACED
            rospy.loginfo(f"Brick '{brick_name}' marked as PLACED")
            return True
        return False
    
    def get_collision_boxes(self):
        """
        Get bounding boxes for all tracked bricks (for collision checking)
        
        Returns:
            List of bounding boxes [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        current_time = rospy.Time.now()
        boxes = []
        
        for brick in self.bricks.values():
            # Only include fresh bricks that exist in the world
            if brick.state in [BrickState.LOST, BrickState.PLACED]:
                continue
            if not brick.is_fresh(current_time, self.freshness_threshold):
                continue
            
            bbox = brick.get_bounding_box()
            if bbox is not None:
                boxes.append(bbox)
        
        return boxes
    
    def publish_bricks(self, event=None):
        """Publish current brick poses to /bricks/poses topic"""
        current_time = rospy.Time.now()
        
        pose_array = PoseArray()
        pose_array.header.stamp = current_time
        pose_array.header.frame_id = "world"
        
        for brick in self.bricks.values():
            # Only publish fresh, available/claimed bricks
            if not brick.is_fresh(current_time, self.freshness_threshold):
                continue
            if brick.state in [BrickState.LOST, BrickState.PLACED]:
                continue
            
            position = brick.get_filtered_position()
            if position is not None:
                pose = Pose()
                pose.position.x = position[0]
                pose.position.y = position[1]
                pose.position.z = position[2]
                pose.orientation.w = 1.0  # Identity quaternion
                pose_array.poses.append(pose)
        
        self.bricks_pub.publish(pose_array)
    
    # ROS Service handlers
    
    def handle_claim(self, req):
        """Handle /bricks/claim service request"""
        success, message = self.claim_brick(req.brick_name, req.requester)
        
        response = ClaimBrickResponse()
        response.success = success
        response.message = message
        
        if success:
            brick = self.bricks[req.brick_name]
            pos = brick.get_filtered_position()
            response.position = [pos[0], pos[1], pos[2]]
        
        return response
    
    def handle_release_all(self, req):
        """Handle /bricks/release_all service request"""
        count = self.release_all_bricks()
        return TriggerResponse(success=True, message=f"Released {count} bricks")
    
    def handle_refresh(self, req):
        """Handle /bricks/refresh service request - clear stale bricks"""
        current_time = rospy.Time.now()
        removed = []
        
        for name, brick in list(self.bricks.items()):
            if brick.state == BrickState.LOST:
                removed.append(name)
                del self.bricks[name]
        
        return TriggerResponse(
            success=True,
            message=f"Removed {len(removed)} stale bricks: {removed}"
        )
    
    def get_stats(self):
        """Get current statistics"""
        current_time = rospy.Time.now()
        
        stats = {
            'total_bricks': len(self.bricks),
            'available': 0,
            'claimed': 0,
            'in_transit': 0,
            'placed': 0,
            'lost': 0,
            'fresh': 0,
            'stale': 0
        }
        
        for brick in self.bricks.values():
            stats[brick.state.value] += 1
            if brick.is_fresh(current_time, self.freshness_threshold):
                stats['fresh'] += 1
            else:
                stats['stale'] += 1
        
        return stats
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        rospy.loginfo("World Model running...")
        
        while not rospy.is_shutdown():
            # Periodic stats logging
            if rospy.Time.now().to_sec() % 30 < 0.1:  # Every 30 seconds
                stats = self.get_stats()
                rospy.loginfo(f"World Model Stats: {stats}")
            
            rate.sleep()


def main():
    """Main entry point"""
    try:
        world_model = WorldModel()
        world_model.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
