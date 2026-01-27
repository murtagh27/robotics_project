#!/usr/bin/env python3
"""
Quick test script to run p.start_test() with velocity limit checking.
"""
import os
import sys

# Set up environment
os.environ['LOCOSIM_DIR'] = '/opt/ros/locosim'
sys.path.insert(0, '/opt/ros/locosim/robot_control')

import rospy
from controller import main

if __name__ == '__main__':
    # Initialize controller
    p = main()
    
    # Run test
    rospy.sleep(2.0)  # Wait for everything to settle
    rospy.loginfo("\n" + "="*60)
    rospy.loginfo("Starting test with SLOW movements (10s duration)")
    rospy.loginfo("Testing velocity limit hypothesis...")
    rospy.loginfo("="*60 + "\n")
    
    p.start_test()
    
    rospy.loginfo("\nTest complete. Check logs for velocity warnings.")
