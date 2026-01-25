#!/usr/bin/env python3
"""
@file test_perception.py
@brief Test script for validating the perception module's detection accuracy.
@details This script compares detected objects from the perception module against ground truth
         from Gazebo simulation. It evaluates position accuracy (XY, Z), orientation accuracy,
         and classification accuracy. The script handles object matching using planar distance
         and accounts for object symmetry when calculating orientation errors.
@author Benjamin Krech
@date January 2026
"""

import rospy
import numpy as np
import math
import time

# Try importing scipy, handle graceful failure if not installed
try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] Scipy not found. Using simple greedy matching.")

# Try importing TF transformations for accurate quaternion handling
try:
    from tf.transformations import euler_from_quaternion

    HAS_TF = True
except ImportError:
    HAS_TF = False


def get_yaw(orientation_q):
    """
    @brief Extracts yaw angle from a quaternion orientation.
    @details Converts a quaternion representation to yaw angle in degrees. Uses TF transformations
             library if available, otherwise falls back to manual calculation. The quaternion is
             normalized before conversion to ensure numerical stability.
    @param orientation_q Quaternion orientation as [x, y, z, w] (array or list).
    @return Yaw angle in degrees.
    """
    # Normalize quaternion to be safe
    q = np.array(orientation_q)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    if HAS_TF:
        # ROS standard [x, y, z, w]
        (_, _, yaw) = euler_from_quaternion(q)
        return math.degrees(yaw)
    else:
        # Manual fallback
        # Assumes q = [x, y, z, w]
        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def calculate_symmetry_error(gt_angle, det_angle, shape_class):
    """
    @brief Calculates angular error while accounting for object symmetry.
    @details Different objects have different rotational symmetries. Cubes have 90-degree symmetry
             (4-fold), while rectangles have 180-degree symmetry (2-fold). This function maps
             the angular difference to the smallest equivalent error considering the object's
             symmetry properties.
    @param gt_angle Ground truth angle in degrees.
    @param det_angle Detected angle in degrees.
    @param shape_class Object class name (e.g., 'cube', 'rectangle') used to determine symmetry.
    @return Symmetry-adjusted angular error in degrees (smallest equivalent difference).
    """
    diff = abs(gt_angle - det_angle) % 360
    if diff > 180:
        diff = 360 - diff

    # Lowercase check for safety
    shape_class = shape_class.lower()

    # CUBE: 90 degree symmetry
    if "cube" in shape_class:
        # Map error to 0-45 range
        # ex: 89 deg error -> 1 deg error
        while diff > 45:
            diff = abs(diff - 90)

    # RECTANGLE: 180 degree symmetry
    # ex: 179 deg error -> 1 deg error (flipped is fine)
    # ex: 90 deg error -> 90 deg error (sideways is BAD)
    else:
        if diff > 90:
            diff = abs(diff - 180)

    return diff


def run_test():
    """
    @brief Main test function that validates perception module accuracy.
    @details Executes the complete test pipeline:
             1. Initializes ROS node and perception module
             2. Retrieves ground truth object poses from Gazebo
             3. Obtains detected objects from the perception module
             4. Matches detected objects to ground truth using planar (XY) distance
             5. Calculates position errors (XY, Z), orientation errors, and classification accuracy
             6. Displays results in a formatted table with statistics
             7. Computes systematic biases and accuracy metrics

             The matching algorithm uses planar distance only to be robust against Z-height errors.
             Results include mean errors, standard deviations, and systematic bias detection.
    @return None
    """
    # Force stdout to flush immediately
    print("========================================", flush=True)
    print("   STARTING PERCEPTION TESTER...        ", flush=True)
    print("========================================", flush=True)

    print("[INFO] Importing Perception Module (this may take a moment)...", flush=True)

    # MOVE the import here. Now we see the prints above first!
    from perception_module import PerceptionModule

    print("[INFO] Initializing Node...", flush=True)
    rospy.init_node('perception_tester', anonymous=True)

    print("[INFO] Loading AI Models...", flush=True)
    perception = PerceptionModule(config=None)

    rospy.init_node('perception_tester', anonymous=True)

    # 1. Initialize Perception
    perception = PerceptionModule(config=None)
    time.sleep(1.0)  # Allow connections to stabilize

    # 2. Get Ground Truth from Gazebo
    try:
        from gazebo_msgs.msg import ModelStates

        print("[INFO] Waiting for Gazebo states...")
        msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5.0)
        perception.update_ground_truth(msg)
    except Exception as e:
        print(f"[ERROR] Could not get Ground Truth: {e}")
        return

    # 3. Get Detections
    detections = perception.get_detected_objects()
    ground_truth = perception.get_ground_truth_objects()

    n_det = len(detections)
    n_gt = len(ground_truth)
    print(f"\n[STATUS] Detected: {n_det} | Ground Truth: {n_gt}")

    if n_det == 0:
        print("[WARN] No objects detected. Check camera connection.")
        return

    # --- MATCHING LOGIC (PLANAR PRIORITY) ---
    # We use only X and Y for matching to be robust against Z-height errors
    cost_matrix = np.zeros((n_gt, n_det))

    for r, gt in enumerate(ground_truth):
        for c, det in enumerate(detections):
            # PLANAR DISTANCE only (ignore Z for matching purposes)
            dx = gt['position'][0] - det['position'][0]
            dy = gt['position'][1] - det['position'][1]
            dist_xy = math.sqrt(dx * dx + dy * dy)
            cost_matrix[r, c] = dist_xy

    # Assignment
    if HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # Simple greedy fallback if scipy missing
        row_ind = list(range(n_gt))
        col_ind = np.argmin(cost_matrix, axis=1)

    # --- RESULTS TABLE ---
    print("\n" + "-" * 120)
    print(
        f"{'GT NAME':<28} | {'MATCHED':<22} | {'XY-ERR':<9} | {'Z-ERR':<9} | {'YAW-ERR':<9} | {'CONF':<5} | {'CLASS OK?'}"
    )
    print("-" * 120)

    stats = {
        'count': 0,
        'xy_err': [],
        'z_err': [],
        'yaw_err': [],
        'correct_class': 0,
        'bias_x': [],
        'bias_y': [],
        'bias_z': [],
    }

    matched_det_indices = set()

    for i in range(len(row_ind)):
        gt_idx = row_ind[i]
        det_idx = col_ind[i]

        # Guard against index out of bounds if sizes differ
        if gt_idx >= n_gt or det_idx >= n_det:
            continue

        gt = ground_truth[gt_idx]
        det = detections[det_idx]

        # Calculate full 3D distance for validation
        pos_err_vec = det['position'] - gt['position']  # [dx, dy, dz]
        dist_3d = np.linalg.norm(pos_err_vec)

        # PLANAR match distance
        dist_xy = cost_matrix[gt_idx, det_idx]

        # Thresholds: Match is valid if XY < 15cm (generous to allow for calibration errors)
        if dist_xy < 0.15:
            matched_det_indices.add(det_idx)
            stats['count'] += 1

            # Errors
            xy_mm = dist_xy * 1000
            z_mm = pos_err_vec[2] * 1000  # Signed Z error to check bias

            # Yaw
            gt_yaw = get_yaw(gt['orientation'])
            det_yaw = get_yaw(det['orientation'])
            yaw_diff = calculate_symmetry_error(gt_yaw, det_yaw, gt['class'])

            # Class Check
            class_match = gt['class'] == det['class']
            class_str = "YES" if class_match else f"NO ({det['class']})"

            # Update Stats
            stats['xy_err'].append(xy_mm)
            stats['z_err'].append(abs(z_mm))
            stats['yaw_err'].append(yaw_diff)
            stats['bias_x'].append(pos_err_vec[0])
            stats['bias_y'].append(pos_err_vec[1])
            stats['bias_z'].append(pos_err_vec[2])
            if class_match:
                stats['correct_class'] += 1

            print(
                f"{gt['name']:<28} | {det['name']:<22} | {xy_mm:5.1f}mm   | {z_mm:+6.1f}mm  | {yaw_diff:5.1f}°   | {det['conf']:.2f}  | {class_str}"
            )
        else:
            print(
                f"{gt['name']:<28} | {'--- NO MATCH ---':<22} | -        | -        | -        | -     | -"
            )

    # Print Ghosts
    for i in range(n_det):
        if i not in matched_det_indices:
            print(
                f"{'??? (GHOST)':<28} | {detections[i]['name']:<22} | -        | -        | -        | {detections[i]['conf']:.2f}  | -"
            )

    print("-" * 120)

    # --- SUMMARY & DIAGNOSTICS ---
    if stats['count'] > 0:
        avg_xy = np.mean(stats['xy_err'])
        avg_z_abs = np.mean(stats['z_err'])
        avg_yaw = np.mean(stats['yaw_err'])
        accuracy = (stats['correct_class'] / stats['count']) * 100

        # BIAS CALCULATION (The Magic Numbers


if __name__ == '__main__':
    run_test()
