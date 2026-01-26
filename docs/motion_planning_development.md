# Motion Planning

## Approach Overview

The motion planning module is responsible for generating collision-free,
kinematically valid movements for the UR5 manipulator during
pick-and-place operations. The system combines analytical inverse
kinematics, coordinate frame transformations, and trajectory generation
to determine feasible motions between grasp and placement poses.

The main responsibilities are:

-   Converting 3D target poses into joint configurations using
    analytical IK

-   Selecting a valid and consistent IK solution from the eight
    available configurations

-   Producing smooth joint-space trajectories for approach, grasp, lift,
    and place motions

-   Executing these trajectories through the robot controller

## Analytical Inverse Kinematics

The IK subsystem is based on the analytical solution provided by the
reference script `ur5Inverse.m`. This formulation exploits the geometry
of the UR5 to compute closed-form joint solutions, yielding up to eight
distinct configurations for any valid end-effector pose.

## Coordinate System Handling

In the simulation environment, the UR5 is mounted in an inverted
configuration. As a result, the robot's Denavit--Hartenberg (DH)
coordinate frame has its $+Z$ axis pointing downward relative to the
world frame. To correctly interpret a target point expressed in the
robot-relative frame, a frame adjustment is required. This is done by
simply negating the Z-axis.

## IK Solution Selection

### Angle Normalization

Many IK solutions differ only by multiples of $2\pi$. To ensure
consistency and avoid unnecessary long rotations, each candidate
solution is normalized relative to a reference configuration (typically
the home position). For each joint:
$\tilde{q}_i = q_i + 2\pi k \quad \text{such that} \quad |\tilde{q}_i - q_{i,\text{ref}}| \text{ is minimal}$

### Base Joint Constraint

To prevent the robot from rotating around its base unnecessarily,
potentially colliding with the environment, a soft constraint is applied
on the deviation of $q_1$ from the reference posture:
$|q_1 - q_{1,\text{ref}}| < \frac{\pi}{3}$ This constraint is still
being tuned to accommodate the full reachable workspace.

### Joint Limit Cost

To prioritize solutions that remain well within joint limits, a
quadratic penalty is applied: $$w(q) = \frac{1}{2n} \sum_{i=1}^{n}
\left(
\frac{q_i - q_{\text{mid},i}}{q_{\max,i} - q_{\min,i}}
\right)^2$$

## Trajectory Generation

Once a target joint configuration is selected, the trajectory is
generated using quintic interpolation. The time-scaling function ensures
smooth transitions with zero velocity and acceleration at the start and
end: $s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$

This interpolation is used throughout all motion segments, including
approach, descent, grasp, retreat, and placement.

## Error Sources and Mitigation

-   **IK unreachable poses:** The UR5 has an inner cylindrical region of
    radius $0.1333\,\text{m}$ that is geometrically unreachable. To
    avoid infeasible targets, objects are spawned only outside this
    inner workspace boundary.

-   **Incorrect placement frame:** Placement poses must be interpreted
    directly in the world frame. The error was resolved by disabling
    robot-relative transforms during placement, using
    `robot_relative=False`.

## Testing and Diagnostic Tools

There are diagnostic tools implemented that help with the validation of
the motion planning system.

-   **Forward kinematics and IK validation:** Tools that compute forward
    kinematics, compare it with simulation data, and verify the
    consistency of IK by performing FK → IK → FK roundtrip tests.

-   **Coordinate frame evaluation:** Scripts that systematically test
    different world-to-robot and camera-to-robot frame mappings. These
    tests were used to identify the correct transformation for the
    inverted robot setup.

-   **Interactive frame testing:** Tools that move the robot to
    predicted positions and require user confirmation to validate
    whether the gripper aligns correctly with the perceived object.

-   **End-to-end motion pipeline testing:** High-level tests that detect
    objects, compute approach poses, run IK, and execute trajectories
    for one or multiple objects to assess overall system performance.
