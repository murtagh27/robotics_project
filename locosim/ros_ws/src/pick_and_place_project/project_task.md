# Project Description

## Project

A number of objects (e.g., mega-blocks) are stored without any specific order on a stand
(initial stand) located within the workspace of a robotic manipulator.
The manipulator is an anthropomorphic arm with a spherical wrist and a two-fingered gripper as an end-effector.
The objects can belong to different classes but have a known geometry (coded in the STL
files).

The objective of the project is to use the manipulator to pick the objects in sequence and to position them on a different stand according to a specified order (final stand).
A calibrated 3D sensor is used to locate the different objects and to detect their position in the initial stand.

## Assignment

There are multiple objects on the initial stand, one for each class.
There is no specific order in the initial configuration, except that the base of the object is “naturally” in contact with the ground.
Each object must be picked up and stored in the position prescribed for its class and marked by the object’s silhouette.

## Delivery rules

The project is developed in groups.
The typical group size consists of three to four members.
We can also accept groups with a smaller number of members.
The group is supposed to work in perfect cooperation, and the workload is required to be fairly distributed.
The specific contribution of each member will be exposed during the project discussion.
The delivery phase is as follows:

1. The project can be implemented both in simulation and on the real robot. In the second case, it
   will need to be tested in the laboratory with the Teaching Assistant at least five days before the exam date. During the tests, small videos can be shot and used for the presentation.
2. Each group will have to deliver the package containing the full code (with doxygen documentation and a readme for use), plus a 5–6-page report describing
   - The technique used for perception
   - The technique used for robot motion
   - The technique used for high-level planning
3. The delivery deadline is three days before the (oral) exam presentation
4. On the day of the exam, the students will give a 10-minute presentation highlighting the contribution of each member in the oral session.
5. If allowed by the time, the group could also be asked to perform a small demo session. Otherwise, we will rely on the clip shot before the exam.
