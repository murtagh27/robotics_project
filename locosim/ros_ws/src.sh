source /opt/ros/noetic/setup.bash
source /home/ubuntu/ros_ws/devel/setup.bash
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3.8/site-packages
roscore
#Ex python3 -i $LOCOSIM_DIR/robot_control/base_controllers/base_controller.py