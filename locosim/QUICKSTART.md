# Locosim Docker Setup - Quick Reference

## ✅ Setup Complete

All submodules fetched, workspace built successfully, and ready to run examples.

---

## 🚀 Running Examples

### Method 1: Using the helper script (recommended)

```bash
cd ~/Documents/Uni/Sem_05_Erasmus/Robotic/project/robotics_project/locosim/ros_ws
./RUN_EXAMPLE.sh base_controller.py
# or
./RUN_EXAMPLE.sh ur5_generic.py
```

### Method 2: Manual commands (inside container)

```bash
# Open a shell in the running container
docker exec -it locosim_c bash

# Inside the container:
source /home/ubuntu/ros_ws/src.sh
python3 -i $LOCOSIM_DIR/robot_control/base_controllers/base_controller.py
# or
python3 -i $LOCOSIM_DIR/robot_control/base_controllers/ur5_generic.py
```

The `src.sh` script automatically:

- Sources ROS and workspace environments
- Sets `LOCOSIM_DIR` variable
- Configures Python paths for imports

---

## 🖥️ Viewing GUIs (Gazebo, RViz, etc.)

Open in your browser: **<http://localhost:6080>**

- Username: `ubuntu`
- Password: `ubuntu`

**Note:** GDK/Qt warnings like `"could not connect to display"` are normal in VNC mode and can be safely ignored.

---

## 📂 Available Example Controllers

Located in `/home/ubuntu/ros_ws/src/locosim/robot_control/`:

**Base controllers:**

- `base_controllers/base_controller.py` - Generic base controller
- `base_controllers/ur5_generic.py` - UR5 manipulator controller
- `base_controllers/quadruped_controller.py` - Quadruped robot controller

**Lab exercises:**

- `lab_exercises/lab_palopoli/ur5_generic.py` - UR5 lab examples

Explore other files in those directories for more examples.

---

## 🔧 Common Issues & Fixes

### 1. GDK/Display/Qt errors (safe to ignore)

```text
Unable to init server: Could not connect: Connection refused
Gdk-CRITICAL **: gdk_cursor_new_for_display: assertion failed
qt.qpa.xcb: could not connect to display
```

These are normal when launching from a headless terminal. Use the VNC browser UI at <http://localhost:6080> to view GUIs.

### 2. Container not running

Start the container:

```bash
docker run --name locosim_c --rm \
  -v ~/Documents/Uni/Sem_05_Erasmus/Robotic/project/robotics_project/locosim/ros_ws/:/home/ubuntu/ros_ws/ \
  -p 6080:80 \
  --shm-size=512m \
  --platform linux/amd64 \
  locosim:noetic
```

Leave this terminal open (or remove `--rm` to persist the container).

### 3. Import errors after code changes

If you modify C++ code or add new packages, rebuild:

```bash
docker exec -it locosim_c bash
cd /home/ubuntu/ros_ws
source /opt/ros/noetic/setup.bash
catkin_make
catkin_make install
```

### 4. Submodule update needed

If you pull new commits:

```bash
# On host:
cd ~/Documents/Uni/Sem_05_Erasmus/Robotic/project/robotics_project/locosim/ros_ws/src/locosim
git submodule update --init --recursive
# Then rebuild inside container as in #3
```

### 5. Missing Python packages (pinocchio, etc.)

Install in the container:

```bash
docker exec -it locosim_c bash
apt-get update
apt-get install -y ros-noetic-pinocchio
```

---

## ✏️ Editing Code

Edit files on your host with PyCharm or any editor:

```text
~/Documents/Uni/Sem_05_Erasmus/Robotic/project/robotics_project/locosim/ros_ws/src/locosim/robot_control/
```

Changes are immediately visible in the container because the folder is mounted with `-v`.

---

## 📚 Next Steps

1. Try running `base_controller.py` or `ur5_generic.py`
2. Open <http://localhost:6080> to view Gazebo/RViz
3. Explore lab exercises in `robot_control/lab_exercises/`
4. Read the main README: `locosim/ros_ws/src/locosim/Readme.md`

Enjoy!
