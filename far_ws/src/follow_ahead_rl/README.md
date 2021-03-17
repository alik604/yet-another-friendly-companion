- Install [ROS1](http://wiki.ros.org/ROS/Installation)
- Install [Gazebo 9.16](http://gazebosim.org/tutorials?cat=install&tut=install_ubuntu&ver=9.0)
    - Use Alternative Install

**Download dependencies**
- `sudo apt install ros-melodic-turtlebot3-description ros-melodic-control-msgs  ros-melodic-control-toolbox  ros-melodic-controller-interface  ros-melodic-controller-manager  ros-melodic-controller-manager-msgs  ros-melodic-diff-drive-controller  ros-melodic-forward-command-controller  ros-melodic-gazebo-ros-control  ros-melodic-joint-state-controller  ros-melodic-position-controllers ros-melodic-robot-localization ros-melodic-move-base ros-melodic-lms1xx ros-melodic-pointgrey-camera-driver ros-melodic-pointgrey-camera-description ros-melodic-hector-gazebo-plugins ros-melodic-interactive-marker-twist-server`

**Build project:**
- Inside /far_ws
- Build project: `catkin_make`
- Source workspace: `source devel/setup.bash`
- Launch project: `roslaunch src/follow_ahead_rl/launch/turtlebot.launch`
