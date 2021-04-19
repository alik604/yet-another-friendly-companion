cmake_minimum_required(VERSION 2.8.3)
project(follow_ahead_rl)

find_package(catkin REQUIRED COMPONENTS rospy)

# add_message_files(
#   FILES  # e.g. Floats.msg HeaderString.msg
# )
# add_service_files(
#   DIRECTORY srv
#   FILES AddTwoInts.srv BadTwoInts.srv
# )

## Generate services in the 'srv' folder
# add_service_files(
#  FILES  # e.g. Floats.srv HeaderString.srv
#)

## Generate added messages and services with any dependencies
# generate_messages()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES python_ekf
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  old_script/tf_node.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_package(
  CATKIN_DEPENDS
)
