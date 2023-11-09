# FILEPATH: /home/jiuzl/scarab_ws/src/generalized_vo/CMakeLists.txt

cmake_minimum_required(VERSION 3.5)
project(generalized_vo)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gazebo REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  geometry_msgs
  nav_msgs
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS roscpp geometry_msgs nav_msgs tf2_ros gazebo_ros
)

# Add include directories
include_directories(include ${EIGEN3_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

# add_executable(generalized_vo_src src/GVO.cpp )

# Add executables
add_executable(generalized_vo_node src/gvo_node.cpp src/GVO.cpp)
target_link_libraries(generalized_vo_node ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${CERES_LIBRARIES})
