cmake_minimum_required(VERSION 3.5)
project(plen_ros)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# (Optional) Add ROS2 include directory
include_directories(/opt/ros/humble/include)

# Explicitly set gazebo_ros directory for Humble if needed
set(gazebo_ros_DIR "/opt/ros/humble/share/gazebo_ros/cmake")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)

# Generate messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RLExperimentInfo.msg"
  "srv/Iterate.srv"
  DEPENDENCIES std_msgs
)

# Include the generated interface headers
include_directories(
  include
  ${gazebo_ros_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)

# Add the C++ node executable
add_executable(gazebo_tools_test src/gazebo_tools_test.cpp)
target_link_libraries(gazebo_tools_test
  ${gazebo_msgs_LIBRARIES}
  ${gazebo_ros_LIBRARIES}
  rclcpp::rclcpp
)
ament_target_dependencies(gazebo_tools_test
  rclcpp
  std_msgs
  std_srvs
  gazebo_ros
  gazebo_msgs
  controller_manager_msgs
  rosidl_default_runtime
)

# Link to the generic rosidl_typesupport_cpp library
rosidl_get_typesupport_target(
  typesupport_target
  "${PROJECT_NAME}"
  "rosidl_typesupport_cpp"
)
target_link_libraries(gazebo_tools_test ${typesupport_target})

# Install the C++ node
install(TARGETS gazebo_tools_test
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install your Python executables (must have +x permissions)
install(PROGRAMS
  src/joint_test.py
  src/joint_trajectory_test.py
  src/plen_td3.py
  src/walk_eval.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Note: Removed the ament_python_install_package call here.

# Install your URDF folder
install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

# Install your ROS2 launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Optionally install any headers in include/
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

# Export rosidl dependencies to ensure interfaces are recognized system-wide
ament_export_dependencies(rosidl_default_runtime)

# Finalize package
ament_package()

