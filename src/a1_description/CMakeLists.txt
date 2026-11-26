cmake_minimum_required(VERSION 3.8)
project(a1_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch maps meshes rviz urdf worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/rl_locomotion_controller.py
  scripts/lidar_obstacle_filter.py
  scripts/elevation_map_node.py
  scripts/floor_manager_node.py
  scripts/floor_goal_gui.py
  scripts/dynamic_obstacles_node.py
  scripts/multi_floor_patrol.py
  scripts/mapping_patrol.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
