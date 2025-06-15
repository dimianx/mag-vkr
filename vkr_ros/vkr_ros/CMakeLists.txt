cmake_minimum_required(VERSION 3.8)
project(vkr_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(vkr_msgs REQUIRED)
find_package(vkr_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(TBB REQUIRED)
find_package(GDAL REQUIRED)

set(VKR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vkr_lib/include")
set(VKR_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vkr_lib/lib")

include_directories(
  include
  ${VKR_INCLUDE_DIR}
)

link_directories(${VKR_LIB_DIR})

set(dependencies
  rclcpp
  rclcpp_components
  std_msgs
  geometry_msgs
  sensor_msgs
  mavros_msgs
  vkr_msgs
  vkr_srvs
  tf2
  tf2_ros
  tf2_eigen
  pcl_conversions
  pcl_ros
  yaml-cpp
)

add_library(collision_server_component SHARED
  src/collision_server.cpp
)
ament_target_dependencies(collision_server_component ${dependencies})
target_link_libraries(collision_server_component
  vkr_geometry
  vkr_terrain
  ${PCL_LIBRARIES}
  ${GDAL_LIBRARIES}
)
rclcpp_components_register_node(collision_server_component
  PLUGIN "vkr_ros::CollisionServer"
  EXECUTABLE collision_server_node
)

add_executable(collision_server
  src/collision_server_main.cpp
)
ament_target_dependencies(collision_server ${dependencies})
target_link_libraries(collision_server
  collision_server_component
)

add_library(corridor_manager_component SHARED
  src/corridor_manager.cpp
)
ament_target_dependencies(corridor_manager_component ${dependencies})
target_link_libraries(corridor_manager_component
  vkr_geometry
)
rclcpp_components_register_node(corridor_manager_component
  PLUGIN "vkr_ros::CorridorManager"
  EXECUTABLE corridor_manager_node
)

add_executable(corridor_manager
  src/corridor_manager_main.cpp
)
ament_target_dependencies(corridor_manager ${dependencies})
target_link_libraries(corridor_manager
  corridor_manager_component
)

add_library(planner_node_component SHARED
  src/planner_node.cpp
)
ament_target_dependencies(planner_node_component ${dependencies})
target_link_libraries(planner_node_component
  vkr_planning
  vkr_geometry
  vkr_terrain
)
rclcpp_components_register_node(planner_node_component
  PLUGIN "vkr_ros::PlannerNode"
  EXECUTABLE planner_node_node
)

add_executable(planner_node
  src/planner_node_main.cpp
)
ament_target_dependencies(planner_node ${dependencies})
target_link_libraries(planner_node
  planner_node_component
)

add_library(state_machine_component SHARED
  src/state_machine.cpp
)
ament_target_dependencies(state_machine_component ${dependencies})
target_link_libraries(state_machine_component
  ${YAML_CPP_LIBRARIES}
)
rclcpp_components_register_node(state_machine_component
  PLUGIN "vkr_ros::StateMachine"
  EXECUTABLE state_machine_node
)

add_executable(state_machine
  src/state_machine_main.cpp
)
ament_target_dependencies(state_machine ${dependencies})
target_link_libraries(state_machine
  state_machine_component
)

add_library(landing_detector_component SHARED
  src/landing_detector.cpp
)
ament_target_dependencies(landing_detector_component ${dependencies})
target_link_libraries(landing_detector_component
  vkr_landing
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
rclcpp_components_register_node(landing_detector_component
  PLUGIN "vkr_ros::LandingDetector"
  EXECUTABLE landing_detector_node
)

add_executable(landing_detector
  src/landing_detector_main.cpp
)
ament_target_dependencies(landing_detector ${dependencies})
target_link_libraries(landing_detector
  landing_detector_component
)

install(TARGETS
  collision_server_component
  corridor_manager_component
  planner_node_component
  state_machine_component
  landing_detector_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  collision_server
  corridor_manager
  planner_node
  state_machine
  landing_detector
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  worlds
  data
  scripts
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()