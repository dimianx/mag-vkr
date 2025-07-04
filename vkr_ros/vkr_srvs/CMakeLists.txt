cmake_minimum_required(VERSION 3.8)
project(vkr_srvs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(vkr_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(srv_files
  "srv/CheckCapsule.srv"
  "srv/CheckSegment.srv"
  "srv/CheckPath.srv"
  "srv/GetDistance.srv"
  "srv/CreateCorridor.srv"
  "srv/UpdateCorridor.srv"
  "srv/RemoveCorridor.srv"
  "srv/CheckIntrusion.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
  DEPENDENCIES
    std_msgs
    geometry_msgs
    builtin_interfaces
    vkr_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()