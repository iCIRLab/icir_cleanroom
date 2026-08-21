cmake_minimum_required(VERSION 3.8)
project(icir_cleanroom)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(GAZEBO REQUIRED)


include_directories(
    include
    ${GAZEBO_INCLUDE_DIRS}
)
link_directories(${GAZEBO_LIBRARY_DIRS})

install(DIRECTORY
    launch
    worlds
    urdf
    rviz
    DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY include/
    DESTINATION include
)

add_library(gas_sensor_plugin SHARED
    src/gas_sensor_plugin.cpp
)
ament_target_dependencies(gas_sensor_plugin
    rclcpp
    gazebo_ros
    std_msgs
    geometry_msgs
)
target_link_libraries(gas_sensor_plugin ${GAZEBO_LIBRARIES})

install(TARGETS
    gas_sensor_plugin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    scripts/gas_empty_map_node.py
    scripts/gas_mapping_controller_node.py
    scripts/lrs_path_planner_node.py
    DESTINATION lib/${PROJECT_NAME}
)

install(FILES
    scripts/gas_history.py
    scripts/gmrf_grid.py
    scripts/hrs_path_planner.py
    scripts/lrs_priority_planner.py
    scripts/lrs_tsp_planner.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
