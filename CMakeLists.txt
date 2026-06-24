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
find_package(nav_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(GAZEBO REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)


include_directories(
    include
    ${GAZEBO_INCLUDE_DIRS}
)
link_directories(${GAZEBO_LIBRARY_DIRS})

install(DIRECTORY
    launch
    worlds
    urdf
    models
    rviz
    data
    map
    DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY include/
    DESTINATION include
)

add_library(gas_source_plugin SHARED
    src/gas_source_plugin.cpp
)
ament_target_dependencies(gas_source_plugin
    rclcpp
    gazebo_ros
    std_msgs
    geometry_msgs
)

target_link_libraries(gas_source_plugin ${GAZEBO_LIBRARIES})

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

add_library(gas_attraction_layer SHARED
    src/gas_attraction_layer.cpp
)
ament_target_dependencies(gas_attraction_layer
    rclcpp
    nav2_costmap_2d
    pluginlib
    std_msgs
    geometry_msgs
    ament_index_cpp
)
target_link_libraries(gas_attraction_layer yaml-cpp)

pluginlib_export_plugin_description_file(nav2_costmap_2d plugin_description.xml)

install(TARGETS
    gas_source_plugin
    gas_sensor_plugin
    gas_attraction_layer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    scripts/gas_ground_truth_debug_node.py
    scripts/gas_patrol_node.py
    scripts/gas_concentration_marker_node.py
    DESTINATION lib/${PROJECT_NAME}
)

install(FILES plugin_description.xml
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
