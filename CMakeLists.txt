cmake_minimum_required(VERSION 2.8)
project(VisualOdometry)
find_package(OpenCV REQUIRED)
message(${OpenCV_LIBS})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -Wall")
add_subdirectory(visual_odometry_8point)

add_subdirectory(visual_odometry_essential_matrix)
