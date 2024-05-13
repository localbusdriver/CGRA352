cmake_minimum_required(VERSION 3.10)
project(a3)
find_package(OpenCV REQUIRED)

set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.9.0_8/share/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/opt/homebrew/Cellar/opencv/4.9.0_8/lib)

add_executable(a3 Assignment3.cpp)
target_link_libraries(a3 ${OpenCV_LIBS})

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
