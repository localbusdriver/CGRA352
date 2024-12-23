cmake_minimum_required(VERSION 3.10)
project(a3)

# set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.9.0_8/share/opencv4)
set(Open_CV_DIR C:/Program Files/opencv/build/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(/opt/homebrew/Cellar/opencv/4.9.0_8/lib)
# link_directories(C:/Program Files/opencv/build/x64/vc16/lib)
add_executable(a3 Assignment3.cpp)
target_link_directories(a3 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(a3 PRIVATE ${OpenCV_LIBS})

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
