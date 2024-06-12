cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)
add_compile_options(-std=c++17)
project(a4)
find_package(OpenCV REQUIRED)

add_executable(a4 main.cpp FeatureExtractionAndMatching.cpp VideoStabilizer.cpp CroppingWindow.cpp)
target_link_libraries(a4 ${OpenCV_LIBS})

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
