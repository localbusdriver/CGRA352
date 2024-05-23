cmake_minimum_required(VERSION 3.10)
project(a4)
find_package(OpenCV REQUIRED)

add_executable(a4 main.cpp FeatureExtractionAndMatching.cpp VideoStabilizer.cpp)
# add_executable(a4 FeatureExtractionAndMatching.cpp )
target_link_libraries(a4 ${OpenCV_LIBS})

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
