cmake_minimum_required(VERSION 3.6)
project(sort)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
add_executable(sort ${SOURCE_FILES})

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# for openmp
#find_package(OpenMP REQUIRED)
#target_link_libraries(sort PRIVATE OpenMP::OpenMP_CXX)