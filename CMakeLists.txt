cmake_minimum_required(VERSION 3.10)

# Project Name
project(InferenceServer VERSION 1.0)

# Force C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Where to look for header files
include_directories(include)

# Find all source files in 'src' folder
file(GLOB SOURCES "src/*.cpp")

# Create the executable
add_executable(server_engine ${SOURCES})