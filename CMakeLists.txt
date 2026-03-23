cmake_minimum_required(VERSION 3.10)

# Project Name
project(InferenceServer VERSION 1.0)

# Force C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Where to look for header files
include_directories(include)

# 1. Find the gRPC and Protobuf libraries on your system
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
# ---> ADDED THIS LINE FOR AI MATH <---
find_package(onnxruntime CONFIG REQUIRED)
find_package(Crow CONFIG REQUIRED)

# 2. Point CMake to your contract file
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_FILE "${PROTO_DIR}/inference.proto")

# 3. Define the output files we expect the compiler to generate
set(PROTO_SRCS 
    "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.cc" 
    "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.cc"
)
set(PROTO_HDRS 
    "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.h" 
    "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.h"
)

# 4. Tell CMake to run protoc WITH the gRPC network plugin
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND protobuf::protoc
    ARGS --proto_path="${PROTO_DIR}"
         --cpp_out="${CMAKE_CURRENT_BINARY_DIR}"
         --grpc_out="${CMAKE_CURRENT_BINARY_DIR}"
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
)

# 5. Find all source files in 'src' folder
file(GLOB SOURCES "src/*.cpp")

# Create the executable (now including BOTH sets of generated files)
add_executable(server_engine ${SOURCES} ${PROTO_SRCS})

# Tell the C++ compiler where to find the newly generated header files
target_include_directories(server_engine PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# 6. Link the actual gRPC and Protobuf network engines to your code
# ---> ADDED 'onnxruntime' TO THE END OF THIS LINE <---
target_link_libraries(server_engine PRIVATE gRPC::grpc++ protobuf::libprotobuf onnxruntime::onnxruntime Crow::Crow)

