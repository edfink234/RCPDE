cmake_minimum_required(VERSION 3.10)
project(LibTorchExample)
# Print which C++ compiler is being used
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)

# Specify the path to libtorch
set(CMAKE_PREFIX_PATH "/usr/local/libtorch")

# Find the Torch package
find_package(Torch REQUIRED)

# Add the executable
add_executable(LibTorchExample libtorch_demo.cpp)

# Link the libraries to the executable
target_link_libraries(LibTorchExample "${TORCH_LIBRARIES}")

# Set the optimization level to -O2 for all compilers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Ensure C++17 standard for the target
set_property(TARGET LibTorchExample PROPERTY CXX_STANDARD 17)
