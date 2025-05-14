cmake_minimum_required(VERSION 3.23)
project(MathTests)

set(CMAKE_C_STANDARD 17)

# Include the Math library
include_directories(../src)

# Add the TensorTest executable
add_executable(TensorTest TensorTest.c)
target_link_libraries(TensorTest Math util_c::util_c data_structure_c::data_structure_c)