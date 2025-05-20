cmake_minimum_required(VERSION 3.23)
project(Math)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-O3")

set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/Release/generators)
find_package(util_c REQUIRED)

add_library(Math 
    src/Distribution.c 
    src/DiscreteDistribution.c 
    src/Vector.c 
    src/Eigenvector.c 
    src/Matrix.c 
    src/Tensor.c)
target_link_libraries(Math util_c::util_c)

add_executable(MatrixTest Test/MatrixTest.c)
target_link_libraries(MatrixTest Math util_c::util_c)

add_executable(VectorTest Test/VectorTest.c)
target_link_libraries(VectorTest Math util_c::util_c)

add_executable(DiscreteDistributionTest Test/DiscreteDistributionTest.c)
target_link_libraries(DiscreteDistributionTest Math util_c::util_c)

add_executable(TensorTest Test/TensorTest.c)
target_link_libraries(TensorTest Math util_c::util_c)

add_subdirectory(Test)
