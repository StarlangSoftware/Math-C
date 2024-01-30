cmake_minimum_required(VERSION 3.23)
project(Math)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")


find_package(util_c REQUIRED)

add_library(Math Distribution.c Distribution.h DiscreteDistribution.c DiscreteDistribution.h Vector.c Vector.h Eigenvector.c Eigenvector.h Matrix.c Matrix.h)
target_link_libraries(Math util_c::util_c)
