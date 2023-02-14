cmake_minimum_required(VERSION 3.23)
project(Math C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_FLAGS "-O3")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(Math src/Distribution.c src/Distribution.h)
target_link_libraries(Math ${CONAN_LIBS})
