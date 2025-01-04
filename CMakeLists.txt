cmake_minimum_required(VERSION 3.10)
project(cpp_nn_mnsit)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(matrix STATIC
    #  ${CMAKE_CURRENT_SOURCE_DIR}/lib/matrix.cpp
    ../lib/matrix.cpp
)

add_executable(matrix_init_test
    #  ${CMAKE_CURRENT_SOURCE_DIR}/tests/matrix_init.cpp
    ../tests/matrix_init.cpp
)

target_link_libraries(matrix_init_test
  matrix
  gtest
  gtest_main
  pthread
)

enable_testing()

add_test(NAME matrix_init_test COMMAND matrix_init_test)

