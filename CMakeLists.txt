cmake_minimum_required(VERSION 3.5)
project(sudoku_c)
find_package( OpenCV REQUIRED )
add_executable( main main.cpp imageProcessor.cpp)
target_link_libraries( main ${OpenCV_LIBS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")