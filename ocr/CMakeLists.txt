cmake_minimum_required(VERSION 3.5)
project(ocr)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(ocr ${SOURCE_FILES})
set(AUX_FILES train_test.cpp)
add_executable(train_test ${AUX_FILES})

target_link_libraries( ocr ${OpenCV_LIBS} )
target_link_libraries( train_test ${OpenCV_LIBS} )

