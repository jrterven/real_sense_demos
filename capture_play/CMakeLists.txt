#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(capture_play)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

FIND_PACKAGE(OpenCV REQUIRED)

# Make project require C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()



find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()

# getting-started
include_directories(capture_play ${ADDITIONAL_INCLUDE_DIRS} "/home2/aifi_code/depth")
set(CODEFILES capture_play.cpp ../../utils.cpp)
add_executable(capture_play  ${CODEFILES})
target_link_libraries(capture_play ${REALSENSE2_FOUND} ${OpenCV_LIBS})


install(
        TARGETS

        capture_play

        RUNTIME DESTINATION
        ${CMAKE_INSTALL_PREFIX}/bin
)
