cmake_minimum_required(VERSION 3.10)
project(ARIMA_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

# Set Boost paths
set(BOOST_ROOT "$ENV{HOME}/boost_1_74_0")
set(BOOST_INCLUDEDIR "$ENV{HOME}/boost_1_74_0")
set(BOOST_LIBRARYDIR "$ENV{HOME}/boost_1_74_0/stage/lib")

# Find required packages
find_package(Boost 1.74.0 REQUIRED COMPONENTS math_c99 math_c99f)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ARMADILLO REQUIRED armadillo)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Source files
set(SOURCES
    main.cpp
    data_loader.cpp
    adf_test.cpp
    arima_modeling.cpp
)

# Create executable
add_executable(main ${SOURCES})

# Link libraries
target_link_libraries(main 
    ${Boost_LIBRARIES}
    ${ARMADILLO_LIBRARIES}
)

# Set compiler definitions
target_compile_definitions(main PRIVATE ${ARMADILLO_CFLAGS_OTHER})