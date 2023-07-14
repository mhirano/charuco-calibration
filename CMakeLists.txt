cmake_minimum_required(VERSION 3.5)

#if(UNIX)
#set(OpenCV_LIBS /usr/local/lib)
#set(OpenCV_INCLUDE_DIRS /usr/local/include)
#endif()

# IF OpenCV_DIR IS NOT DECLARED IN YOUR ENVIRONMENT, SET ENVIRONMENT VARIABLE AS BELOW

set(OpenCV_DIR /Users/mhirano/opencv-4.5.3/build)
project( CHARUCO-CALIB CXX )
find_package( OpenCV REQUIRED )

# spdlog
include_directories(3rdparty/spdlog/include)

enable_language(CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
	set(CMAKE_CXX_STANDARD 17) # C++11...
	set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
	set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++17 support.")
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-DRELEASE -O3 -DNDEBUG -march=native")

include_directories( ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include )
message(STATUS "${PROJECT_SOURCE_DIR}")

set( src
		${PROJECT_SOURCE_DIR}/src/main.cpp
		${PROJECT_SOURCE_DIR}/src/CharucoCalibrator.cpp
		)

add_executable( CHARUCO-CALIB ${src} )
target_link_libraries( CHARUCO-CALIB ${OpenCV_LIBS})