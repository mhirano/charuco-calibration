cmake_minimum_required(VERSION 2.8)

#if(UNIX)
#set(OpenCV_LIBS /usr/local/lib)
#set(OpenCV_INCLUDE_DIRS /usr/local/include)
#endif()

# IF OpenCV_DIR IS NOT DECLARED IN YOUR ENVIRONMENT, SET ENVIRONMENT VARIABLE AS BELOW
# set(OpenCV_DIR /Users/mhirano/opencv-3.4.1/build)

project( SAMPLE CXX )
find_package( OpenCV REQUIRED )

enable_language(CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	set(CMAKE_CXX_STANDARD 11) # C++11...
	set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
	set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has C++11 support.")
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g -pg -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-DRELEASE -O3 -DNDEBUG -march=native")

include_directories( ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include )
message(STATUS "${PROJECT_SOURCE_DIR}")

set( src
		${PROJECT_SOURCE_DIR}/src/main.cpp
		${PROJECT_SOURCE_DIR}/include/InterThreadMessenger.hpp
		)

add_executable( SAMPLE ${src} )
target_link_libraries( SAMPLE ${OpenCV_LIBS})