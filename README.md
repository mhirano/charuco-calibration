# Asynchronous Display Template

This project serves as a starting point of high speed image processing and display results asynchronously.

High speed image processing is designed to be done in a dedicated worker thread. 
Display the result asynchronously irrerevelant to the state of the working thread.

This is a modified version of Shingo W. Kagami's code.
Unlike the code, this provides universal version, e.g. no boost, no OS dependency.
See original in https://bitbucket.org/swkagami/highspeed-cam-opencv

Some remarks:
- In compatible with modern C++ compiler (that supports C++11)

- No additinoal library required. No Boost, No Qt.

- Primarily designed for CLion and Mac, but should work with other IDEs and operating system that can handle CMake.

- You should add paths before compile. For ease of use, here is excerpt from my .bash_profile
```
# OpenCV
export OpenCV_DIR=/Users/mhirano/opencv-3.2.0/build
export OpenCV_LIBS=/Users/mhirano/opencv-3.2.0/build/lib
export OpenCV_INCLUDE_DIRS=/Users/mhirano/opencv-3.2.0/include
```
Otherwise, set environmental variable in CMakeLists.txt as directed within the file.

- Use out-of-source build. Namely, run cmake within debug/release directory as `cmake ..`
