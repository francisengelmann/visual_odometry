add_executable(EightPoint main.cpp)
add_executable(Util utils.cpp)

set(OpenCV_LIBS)
set(OpenCV_LIBS "opencv_core" "opencv_xfeatures2d" "opencv_highgui")
target_link_libraries(EightPoint ${OpenCV_LIBS})
target_link_libraries(Util ${OpenCV_LIBS})
