add_executable(EightPoint main.cpp io.cpp io.h geometry.cpp geometry.h visualize.cpp visualize.h utils.cpp utils.h)

set(OpenCV_LIBS)
set(OpenCV_LIBS "opencv_core" "opencv_features2d" "opencv_highgui")
target_link_libraries(EightPoint ${OpenCV_LIBS})
