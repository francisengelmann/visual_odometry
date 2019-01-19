//
// Created by Francis Engelmann on 2019-01-07.
//

#ifndef VISUALODOMETRY_VISUALIZE_H
#define VISUALODOMETRY_VISUALIZE_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <eigen3/Eigen/Dense>

class Visualize {

    void visualize(const cv::Mat& image, double rs = 1.0){
        cv::Mat tmp;
        cv::resize(image, tmp, cv::Size(), rs, rs);
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display window", tmp);
        cv::waitKey(0);
    }

};


#endif //VISUALODOMETRY_VISUALIZE_H
