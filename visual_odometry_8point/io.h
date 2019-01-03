//
// Created by Francis Engelmann on 2018-12-31.
//

#ifndef VISUALODOMETRY_IO_H
#define VISUALODOMETRY_IO_H

#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class IO {

public:
    static void write_file(Eigen::MatrixXd& depths, Eigen::MatrixXd& points, char seperator=','){
        std::ofstream myfile;
        myfile.open("example.txt");
        for (int i=0; i<points.cols(); i++){
            double x = points(0, i);
            double y = points(1, i);
            double z = depths(i, 0);
            myfile << x << seperator << y << seperator << z << std::endl;
        }
        myfile.close();
    }

    static void load_data(cv::Mat& image1, cv::Mat& image2, const double rs = 0.3){
        image1 = cv::imread("../../visual_odometry_8point/data/0005.png", CV_LOAD_IMAGE_COLOR);
        image2 = cv::imread("../../visual_odometry_8point/data/0007.png", CV_LOAD_IMAGE_COLOR);
        if(!image1.data){
            std::cout <<  "Could not open or find the image1" << std::endl;
            exit(1);
        }
        if(!image2.data){
            std::cout <<  "Could not open or find the image2" << std::endl;
            exit(1);
        }
        cv::resize(image1, image1, cv::Size(), rs, rs);
        cv::resize(image2, image2, cv::Size(), rs, rs);
    }

    static void read_disparity(const std::string& path, cv::Mat& disparity){
        auto disparity_raw = cv::imread(path, CV_LOAD_IMAGE_ANYDEPTH);
        cv::imshow("disp", disparity_raw);
        cv::waitKey(0);
    }
};

#endif //VISUALODOMETRY_IO_H
