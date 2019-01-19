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
        std::string path = "/Users/francis/example.txt";
        std::cout << "Writing to:" << path << std::endl;
        myfile.open(path);
        for (int i=0; i<points.cols(); i++){
            double x = points(0, i) * depths(i, 0);
            double y = points(1, i) * depths(i, 0);
            double z = depths(i, 0);
            myfile << x << seperator << y << seperator << z*100 << std::endl;
        }
        myfile.close();
    }

    static void write_pc(const Eigen::MatrixXf& points, const std::string& path, const char seperator=','){
        std::ofstream myfile;
        myfile.open(path);
        for (int i=0; i<points.rows(); i++) {
            double x = points(i, 0);
            double y = points(i, 1);
            double z = points(i, 2);
            myfile << x << seperator << y << seperator << z;
            if (points.cols() == 6) {
                double r = points(i, 3);
                double g = points(i, 4);
                double b = points(i, 5);
                myfile << seperator << r << seperator << g << seperator << b;
            }
            myfile  << std::endl;
        }
        myfile.close();
        std::cout << "Wrote to: " << path << std::endl;
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

    static cv::Mat* read_disparity(const std::string& path){
        auto raw = cv::imread(path, CV_LOAD_IMAGE_ANYDEPTH);
        cv::imshow("disp", raw);
        cv::waitKey(0);

    }

    static cv::Mat* read_color(const std::string& path){
        auto raw = cv::imread(path, CV_LOAD_IMAGE_COLOR);
        cv::imshow("disp", raw);
        cv::waitKey(0);
    }
};

#endif //VISUALODOMETRY_IO_H
