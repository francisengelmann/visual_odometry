//
// Created by Francis Engelmann on 2019-01-07.
//

#ifndef VISUALODOMETRY_UTILS_H
#define VISUALODOMETRY_UTILS_H

#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "geometry.h"
#include "io.h"

class Utils {
public:
    static void get_predefined_points(Eigen::MatrixXd& points1, Eigen::MatrixXd& points2){

        points1.resize(12, 3);
        points2.resize(12, 3);

        points1(0, 0) = 0.9535;
        points1(1, 0) = 1.8815;
        points1(2, 0) = 0.9535;
        points1(3, 0) = 1.8855;
        points1(4, 0) = 0.9475;
        points1(5, 0) = 0.6815;
        points1(6, 0) = 1.9055;
        points1(7, 0) = 2.1575;
        points1(8, 0) = 1.1075;
        points1(9, 0) = 1.7415;
        points1(10, 0) = 1.1435;
        points1(11, 0) = 1.7015;

        points1(0, 1) = 1.4055;
        points1(1, 1) = 1.4035;
        points1(2, 1) = 1.8895;
        points1(3, 1) = 1.8875;
        points1(4, 1) = 0.3455;
        points1(5, 1) = 0.8135;
        points1(6, 1) = 0.3515;
        points1(7, 1) = 0.8115;
        points1(8, 1) = 1.2055;
        points1(9, 1) = 1.2035;
        points1(10, 1) = 0.6635;
        points1(11, 1) = 0.6655;

        points2(0, 0) = 1.6295;
        points2(1, 0) = 2.4355;
        points2(2, 0) = 1.6295;
        points2(3, 0) = 2.4415;
        points2(4, 0) = 1.1755;
        points2(5, 0) = 0.8835;
        points2(6, 0) = 2.0875;
        points2(7, 0) = 2.2875;
        points2(8, 0) = 1.3355;
        points2(9, 0) = 1.9375;
        points2(10, 0) = 1.3775;
        points2(11, 0) = 1.9075;

        points2(0, 1) = 1.3855;
        points2(1, 1) = 1.3535;
        points2(2, 1) = 1.9135;
        points2(3, 1) = 1.8175;
        points2(4, 1) = 0.1955;
        points2(5, 1) = 0.7075;
        points2(6, 1) = 0.3115;
        points2(7, 1) = 0.7755;
        points2(8, 1) = 1.1595;
        points2(9, 1) = 1.1515;
        points2(10, 1) = 0.5655;
        points2(11, 1) = 0.6075;

        points1 = 1000 * points1;
        points2 = 1000 * points2;

        for(int i=0; i<12; i++){
            points1(i,2) = 1.0;
            points2(i,2) = 1.0;
        }
    }

    static void get_predefined_points2(Eigen::MatrixXd& points1, Eigen::MatrixXd& points2, cv::Mat& disp){

        int sc = 16;
        const int num_pts_max = (int)ceil(disp.rows/(float)sc) * (int)ceil(disp.cols/(float)sc);
        int num_pts = 0;

        points1.resize(num_pts_max, 3);
        points2.resize(num_pts_max, 3);

        for (int u = 0; u < disp.cols; u+=sc) {
            for (int v = 0; v < disp.rows; v+=sc) {
                float d = (float) (disp.at<unsigned char>(cv::Point(u, v)) / 4.0);
                //if (d < 0.25) continue; // skip points with invalid disparity
                float x = (float) (u);  // * z / f;
                float y = (float) (v);  // * z / f;
                float z = 1.0;

                points1(num_pts, 0) = x;
                points1(num_pts, 1) = y;
                points1(num_pts, 2) = z;
                points2(num_pts, 0) = x + d;
                points2(num_pts, 1) = y;
                points2(num_pts, 2) = z;

                num_pts++;
            }
        }
    }

    static void generate_matches(cv::Mat& image1, cv::Mat& image2, Eigen::MatrixXd& points1, Eigen::MatrixXd& points2){

        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        int minHessian = 400;
        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
        detector->setHessianThreshold(minHessian);
        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;
        detector->detectAndCompute(image1, cv::Mat(), keypoints_1, descriptors_1);
        detector->detectAndCompute(image2, cv::Mat(), keypoints_2, descriptors_2);

        //-- Step 2: Matching descriptor vectors using FLANN matcher
        cv::FlannBasedMatcher matcher;
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher.knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

        double max_dist = 0;
        double min_dist = DBL_MAX;
        //-- Quick calculation of max and min distances between keypoints
        for (auto& knn_match : knn_matches) {
            double dist = knn_match[0].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        // printf("-- Max dist : %f \n", max_dist );
        // printf("-- Min dist : %f \n", min_dist );
        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.
        std::vector<cv::DMatch> good_matches;
        for (auto knn_match : knn_matches) {
            if(knn_match[0].distance <= std::max(2*min_dist, 1.3)){
                double dist1 = knn_match[0].distance;
                double dist2 = knn_match[1].distance;
                double ratio = dist1 / dist2;
                if (ratio < 0.8) {
                    good_matches.push_back(knn_match[0]);
                }
            }
        }

        //-- Draw only "good" matches
        cv::Mat img_matches;
        drawMatches(image1, keypoints_1, image2, keypoints_2, good_matches, img_matches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //visualize(img_matches);

        //-- Extract feature points
        points1.resize(good_matches.size(), 3);
        points2.resize(good_matches.size(), 3);
        for(std::size_t i = 0; i < good_matches.size(); ++i){
            auto& match = good_matches[i];
            int img1_idx = match.queryIdx;
            int img2_idx = match.trainIdx;
            points1(i, 0) = keypoints_1[img1_idx].pt.x;
            points1(i, 1) = keypoints_1[img1_idx].pt.y;
            points1(i, 2) = 1.0;
            points2(i, 0) = keypoints_2[img2_idx].pt.x;
            points2(i, 1) = keypoints_2[img2_idx].pt.y;
            points2(i, 2) = 1.0;
        }
    }

    static bool reconstruction(Eigen::Matrix<double, 3, 3>& R,
                               Eigen::Matrix<double, 3, 1>& T,
                               Eigen::MatrixXd& points1,
                               Eigen::MatrixXd& points2,
                               Eigen::MatrixXd& lambda){
        int num_points = (int)points1.cols();
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3 * num_points, num_points + 1);
        for (int i=0; i<num_points; i++) {
            Eigen::Matrix3d x2_hat;
            Eigen::Matrix<double, 3, 1> x2 = points2.col(i);
            Eigen::Matrix<double, 3, 1> x1 = points1.col(i);
            Geometry::hat(x2, x2_hat);

            M.block(3*i, i, 3, 1) = x2_hat * R * x1;
            M.block(3*i, num_points, 3, 1) = x2_hat * T;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd.matrixV();
        lambda = V.block(0, 0, num_points, 1);
        double gamma = V(num_points, 0);

        bool correct = true;
        int wrong = 0;
        for (int i=0; i<num_points; i++){
            // std::cout << lambda(i, 0) << std::endl;
            if (lambda(i, 0) < 0) {
                correct = false;
                wrong++;
            }
        }
        std::cout << "Negative: " << wrong << "/" << num_points << std::endl;
        if (wrong > num_points * 0.25) {
            return false;
        }

        return true;
    }

};


#endif //VISUALODOMETRY_UTILS_H
