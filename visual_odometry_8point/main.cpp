#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <eigen3/Eigen/Dense>

#include "io.h"
#include "geometry.h"

using namespace cv::xfeatures2d;

double time_diff(std::chrono::high_resolution_clock::time_point t1,
                 std::chrono::high_resolution_clock::time_point t2){
    std::chrono::duration<double> time_diff = t2 - t1;
    return time_diff.count();
}

void visualize(const cv::Mat& image, double rs = 1.0){
    cv::Mat tmp;
    cv::resize(image, tmp, cv::Size(), rs, rs);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", tmp);
    cv::waitKey(0);
}

void generate_matches(cv::Mat& image1, cv::Mat& image2, Eigen::MatrixXd& points1, Eigen::MatrixXd& points2){

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = SURF::create();
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

void get_predefined_points(Eigen::MatrixXd& points1, Eigen::MatrixXd& points2){

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

bool reconstruction(Eigen::Matrix<double, 3, 3>& R, Eigen::Matrix<double, 3, 1>& T, Eigen::MatrixXd& points1, Eigen::MatrixXd& points2){
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
    Eigen::MatrixXd lambda = V.block(0, 0, num_points, 1);
    double gamma = V(num_points, 0);

    bool correct = true;
    int wrong = 0;
    for (int i=0; i<num_points; i++){
        if (lambda(i, 0) < 0) {
            correct = false;
            wrong++;
        }
    }

    if (wrong > num_points * 0.25) {
        return false;
    }

    IO::write_file(lambda, points1);
    return true;
}

int main(int argc, char** argv){

    cv::Mat disp;
    IO::read_disparity("../../visual_odometry_8point/data/teddy/disp2.png", disp);

    cv::Mat image1, image2;
    Eigen::MatrixXd points1, points2;
    IO::load_data(image1, image2);
    generate_matches(image1, image2, points1, points2);
    // get_predefined_points(points1, points2);

    auto time_start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix3d K1, K2;
    K1 << 2759.48,     0.0, 1520.69,
              0.0, 2764.16, 1006.81,
              0.0,     0.0,     1.0;
    K2 = K1;
    auto K1_inv = K1.inverse();
    auto K2_inv = K2.inverse();

    // Transform image coordinates with inverse camera matrices:
    points1 = K1_inv * points1.transpose();  // shape: [3, N]
    points2 = K2_inv * points2.transpose();  // shape: [3, N]

    // Compute constraint matrix A
    int num_points = (int)points1.cols();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_points, 9);
    for (int i=0; i<num_points; i++){
        auto a = points1.col(i);
        auto b = points2.col(i);
        Eigen::MatrixXd kr;
        Geometry::kron(a, b, kr);
        A.row(i) = kr.transpose();
    }

    // Find minimizer for A*E:
    // BDCSVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::MatrixXd E = svd.matrixV().col(8);
    E.resize(3, 3);

    // SVD E
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(E, Eigen::ComputeFullV | Eigen::ComputeFullU );
    Eigen::MatrixXd U, D, V;
    U = svd2.matrixU();
    D = svd2.singularValues().asDiagonal();
    V = svd2.matrixV();

    if (U.determinant() < 0 || V.determinant() < 0){
        Eigen::JacobiSVD<Eigen::MatrixXd> svd3(E*-1, Eigen::ComputeFullV | Eigen::ComputeFullU );
        U = svd3.matrixU();
        D = svd3.singularValues().asDiagonal();
        V = svd3.matrixV();
    }

    D(0, 0) = 1.0;
    D(1, 1) = 1.0;
    D(2, 2) = 0.0;

    // Final essential matrix
    E = U * D * V.transpose();

    // Recover R and T from the essential matrix E
    Eigen::Matrix3d Rz1;
    Rz1 << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Rz1 = Rz1.transpose().eval();
    Eigen::Matrix3d Rz2;
    Rz2 << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Rz2 = Rz2.transpose().eval();

    Eigen::Matrix3d R1 = U * Rz1.transpose() * V.transpose();
    Eigen::Matrix3d R2 = U * Rz2.transpose() * V.transpose();

    Eigen::Matrix3d T_hat1 = U * Rz1 * D * U.transpose();
    Eigen::Matrix3d T_hat2 = U * Rz2 * D * U.transpose();

    Eigen::Matrix<double, 3, 1> T1;
    Eigen::Matrix<double, 3, 1> T2;
    T1 << -T_hat1(1, 2), T_hat1(0, 2), -T_hat1(0, 1);
    T2 << -T_hat2(1, 2), T_hat2(0, 2), -T_hat2(0, 1);

    reconstruction(R1, T1, points1, points2);
    reconstruction(R2, T1, points1, points2);

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Processing time:" << time_diff(time_start, time_end) << std::endl;
    return 0;
}
