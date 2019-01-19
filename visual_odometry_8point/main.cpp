#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <eigen3/Eigen/Dense>

#include "io.h"
#include "geometry.h"
#include "utils.h"

double time_diff(std::chrono::high_resolution_clock::time_point t1,
                 std::chrono::high_resolution_clock::time_point t2){
    std::chrono::duration<double> time_diff = t2 - t1;
    return time_diff.count();
}

int main(int argc, char** argv){

    auto disp = cv::imread("../../visual_odometry_8point/data/teddy/disp2.png", CV_LOAD_IMAGE_ANYDEPTH);
    auto color = cv::imread("../../visual_odometry_8point/data/teddy/im2.png", CV_LOAD_IMAGE_COLOR);

    /*int sc = 64;
    const int num_pts_max = (int)ceil(disp.rows/(float)sc) * (int)ceil(disp.cols/(float)sc);
    std::cout << "number of points: " << num_pts_max << std::endl;
    Eigen::MatrixXf points(num_pts_max, 6);
    int num_pts = 0;
    float f = 500.0;
    float b = 100.0;
    for (int u = 0; u < disp.cols; u+=sc) {
        for (int v = 0; v < disp.rows; v+=sc) {
            float d = (float)(disp.at<uchar>(cv::Point(u,v)) / 4.0);
            if (d < 0.25) {
                // std::cout << "too small" << std::endl;
                continue; // skip points with invalid disparity
            }
            float z = f * b / d;
            // std::cout << z << std::endl;
            float x = (float)(u - disp.cols/2.0) * z / f;
            float y = (float)(v - disp.rows/2.0) * z / f;
            float r = color.at<cv::Vec3b>(cv::Point(u,v))[2];
            float g = color.at<cv::Vec3b>(cv::Point(u,v))[1];
            float b = color.at<cv::Vec3b>(cv::Point(u,v))[0];
            points(num_pts, 0) = x;
            points(num_pts, 1) = y;
            points(num_pts, 2) = z;
            points(num_pts, 3) = r;
            points(num_pts, 4) = g;
            points(num_pts, 5) = b;
            num_pts++;
        }
    }
    std::cout << "number of points: " << num_pts << std::endl;
    //points.resize(num_pts, 6);

    IO::write_pc(points, "/Users/francis/cloud_small.txt");

    //cv::Mat disp, color;
    //IO::read_disparity("../../visual_odometry_8point/data/teddy/disp2.png", disp);
    //IO::read_color("../../visual_odometry_8point/data/teddy/im2.png", color);
    exit(0);*/


    //cv::Mat image1, image2;
    Eigen::MatrixXd points1, points2;
    //IO::load_data(image1, image2);
    //generate_matches(image1, image2, points1, points2);
    // get_predefined_points(points1, points2);
    Utils::get_predefined_points2(points1, points2, disp);

    auto time_start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix3d K1, K2;
//    K1 << 2759.48,     0.0, 1520.69,
//              0.0, 2764.16, 1006.81,
//              0.0,     0.0,     1.0;
    K1 <<     1,     0.0, disp.cols/2.0,
              0.0,     1, disp.rows/2.0,
              0.0,     0.0,         1.0;

    std::cout << K1 << std::endl;

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

    Eigen::MatrixXd depths((int)points1.cols(), 1);

    if(Utils::reconstruction(R1, T1, points1, points2, depths)){
        std::cout << std::endl << "> R1, T1" << std::endl;
        IO::write_file(depths, points1);
    }else if(Utils::reconstruction(R2, T1, points1, points2, depths)){
        std::cout << std::endl << "> R2, T1" << std::endl;
        IO::write_file(depths, points1);
    }else if(Utils::reconstruction(R1, T2, points1, points2, depths)){
        std::cout << std::endl << "> R1, T2" << std::endl;
        IO::write_file(depths, points1);
    }else if(Utils::reconstruction(R2, T2, points1, points2, depths)){
        std::cout << std::endl << "> R2, T2" << std::endl;
        IO::write_file(depths, points1);
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "Processing time:" << time_diff(time_start, time_end) << std::endl;
    return 0;

}
