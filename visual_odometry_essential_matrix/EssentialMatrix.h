//
// Created by engelmann on 21.01.19.
//

#ifndef VISUALODOMETRY_ESSENTIALMATRIX_H
#define VISUALODOMETRY_ESSENTIALMATRIX_H

#include <Eigen/Dense>

#include <vector>
#include <memory>

class EssentialMatrix {

public:
    Eigen::MatrixXd points1, points2;
    Eigen::Matrix3d K1, K2;
    Eigen::MatrixXd E;
    Eigen::MatrixXd U, D, V;
    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d T1, T2;

    EssentialMatrix(const Eigen::Matrix3d &K1,
                    const Eigen::Matrix3d &K2,
                    const Eigen::MatrixXd &points1,
                    const Eigen::MatrixXd &points2);

    void estimateEssentialMatrix();

    void constructPoses();

    std::shared_ptr<Eigen::MatrixXd> estimateScale(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &T);

    std::tuple<Eigen::Matrix4d, std::shared_ptr<Eigen::MatrixXd>> getReconstruction();

    Eigen::Matrix4d getPoseFromRT(const Eigen::Matrix3d &R, const Eigen::Vector3d &T);

    static void hat(Eigen::Matrix<double, 3, 1> &x, Eigen::Matrix3d &x_hat);

    static void kron(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, Eigen::MatrixXd &res);

};
#endif //VISUALODOMETRY_ESSENTIALMATRIX_H
