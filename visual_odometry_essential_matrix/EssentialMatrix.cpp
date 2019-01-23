//
// Created by engelmann on 21.01.19.
//


#include "EssentialMatrix.h"
#include <iostream>
#include <memory>

EssentialMatrix::EssentialMatrix(const Eigen::Matrix3d &K1,
                                 const Eigen::Matrix3d &K2,
                                 const Eigen::MatrixXd &points1,
                                 const Eigen::MatrixXd &points2)
{
    this->K1 = K1;
    this->K2 = K2;
    this->points1 = points1;
    this->points2 = points2;
}

void EssentialMatrix::estimateEssentialMatrix()
{
    auto K1_inv = this->K1.inverse();
    auto K2_inv = this->K2.inverse();

    // Transform image coordinates with inverse camera matrices:
    this->points1 = K1_inv * this->points1.transpose();  // shape: [3, N]
    this->points2 = K2_inv * this->points2.transpose();  // shape: [3, N]

    // Compute constraint matrix A
    int num_points = (int)points1.cols();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_points, 9);
    for (int i=0; i<num_points; i++){
        auto a = points1.col(i);
        auto b = points2.col(i);
        Eigen::MatrixXd kr;
        EssentialMatrix::kron(a, b, kr);
        A.row(i) = kr.transpose();
    }

    // Find minimizer for A*E:
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    this->E = svd.matrixV().col(8);
    this->E.resize(3, 3);

    // SVD E
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(this->E, Eigen::ComputeFullV | Eigen::ComputeFullU );
    //Eigen::MatrixXd U, D, V;
    this->U = svd2.matrixU();
    this->D = svd2.singularValues().asDiagonal();
    this->V = svd2.matrixV();

    if (this->U.determinant() < 0 || this->V.determinant() < 0){
        Eigen::JacobiSVD<Eigen::MatrixXd> svd3(this->E*-1, Eigen::ComputeFullV | Eigen::ComputeFullU );
        this->U = svd3.matrixU();
        this->D = svd3.singularValues().asDiagonal();
        this->V = svd3.matrixV();
    }

    this->D(0, 0) = 1.0;
    this->D(1, 1) = 1.0;
    this->D(2, 2) = 0.0;

    // Final essential matrix
    this->E = this->U * this->D * this->V.transpose();
}

void EssentialMatrix::constructPoses()
{
    Eigen::Matrix3d Rz1;
    Rz1 << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Rz1 = Rz1.transpose().eval();
    Eigen::Matrix3d Rz2;
    Rz2 << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Rz2 = Rz2.transpose().eval();

    this->R1 = this->U * Rz1.transpose() * this->V.transpose();
    this->R2 = this->U * Rz2.transpose() * this->V.transpose();

    Eigen::Matrix3d T_hat1 = this->U * Rz1 * this->D * this->U.transpose();
    Eigen::Matrix3d T_hat2 = this->U * Rz2 * this->D * this->U.transpose();

    this->T1 << -T_hat1(1, 2), T_hat1(0, 2), -T_hat1(0, 1);
    this->T2 << -T_hat2(1, 2), T_hat2(0, 2), -T_hat2(0, 1);
}

std::shared_ptr<Eigen::MatrixXd> EssentialMatrix::estimateScale(
        const Eigen::Matrix<double, 3, 3> &R,
        const Eigen::Matrix<double, 3, 1> &T)
{
    int num_points = (int)(this->points1.cols());
    std::shared_ptr<Eigen::MatrixXd> M = std::make_shared<Eigen::MatrixXd>();
    M->resize(3 * num_points, num_points + 1);
    M->setZero();

    for (int i=0; i<num_points; i++) {
        Eigen::Matrix3d x2_hat;
        Eigen::Matrix<double, 3, 1> x2 = this->points2.col(i);
        Eigen::Matrix<double, 3, 1> x1 = this->points1.col(i);
        EssentialMatrix::hat(x2, x2_hat);
        M->block(3*i, i, 3, 1) = x2_hat * R * x1;
        M->block(3*i, num_points, 3, 1) = x2_hat * T;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(*M, Eigen::ComputeFullV);
    std::shared_ptr<Eigen::MatrixXd> V = std::make_shared<Eigen::MatrixXd>();
    *V = svd.matrixV().col(num_points);
    return V;
}

Eigen::Matrix4d EssentialMatrix::getPoseFromRT(const Eigen::Matrix3d &R, const Eigen::Vector3d &T)
{
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity(4, 4);
    P.block<3, 3>(0, 0) = R;
    P.block<3, 1>(0, 3) = T;
    return P;
}

std::tuple<Eigen::Matrix4d, std::shared_ptr<Eigen::MatrixXd>>  EssentialMatrix::getReconstruction()
{
    long num_points = this->points2.cols();
    auto reconstruction = std::make_shared<Eigen::MatrixXd>();
    auto scales = std::make_shared<Eigen::MatrixXd>();
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity(4, 4);
    reconstruction->resize(num_points, 3);

    // Iterate over four potential pose solutions, find correct one using positive depths constraint
    for (int p = 0; p < 4; p++) {
        if (p == 0) { scales = estimateScale(this->R1, this->T1); pose = getPoseFromRT(this->R1, this->T1); }
        if (p == 1) { scales = estimateScale(this->R1, this->T2); pose = getPoseFromRT(this->R1, this->T2); }
        if (p == 2) { scales = estimateScale(this->R2, this->T1); pose = getPoseFromRT(this->R2, this->T1); }
        if (p == 3) { scales = estimateScale(this->R2, this->T2); pose = getPoseFromRT(this->R2, this->T2); }

        bool allPos = true;
        for (int i = 0; i < num_points + 1 && allPos; i++) {
            allPos = ((*scales)(i, 0) > 0);
            std::cout << (*scales)(i, 0) << std::endl;
        }
        if (!allPos) { std::cout << "Negative depth for p = " << p << std::endl; continue; }

        double gamma = (*scales)(num_points, 0);
        std::cout << "gamma = " << gamma << std::endl;
        for (size_t i = 0; i < num_points; i++){
            (*reconstruction)(num_points * 0 + i, 0) = this->points1(0, i) * (*scales)(i, 0) / gamma;
            (*reconstruction)(num_points * 0 + i, 1) = this->points1(1, i) * (*scales)(i, 0) / gamma;
            (*reconstruction)(num_points * 0 + i, 2) = (*scales)(i, 0) / gamma;
        }
        break;
    }

    return std::make_tuple(pose, reconstruction);
}


void EssentialMatrix::hat(
        Eigen::Matrix<double, 3, 1> &x,
        Eigen::Matrix3d &x_hat)
{
    x_hat(0, 0) = 0.0;
    x_hat(1, 0) = x(2, 0);
    x_hat(2, 0) = -x(1, 0);

    x_hat(0, 1) = -x(2, 0);
    x_hat(1, 1) = 0.0;
    x_hat(2, 1) = x(0, 0);

    x_hat(0, 2) = x(1, 0);
    x_hat(1, 2) = -x(0, 0);
    x_hat(2, 2) = 0.0;
}

void EssentialMatrix::kron(
        const Eigen::MatrixXd &a,
        const Eigen::MatrixXd &b,
        Eigen::MatrixXd &res)
{
    res.resize(a.rows()*b.rows(), a.cols()*b.cols());
    for (int x=0; x<a.rows(); x++) {
        for (int y=0; y<a.cols(); y++) {
            res.block(x*b.rows(), y*b.cols(), b.rows(), b.cols()) = a(x, y) * b;
        }
    }
}
