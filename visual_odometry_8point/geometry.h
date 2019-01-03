//
// Created by Francis Engelmann on 2018-12-31.
//

#ifndef VISUALODOMETRY_GEOMETRY_H
#define VISUALODOMETRY_GEOMETRY_H

#include <iostream>
#include <eigen3/Eigen/Dense>

class Geometry {

public:
    static void hat(Eigen::Matrix<double, 3, 1>& x, Eigen::Matrix3d& x_hat){
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

    static void kron(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, Eigen::MatrixXd& res){
        res.resize(a.rows()*b.rows(), a.cols()*b.cols());
        for (int x=0; x<a.rows(); x++) {
            for (int y=0; y<a.cols(); y++) {
                res.block(x*b.rows(), y*b.cols(), b.rows(), b.cols()) = a(x, y) * b;
            }
        }
    }

    static void kron_test(){
        Eigen::Matrix2d a, b;
        Eigen::MatrixXd c;
        a << 1, 2, 3, 4;
        b << 1, 2, 3, 4;
        kron(a, b, c);
        std::cout << "a:" << std::endl << a << std::endl;
        std::cout << "b:" << std::endl << b << std::endl;
        std::cout << "c:" << std::endl << c << std::endl;
    }
};


#endif //VISUALODOMETRY_GEOMETRY_H
