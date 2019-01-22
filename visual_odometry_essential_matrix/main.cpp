#include <iostream>
#include <memory>
#include <thread>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include "EssentialMatrix.h"

namespace o3d = open3d;

void CreateTransformation(
        double a_x, double a_y, double a_z,
        double t_x, double t_y, double t_z,
        double s_x, double s_y, double s_z,
        Eigen::Matrix4d& trafo) {

    // Init as identity
    trafo = Eigen::Matrix4d::Identity();

    // Translation
    trafo(0, 3) = t_x;
    trafo(1, 3) = t_y;
    trafo(2, 3) = t_z;

    // Rot x
    Eigen::Matrix4d rot_x = Eigen::Matrix4d::Identity();
    a_x *= M_PI/180.0;
    rot_x(1, 1) = cos(a_x);
    rot_x(2, 1) = sin(a_x);
    rot_x(1, 2) = -sin(a_x);
    rot_x(2, 2) = cos(a_x);

    // Rot y
    Eigen::Matrix4d rot_y = Eigen::Matrix4d::Identity();
    a_y *= M_PI/180.0;
    rot_y(0, 0) = cos(a_y);
    rot_y(2, 0) = sin(a_y);
    rot_y(0, 2) = -sin(a_y);
    rot_y(2, 2) = cos(a_y);

    // Rot z
    Eigen::Matrix4d rot_z = Eigen::Matrix4d::Identity();
    a_z *= M_PI/180.0;
    rot_z(0, 0) = cos(a_z);
    rot_z(1, 0) = sin(a_z);
    rot_z(0, 1) = -sin(a_z);
    rot_z(1, 1) = cos(a_z);

    // Scaling
    Eigen::Matrix4d scaling = Eigen::Matrix4d::Identity();
    scaling(0, 0) = s_x;
    scaling(1, 1) = s_y;
    scaling(2, 2) = s_z;
    trafo *= rot_x * rot_y * rot_z * scaling; // this can break
}

void ProjectObject(
        const Eigen::Matrix3d &K,
        const Eigen::Matrix4d &P,
        const o3d::PointCloud &object,
        o3d::PointCloud &projection)
{
    Eigen::MatrixXd I0(3, 4);
    I0 << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0;
    projection.points_.clear();
    for (const auto& X : object.points_) {
        Eigen::Vector4d Xh(X(0), X(1), X(2), 1);
        Eigen::Vector3d xh = K*I0*P*Xh;
        xh /= xh(2);
        projection.points_.emplace_back(xh);
    }
    // projection.Transform(P);
}

void ConvertPointcloudToMatrix(const o3d::PointCloud &pointcloud, // 3D points, z is ignored
                               Eigen::MatrixXd &matrix)           // homogeneous 2D coordinates (Nx3)
{
    matrix.resize(pointcloud.points_.size(), 3);
    for (size_t i = 0; i < pointcloud.points_.size(); ++i) {
        matrix.block<1, 3>(i, 0) = pointcloud.points_.at(i);
        matrix(i, 2) = 1.0;
    }
}

std::shared_ptr<o3d::PointCloud> ConvertMatrixToPointcloud(const Eigen::MatrixXd &matrix)  // (N x 3)
{
    auto pointcloud = std::make_shared<o3d::PointCloud>();
    for (int i=0; i < matrix.rows(); i++){
        Eigen::Vector3d v = matrix.block<1, 3>(i, 0);
        pointcloud->points_.push_back(v);
    }
    return pointcloud;
}

int main(int argc, char *argv[])
{
    auto trafo = std::make_shared<Eigen::Matrix4d>();
    CreateTransformation(
            -90, 0, -90,
            0, 1, -5,
            10, 10, 10, *trafo);

    CreateTransformation(
            0, 0, 0,
            0, 0, -5,
            0.1, 0.1, 0.1, *trafo);

    auto cloud1 = std::make_shared<o3d::PointCloud>();
    // std::string path = "data/head2.ply";
    //std::string path = "data/big_porsche.ply";
    std::string path = "data/ant.ply";
    o3d::ReadPointCloud(path, *cloud1);
    cloud1->NormalizeNormals();
    cloud1->PaintUniformColor(Eigen::Vector3d(0.8, 0.2, 0.3));
    cloud1->Transform(*trafo);
    o3d::EstimateNormals(*cloud1);
    cloud1->NormalizeNormals();

    // Define matrixes
    Eigen::Matrix3d K;
    K <<  2, 0, 0,
          0, 2, 0,
          0, 0, 1;

    Eigen::Matrix4d P1;
    CreateTransformation(0, 0, 0,
                         0, 0, 0,
                         1, 1, 1,
                         P1);
    Eigen::Matrix4d P2;
    CreateTransformation(0, -45, 0,
                         3, 0, -2.0,
                         1, 1, 1,
                         P2);

    auto proj_1 = std::make_shared<o3d::PointCloud>();
    auto proj_2 = std::make_shared<o3d::PointCloud>();

    auto cam_1_coords = o3d::CreateMeshCoordinateFrame();
    auto cam_2_coords = o3d::CreateMeshCoordinateFrame();

    cam_1_coords->Transform(P1);
    cam_2_coords->Transform(P2);

    ProjectObject(K, P1, *cloud1, *proj_1);
    ProjectObject(K, P2, *cloud1, *proj_2);

    // Correspondences between two camera images
    auto points_1 = std::make_shared<Eigen::MatrixXd>();
    auto points_2 = std::make_shared<Eigen::MatrixXd>();
    ConvertPointcloudToMatrix(*proj_1, *points_1);
    ConvertPointcloudToMatrix(*proj_2, *points_2);

    // Compute essential matrix
    auto ess = std::make_shared<EssentialMatrix>(K, K, *points_1, *points_2);
    ess->estimateEssentialMatrix();
    ess->constructPoses();
    auto poses = ess->getPoses();

    std::vector<std::shared_ptr<o3d::TriangleMesh>> cyls;
    cyls.resize(4);
    int i = 0;
    for (auto &cyl : cyls) {
        cyl = o3d::CreateMeshArrow(0.050, 0.1, 1.0, 0.3);
        cyl->Transform(poses->at(i));
        cyl->PaintUniformColor(Eigen::Vector3d(0.0, 0.5, 1.0));
        cyl->ComputeVertexNormals();
        i++;
    }

    auto matrix = ess->getReconstruction();
    (*matrix) = (*matrix)*1000;
    std::cout << (*matrix) << std::endl;
    auto reconstruction = ConvertMatrixToPointcloud(*matrix);
    for (auto p : reconstruction->points_) {
        std::cout << p[0] << " \t " << p[1] << " \t " << p[2] << std::endl;
    }

    o3d::DrawGeometries(
            {cam_1_coords, cam_2_coords, cloud1, //proj_1, proj_2,
             cyls[0], cyls[1], cyls[2], cyls[3],
             reconstruction},
            "PointCloud", 1600, 900);
}