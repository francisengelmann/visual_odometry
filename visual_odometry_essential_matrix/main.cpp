#include <iostream>
#include <memory>
#include <thread>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include "EssentialMatrix.h"

namespace o3d = open3d;

std::shared_ptr<Eigen::Matrix4d> CreateTransformation(
        Eigen::Vector3d a, Eigen::Vector3d t, Eigen::Vector3d s) {

    auto res = std::make_shared<Eigen::Matrix4d>();

    // Init as identity
    *res = Eigen::Matrix4d::Identity();

    // Translation
    (*res)(0, 3) = t[0];
    (*res)(1, 3) = t[1];
    (*res)(2, 3) = t[2];

    // Rot x
    Eigen::Matrix4d rot_x = Eigen::Matrix4d::Identity();
    a[0] *= M_PI/180.0;
    rot_x(1, 1) = cos(a[0]);
    rot_x(2, 1) = sin(a[0]);
    rot_x(1, 2) = -sin(a[0]);
    rot_x(2, 2) = cos(a[0]);

    // Rot y
    Eigen::Matrix4d rot_y = Eigen::Matrix4d::Identity();
    a[1] *= M_PI/180.0;
    rot_y(0, 0) = cos(a[1]);
    rot_y(2, 0) = sin(a[1]);
    rot_y(0, 2) = -sin(a[1]);
    rot_y(2, 2) = cos(a[1]);

    // Rot z
    Eigen::Matrix4d rot_z = Eigen::Matrix4d::Identity();
    a[2] *= M_PI/180.0;
    rot_z(0, 0) = cos(a[2]);
    rot_z(1, 0) = sin(a[2]);
    rot_z(0, 1) = -sin(a[2]);
    rot_z(1, 1) = cos(a[2]);

    // Scaling
    Eigen::Matrix4d scaling = Eigen::Matrix4d::Identity();
    scaling(0, 0) = s[0];
    scaling(1, 1) = s[1];
    scaling(2, 2) = s[2];

    *res *= rot_x * rot_y * rot_z * scaling; // this can break
    return res;
}

std::shared_ptr<o3d::PointCloud> ProjectObject(const Eigen::Matrix3d &K,
                                               const Eigen::Matrix4d &P,
                                               const o3d::PointCloud &object)
{
    auto projection = std::make_shared<o3d::PointCloud>();
    Eigen::MatrixXd I0(3, 4);
    I0 << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0;
    projection->points_.clear();
    for (const auto& X : object.points_) {
        Eigen::Vector4d Xh(X(0), X(1), X(2), 1);
        Eigen::Vector3d xh = K*I0*P*Xh;
        xh /= xh(2);
        projection->points_.emplace_back(xh);
    }
}

std::shared_ptr<Eigen::MatrixXd> ConvertPointcloudToMatrix(
        const o3d::PointCloud &pointcloud) // 3D points, z is ignored
{
    auto matrix = std::make_shared<Eigen::MatrixXd>();
    matrix->resize(pointcloud.points_.size(), 3);
    for (size_t i = 0; i < pointcloud.points_.size(); ++i) {
        matrix->block<1, 3>(i, 0) = pointcloud.points_.at(i);
        (*matrix)(i, 2) = 1.0;
    }
    return matrix;
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
    // Read ground truth point cloud
    auto cloud1 = std::make_shared<o3d::PointCloud>();
    std::string path = "data/ant.ply"; // {head2, big_porsche}.ply
    o3d::ReadPointCloud(path, *cloud1);
    cloud1->PaintUniformColor(Eigen::Vector3d(0.8, 0.2, 0.3));
    auto trafo = CreateTransformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 5), Eigen::Vector3d(0.1, 0.1, 0.1));
    cloud1->Transform(*trafo);

    // Camera/intrinsics matrix (here same for both cameras)
    Eigen::Matrix3d K;
    K <<  2, 0, 0,
          0, 2, 0,
          0, 0, 1;

    // Pose of camera 1
    Eigen::Matrix4d P1 = *CreateTransformation(Eigen::Vector3d(0, 0, 0),
                                               Eigen::Vector3d(0, 0, 0),
                                               Eigen::Vector3d(1, 1, 1));
    // Pose of camera 2
    Eigen::Matrix4d P2 = *CreateTransformation(Eigen::Vector3d(30,  315, 0),
                                               Eigen::Vector3d(-3, 2, 2.0),
                                               Eigen::Vector3d(1, 1, 1));

    // Project 3D to 2D to synthesize correspondences between two camera images
    auto proj_1 = ProjectObject(K, P1, *cloud1);
    auto proj_2 = ProjectObject(K, P2, *cloud1);
    auto points_1 = ConvertPointcloudToMatrix(*proj_1);
    auto points_2 = ConvertPointcloudToMatrix(*proj_2);

    // Compute essential matrix
    auto ess = std::make_shared<EssentialMatrix>(K, K, *points_1, *points_2);
    ess->estimateEssentialMatrix();
    ess->constructPoses();
    auto poses = ess->getPoses();
    auto reconstruction = ess->getReconstruction();

    std::vector<std::shared_ptr<o3d::TriangleMesh>> cyls;
    cyls.resize(4);
    int i = 0;
    for (auto &cyl : cyls) {
        cyl = o3d::CreateMeshArrow(0.03, 0.08, 0.7, 0.3);
        cyl->Transform(poses->at(i));
        cyl->PaintUniformColor(Eigen::Vector3d(0.0, 0.5, 1.0));
        cyl->ComputeVertexNormals();
        i++;
    }
    
    // Visualization
    auto pointcloud = ConvertMatrixToPointcloud(*reconstruction);
    auto cam_1_coords = o3d::CreateMeshCoordinateFrame();
    auto cam_2_coords = o3d::CreateMeshCoordinateFrame();
    cam_1_coords->Transform(P1);
    cam_2_coords->Transform(P2);
    o3d::DrawGeometries(
            {cam_1_coords, cam_2_coords, cloud1, //proj_1, proj_2,
             //cyls[0], cyls[1],
             cyls[2], //cyls[3],
             pointcloud
             },
            "PointCloud", 1600, 900);
}