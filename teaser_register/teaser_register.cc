#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cstdlib>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>

#include <teaser/registration.h>
#include <teaser/matcher.h>
#include <teaser/geometry.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

using PointT = pcl::PointXYZRGB;
using NormalT = pcl::Normal;
using FeatureT = pcl::FPFHSignature33;

using CloudT = pcl::PointCloud<PointT>;
using NormalCloudT = pcl::PointCloud<NormalT>;
using FeatureCloudT = pcl::PointCloud<FeatureT>;

// Загрузка с проверкой RGB
CloudT::Ptr loadCloudRGB(const std::string &path, bool &has_rgb)
{
    pcl::PCLPointCloud2 blob;
    if (pcl::io::loadPCDFile(path, blob) < 0)
    {
        std::cerr << "[!] Не удалось прочитать " << path << "\n";
        return nullptr;
    }

    has_rgb = false;
    for (const auto &f : blob.fields)
    {
        if (f.name == "rgb" || f.name == "rgba")
        {
            has_rgb = true;
            break;
        }
    }

    CloudT::Ptr cloud(new CloudT);
    pcl::fromPCLPointCloud2(blob, *cloud);

    std::cout << "[load] " << path << ": " << cloud->size() << " точек"
              << (has_rgb ? " (с цветом)" : " (без цвета — будет чёрным)") << "\n";
    return cloud;
}

teaser::PointCloud toTeaserCloud(const CloudT &cloud)
{
    teaser::PointCloud out;
    out.reserve(cloud.size());
    for (const auto &p : cloud)
    {
        out.push_back({static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z)});
    }
    return out;
}

void printMatrix(const Eigen::Matrix4d &T, const std::string &name)
{
    std::cout << name << ":\n";
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
            std::cout << std::setw(12) << std::setprecision(6) << std::fixed << T(i, j) << " ";
        std::cout << "\n";
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " src.pcd tgt.pcd [output.pcd] [voxel_leaf] [noise_bound]\n";
        return 1;
    }
    const std::string src_path = argv[1];
    const std::string tgt_path = argv[2];
    const std::string out_path = (argc > 3) ? argv[3] : "aligned.pcd";
    const float voxel_leaf = (argc > 4) ? std::atof(argv[4]) : 0.3f;
    const double noise_bound = (argc > 5) ? std::atof(argv[5]) : 1.0;

    const float normal_radius = 2.5f * voxel_leaf;
    const float fpfh_radius = 5.0f * voxel_leaf;

    auto t0 = std::chrono::steady_clock::now();

    // Загрузка (с цветами)
    bool src_has_rgb = false, tgt_has_rgb = false;
    CloudT::Ptr src_full = loadCloudRGB(src_path, src_has_rgb);
    CloudT::Ptr tgt_full = loadCloudRGB(tgt_path, tgt_has_rgb);
    if (!src_full || !tgt_full)
        return 1;

    // Даунсэмплинг (цвета сохраняются автоматически)
    CloudT::Ptr src(new CloudT);
    CloudT::Ptr tgt(new CloudT);
    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
    vg.setInputCloud(src_full);
    vg.filter(*src);
    vg.setInputCloud(tgt_full);
    vg.filter(*tgt);
    std::cout << "[voxel " << voxel_leaf << "] src=" << src->size() << ", tgt=" << tgt->size()
              << "\n";

    // Нормали
    NormalCloudT::Ptr src_n(new NormalCloudT);
    NormalCloudT::Ptr tgt_n(new NormalCloudT);
    pcl::NormalEstimationOMP<PointT, NormalT> ne;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(normal_radius);
    ne.setInputCloud(src);
    ne.compute(*src_n);
    ne.setInputCloud(tgt);
    ne.compute(*tgt_n);

    // FPFH
    FeatureCloudT::Ptr src_f(new FeatureCloudT);
    FeatureCloudT::Ptr tgt_f(new FeatureCloudT);
    pcl::FPFHEstimationOMP<PointT, NormalT, FeatureT> fpfh;
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(fpfh_radius);
    fpfh.setInputCloud(src);
    fpfh.setInputNormals(src_n);
    fpfh.compute(*src_f);
    fpfh.setInputCloud(tgt);
    fpfh.setInputNormals(tgt_n);
    fpfh.compute(*tgt_f);
    std::cout << "[fpfh r=" << fpfh_radius << "] computed\n";

    // Соответствия через teaser::Matcher
    auto src_t = toTeaserCloud(*src);
    auto tgt_t = toTeaserCloud(*tgt);

    teaser::Matcher matcher;
    auto correspondences =
        matcher.calculateCorrespondences(src_t, tgt_t, *src_f, *tgt_f, false, true, false, 0.95f);
    std::cout << "[match] пар-кандидатов: " << correspondences.size() << "\n";

    if (correspondences.size() < 3)
    {
        std::cerr << "[!] Слишком мало пар — TEASER++ не сойдётся\n";
        return 1;
    }

    // TEASER++
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = noise_bound;
    params.cbar2 = 1.0;
    params.estimate_scaling = false;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_gnc_factor = 1.4;
    params.rotation_max_iterations = 100;
    params.rotation_cost_threshold = 1e-12;

    teaser::RobustRegistrationSolver solver(params);
    auto solution = solver.solve(src_t, tgt_t, correspondences);

    Eigen::Matrix4d T_teaser = Eigen::Matrix4d::Identity();
    T_teaser.block<3, 3>(0, 0) = solution.rotation;
    T_teaser.block<3, 1>(0, 3) = solution.translation;

    auto t_teaser = std::chrono::steady_clock::now();
    std::cout << "[teaser] готово за " << std::chrono::duration<double>(t_teaser - t0).count()
              << " s\n";
    printMatrix(T_teaser, "[teaser] T");

    Eigen::Matrix4d T_final = T_teaser;
    printMatrix(T_final, "[final] T = TEASER");

    // Сохранение (цвета сохраняются автоматически)
    CloudT::Ptr src_final(new CloudT);
    pcl::transformPointCloud(*src_full, *src_final, T_final.cast<float>());

    CloudT merged;
    merged.reserve(src_final->size() + tgt_full->size());
    for (const auto &p : *src_final)
        merged.push_back(p);
    for (const auto &p : *tgt_full)
        merged.push_back(p);
    merged.width = merged.size();
    merged.height = 1;
    merged.is_dense = true;

    pcl::io::savePCDFileBinary(out_path, merged);
    std::cout << "[save] " << out_path << " (" << merged.size() << " точек)\n";

    if (!src_has_rgb || !tgt_has_rgb)
    {
        std::cout << "[warn] Одно из облаков было без RGB — в merged эти точки будут чёрными\n";
    }

    auto t_end = std::chrono::steady_clock::now();
    std::cout << "[total] " << std::chrono::duration<double>(t_end - t0).count() << " s\n";
    return 0;
}
