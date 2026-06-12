#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <maskgen/camera.h>
#include <maskgen/mask_generator.h>
#include <maskgen/mesh.h>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <fstream>
#include <string>

TEST_CASE("Benchmark MaskGenerator with model/ad.step", "[benchmark]") {
  const std::string kMeshPath = BENCHMARK_MESH_PATH;
  constexpr float kMeshScale = 0.001f;

  maskgen::CameraParams params;
  {
    std::ifstream ifs(BENCHMARK_CAMERA_PATH);
    REQUIRE(ifs);
    nlohmann::json j;
    ifs >> j;
    params.width = j.value("width", 640);
    params.height = j.value("height", 480);
    params.fx = j.value("fx", 500.0);
    params.fy = j.value("fy", 500.0);
    params.cx = j.value("cx", 320.0);
    params.cy = j.value("cy", 240.0);
  }

  maskgen::MeshPose pose;

  maskgen::Mesh mesh;
  REQUIRE(mesh.LoadFromFile(kMeshPath, kMeshScale));

  maskgen::MaskGenerator generator(params);

  BENCHMARK("Generate (upload + render)") {
    cv::Mat mask = generator.Generate(mesh, pose);
    return mask.rows;
  };

  BENCHMARK("GeneratePose (render only)") {
    generator.SetMesh(mesh);
    cv::Mat mask = generator.GeneratePose(pose);
    return mask.rows;
  };
}
