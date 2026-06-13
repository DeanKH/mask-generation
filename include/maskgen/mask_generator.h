#pragma once

#include <memory>

#include <maskgen/camera.h>
#include <maskgen/mesh.h>

#include <opencv2/core.hpp>

namespace maskgen {

struct MeshPose {
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  double rx = 0.0;
  double ry = 0.0;
  double rz = 0.0;
};

struct CostResult {
  float chamfer_sum = 0.0f;
  float rendered_area = 0.0f;
  float intersection = 0.0f;
};

class MaskGenerator {
 public:
  explicit MaskGenerator(const CameraParams& params);
  ~MaskGenerator();

  MaskGenerator(const MaskGenerator&) = delete;
  MaskGenerator& operator=(const MaskGenerator&) = delete;
  MaskGenerator(MaskGenerator&&) noexcept;
  MaskGenerator& operator=(MaskGenerator&&) noexcept;

  cv::Mat Generate(const Mesh& mesh, const MeshPose& pose);

  void SetMesh(const Mesh& mesh);
  cv::Mat GeneratePose(const MeshPose& pose);

  void SetCostInputs(const cv::Mat& dt_input, const cv::Mat& input_mask);
  CostResult GeneratePoseWithCost(const MeshPose& pose, float scale_factor);
  bool HasComputeCost() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace maskgen
