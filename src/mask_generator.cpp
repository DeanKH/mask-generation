#include <maskgen/mask_generator.h>

#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "vulkan_context.h"

namespace maskgen {

class MaskGenerator::Impl {
 public:
  explicit Impl(const CameraParams& params) : params_(params) {
    ctx_ = std::make_unique<VulkanContext>(params.width, params.height);
  }

  cv::Mat Generate(const Mesh& mesh, const MeshPose& pose) {
    if (mesh.empty()) {
      throw std::runtime_error("Mesh is empty");
    }

    glm::mat4 model = ComputeModelMatrix(pose);
    glm::mat4 view = ComputeViewMatrix();
    glm::mat4 proj = ComputeProjectionMatrix();
    glm::mat4 mvp = proj * view * model;

    return ctx_->Render(mesh.vertices().data(), mesh.vertices().size() / 3,
                        mesh.indices().data(), mesh.indices().size(), mvp);
  }

 private:
  glm::mat4 ComputeModelMatrix(const MeshPose& pose) const {
    glm::mat4 model(1.0f);
    model = glm::translate(model,
                           glm::vec3(pose.tx, pose.ty, pose.tz));
    model = glm::rotate(model, static_cast<float>(pose.rz),
                        glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::rotate(model, static_cast<float>(pose.ry),
                        glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, static_cast<float>(pose.rx),
                        glm::vec3(1.0f, 0.0f, 0.0f));
    return model;
  }

  glm::mat4 ComputeViewMatrix() const {
    return glm::lookAt(
        glm::vec3(params_.eye_x, params_.eye_y, params_.eye_z),
        glm::vec3(params_.target_x, params_.target_y, params_.target_z),
        glm::vec3(params_.up_x, params_.up_y, params_.up_z));
  }

  glm::mat4 ComputeProjectionMatrix() const {
    float w = static_cast<float>(params_.width);
    float h = static_cast<float>(params_.height);
    float fx = static_cast<float>(params_.fx);
    float fy = static_cast<float>(params_.fy);
    float cx = static_cast<float>(params_.cx);
    float cy = static_cast<float>(params_.cy);
    float n = static_cast<float>(params_.near_plane);
    float f = static_cast<float>(params_.far_plane);

    glm::mat4 proj(0.0f);
    proj[0][0] = 2.0f * fx / w;
    proj[2][0] = 1.0f - 2.0f * cx / w;
    proj[1][1] = -2.0f * fy / h;
    proj[2][1] = 1.0f - 2.0f * cy / h;
    proj[2][2] = -(f + n) / (f - n);
    proj[2][3] = -1.0f;
    proj[3][2] = -2.0f * f * n / (f - n);

    return proj;
  }

  CameraParams params_;
  std::unique_ptr<VulkanContext> ctx_;
};

MaskGenerator::MaskGenerator(const CameraParams& params)
    : impl_(std::make_unique<Impl>(params)) {}

MaskGenerator::~MaskGenerator() = default;

MaskGenerator::MaskGenerator(MaskGenerator&&) noexcept = default;
MaskGenerator& MaskGenerator::operator=(MaskGenerator&&) noexcept = default;

cv::Mat MaskGenerator::Generate(const Mesh& mesh, const MeshPose& pose) {
  return impl_->Generate(mesh, pose);
}

}  // namespace maskgen
