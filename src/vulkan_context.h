#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <opencv2/core.hpp>

namespace maskgen {

class VulkanContext {
 public:
  VulkanContext(int width, int height);
  ~VulkanContext();

  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  cv::Mat Render(const float* vertices, size_t vertex_count, const uint32_t* indices,
                 size_t index_count, const glm::mat4& mvp);

  void UploadMesh(const float* vertices, size_t vertex_count, const uint32_t* indices,
                  size_t index_count);
  cv::Mat RenderPose(const glm::mat4& mvp);
  void CleanupMeshBuffers();

  void SetCostInputs(const cv::Mat& dt_input, const cv::Mat& input_mask);
  void RenderPoseWithCost(const glm::mat4& mvp, float scale_factor,
                          float& chamfer_sum, float& rendered_area,
                          float& intersection);
  bool HasComputeCost() const { return has_compute_cost_; }

 private:
  void CreateInstance();
  void PickPhysicalDevice();
  void CreateLogicalDevice();
  void CreateCommandPool();
  void CompileShaders();
  void CreateRenderPass();
  void CreatePipelineLayout();
  void CreatePipeline();
  void CreateOffscreenResources();
  void CreateFence();
  void CleanupOffscreenResources();

  void CreateComputeResources();
  void CleanupComputeResources();

  uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);
  void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer* buffer,
                    VkDeviceMemory* memory);
  void CreateImage(VkFormat format, VkImageUsageFlags usage, VkImage* image,
                   VkDeviceMemory* memory);
  VkImageView CreateImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspect_flags);
  VkShaderModule CreateShaderModule(const std::vector<uint32_t>& code);
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  uint32_t queue_family_index_ = 0;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkImage color_image_ = VK_NULL_HANDLE;
  VkDeviceMemory color_image_memory_ = VK_NULL_HANDLE;
  VkImageView color_image_view_ = VK_NULL_HANDLE;
  VkImage depth_image_ = VK_NULL_HANDLE;
  VkDeviceMemory depth_image_memory_ = VK_NULL_HANDLE;
  VkImageView depth_image_view_ = VK_NULL_HANDLE;
  VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  int width_;
  int height_;

  std::vector<uint32_t> vert_spv_;
  std::vector<uint32_t> frag_spv_;

  VkBuffer persistent_vertex_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory persistent_vertex_memory_ = VK_NULL_HANDLE;
  VkBuffer persistent_index_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory persistent_index_memory_ = VK_NULL_HANDLE;
  VkBuffer persistent_readback_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory persistent_readback_memory_ = VK_NULL_HANDLE;
  bool has_persistent_mesh_ = false;
  uint32_t persistent_index_count_ = 0;

  bool has_compute_cost_ = false;

  std::vector<uint32_t> cost_compute_spv_;
  std::vector<uint32_t> cost_reduce_spv_;

  VkDescriptorSetLayout compute_desc_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool compute_desc_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet compute_desc_set_ = VK_NULL_HANDLE;
  VkPipelineLayout compute_pipeline_layout_ = VK_NULL_HANDLE;

  VkPipeline cost_compute_pipeline_ = VK_NULL_HANDLE;
  VkPipeline cost_reduce_pipeline_ = VK_NULL_HANDLE;

  VkBuffer dt_input_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory dt_input_buffer_memory_ = VK_NULL_HANDLE;
  VkBuffer input_mask_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory input_mask_buffer_memory_ = VK_NULL_HANDLE;

  VkBuffer cost_output_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory cost_output_memory_ = VK_NULL_HANDLE;

  VkBuffer partial_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory partial_buffer_memory_ = VK_NULL_HANDLE;
  int num_partial_groups_ = 0;
};

}  // namespace maskgen
