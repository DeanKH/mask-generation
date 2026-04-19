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
};

}  // namespace maskgen
