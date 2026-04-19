#include "vulkan_context.h"

#include <array>
#include <cstring>
#include <stdexcept>

#include <shaderc/shaderc.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace maskgen {

static const char* kVertSource = R"(#version 450
layout(push_constant) uniform PushConstants {
  mat4 mvp;
} pc;
layout(location = 0) in vec3 inPosition;
void main() {
  gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
)";

static const char* kFragSource = R"(#version 450
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)";

static std::vector<uint32_t> CompileGlsl(const std::string& source,
                                          shaderc_shader_kind kind) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;
  options.SetOptimizationLevel(shaderc_optimization_level_performance);

  auto result = compiler.CompileGlslToSpv(source, kind, "shader.glsl", options);
  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    throw std::runtime_error("Shader compilation failed: " + result.GetErrorMessage());
  }
  return {result.cbegin(), result.cend()};
}

VulkanContext::VulkanContext(int width, int height) : width_(width), height_(height) {
  CompileShaders();
  CreateInstance();
  PickPhysicalDevice();
  CreateLogicalDevice();
  CreateCommandPool();
  CreateRenderPass();
  CreatePipelineLayout();
  CreatePipeline();
  CreateOffscreenResources();
  CreateFence();
}

VulkanContext::~VulkanContext() {
  CleanupOffscreenResources();

  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_, fence_, nullptr);
  }
  vkDestroyPipeline(device_, pipeline_, nullptr);
  vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
  vkDestroyRenderPass(device_, render_pass_, nullptr);
  vkDestroyCommandPool(device_, command_pool_, nullptr);
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
  }
}

void VulkanContext::CompileShaders() {
  vert_spv_ = CompileGlsl(kVertSource, shaderc_vertex_shader);
  frag_spv_ = CompileGlsl(kFragSource, shaderc_fragment_shader);
}

void VulkanContext::CreateInstance() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "maskgen";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.pEngineName = "maskgen";
  app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  app_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan instance: " +
                             std::to_string(result));
  }
}

void VulkanContext::PickPhysicalDevice() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  if (device_count == 0) {
    throw std::runtime_error("No Vulkan-capable GPUs found");
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  for (const auto& device : devices) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
      if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        physical_device_ = device;
        queue_family_index_ = i;
        return;
      }
    }
  }

  throw std::runtime_error("No suitable GPU found with graphics queue");
}

void VulkanContext::CreateLogicalDevice() {
  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = queue_family_index_;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.queueCreateInfoCount = 1;

  VkResult result = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to create logical device: " +
                             std::to_string(result));
  }

  vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);
}

void VulkanContext::CreateCommandPool() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = queue_family_index_;

  if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool");
  }
}

void VulkanContext::CreateRenderPass() {
  std::array<VkAttachmentDescription, 2> attachments{};

  attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
  attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

  attachments[1].format = VK_FORMAT_D32_SFLOAT;
  attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
  attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_ref{};
  color_ref.attachment = 0;
  color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_ref{};
  depth_ref.attachment = 1;
  depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_ref;
  subpass.pDepthStencilAttachment = &depth_ref;

  VkRenderPassCreateInfo rp_info{};
  rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  rp_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  rp_info.pAttachments = attachments.data();
  rp_info.subpassCount = 1;
  rp_info.pSubpasses = &subpass;

  if (vkCreateRenderPass(device_, &rp_info, nullptr, &render_pass_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create render pass");
  }
}

void VulkanContext::CreatePipelineLayout() {
  VkPushConstantRange push_constant{};
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  push_constant.offset = 0;
  push_constant.size = sizeof(glm::mat4);

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_constant;

  if (vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout");
  }
}

VkShaderModule VulkanContext::CreateShaderModule(const std::vector<uint32_t>& code) {
  VkShaderModuleCreateInfo module_info{};
  module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_info.codeSize = code.size() * sizeof(uint32_t);
  module_info.pCode = code.data();

  VkShaderModule module;
  if (vkCreateShaderModule(device_, &module_info, nullptr, &module) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module");
  }
  return module;
}

void VulkanContext::CreatePipeline() {
  VkShaderModule vert_module = CreateShaderModule(vert_spv_);
  VkShaderModule frag_module = CreateShaderModule(frag_spv_);

  std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert_module;
  stages[0].pName = "main";

  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag_module;
  stages[1].pName = "main";

  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = 3 * sizeof(float);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attribute{};
  attribute.binding = 0;
  attribute.location = 0;
  attribute.format = VK_FORMAT_R32G32B32_SFLOAT;
  attribute.offset = 0;

  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &binding;
  vertex_input.vertexAttributeDescriptionCount = 1;
  vertex_input.pVertexAttributeDescriptions = &attribute;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(width_);
  viewport.height = static_cast<float>(height_);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState blend_attachment{};
  blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blend_attachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &blend_attachment;

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = static_cast<uint32_t>(stages.size());
  pipeline_info.pStages = stages.data();
  pipeline_info.pVertexInputState = &vertex_input;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_stencil;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.layout = pipeline_layout_;
  pipeline_info.renderPass = render_pass_;
  pipeline_info.subpass = 0;

  if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
                                &pipeline_) != VK_SUCCESS) {
    vkDestroyShaderModule(device_, vert_module, nullptr);
    vkDestroyShaderModule(device_, frag_module, nullptr);
    throw std::runtime_error("Failed to create graphics pipeline");
  }

  vkDestroyShaderModule(device_, vert_module, nullptr);
  vkDestroyShaderModule(device_, frag_module, nullptr);
}

uint32_t VulkanContext::FindMemoryType(uint32_t type_filter,
                                       VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanContext::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags properties, VkBuffer* buffer,
                                 VkDeviceMemory* memory) {
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device_, &buffer_info, nullptr, buffer) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create buffer");
  }

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device_, *buffer, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = FindMemoryType(mem_reqs.memoryTypeBits, properties);

  if (vkAllocateMemory(device_, &alloc_info, nullptr, memory) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate buffer memory");
  }

  vkBindBufferMemory(device_, *buffer, *memory, 0);
}

void VulkanContext::CreateImage(VkFormat format, VkImageUsageFlags usage, VkImage* image,
                                VkDeviceMemory* memory) {
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = format;
  image_info.extent = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (vkCreateImage(device_, &image_info, nullptr, image) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image");
  }

  VkMemoryRequirements mem_reqs;
  vkGetImageMemoryRequirements(device_, *image, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = FindMemoryType(mem_reqs.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device_, &alloc_info, nullptr, memory) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate image memory");
  }

  vkBindImageMemory(device_, *image, *memory, 0);
}

VkImageView VulkanContext::CreateImageView(VkImage image, VkFormat format,
                                           VkImageAspectFlags aspect_flags) {
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.subresourceRange.aspectMask = aspect_flags;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.layerCount = 1;

  VkImageView view;
  if (vkCreateImageView(device_, &view_info, nullptr, &view) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image view");
  }
  return view;
}

void VulkanContext::CreateOffscreenResources() {
  CreateImage(VK_FORMAT_R8G8B8A8_UNORM,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
              &color_image_, &color_image_memory_);
  color_image_view_ =
      CreateImageView(color_image_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

  CreateImage(VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              &depth_image_, &depth_image_memory_);
  depth_image_view_ =
      CreateImageView(depth_image_, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);

  std::array<VkImageView, 2> attachments = {color_image_view_, depth_image_view_};
  VkFramebufferCreateInfo fb_info{};
  fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fb_info.renderPass = render_pass_;
  fb_info.attachmentCount = static_cast<uint32_t>(attachments.size());
  fb_info.pAttachments = attachments.data();
  fb_info.width = static_cast<uint32_t>(width_);
  fb_info.height = static_cast<uint32_t>(height_);
  fb_info.layers = 1;

  if (vkCreateFramebuffer(device_, &fb_info, nullptr, &framebuffer_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create framebuffer");
  }
}

void VulkanContext::CleanupOffscreenResources() {
  if (framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(device_, framebuffer_, nullptr);
    framebuffer_ = VK_NULL_HANDLE;
  }
  if (depth_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, depth_image_view_, nullptr);
    depth_image_view_ = VK_NULL_HANDLE;
  }
  if (depth_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device_, depth_image_, nullptr);
    depth_image_ = VK_NULL_HANDLE;
  }
  if (depth_image_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, depth_image_memory_, nullptr);
    depth_image_memory_ = VK_NULL_HANDLE;
  }
  if (color_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, color_image_view_, nullptr);
    color_image_view_ = VK_NULL_HANDLE;
  }
  if (color_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device_, color_image_, nullptr);
    color_image_ = VK_NULL_HANDLE;
  }
  if (color_image_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, color_image_memory_, nullptr);
    color_image_memory_ = VK_NULL_HANDLE;
  }
}

void VulkanContext::CreateFence() {
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  if (vkCreateFence(device_, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create fence");
  }
}

static void TransitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                  VkImageLayout old_layout, VkImageLayout new_layout,
                                  VkAccessFlags src_access, VkAccessFlags dst_access,
                                  VkPipelineStageFlags src_stage,
                                  VkPipelineStageFlags dst_stage) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);
}

cv::Mat VulkanContext::Render(const float* vertices, size_t vertex_count,
                              const uint32_t* indices, size_t index_count,
                              const glm::mat4& mvp) {
  VkDeviceSize vertex_buffer_size = static_cast<VkDeviceSize>(vertex_count * 3) * sizeof(float);
  VkDeviceSize index_buffer_size = static_cast<VkDeviceSize>(index_count) * sizeof(uint32_t);
  VkDeviceSize image_size =
      static_cast<VkDeviceSize>(width_) * height_ * 4;

  VkBuffer vertex_buffer = VK_NULL_HANDLE;
  VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
  VkBuffer index_buffer = VK_NULL_HANDLE;
  VkDeviceMemory index_memory = VK_NULL_HANDLE;
  VkBuffer readback_buffer = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory = VK_NULL_HANDLE;

  CreateBuffer(vertex_buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &vertex_buffer, &vertex_memory);
  CreateBuffer(index_buffer_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &index_buffer, &index_memory);
  CreateBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &readback_buffer, &readback_memory);

  void* data = nullptr;
  vkMapMemory(device_, vertex_memory, 0, vertex_buffer_size, 0, &data);
  std::memcpy(data, vertices, static_cast<size_t>(vertex_buffer_size));
  vkUnmapMemory(device_, vertex_memory);

  vkMapMemory(device_, index_memory, 0, index_buffer_size, 0, &data);
  std::memcpy(data, indices, static_cast<size_t>(index_buffer_size));
  vkUnmapMemory(device_, index_memory);

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = command_pool_;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;

  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(device_, &cmd_alloc, &cmd) != VK_SUCCESS) {
    vkDestroyBuffer(device_, readback_buffer, nullptr);
    vkFreeMemory(device_, readback_memory, nullptr);
    vkDestroyBuffer(device_, index_buffer, nullptr);
    vkFreeMemory(device_, index_memory, nullptr);
    vkDestroyBuffer(device_, vertex_buffer, nullptr);
    vkFreeMemory(device_, vertex_memory, nullptr);
    throw std::runtime_error("Failed to allocate command buffer");
  }

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin_info);

  TransitionImageLayout(cmd, color_image_, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

  VkImageMemoryBarrier depth_barrier{};
  depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  depth_barrier.srcAccessMask = 0;
  depth_barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  depth_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depth_barrier.image = depth_image_;
  depth_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  depth_barrier.subresourceRange.levelCount = 1;
  depth_barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &depth_barrier);

  std::array<VkClearValue, 2> clear_values{};
  clear_values[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
  clear_values[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo rp_begin{};
  rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rp_begin.renderPass = render_pass_;
  rp_begin.framebuffer = framebuffer_;
  rp_begin.renderArea.offset = {0, 0};
  rp_begin.renderArea.extent = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_)};
  rp_begin.clearValueCount = static_cast<uint32_t>(clear_values.size());
  rp_begin.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(glm::mat4), &mvp);

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
  vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);

  vkCmdDrawIndexed(cmd, static_cast<uint32_t>(index_count), 1, 0, 0, 0);

  vkCmdEndRenderPass(cmd);

  TransitionImageLayout(cmd, color_image_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkBufferImageCopy copy_region{};
  copy_region.bufferOffset = 0;
  copy_region.bufferRowLength = 0;
  copy_region.bufferImageHeight = 0;
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.mipLevel = 0;
  copy_region.imageSubresource.baseArrayLayer = 0;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageOffset = {0, 0, 0};
  copy_region.imageExtent = {static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), 1};

  vkCmdCopyImageToBuffer(cmd, color_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         readback_buffer, 1, &copy_region);

  VkBufferMemoryBarrier readback_barrier{};
  readback_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  readback_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  readback_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  readback_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  readback_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  readback_barrier.buffer = readback_buffer;
  readback_barrier.offset = 0;
  readback_barrier.size = image_size;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                       &readback_barrier, 0, nullptr);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;

  vkQueueSubmit(queue_, 1, &submit_info, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  vkMapMemory(device_, readback_memory, 0, image_size, 0, &data);

  cv::Mat mask(height_, width_, CV_8UC1);
  auto* src = static_cast<const uint8_t*>(data);
  for (int row = 0; row < height_; ++row) {
    auto* dst_row = mask.ptr<uint8_t>(row);
    for (int col = 0; col < width_; ++col) {
      int idx = (row * width_ + col) * 4;
      dst_row[col] = (src[idx] > 127) ? 255 : 0;
    }
  }

  vkUnmapMemory(device_, readback_memory);

  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
  vkDestroyBuffer(device_, readback_buffer, nullptr);
  vkFreeMemory(device_, readback_memory, nullptr);
  vkDestroyBuffer(device_, index_buffer, nullptr);
  vkFreeMemory(device_, index_memory, nullptr);
  vkDestroyBuffer(device_, vertex_buffer, nullptr);
  vkFreeMemory(device_, vertex_memory, nullptr);

  return mask;
}

}  // namespace maskgen
