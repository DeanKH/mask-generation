#include "vulkan_context.h"

#include <array>
#include <cstring>
#include <iostream>
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

static const char* kCostComputeSource = R"(#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0, r8) readonly uniform image2D rendered_img;
layout(set = 0, binding = 3) readonly restrict buffer DtInput { float dt_input[]; };
layout(set = 0, binding = 4) readonly restrict buffer InputMask { float input_mask[]; };
layout(set = 0, binding = 6) writeonly restrict buffer Partial { vec4 partial[]; };
layout(push_constant) uniform PC {
  ivec2 img_size;
  int radius;
  float scale;
} pc;
const int R = 16;
shared float s_sum[256];
shared float s_area[256];
shared float s_inter[256];
void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  uint lid = gl_LocalInvocationIndex;
  float chamfer = 0.0;
  float area = 0.0;
  float inter = 0.0;
  if (coord.x < pc.img_size.x && coord.y < pc.img_size.y) {
    float rendered_val = imageLoad(rendered_img, coord).r;
    uint idx = uint(coord.y * pc.img_size.x + coord.x);
    float input_val = input_mask[idx];
    float dt_i = dt_input[idx];
    float diff = abs(rendered_val - input_val);
    float dt_r = 0.0;
    if (rendered_val <= 0.5 && diff > 0.5) {
      float min_dist = float(R * R) * 2.0 + 1.0;  // cap: beyond window
      for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
          ivec2 nc = coord + ivec2(dx, dy);
          nc = clamp(nc, ivec2(0), pc.img_size - 1);
          if (imageLoad(rendered_img, nc).r > 0.5) {
            float d = float(dx * dx + dy * dy);
            if (d < min_dist) min_dist = d;
          }
        }
      }
      dt_r = sqrt(min_dist);
    }
    float dt_max = max(dt_i, dt_r);
    chamfer = dt_max * diff * pc.scale;
    area = (rendered_val > 0.5) ? 1.0 : 0.0;
    inter = (rendered_val > 0.5 && input_val > 0.5) ? 1.0 : 0.0;
  }
  s_sum[lid] = chamfer;
  s_area[lid] = area;
  s_inter[lid] = inter;
  barrier();
  for (uint s = 128; s > 0; s >>= 1) {
    if (lid < s) {
      s_sum[lid] += s_sum[lid + s];
      s_area[lid] += s_area[lid + s];
      s_inter[lid] += s_inter[lid + s];
    }
    barrier();
  }
  if (lid == 0) {
    uint wgs_x = (uint(pc.img_size.x) + 15u) / 16u;
    uint gid = gl_WorkGroupID.y * wgs_x + gl_WorkGroupID.x;
    partial[gid] = vec4(s_sum[0], s_area[0], s_inter[0], 0.0);
  }
}
)";

static const char* kCostReduceSource = R"(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 6) readonly restrict buffer Partial { vec4 partial[]; };
layout(set = 0, binding = 5) writeonly restrict buffer Output { vec4 result; };
layout(push_constant) uniform PC {
  ivec2 img_size;
  int num_groups;
  float scale;
} pc;
shared vec3 s_data[256];
void main() {
  uint tid = gl_LocalInvocationIndex;
  vec3 sum = vec3(0.0);
  for (uint i = tid; i < uint(pc.num_groups); i += 256u) {
    sum += partial[i].xyz;
  }
  s_data[tid] = sum;
  barrier();
  for (uint s = 128; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    barrier();
  }
  if (tid == 0) {
    result = vec4(s_data[0], 0.0);
  }
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

  VkImageFormatProperties img_props;
  has_compute_cost_ =
      (vkGetPhysicalDeviceImageFormatProperties(
           physical_device_, VK_FORMAT_R8_UNORM, VK_IMAGE_TYPE_2D,
           VK_IMAGE_TILING_OPTIMAL,
           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_STORAGE_BIT,
           0, &img_props) == VK_SUCCESS);

  CreateLogicalDevice();
  CreateCommandPool();
  CreateRenderPass();
  CreatePipelineLayout();
  CreatePipeline();
  CreateOffscreenResources();
  CreateFence();

  if (has_compute_cost_) {
    CreateComputeResources();
  }
}

VulkanContext::~VulkanContext() {
  if (has_compute_cost_) {
    CleanupComputeResources();
  }
  CleanupMeshBuffers();
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
  cost_compute_spv_ = CompileGlsl(kCostComputeSource, shaderc_compute_shader);
  cost_reduce_spv_ = CompileGlsl(kCostReduceSource, shaderc_compute_shader);
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

  int best_score = -1;
  for (const auto& device : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             queue_families.data());

    bool has_graphics = false;
    uint32_t graphics_family = 0;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
      if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        has_graphics = true;
        graphics_family = i;
        break;
      }
    }
    if (!has_graphics) continue;

    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score = 1000;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      score = 100;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
      score = 50;
    else
      score = 10;

    if (score > best_score) {
      best_score = score;
      physical_device_ = device;
      queue_family_index_ = graphics_family;
    }
  }

  if (best_score < 0) {
    throw std::runtime_error("No suitable GPU found with graphics queue");
  }

  VkPhysicalDeviceProperties picked_props;
  vkGetPhysicalDeviceProperties(physical_device_, &picked_props);
  std::cout << "[Vulkan] Selected device: " << picked_props.deviceName << "\n";
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

  attachments[0].format = VK_FORMAT_R8_UNORM;
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
  blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
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
  VkImageUsageFlags color_usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (has_compute_cost_) {
    color_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  CreateImage(VK_FORMAT_R8_UNORM, color_usage, &color_image_, &color_image_memory_);
  color_image_view_ =
      CreateImageView(color_image_, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

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

void VulkanContext::UploadMesh(const float* vertices, size_t vertex_count,
                               const uint32_t* indices, size_t index_count) {
  CleanupMeshBuffers();

  VkDeviceSize vertex_buffer_size = static_cast<VkDeviceSize>(vertex_count * 3) * sizeof(float);
  VkDeviceSize index_buffer_size = static_cast<VkDeviceSize>(index_count) * sizeof(uint32_t);
  VkDeviceSize image_size = static_cast<VkDeviceSize>(width_) * height_;

  CreateBuffer(vertex_buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &persistent_vertex_buffer_, &persistent_vertex_memory_);
  CreateBuffer(index_buffer_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &persistent_index_buffer_, &persistent_index_memory_);
  CreateBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &persistent_readback_buffer_, &persistent_readback_memory_);

  void* data = nullptr;
  vkMapMemory(device_, persistent_vertex_memory_, 0, vertex_buffer_size, 0, &data);
  std::memcpy(data, vertices, static_cast<size_t>(vertex_buffer_size));
  vkUnmapMemory(device_, persistent_vertex_memory_);

  vkMapMemory(device_, persistent_index_memory_, 0, index_buffer_size, 0, &data);
  std::memcpy(data, indices, static_cast<size_t>(index_buffer_size));
  vkUnmapMemory(device_, persistent_index_memory_);

  persistent_index_count_ = static_cast<uint32_t>(index_count);
  has_persistent_mesh_ = true;
}

void VulkanContext::CleanupMeshBuffers() {
  if (persistent_vertex_buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, persistent_vertex_buffer_, nullptr);
    persistent_vertex_buffer_ = VK_NULL_HANDLE;
  }
  if (persistent_vertex_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, persistent_vertex_memory_, nullptr);
    persistent_vertex_memory_ = VK_NULL_HANDLE;
  }
  if (persistent_index_buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, persistent_index_buffer_, nullptr);
    persistent_index_buffer_ = VK_NULL_HANDLE;
  }
  if (persistent_index_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, persistent_index_memory_, nullptr);
    persistent_index_memory_ = VK_NULL_HANDLE;
  }
  if (persistent_readback_buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, persistent_readback_buffer_, nullptr);
    persistent_readback_buffer_ = VK_NULL_HANDLE;
  }
  if (persistent_readback_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, persistent_readback_memory_, nullptr);
    persistent_readback_memory_ = VK_NULL_HANDLE;
  }
  has_persistent_mesh_ = false;
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

cv::Mat VulkanContext::RenderPose(const glm::mat4& mvp) {
  if (!has_persistent_mesh_) {
    throw std::runtime_error("UploadMesh must be called before RenderPose");
  }

  VkDeviceSize image_size = static_cast<VkDeviceSize>(width_) * height_;

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = command_pool_;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;

  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(device_, &cmd_alloc, &cmd) != VK_SUCCESS) {
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
  vkCmdBindVertexBuffers(cmd, 0, 1, &persistent_vertex_buffer_, &offset);
  vkCmdBindIndexBuffer(cmd, persistent_index_buffer_, 0, VK_INDEX_TYPE_UINT32);

  vkCmdDrawIndexed(cmd, persistent_index_count_, 1, 0, 0, 0);

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
                         persistent_readback_buffer_, 1, &copy_region);

  VkBufferMemoryBarrier readback_barrier{};
  readback_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  readback_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  readback_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  readback_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  readback_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  readback_barrier.buffer = persistent_readback_buffer_;
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

  void* data = nullptr;
  vkMapMemory(device_, persistent_readback_memory_, 0, image_size, 0, &data);

  cv::Mat mask(height_, width_, CV_8UC1);
  std::memcpy(mask.data, data, static_cast<size_t>(image_size));

  vkUnmapMemory(device_, persistent_readback_memory_);

  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);

  return mask;
}

cv::Mat VulkanContext::Render(const float* vertices, size_t vertex_count,
                              const uint32_t* indices, size_t index_count,
                              const glm::mat4& mvp) {
  VkDeviceSize vertex_buffer_size = static_cast<VkDeviceSize>(vertex_count * 3) * sizeof(float);
  VkDeviceSize index_buffer_size = static_cast<VkDeviceSize>(index_count) * sizeof(uint32_t);
  VkDeviceSize image_size =
      static_cast<VkDeviceSize>(width_) * height_;

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
  std::memcpy(mask.data, data, static_cast<size_t>(image_size));

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

namespace {
struct ComputePC {
  int32_t width;
  int32_t height;
  int32_t radius_or_groups;
  float scale;
};
}  // namespace

void VulkanContext::CreateComputeResources() {
  // Bindings: 0=color image, 3=dt_input buf, 4=input_mask buf,
  //           5=cost_output buf, 6=partial buf
  std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  for (int i = 1; i < 5; i++) {
    bindings[i].binding = i + 2;  // 3,4,5,6
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo desc_layout_info{};
  desc_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  desc_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  desc_layout_info.pBindings = bindings.data();
  if (vkCreateDescriptorSetLayout(device_, &desc_layout_info, nullptr,
                                  &compute_desc_layout_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create compute descriptor set layout");
  }

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(ComputePC);

  VkPipelineLayoutCreateInfo pipe_layout_info{};
  pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipe_layout_info.setLayoutCount = 1;
  pipe_layout_info.pSetLayouts = &compute_desc_layout_;
  pipe_layout_info.pushConstantRangeCount = 1;
  pipe_layout_info.pPushConstantRanges = &push_range;
  if (vkCreatePipelineLayout(device_, &pipe_layout_info, nullptr,
                             &compute_pipeline_layout_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create compute pipeline layout");
  }

  auto make_pipeline = [&](const std::vector<uint32_t>& spv) {
    VkShaderModule module = CreateShaderModule(spv);
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = compute_pipeline_layout_;
    VkPipeline pipeline;
    VkResult r =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device_, module, nullptr);
    if (r != VK_SUCCESS) throw std::runtime_error("Failed to create compute pipeline");
    return pipeline;
  };
  cost_compute_pipeline_ = make_pipeline(cost_compute_spv_);
  cost_reduce_pipeline_ = make_pipeline(cost_reduce_spv_);

  VkDeviceSize float_buf_size = static_cast<VkDeviceSize>(width_) * height_ * sizeof(float);
  CreateBuffer(float_buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &dt_input_buffer_, &dt_input_buffer_memory_);
  CreateBuffer(float_buf_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &input_mask_buffer_, &input_mask_buffer_memory_);

  CreateBuffer(sizeof(float) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &cost_output_buffer_, &cost_output_memory_);

  // 2D workgroups: 16x16 threads each
  num_partial_groups_ = ((width_ + 15) / 16) * ((height_ + 15) / 16);
  VkDeviceSize partial_size = static_cast<VkDeviceSize>(num_partial_groups_) * 16;
  CreateBuffer(partial_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &partial_buffer_,
               &partial_buffer_memory_);

  std::array<VkDescriptorPoolSize, 2> pool_sizes{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  pool_sizes[0].descriptorCount = 1;
  pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_sizes[1].descriptorCount = 4;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = 1;
  if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &compute_desc_pool_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = compute_desc_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &compute_desc_layout_;
  if (vkAllocateDescriptorSets(device_, &alloc_info, &compute_desc_set_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor set");
  }

  VkDescriptorImageInfo img0{};
  img0.imageView = color_image_view_;
  img0.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  VkDescriptorBufferInfo buf3{};
  buf3.buffer = dt_input_buffer_;
  buf3.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo buf4{};
  buf4.buffer = input_mask_buffer_;
  buf4.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo buf5{};
  buf5.buffer = cost_output_buffer_;
  buf5.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo buf6{};
  buf6.buffer = partial_buffer_;
  buf6.range = VK_WHOLE_SIZE;

  std::array<VkWriteDescriptorSet, 5> writes{};
  for (int i = 0; i < 5; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = compute_desc_set_;
    writes[i].descriptorCount = 1;
  }
  writes[0].dstBinding = 0;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[0].pImageInfo = &img0;
  writes[1].dstBinding = 3;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &buf3;
  writes[2].dstBinding = 4;
  writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[2].pBufferInfo = &buf4;
  writes[3].dstBinding = 5;
  writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[3].pBufferInfo = &buf5;
  writes[4].dstBinding = 6;
  writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[4].pBufferInfo = &buf6;
  vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                         nullptr);
}

void VulkanContext::CleanupComputeResources() {
  if (partial_buffer_) {
    vkDestroyBuffer(device_, partial_buffer_, nullptr);
    vkFreeMemory(device_, partial_buffer_memory_, nullptr);
  }
  if (cost_output_buffer_) {
    vkDestroyBuffer(device_, cost_output_buffer_, nullptr);
    vkFreeMemory(device_, cost_output_memory_, nullptr);
  }
  if (input_mask_buffer_) {
    vkDestroyBuffer(device_, input_mask_buffer_, nullptr);
    vkFreeMemory(device_, input_mask_buffer_memory_, nullptr);
  }
  if (dt_input_buffer_) {
    vkDestroyBuffer(device_, dt_input_buffer_, nullptr);
    vkFreeMemory(device_, dt_input_buffer_memory_, nullptr);
  }
  if (cost_reduce_pipeline_) vkDestroyPipeline(device_, cost_reduce_pipeline_, nullptr);
  if (cost_compute_pipeline_) vkDestroyPipeline(device_, cost_compute_pipeline_, nullptr);
  if (compute_desc_pool_) vkDestroyDescriptorPool(device_, compute_desc_pool_, nullptr);
  if (compute_desc_layout_)
    vkDestroyDescriptorSetLayout(device_, compute_desc_layout_, nullptr);
  if (compute_pipeline_layout_)
    vkDestroyPipelineLayout(device_, compute_pipeline_layout_, nullptr);
}

void VulkanContext::SetCostInputs(const cv::Mat& dt_input, const cv::Mat& input_mask) {
  cv::Mat dt_input_f;
  dt_input.convertTo(dt_input_f, CV_32F);

  VkDeviceSize buf_size = static_cast<VkDeviceSize>(width_) * height_ * sizeof(float);

  void* data = nullptr;
  vkMapMemory(device_, dt_input_buffer_memory_, 0, buf_size, 0, &data);
  std::memcpy(data, dt_input_f.data, static_cast<size_t>(buf_size));
  vkUnmapMemory(device_, dt_input_buffer_memory_);

  std::vector<float> mask_f(static_cast<size_t>(width_) * height_);
  for (int i = 0; i < width_ * height_; i++) {
    mask_f[i] = input_mask.data[i] > 127 ? 1.0f : 0.0f;
  }
  vkMapMemory(device_, input_mask_buffer_memory_, 0, buf_size, 0, &data);
  std::memcpy(data, mask_f.data(), static_cast<size_t>(buf_size));
  vkUnmapMemory(device_, input_mask_buffer_memory_);
}

void VulkanContext::RenderPoseWithCost(const glm::mat4& mvp, float scale_factor,
                                       float& chamfer_sum, float& rendered_area,
                                       float& intersection) {
  if (!has_persistent_mesh_) {
    throw std::runtime_error("UploadMesh must be called before RenderPoseWithCost");
  }

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = command_pool_;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;

  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(device_, &cmd_alloc, &cmd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate command buffer");
  }

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin_info);

  // --- Render pass (same as RenderPose) ---
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
  vkCmdBindVertexBuffers(cmd, 0, 1, &persistent_vertex_buffer_, &offset);
  vkCmdBindIndexBuffer(cmd, persistent_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, persistent_index_count_, 1, 0, 0, 0);
  vkCmdEndRenderPass(cmd);

  // Transition color_image_ from TRANSFER_SRC_OPTIMAL → GENERAL for compute
  TransitionImageLayout(cmd, color_image_,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // --- Compute: local-window DT + cost (2 dispatches) ---
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_layout_, 0,
                          1, &compute_desc_set_, 0, nullptr);

  ComputePC pc{width_, height_, 0, scale_factor};
  uint32_t wg2d_x = (static_cast<uint32_t>(width_) + 15) / 16;
  uint32_t wg2d_y = (static_cast<uint32_t>(height_) + 15) / 16;

  // 1. cost_compute (per-pixel local-window DT + workgroup reduction)
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cost_compute_pipeline_);
  vkCmdPushConstants(cmd, compute_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(pc), &pc);
  vkCmdDispatch(cmd, wg2d_x, wg2d_y, 1);

  // Buffer barrier on partial_buffer_
  {
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = partial_buffer_;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &b, 0,
                         nullptr);
  }

  // 2. cost_reduce (final reduction)
  pc.radius_or_groups = num_partial_groups_;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cost_reduce_pipeline_);
  vkCmdPushConstants(cmd, compute_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(pc), &pc);
  vkCmdDispatch(cmd, 1, 1, 1);

  // Output barrier: SHADER_WRITE → HOST_READ
  {
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = cost_output_buffer_;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
  }

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;
  vkQueueSubmit(queue_, 1, &submit_info, fence_);
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  vkResetFences(device_, 1, &fence_);

  void* data = nullptr;
  vkMapMemory(device_, cost_output_memory_, 0, sizeof(float) * 4, 0, &data);
  float* out = static_cast<float*>(data);
  chamfer_sum = out[0];
  rendered_area = out[1];
  intersection = out[2];
  vkUnmapMemory(device_, cost_output_memory_);

  vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

}  // namespace maskgen
