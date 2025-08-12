#include "vulkan_renderer.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>

const std::vector<const char *> VulkanRenderer::validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool VulkanRenderer::enableValidationLayers = false;
#else
const bool VulkanRenderer::enableValidationLayers = true;
#endif

VulkanRenderer::VulkanRenderer() {}

VulkanRenderer::~VulkanRenderer() { cleanup(); }

bool VulkanRenderer::initialize() {
  if (!createInstance()) {
    std::cerr << "Failed to create Vulkan instance" << std::endl;
    return false;
  }

  if (!pickPhysicalDevice()) {
    std::cerr << "Failed to find suitable GPU" << std::endl;
    return false;
  }

  if (!createLogicalDevice()) {
    std::cerr << "Failed to create logical device" << std::endl;
    return false;
  }

  if (!createCommandPool()) {
    std::cerr << "Failed to create command pool" << std::endl;
    return false;
  }

  if (!createVertexBuffer()) {
    std::cerr << "Failed to create vertex buffer" << std::endl;
    return false;
  }

  if (!createRenderPass()) {
    std::cerr << "Failed to create render pass" << std::endl;
    return false;
  }

  std::cout << "Vulkan renderer initialized successfully" << std::endl;
  return true;
}

void VulkanRenderer::cleanup() {
  if (device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device);

    if (renderPass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(device, renderPass, nullptr);
    }

    if (vertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, vertexBuffer, nullptr);
    }

    if (vertexBufferMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    if (commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, commandPool, nullptr);
    }

    vkDestroyDevice(device, nullptr);
  }

  if (instance != VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
  }
}

VulkanError VulkanRenderer::test_invalid_buffer_access() {
  VkBuffer invalidBuffer = VK_NULL_HANDLE;

  VkCommandBuffer commandBuffer;
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkResult result =
      vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Failed to allocate command buffer",
            "VK_COMMAND_BUFFER_ALLOC_ERROR"};
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkDeviceSize invalidOffset = 1000000;
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, &invalidBuffer, &invalidOffset);

  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphicsQueue);

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Invalid buffer access error",
            "VK_INVALID_BUFFER_ACCESS"};
  }

  return {0, "Success", "NONE"};
}

VulkanError VulkanRenderer::test_invalid_command_buffer() {
  VkCommandBuffer invalidCommandBuffer = VK_NULL_HANDLE;

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &invalidCommandBuffer;

  VkResult result =
      vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Invalid command buffer error",
            "VK_INVALID_COMMAND_BUFFER"};
  }

  return {0, "Success", "NONE"};
}

VulkanError VulkanRenderer::test_out_of_bounds_descriptor() {
  VkDescriptorSetLayout invalidLayout = VK_NULL_HANDLE;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &invalidLayout;

  VkPipelineLayout pipelineLayout;
  VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                           &pipelineLayout);

  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Out of bounds descriptor error",
            "VK_OUT_OF_BOUNDS_DESCRIPTOR"};
  }

  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  return {0, "Success", "NONE"};
}

VulkanError VulkanRenderer::test_invalid_render_pass() {
  VkRenderPass invalidRenderPass = VK_NULL_HANDLE;

  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = invalidRenderPass;
  framebufferInfo.attachmentCount = 0;
  framebufferInfo.width = 800;
  framebufferInfo.height = 600;
  framebufferInfo.layers = 1;

  VkFramebuffer framebuffer;
  VkResult result =
      vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer);

  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Invalid render pass error",
            "VK_INVALID_RENDER_PASS"};
  }

  vkDestroyFramebuffer(device, framebuffer, nullptr);
  return {0, "Success", "NONE"};
}

VulkanError VulkanRenderer::test_device_lost_simulation() {
  std::vector<VkCommandBuffer> commandBuffers(1000);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = commandBuffers.size();

  VkResult result =
      vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());

  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Device lost simulation error",
            "VK_DEVICE_LOST_SIMULATION"};
  }

  return {0, "Success", "NONE"};
}

bool VulkanRenderer::createInstance() {
  if (enableValidationLayers && !checkValidationLayerSupport()) {
    std::cerr << "Validation layers requested but not available!" << std::endl;
  }

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Sentry Vulkan Demo";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  auto extensions = getRequiredExtensions();
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  if (enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
  return result == VK_SUCCESS;
}

bool VulkanRenderer::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  if (deviceCount == 0) {
    return false;
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  for (const auto &device : devices) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        physicalDevice = device;
        graphicsFamily = i;
        return true;
      }
    }
  }

  return false;
}

bool VulkanRenderer::createLogicalDevice() {
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = graphicsFamily;
  queueCreateInfo.queueCount = 1;

  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;
  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount = 0;

  if (enableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  VkResult result =
      vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
  if (result == VK_SUCCESS) {
    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
    return true;
  }

  return false;
}

bool VulkanRenderer::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = graphicsFamily;

  return vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) ==
         VK_SUCCESS;
}

bool VulkanRenderer::createVertexBuffer() {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = 1024;
  bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) !=
      VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) !=
      VK_SUCCESS) {
    return false;
  }

  vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
  return true;
}

bool VulkanRenderer::createRenderPass() {
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = VK_FORMAT_R8G8B8A8_SRGB;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;

  return vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) ==
         VK_SUCCESS;
}

uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter,
                                        VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  return 0;
}

std::vector<const char *> VulkanRenderer::getRequiredExtensions() {
  std::vector<const char *> extensions;
  return extensions;
}

bool VulkanRenderer::checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : validationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

// NVIDIA Driver crash scenarios - WARNING: These may crash the driver or system
VulkanError VulkanRenderer::test_nvidia_driver_memory_exhaustion() {
  // Attempt to allocate massive amounts of GPU memory to exhaust NVIDIA driver
  std::vector<VkBuffer> buffers;
  std::vector<VkDeviceMemory> memories;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = 1024 * 1024 * 1024; // 1GB per buffer
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  // Try to allocate many large buffers
  for (int i = 0; i < 100; i++) {
    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
      // Clean up allocated buffers
      for (size_t j = 0; j < buffers.size(); j++) {
        vkDestroyBuffer(device, buffers[j], nullptr);
        if (j < memories.size()) {
          vkFreeMemory(device, memories[j], nullptr);
        }
      }
      return {static_cast<int>(result), "NVIDIA driver memory exhaustion",
              "VK_NVIDIA_DRIVER_MEMORY_EXHAUSTION"};
    }

    buffers.push_back(buffer);

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDeviceMemory memory;
    result = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
      // Clean up
      for (auto buf : buffers)
        vkDestroyBuffer(device, buf, nullptr);
      for (auto mem : memories)
        vkFreeMemory(device, mem, nullptr);
      return {static_cast<int>(result), "NVIDIA driver memory exhaustion",
              "VK_NVIDIA_DRIVER_MEMORY_EXHAUSTION"};
    }

    memories.push_back(memory);
    vkBindBufferMemory(device, buffer, memory, 0);
  }

  // Clean up
  for (auto buf : buffers)
    vkDestroyBuffer(device, buf, nullptr);
  for (auto mem : memories)
    vkFreeMemory(device, mem, nullptr);

  return {0, "Success", "NONE"};
}

VulkanError VulkanRenderer::test_nvidia_driver_invalid_memory_mapping() {
  // Create buffer with invalid memory mapping to trigger NVIDIA driver issues
  VkBuffer buffer;
  VkDeviceMemory memory;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = 4096;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Failed to create buffer",
            "VK_NVIDIA_DRIVER_INVALID_MEMORY_MAPPING"};
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  result = vkAllocateMemory(device, &allocInfo, nullptr, &memory);
  if (result != VK_SUCCESS) {
    vkDestroyBuffer(device, buffer, nullptr);
    return {static_cast<int>(result), "Failed to allocate memory",
            "VK_NVIDIA_DRIVER_INVALID_MEMORY_MAPPING"};
  }

  // Bind buffer to memory
  result = vkBindBufferMemory(device, buffer, memory, 0);
  if (result != VK_SUCCESS) {
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    return {static_cast<int>(result), "Failed to bind buffer memory",
            "VK_NVIDIA_DRIVER_INVALID_MEMORY_MAPPING"};
  }

  // Map memory and attempt invalid operations
  void *mappedMemory;
  result =
      vkMapMemory(device, memory, 0, memRequirements.size, 0, &mappedMemory);
  if (result != VK_SUCCESS) {
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    return {static_cast<int>(result), "Failed to map memory",
            "VK_NVIDIA_DRIVER_INVALID_MEMORY_MAPPING"};
  }

  // Attempt to access memory outside allocated range
  char *invalidPtr =
      static_cast<char *>(mappedMemory) + memRequirements.size + 1000000;
  *invalidPtr = 0x42; // This should cause driver-level protection fault

  vkUnmapMemory(device, memory);
  vkDestroyBuffer(device, buffer, nullptr);
  vkFreeMemory(device, memory, nullptr);

  return {static_cast<int>(VK_ERROR_MEMORY_MAP_FAILED),
          "Invalid memory mapping", "VK_NVIDIA_DRIVER_INVALID_MEMORY_MAPPING"};
}

VulkanError VulkanRenderer::test_nvidia_driver_command_buffer_corruption() {
  // Create many command buffers and corrupt their state
  std::vector<VkCommandBuffer> commandBuffers(1000);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = commandBuffers.size();

  VkResult result =
      vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Failed to allocate command buffers",
            "VK_NVIDIA_DRIVER_COMMAND_BUFFER_CORRUPTION"};
  }

  // Begin recording on all command buffers simultaneously
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  for (auto &cmdBuffer : commandBuffers) {
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // Add some invalid commands to corrupt the command buffer
    VkBufferCopy copyRegion{};
    copyRegion.size = 1000000000; // Extremely large size
    vkCmdCopyBuffer(cmdBuffer, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &copyRegion);

    vkEndCommandBuffer(cmdBuffer);
  }

  // Submit all command buffers simultaneously to different queues
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;

  for (auto &cmdBuffer : commandBuffers) {
    submitInfo.pCommandBuffers = &cmdBuffer;
    result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
      return {static_cast<int>(result), "Command buffer corruption",
              "VK_NVIDIA_DRIVER_COMMAND_BUFFER_CORRUPTION"};
    }
  }

  // Wait for completion (this might never complete due to corruption)
  vkQueueWaitIdle(graphicsQueue);

  return {0, "Success", "NONE"};
}

VulkanError VulkanRenderer::test_nvidia_driver_descriptor_set_overflow() {
  // Create excessive descriptor sets to overflow NVIDIA driver limits
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1000000; // Extremely large count
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &binding;

  VkDescriptorSetLayout descriptorSetLayout;
  VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                                &descriptorSetLayout);
  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Descriptor set overflow",
            "VK_NVIDIA_DRIVER_DESCRIPTOR_SET_OVERFLOW"};
  }

  // Create descriptor pool with excessive descriptors
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1000000;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 1000000;

  VkDescriptorPool descriptorPool;
  result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
  if (result != VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    return {static_cast<int>(result), "Descriptor set overflow",
            "VK_NVIDIA_DRIVER_DESCRIPTOR_SET_OVERFLOW"};
  }

  // Cleanup
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

  return {0, "Success", "NONE"};
}

VulkanError VulkanRenderer::test_nvidia_driver_concurrent_queue_destruction() {
  // This is extremely dangerous - concurrent queue operations
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = graphicsFamily;
  queueCreateInfo.queueCount = 1;

  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  // Create command buffer to submit work
  VkCommandBuffer cmdBuffer;
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer);
  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Failed to allocate command buffer",
            "VK_NVIDIA_DRIVER_CONCURRENT_QUEUE_DESTRUCTION"};
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(cmdBuffer, &beginInfo);

  // Add a long-running operation
  for (int i = 0; i < 1000; i++) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

    vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
  }

  vkEndCommandBuffer(cmdBuffer);

  // Submit the work
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuffer;

  result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  if (result != VK_SUCCESS) {
    return {static_cast<int>(result), "Failed to submit command buffer",
            "VK_NVIDIA_DRIVER_CONCURRENT_QUEUE_DESTRUCTION"};
  }

  // Immediately try to destroy device while queue is busy (DANGEROUS!)
  // This should cause driver-level issues
  vkDestroyDevice(device, nullptr);
  device = VK_NULL_HANDLE; // Mark as destroyed

  return {static_cast<int>(VK_ERROR_DEVICE_LOST),
          "Concurrent queue destruction",
          "VK_NVIDIA_DRIVER_CONCURRENT_QUEUE_DESTRUCTION"};
}