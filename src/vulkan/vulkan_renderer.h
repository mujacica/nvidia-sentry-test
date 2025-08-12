#pragma once

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

struct VulkanError {
  int code;
  const char *message;
  const char *type;
};

class VulkanRenderer {
public:
  VulkanRenderer();
  ~VulkanRenderer();

  bool initialize();
  void cleanup();

  VulkanError test_invalid_buffer_access();
  VulkanError test_invalid_command_buffer();
  VulkanError test_out_of_bounds_descriptor();
  VulkanError test_invalid_render_pass();
  VulkanError test_device_lost_simulation();

  // NVIDIA Driver crash scenarios - WARNING: These may crash the driver or
  // system
  VulkanError test_nvidia_driver_memory_exhaustion();
  VulkanError test_nvidia_driver_invalid_memory_mapping();
  VulkanError test_nvidia_driver_command_buffer_corruption();
  VulkanError test_nvidia_driver_descriptor_set_overflow();
  VulkanError test_nvidia_driver_concurrent_queue_destruction();

private:
  VkInstance instance = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkBuffer vertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;

  uint32_t graphicsFamily = UINT32_MAX;

  // Driver crash test resources
  std::vector<VkBuffer> crashTestBuffers;
  std::vector<VkDeviceMemory> crashTestMemories;
  std::vector<VkCommandPool> crashTestCommandPools;
  std::vector<VkQueue> crashTestQueues;

  bool createInstance();
  bool pickPhysicalDevice();
  bool createLogicalDevice();
  bool createCommandPool();
  bool createVertexBuffer();
  bool createRenderPass();

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties);
  std::vector<const char *> getRequiredExtensions();
  bool checkValidationLayerSupport();

  static const std::vector<const char *> validationLayers;
  static const bool enableValidationLayers;
};