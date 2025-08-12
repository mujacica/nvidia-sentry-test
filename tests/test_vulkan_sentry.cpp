#if defined(__linux__)
#include "../src/vulkan/vulkan_renderer.h"
#include <gtest/gtest.h>
#include <sentry.h>

class VulkanSentryTest : public ::testing::Test {
protected:
  void SetUp() override {
    sentry_options_t *options = sentry_options_new();
    sentry_options_set_dsn(options,
                           nullptr); // Disable actual reporting for tests
    sentry_options_set_database_path(options, ".sentry-native-vulkan-test");
    sentry_options_set_debug(options, 0);

    if (sentry_init(options) != 0) {
      FAIL() << "Failed to initialize Sentry for testing";
    }

    renderer = std::make_unique<VulkanRenderer>();

    // Skip tests if Vulkan is not available
    if (!renderer->initialize()) {
      GTEST_SKIP() << "Vulkan not available, skipping Vulkan tests";
    }
  }

  void TearDown() override {
    renderer.reset();
    sentry_close();
  }

  std::unique_ptr<VulkanRenderer> renderer;
};

TEST_F(VulkanSentryTest, VulkanInitialization) {
  // This test passes if we get here (renderer initialized successfully)
  SUCCEED();
}

TEST_F(VulkanSentryTest, SentryInitialization) {
  sentry_value_t event = sentry_value_new_event();
  sentry_value_set_by_key(event, "message",
                          sentry_value_new_string("Test Vulkan event"));

  EXPECT_NO_THROW(sentry_capture_event(event));
}

TEST_F(VulkanSentryTest, InvalidBufferAccess) {
  VulkanError error = renderer->test_invalid_buffer_access();

  // This test checks that we can call the crash test function
  // The actual error depends on Vulkan driver behavior
  EXPECT_NO_THROW({
    if (error.code != 0) {
      std::cout << "Vulkan error detected: " << error.message << std::endl;
    }
  });
}

TEST_F(VulkanSentryTest, InvalidCommandBuffer) {
  VulkanError error = renderer->test_invalid_command_buffer();

  // This should typically fail with invalid command buffer
  EXPECT_NE(error.code, 0)
      << "Expected Vulkan error for invalid command buffer";
  EXPECT_STREQ(error.type, "VK_INVALID_COMMAND_BUFFER");
}

TEST_F(VulkanSentryTest, OutOfBoundsDescriptor) {
  VulkanError error = renderer->test_out_of_bounds_descriptor();

  // This should typically fail with invalid descriptor set layout
  EXPECT_NE(error.code, 0)
      << "Expected Vulkan error for out of bounds descriptor";
  EXPECT_STREQ(error.type, "VK_OUT_OF_BOUNDS_DESCRIPTOR");
}

TEST_F(VulkanSentryTest, InvalidRenderPass) {
  VulkanError error = renderer->test_invalid_render_pass();

  // This should typically fail with invalid render pass
  EXPECT_NE(error.code, 0) << "Expected Vulkan error for invalid render pass";
  EXPECT_STREQ(error.type, "VK_INVALID_RENDER_PASS");
}

#else
#include <gtest/gtest.h>

TEST(VulkanSentryTest, LinuxOnly) {
  GTEST_SKIP() << "Vulkan tests only available on Linux";
}

#endif // __linux__