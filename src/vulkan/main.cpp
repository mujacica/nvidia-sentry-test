#include "vulkan_renderer.h"
#include <iostream>
#include <memory>
#include <sentry.h>
#include <string>

class SentryVulkanDemo {
private:
  bool sentry_initialized;
  std::unique_ptr<VulkanRenderer> renderer;

public:
  SentryVulkanDemo() : sentry_initialized(false) {
    initialize_sentry();
    renderer = std::make_unique<VulkanRenderer>();
  }

  ~SentryVulkanDemo() { cleanup_sentry(); }

  void initialize_sentry() {
    sentry_options_t *options = sentry_options_new();
    sentry_options_set_dsn(options, getenv("SENTRY_DSN"));
    sentry_options_set_database_path(options, ".sentry-native");
    sentry_options_set_release(options, "sentry-vulkan-demo@1.0.0");
    sentry_options_set_debug(options, 1);

    if (sentry_init(options) == 0) {
      sentry_initialized = true;
      std::cout << "Sentry initialized successfully\n";
    } else {
      std::cerr << "Failed to initialize Sentry\n";
    }
  }

  void report_vulkan_error(const VulkanError &error,
                           const std::string &crash_type) {
    if (!sentry_initialized)
      return;

    sentry_value_t event = sentry_value_new_event();
    sentry_value_set_by_key(
        event, "message",
        sentry_value_new_string(
            ("Vulkan Rendering Crash: " + crash_type).c_str()));
    sentry_value_set_by_key(event, "level", sentry_value_new_string("fatal"));

    sentry_value_t extra = sentry_value_new_object();
    sentry_value_set_by_key(extra, "vulkan_error_code",
                            sentry_value_new_int32(error.code));
    sentry_value_set_by_key(extra, "vulkan_error_message",
                            sentry_value_new_string(error.message));
    sentry_value_set_by_key(extra, "vulkan_error_type",
                            sentry_value_new_string(error.type));
    sentry_value_set_by_key(extra, "crash_scenario",
                            sentry_value_new_string(crash_type.c_str()));
    sentry_value_set_by_key(event, "extra", extra);

    sentry_value_t tags = sentry_value_new_object();
    sentry_value_set_by_key(tags, "component",
                            sentry_value_new_string("vulkan_renderer"));
    sentry_value_set_by_key(tags, "crash_type",
                            sentry_value_new_string(crash_type.c_str()));
    sentry_value_set_by_key(tags, "platform", sentry_value_new_string("linux"));
    sentry_value_set_by_key(event, "tags", tags);

    sentry_capture_event(event);
    std::cout << "Reported Vulkan error to Sentry: " << crash_type << "\n";
  }

  void test_invalid_buffer_access() {
    std::cout << "\n--- Testing Invalid Buffer Access ---\n";
    VulkanError error = renderer->test_invalid_buffer_access();
    if (error.code != 0) {
      report_vulkan_error(error, "invalid_buffer_access");
    }
  }

  void test_invalid_command_buffer() {
    std::cout << "\n--- Testing Invalid Command Buffer ---\n";
    VulkanError error = renderer->test_invalid_command_buffer();
    if (error.code != 0) {
      report_vulkan_error(error, "invalid_command_buffer");
    }
  }

  void test_out_of_bounds_descriptor() {
    std::cout << "\n--- Testing Out of Bounds Descriptor ---\n";
    VulkanError error = renderer->test_out_of_bounds_descriptor();
    if (error.code != 0) {
      report_vulkan_error(error, "out_of_bounds_descriptor");
    }
  }

  void test_invalid_render_pass() {
    std::cout << "\n--- Testing Invalid Render Pass ---\n";
    VulkanError error = renderer->test_invalid_render_pass();
    if (error.code != 0) {
      report_vulkan_error(error, "invalid_render_pass");
    }
  }

  void test_device_lost_simulation() {
    std::cout << "\n--- Testing Device Lost Simulation ---\n";
    VulkanError error = renderer->test_device_lost_simulation();
    if (error.code != 0) {
      report_vulkan_error(error, "device_lost_simulation");
    }
  }

  // NVIDIA Driver crash tests - WARNING: These may crash the driver or system
  void test_nvidia_driver_memory_exhaustion() {
    std::cout << "\n--- Testing NVIDIA Driver Memory Exhaustion (WARNING: May "
                 "crash driver) ---\n";
    VulkanError error = renderer->test_nvidia_driver_memory_exhaustion();
    if (error.code != 0) {
      report_vulkan_error(error, "nvidia_driver_memory_exhaustion");
    }
  }

  void test_nvidia_driver_invalid_memory_mapping() {
    std::cout << "\n--- Testing NVIDIA Driver Invalid Memory Mapping (WARNING: "
                 "May crash system) ---\n";
    VulkanError error = renderer->test_nvidia_driver_invalid_memory_mapping();
    if (error.code != 0) {
      report_vulkan_error(error, "nvidia_driver_invalid_memory_mapping");
    }
  }

  void test_nvidia_driver_command_buffer_corruption() {
    std::cout << "\n--- Testing NVIDIA Driver Command Buffer Corruption "
                 "(WARNING: May crash driver) ---\n";
    VulkanError error =
        renderer->test_nvidia_driver_command_buffer_corruption();
    if (error.code != 0) {
      report_vulkan_error(error, "nvidia_driver_command_buffer_corruption");
    }
  }

  void test_nvidia_driver_descriptor_set_overflow() {
    std::cout << "\n--- Testing NVIDIA Driver Descriptor Set Overflow "
                 "(WARNING: May crash system) ---\n";
    VulkanError error = renderer->test_nvidia_driver_descriptor_set_overflow();
    if (error.code != 0) {
      report_vulkan_error(error, "nvidia_driver_descriptor_set_overflow");
    }
  }

  void test_nvidia_driver_concurrent_queue_destruction() {
    std::cout << "\n--- Testing NVIDIA Driver Concurrent Queue Destruction "
                 "(WARNING: May crash system) ---\n";
    VulkanError error =
        renderer->test_nvidia_driver_concurrent_queue_destruction();
    if (error.code != 0) {
      report_vulkan_error(error, "nvidia_driver_concurrent_queue_destruction");
    }
  }

  bool initialize_renderer() {
    if (!renderer->initialize()) {
      std::cerr << "Failed to initialize Vulkan renderer\n";
      return false;
    }
    return true;
  }

  void run_all_tests() {
    std::cout
        << "Starting Vulkan crash demonstration with Sentry reporting...\n";

    if (!initialize_renderer()) {
      std::cerr << "Cannot run tests without initialized renderer\n";
      return;
    }

    test_invalid_buffer_access();
    test_invalid_command_buffer();
    test_out_of_bounds_descriptor();
    test_invalid_render_pass();
    test_device_lost_simulation();

    std::cout << "\nAll Vulkan tests completed. Check Sentry dashboard for "
                 "reported errors.\n";
  }

  void run_nvidia_driver_tests() {
    std::cout << "Starting NVIDIA Driver crash demonstration with Sentry "
                 "reporting...\n";
    std::cout << "WARNING: These tests may crash the NVIDIA driver or entire "
                 "system!\n";
    std::cout << "Make sure to save your work before running these tests.\n\n";

    if (!initialize_renderer()) {
      std::cerr << "Cannot run tests without initialized renderer\n";
      return;
    }

    // Run NVIDIA driver crash tests - these are extremely dangerous
    test_nvidia_driver_memory_exhaustion();
    test_nvidia_driver_invalid_memory_mapping();
    test_nvidia_driver_command_buffer_corruption();
    test_nvidia_driver_descriptor_set_overflow();
    // test_nvidia_driver_concurrent_queue_destruction(); // Most dangerous -
    // may crash system

    std::cout << "\nAll NVIDIA driver tests completed. Check Sentry dashboard "
                 "for reported errors.\n";
  }

private:
  void cleanup_sentry() {
    if (sentry_initialized) {
      sentry_close();
    }
  }
};

int main(int argc, char *argv[]) {
  try {
    if (!getenv("SENTRY_DSN")) {
      std::cout << "Warning: SENTRY_DSN environment variable not set. Errors "
                   "will not be reported to Sentry.\n";
      std::cout << "Set SENTRY_DSN to your Sentry project DSN to enable error "
                   "reporting.\n\n";
    }

    SentryVulkanDemo demo;

    if (argc > 1) {
      std::string test_type = argv[1];
      if (!demo.initialize_renderer()) {
        std::cerr << "Failed to initialize Vulkan renderer\n";
        return 1;
      }

      if (test_type == "invalid_buffer_access") {
        demo.test_invalid_buffer_access();
      } else if (test_type == "invalid_command_buffer") {
        demo.test_invalid_command_buffer();
      } else if (test_type == "out_of_bounds_descriptor") {
        demo.test_out_of_bounds_descriptor();
      } else if (test_type == "invalid_render_pass") {
        demo.test_invalid_render_pass();
      } else if (test_type == "device_lost_simulation") {
        demo.test_device_lost_simulation();
      } else if (test_type == "nvidia_driver_memory_exhaustion") {
        demo.test_nvidia_driver_memory_exhaustion();
      } else if (test_type == "nvidia_driver_invalid_memory_mapping") {
        demo.test_nvidia_driver_invalid_memory_mapping();
      } else if (test_type == "nvidia_driver_command_buffer_corruption") {
        demo.test_nvidia_driver_command_buffer_corruption();
      } else if (test_type == "nvidia_driver_descriptor_set_overflow") {
        demo.test_nvidia_driver_descriptor_set_overflow();
      } else if (test_type == "nvidia_driver_concurrent_queue_destruction") {
        demo.test_nvidia_driver_concurrent_queue_destruction();
      } else if (test_type == "nvidia_driver_tests") {
        demo.run_nvidia_driver_tests();
      } else {
        std::cout << "Unknown test type: " << test_type << std::endl;
        std::cout << "Available tests:" << std::endl;
        std::cout << "  Standard: invalid_buffer_access, "
                     "invalid_command_buffer, out_of_bounds_descriptor, "
                     "invalid_render_pass, device_lost_simulation"
                  << std::endl;
        std::cout
            << "  NVIDIA Driver (DANGEROUS): nvidia_driver_memory_exhaustion, "
               "nvidia_driver_invalid_memory_mapping,"
            << std::endl;
        std::cout << "                             "
                     "nvidia_driver_command_buffer_corruption, "
                     "nvidia_driver_descriptor_set_overflow,"
                  << std::endl;
        std::cout << "                             "
                     "nvidia_driver_concurrent_queue_destruction"
                  << std::endl;
        std::cout
            << "  Batch: nvidia_driver_tests (runs all NVIDIA driver tests)"
            << std::endl;
        return 1;
      }
    } else {
      demo.run_all_tests();
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}