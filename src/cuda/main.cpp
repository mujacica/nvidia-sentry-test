#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <sentry.h>
#include <string>
#include <vector>

class SentryCudaDemo {
private:
  bool sentry_initialized;
  int *d_data;
  const int data_size = 1024;

public:
  SentryCudaDemo() : sentry_initialized(false), d_data(nullptr) {
    initialize_sentry();
    initialize_cuda();
  }

  ~SentryCudaDemo() {
    cleanup_cuda();
    cleanup_sentry();
  }

  void initialize_sentry() {
    sentry_options_t *options = sentry_options_new();
    sentry_options_set_dsn(options, getenv("SENTRY_DSN"));
    sentry_options_set_database_path(options, ".sentry-native");
    sentry_options_set_release(options, "sentry-cuda-demo@1.0.0");
    sentry_options_set_debug(options, 1);

    if (sentry_init(options) == 0) {
      sentry_initialized = true;
      std::cout << "Sentry initialized successfully\n";
    } else {
      std::cerr << "Failed to initialize Sentry\n";
    }
  }

  void initialize_cuda() {
    int device_count = 0;
    cudaError_t cuda_error = cudaGetDeviceCount(&device_count);

    if (cuda_error != cudaSuccess || device_count == 0) {
      report_cuda_initialization_error(cuda_error);
      throw std::runtime_error("No CUDA devices found or CUDA not available");
    }

    std::cout << "Found " << device_count << " CUDA device(s)\n";

    cuda_error = cudaMalloc(&d_data, data_size * sizeof(int));
    if (cuda_error != cudaSuccess) {
      report_cuda_initialization_error(cuda_error);
      throw std::runtime_error("Failed to allocate CUDA memory");
    }

    std::vector<int> h_data(data_size);
    for (int i = 0; i < data_size; i++) {
      h_data[i] = i;
    }

    cuda_error = cudaMemcpy(d_data, h_data.data(), data_size * sizeof(int),
                            cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
      report_cuda_initialization_error(cuda_error);
      throw std::runtime_error("Failed to copy data to GPU");
    }

    std::cout << "CUDA initialized successfully\n";
  }

  void report_cuda_initialization_error(cudaError_t cuda_error) {
    if (!sentry_initialized)
      return;

    sentry_value_t event = sentry_value_new_event();
    sentry_value_set_by_key(
        event, "message", sentry_value_new_string("CUDA Initialization Error"));
    sentry_value_set_by_key(event, "level", sentry_value_new_string("error"));

    sentry_value_t extra = sentry_value_new_object();
    sentry_value_set_by_key(
        extra, "cuda_error_code",
        sentry_value_new_int32(static_cast<int32_t>(cuda_error)));
    sentry_value_set_by_key(
        extra, "cuda_error_string",
        sentry_value_new_string(cudaGetErrorString(cuda_error)));
    sentry_value_set_by_key(event, "extra", extra);

    sentry_capture_event(event);
  }

  void report_cuda_kernel_error(const CudaError &error,
                                const std::string &crash_type) {
    if (!sentry_initialized)
      return;

    sentry_value_t event = sentry_value_new_event();
    sentry_value_set_by_key(
        event, "message",
        sentry_value_new_string(("CUDA Kernel Crash: " + crash_type).c_str()));
    sentry_value_set_by_key(event, "level", sentry_value_new_string("fatal"));

    sentry_value_t extra = sentry_value_new_object();
    sentry_value_set_by_key(extra, "cuda_error_code",
                            sentry_value_new_int32(error.code));
    sentry_value_set_by_key(extra, "cuda_error_message",
                            sentry_value_new_string(error.message));
    sentry_value_set_by_key(extra, "cuda_error_type",
                            sentry_value_new_string(error.type));
    sentry_value_set_by_key(extra, "crash_scenario",
                            sentry_value_new_string(crash_type.c_str()));
    sentry_value_set_by_key(event, "extra", extra);

    sentry_value_t tags = sentry_value_new_object();
    sentry_value_set_by_key(tags, "component",
                            sentry_value_new_string("cuda_kernel"));
    sentry_value_set_by_key(tags, "crash_type",
                            sentry_value_new_string(crash_type.c_str()));
    sentry_value_set_by_key(event, "tags", tags);

    sentry_capture_event(event);
    std::cout << "Reported CUDA error to Sentry: " << crash_type << "\n";
  }

  void test_divide_by_zero_crash() {
    std::cout << "\n--- Testing Divide by Zero Crash ---\n";
    CudaError error = launch_divide_by_zero_crash(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "divide_by_zero");
    }
  }

  void test_out_of_bounds_crash() {
    std::cout << "\n--- Testing Out of Bounds Crash ---\n";
    CudaError error = launch_out_of_bounds_crash(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "out_of_bounds");
    }
  }

  void test_null_pointer_crash() {
    std::cout << "\n--- Testing Null Pointer Crash ---\n";
    CudaError error = launch_null_pointer_crash(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "null_pointer");
    }
  }

  void test_infinite_loop_crash() {
    std::cout << "\n--- Testing Infinite Loop (Timeout) ---\n";
    CudaError error = launch_infinite_loop_crash(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "infinite_loop");
    }
  }

  // NVIDIA Driver crash tests - WARNING: These may crash the driver or system
  void test_driver_memory_exhaustion() {
    std::cout << "\n--- Testing Driver Memory Exhaustion (WARNING: May crash "
                 "driver) ---\n";
    CudaError error = launch_driver_memory_exhaustion(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "driver_memory_exhaustion");
    }
  }

  void test_driver_invalid_memory_access() {
    std::cout << "\n--- Testing Driver Invalid Memory Access (WARNING: May "
                 "crash system) ---\n";
    CudaError error = launch_driver_invalid_memory_access(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "driver_invalid_memory_access");
    }
  }

  void test_driver_context_corruption() {
    std::cout << "\n--- Testing Driver Context Corruption (WARNING: May crash "
                 "driver) ---\n";
    CudaError error = launch_driver_context_corruption(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "driver_context_corruption");
    }
  }

  void test_driver_excessive_recursion() {
    std::cout << "\n--- Testing Driver Excessive Recursion (WARNING: May crash "
                 "system) ---\n";
    CudaError error = launch_driver_excessive_recursion(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "driver_excessive_recursion");
    }
  }

  void test_driver_concurrent_context_destruction() {
    std::cout << "\n--- Testing Driver Concurrent Context Destruction "
                 "(WARNING: May crash system) ---\n";
    CudaError error =
        launch_driver_concurrent_context_destruction(d_data, data_size);
    if (error.code != 0) {
      report_cuda_kernel_error(error, "driver_concurrent_context_destruction");
    }
  }

  void run_all_tests() {
    std::cout << "Starting CUDA crash demonstration with Sentry reporting...\n";

    test_divide_by_zero_crash();
    test_out_of_bounds_crash();
    test_null_pointer_crash();
    // test_infinite_loop_crash(); // Uncomment to test, but it may hang

    std::cout << "\nAll tests completed. Check Sentry dashboard for reported "
                 "errors.\n";
  }

  void run_driver_tests() {
    std::cout << "Starting NVIDIA Driver crash demonstration with Sentry "
                 "reporting...\n";
    std::cout << "WARNING: These tests may crash the NVIDIA driver or entire "
                 "system!\n";
    std::cout << "Make sure to save your work before running these tests.\n\n";

    // Run driver crash tests - these are extremely dangerous
    test_driver_memory_exhaustion();
    test_driver_invalid_memory_access();
    test_driver_context_corruption();
    test_driver_excessive_recursion();
    // test_driver_concurrent_context_destruction(); // Most dangerous - may
    // crash system

    std::cout << "\nAll driver tests completed. Check Sentry dashboard for "
                 "reported errors.\n";
  }

private:
  void cleanup_cuda() {
    if (d_data) {
      cudaFree(d_data);
      d_data = nullptr;
    }
  }

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

    SentryCudaDemo demo;

    if (argc > 1) {
      std::string test_type = argv[1];
      if (test_type == "divide_by_zero") {
        demo.test_divide_by_zero_crash();
      } else if (test_type == "out_of_bounds") {
        demo.test_out_of_bounds_crash();
      } else if (test_type == "null_pointer") {
        demo.test_null_pointer_crash();
      } else if (test_type == "infinite_loop") {
        demo.test_infinite_loop_crash();
      } else if (test_type == "driver_memory_exhaustion") {
        demo.test_driver_memory_exhaustion();
      } else if (test_type == "driver_invalid_memory_access") {
        demo.test_driver_invalid_memory_access();
      } else if (test_type == "driver_context_corruption") {
        demo.test_driver_context_corruption();
      } else if (test_type == "driver_excessive_recursion") {
        demo.test_driver_excessive_recursion();
      } else if (test_type == "driver_concurrent_context_destruction") {
        demo.test_driver_concurrent_context_destruction();
      } else if (test_type == "driver_tests") {
        demo.run_driver_tests();
      } else {
        std::cout << "Unknown test type: " << test_type << std::endl;
        std::cout << "Available tests:" << std::endl;
        std::cout << "  Standard: divide_by_zero, out_of_bounds, null_pointer, "
                     "infinite_loop"
                  << std::endl;
        std::cout << "  Driver (DANGEROUS): driver_memory_exhaustion, "
                     "driver_invalid_memory_access,"
                  << std::endl;
        std::cout << "                      driver_context_corruption, "
                     "driver_excessive_recursion,"
                  << std::endl;
        std::cout
            << "                      driver_concurrent_context_destruction"
            << std::endl;
        std::cout << "  Batch: driver_tests (runs all driver tests)"
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