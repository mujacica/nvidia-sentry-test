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

  void test_invalid_stream_destroy() {
    std::cout << "\n--- Testing Invalid Stream Destroy ---\n";
    
    // Create an invalid stream handle (cast from a bad pointer)
    cudaStream_t invalid_stream = reinterpret_cast<cudaStream_t>(0xDEADBEEF);
    
    std::cout << "Attempting to destroy invalid stream handle...\n";
    cudaError_t cuda_error = cudaStreamDestroy(invalid_stream);
    
    if (cuda_error != cudaSuccess) {
      CudaError error;
      error.code = static_cast<int>(cuda_error);
      error.message = cudaGetErrorString(cuda_error);
      error.type = "invalid_stream_handle";
      
      report_cuda_kernel_error(error, "invalid_stream_destroy");
      std::cout << "CUDA Error: " << error.message << " (Code: " << error.code << ")\n";
    } else {
      std::cout << "Unexpectedly succeeded in destroying invalid stream\n";
    }
  }


  void run_all_tests() {
    std::cout << "Starting CUDA error demonstration with Sentry reporting...\n";

    test_invalid_stream_destroy();

    std::cout << "\nTest completed. Check Sentry dashboard for reported "
                 "errors.\n";
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
      if (test_type == "invalid_stream") {
        demo.test_invalid_stream_destroy();
      } else {
        std::cout << "Unknown test type: " << test_type << std::endl;
        std::cout << "Available tests:" << std::endl;
        std::cout << "  invalid_stream - Test destroying an invalid stream handle" << std::endl;
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