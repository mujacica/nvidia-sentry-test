#if defined(_WIN32) || defined(__linux__)
// CUDA demo is supported on Windows and Linux only (requires NVIDIA GPU)
#include "../src/cuda_kernels.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <sentry.h>

class CudaSentryTest : public ::testing::Test {
protected:
  void SetUp() override {
    sentry_options_t *options = sentry_options_new();
    sentry_options_set_dsn(options,
                           nullptr); // Disable actual reporting for tests
    sentry_options_set_database_path(options, ".sentry-native-test");
    sentry_options_set_debug(options, 0);

    if (sentry_init(options) != 0) {
      FAIL() << "Failed to initialize Sentry for testing";
    }

    int device_count = 0;
    cudaError_t cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    cuda_error = cudaMalloc(&d_data, data_size * sizeof(int));
    ASSERT_EQ(cuda_error, cudaSuccess) << "Failed to allocate CUDA memory";

    std::vector<int> h_data(data_size);
    for (int i = 0; i < data_size; i++) {
      h_data[i] = i;
    }

    cuda_error = cudaMemcpy(d_data, h_data.data(), data_size * sizeof(int),
                            cudaMemcpyHostToDevice);
    ASSERT_EQ(cuda_error, cudaSuccess) << "Failed to copy data to GPU";
  }

  void TearDown() override {
    if (d_data) {
      cudaFree(d_data);
    }
    sentry_close();
  }

  int *d_data = nullptr;
  const int data_size = 1024;
};

TEST_F(CudaSentryTest, CudaDeviceAvailable) {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver) {
    GTEST_SKIP() << "No CUDA devices available or driver issues";
  }

  EXPECT_EQ(error, cudaSuccess);
  EXPECT_GT(device_count, 0);
}

TEST_F(CudaSentryTest, SentryInitialization) {
  sentry_value_t event = sentry_value_new_event();
  sentry_value_set_by_key(event, "message",
                          sentry_value_new_string("Test event"));

  EXPECT_NO_THROW(sentry_capture_event(event));
}

TEST_F(CudaSentryTest, DivideByZeroError) {
  CudaError error = launch_divide_by_zero_crash(d_data, data_size);

  EXPECT_NE(error.code, 0) << "Expected CUDA error for divide by zero";
  EXPECT_STREQ(error.type, "CUDA_DIVIDE_BY_ZERO");
}

TEST_F(CudaSentryTest, OutOfBoundsError) {
  CudaError error = launch_out_of_bounds_crash(d_data, data_size);

  EXPECT_NE(error.code, 0) << "Expected CUDA error for out of bounds access";
  EXPECT_STREQ(error.type, "CUDA_OUT_OF_BOUNDS");
}

TEST_F(CudaSentryTest, NullPointerError) {
  CudaError error = launch_null_pointer_crash(d_data, data_size);

  EXPECT_NE(error.code, 0) << "Expected CUDA error for null pointer access";
  EXPECT_STREQ(error.type, "CUDA_NULL_POINTER");
}

#else
#include <gtest/gtest.h>

TEST(CudaSentryTest, WindowsLinuxOnly) {
  GTEST_SKIP()
      << "CUDA tests only available on Windows and Linux (requires NVIDIA GPU)";
}

#endif // _WIN32 || __linux__