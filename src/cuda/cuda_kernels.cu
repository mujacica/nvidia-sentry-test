#include "cuda_kernels.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void divide_by_zero_kernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = data[idx] / 0;
  }
}

__global__ void out_of_bounds_kernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx + size * 1000] = 42;
  }
}

__global__ void null_pointer_kernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    int *null_ptr = nullptr;
    *null_ptr = data[idx];
  }

  int hats = (int)0xffffffff;
  *((int*) hats) = 12;
}

__global__ void infinite_loop_kernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    while (true) {
      data[idx]++;
    }
  }
}

CudaError launch_divide_by_zero_crash(int *d_data, int size) {
  dim3 block_size(256);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  divide_by_zero_kernel<<<grid_size, block_size>>>(d_data, size);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_DIVIDE_BY_ZERO"};
  }
  return CudaError{0, "Success", "NONE"};
}

CudaError launch_out_of_bounds_crash(int *d_data, int size) {
  dim3 block_size(256);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  out_of_bounds_kernel<<<grid_size, block_size>>>(d_data, size);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_OUT_OF_BOUNDS"};
  }
  return CudaError{0, "Success", "NONE"};
}

CudaError launch_null_pointer_crash(int *d_data, int size) {
  dim3 block_size(256);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  null_pointer_kernel<<<grid_size, block_size>>>(d_data, size);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_NULL_POINTER"};
  }
  return CudaError{0, "Success", "NONE"};
}

CudaError launch_infinite_loop_crash(int *d_data, int size) {
  dim3 block_size(256);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  infinite_loop_kernel<<<grid_size, block_size>>>(d_data, size);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_INFINITE_LOOP"};
  }
  return CudaError{0, "Success", "NONE"};
}

// NVIDIA Driver crash scenarios - WARNING: These may crash the driver or system
__global__ void driver_memory_exhaustion_kernel(int *data) {
  // Attempt to allocate maximum shared memory per block (12KB = 3072 ints)
  __shared__ int shared_memory[3072]; // Close to max shared memory per block
  int idx = threadIdx.x;

  // Create memory pressure by accessing all shared memory
  for (int i = 0; i < 3072; i++) {
    shared_memory[i] = idx * i;
  }

  // Force synchronization to ensure memory allocation
  __syncthreads();

  // Write back to global memory
  if (idx < 3072) {
    data[idx] = shared_memory[idx];
  }
}

__global__ void driver_invalid_memory_access_kernel(int *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Attempt to access memory way outside valid range
  // This can cause driver-level protection faults
  int *invalid_ptr = (int *)0xDEADBEEF; // Invalid pointer

  if (idx < size) {
    // Multiple invalid memory operations
    *invalid_ptr = data[idx];
    data[idx] = *invalid_ptr;

    // Also try accessing extremely high memory addresses
    int *high_addr = (int *)0xFFFFFFFFFFFFFFFF;
    *high_addr = idx;
  }
}

__global__ void driver_excessive_recursion_kernel(int *data, int depth) {
  // Create local memory pressure to exhaust stack
  int large_array[10000];
  for (int i = 0; i < 10000; i++) {
    large_array[i] = threadIdx.x + depth;
  }

  if (threadIdx.x == 0) {
    data[0] = large_array[depth % 10000];
  }
}

CudaError launch_driver_memory_exhaustion(int *d_data, int size) {
  // Launch many blocks to exhaust GPU memory
  dim3 block_size(1024); // Max threads per block
  dim3 grid_size(65535); // Max blocks per grid

  driver_memory_exhaustion_kernel<<<grid_size, block_size>>>(d_data);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_DRIVER_MEMORY_EXHAUSTION"};
  }
  return CudaError{0, "Success", "NONE"};
}

CudaError launch_driver_invalid_memory_access(int *d_data, int size) {
  dim3 block_size(256);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  driver_invalid_memory_access_kernel<<<grid_size, block_size>>>(d_data, size);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_DRIVER_INVALID_MEMORY_ACCESS"};
  }
  return CudaError{0, "Success", "NONE"};
}

CudaError launch_driver_context_corruption(int *d_data, int size) {
  // Attempt to corrupt CUDA context by creating conflicting states
  cudaError_t cuda_error;

  // Create multiple streams and destroy them improperly
  cudaStream_t streams[100];
  for (int i = 0; i < 100; i++) {
    cuda_error = cudaStreamCreate(&streams[i]);
    if (cuda_error != cudaSuccess) {
      return CudaError{static_cast<int>(cuda_error),
                       cudaGetErrorString(cuda_error),
                       "CUDA_DRIVER_CONTEXT_CORRUPTION"};
    }
  }

  // Launch kernels on all streams simultaneously
  dim3 block_size(256);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  for (int i = 0; i < 100; i++) {
    driver_invalid_memory_access_kernel<<<grid_size, block_size, 0,
                                          streams[i]>>>(d_data, size);
  }

  // Destroy streams while kernels might still be running
  for (int i = 0; i < 100; i++) {
    cudaStreamDestroy(streams[i]);
  }

  cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_DRIVER_CONTEXT_CORRUPTION"};
  }
  return CudaError{0, "Success", "NONE"};
}

CudaError launch_driver_excessive_recursion(int *d_data, int size) {
  dim3 block_size(1024); // Max block size for maximum pressure
  dim3 grid_size(65535); // Max grid size

  driver_excessive_recursion_kernel<<<grid_size, block_size>>>(d_data, 1000);

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_DRIVER_EXCESSIVE_RECURSION"};
  }
  return CudaError{0, "Success", "NONE"};
}

CudaError launch_driver_concurrent_context_destruction(int *d_data, int size) {
  // This is extremely dangerous - concurrent context operations
  cudaError_t cuda_error;

  // Create multiple streams to cause context corruption
  cudaStream_t streams[100];
  for (int i = 0; i < 100; i++) {
    cuda_error = cudaStreamCreate(&streams[i]);
    if (cuda_error != cudaSuccess) {
      return CudaError{static_cast<int>(cuda_error),
                       cudaGetErrorString(cuda_error),
                       "CUDA_DRIVER_CONCURRENT_CONTEXT_DESTRUCTION"};
    }
  }

  // Launch kernels on multiple streams simultaneously
  dim3 block_size(256);
  dim3 grid_size((size + block_size.x - 1) / block_size.x);

  for (int i = 0; i < 100; i++) {
    driver_invalid_memory_access_kernel<<<grid_size, block_size, 0, streams[i]>>>(d_data, size);
  }

  // Destroy streams while kernels might still be running
  for (int i = 0; i < 100; i++) {
    cudaStreamDestroy(streams[i]);
  }

  // Try to reset device while kernel is potentially running
  cuda_error = cudaDeviceReset();
  if (cuda_error != cudaSuccess) {
    return CudaError{static_cast<int>(cuda_error),
                     cudaGetErrorString(cuda_error),
                     "CUDA_DRIVER_CONCURRENT_CONTEXT_DESTRUCTION"};
  }

  return CudaError{0, "Success", "NONE"};
}