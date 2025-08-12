#pragma once

struct CudaError {
  int code;
  const char *message;
  const char *type;
};

CudaError launch_divide_by_zero_crash(int *d_data, int size);
CudaError launch_out_of_bounds_crash(int *d_data, int size);
CudaError launch_null_pointer_crash(int *d_data, int size);
CudaError launch_infinite_loop_crash(int *d_data, int size);

// NVIDIA Driver crash scenarios - WARNING: These may crash the driver or system
CudaError launch_driver_memory_exhaustion(int *d_data, int size);
CudaError launch_driver_invalid_memory_access(int *d_data, int size);
CudaError launch_driver_context_corruption(int *d_data, int size);
CudaError launch_driver_excessive_recursion(int *d_data, int size);
CudaError launch_driver_concurrent_context_destruction(int *d_data, int size);