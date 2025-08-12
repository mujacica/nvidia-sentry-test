# CUDA Demo
if(BUILD_CUDA)
    add_subdirectory(cuda)
endif()

# Vulkan Demo  
if(BUILD_VULKAN)
    add_subdirectory(vulkan)
endif()

# DirectX Demo
if(BUILD_DIRECTX)
    add_subdirectory(directx)
endif()

# Python CUDA Demo
if(BUILD_PYTHON_CUDA)
    add_subdirectory(python_cuda)
endif()

# GPU Info Demo
if(BUILD_GPU_INFO)
    add_subdirectory(gpu_info)
endif()