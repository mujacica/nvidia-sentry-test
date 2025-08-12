find_package(GTest QUIET)

if(GTest_FOUND)
    # CUDA Tests
    if(BUILD_CUDA)
        add_executable(test_cuda_sentry test_cuda_sentry.cpp)
        target_link_libraries(test_cuda_sentry 
            sentry::sentry
            ${CUDA_LIBRARIES}
            GTest::gtest 
            GTest::gtest_main
        )
        target_include_directories(test_cuda_sentry PRIVATE 
            ../src/cuda
            ${CUDA_INCLUDE_DIRS}
        )
        add_test(NAME CudaSentryTests COMMAND test_cuda_sentry)
    endif()
    
    # Vulkan Tests
    if(BUILD_VULKAN)
        add_executable(test_vulkan_sentry test_vulkan_sentry.cpp)
        target_link_libraries(test_vulkan_sentry 
            sentry::sentry
            Vulkan::Vulkan
            GTest::gtest 
            GTest::gtest_main
        )
        target_include_directories(test_vulkan_sentry PRIVATE 
            ../src/vulkan
            ${Vulkan_INCLUDE_DIRS}
        )
        add_test(NAME VulkanSentryTests COMMAND test_vulkan_sentry)
    endif()
    
    # DirectX Tests
    if(BUILD_DIRECTX AND WIN32)
        add_executable(test_directx_sentry test_directx_sentry.cpp)
        target_link_libraries(test_directx_sentry 
            sentry::sentry
            d3d11
            dxgi
            d3dcompiler
            GTest::gtest 
            GTest::gtest_main
        )
        target_include_directories(test_directx_sentry PRIVATE 
            ../src/directx
        )
        target_compile_definitions(test_directx_sentry PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        )
        add_test(NAME DirectXSentryTests COMMAND test_directx_sentry)
    endif()
    
    # GPU Info Tests (supported on all platforms)
    if(BUILD_GPU_INFO)
        add_executable(test_gpu_info_sentry test_gpu_info_sentry.cpp)
        target_link_libraries(test_gpu_info_sentry 
            sentry::sentry
            GTest::gtest 
            GTest::gtest_main
        )
        target_include_directories(test_gpu_info_sentry PRIVATE 
            ../src/gpu_info
        )
        add_test(NAME GPUInfoSentryTests COMMAND test_gpu_info_sentry)
    endif()
else()
    message(STATUS "GTest not found. Skipping tests.")
endif()