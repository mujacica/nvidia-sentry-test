add_executable(cuda_crash_demo 
    main.cpp 
    cuda_kernels.cu
)

set_target_properties(cuda_crash_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(cuda_crash_demo 
    sentry::sentry
    ${CUDA_LIBRARIES}
)

target_include_directories(cuda_crash_demo PRIVATE 
    ${CUDA_INCLUDE_DIRS}
)

set_property(TARGET cuda_crash_demo PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS cuda_crash_demo DESTINATION bin)