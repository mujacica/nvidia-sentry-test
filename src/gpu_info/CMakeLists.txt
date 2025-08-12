# GPU Info Demo CMake configuration

add_executable(gpu_info_demo main.cpp)

set_target_properties(gpu_info_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(gpu_info_demo sentry::sentry)

# Platform-specific libraries and settings
if(WIN32)
    # Windows: Link DirectX/DXGI libraries
    target_link_libraries(gpu_info_demo d3d11 dxgi)
    
elseif(APPLE)
    # macOS: Link Metal framework
    find_library(METAL_FRAMEWORK Metal)
    find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
    
    if(METAL_FRAMEWORK AND CORE_FOUNDATION_FRAMEWORK)
        target_link_libraries(gpu_info_demo ${METAL_FRAMEWORK} ${CORE_FOUNDATION_FRAMEWORK})
        set_target_properties(gpu_info_demo PROPERTIES
            COMPILE_FLAGS "-x objective-c++"
        )
    else()
        message(WARNING "Metal or CoreFoundation framework not found on macOS")
    endif()
    
else()
    # Linux: Try to link OpenGL if available
    find_package(OpenGL QUIET)
    if(OPENGL_FOUND)
        target_link_libraries(gpu_info_demo ${OPENGL_LIBRARIES})
        target_include_directories(gpu_info_demo PRIVATE ${OPENGL_INCLUDE_DIRS})
        target_compile_definitions(gpu_info_demo PRIVATE HAS_OPENGL)
    endif()
    
    # Link pthread for threading support
    find_package(Threads REQUIRED)
    target_link_libraries(gpu_info_demo Threads::Threads)
endif()

# Set C++ standard
set_target_properties(gpu_info_demo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Generate dSYM files on macOS for better debugging
if(APPLE)
    add_custom_command(TARGET gpu_info_demo POST_BUILD
        COMMAND dsymutil $<TARGET_FILE:gpu_info_demo>
        COMMENT "Generating dSYM file for gpu_info_demo"
    )
endif()

# Install the executable
install(TARGETS gpu_info_demo DESTINATION bin)

message(STATUS "GPU Info demo configured for ${CMAKE_SYSTEM_NAME}")