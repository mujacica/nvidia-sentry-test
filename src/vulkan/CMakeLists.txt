add_executable(vulkan_crash_demo 
    main.cpp 
    vulkan_renderer.cpp
)

set_target_properties(vulkan_crash_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(vulkan_crash_demo 
    sentry::sentry
    Vulkan::Vulkan
)

target_include_directories(vulkan_crash_demo PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
)

install(TARGETS vulkan_crash_demo DESTINATION bin)