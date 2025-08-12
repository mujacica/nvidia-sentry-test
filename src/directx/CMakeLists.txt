if(WIN32)
    add_executable(directx_crash_demo 
        main.cpp 
        d3d11_renderer.cpp
    )

    set_target_properties(directx_crash_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    target_link_libraries(directx_crash_demo 
        sentry::sentry
        d3d11
        dxgi
        d3dcompiler
    )

    target_compile_definitions(directx_crash_demo PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )

    install(TARGETS directx_crash_demo DESTINATION bin)
else()
    message(STATUS "DirectX demo skipped - not building on Windows")
endif()