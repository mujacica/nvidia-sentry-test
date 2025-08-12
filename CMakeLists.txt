cmake_minimum_required(VERSION 3.18)

project(SentryGraphicsDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for different graphics APIs
option(BUILD_CUDA "Build CUDA demo" ON)
option(BUILD_VULKAN "Build Vulkan demo" ON)
option(BUILD_DIRECTX "Build DirectX demo" ON)
option(BUILD_PYTHON_CUDA "Build Python CUDA demo" ON)
option(BUILD_GPU_INFO "Build GPU info demo" ON)
option(SENTRY_BUILD_TESTS "Build sentry-native tests" OFF)
option(SENTRY_BUILD_EXAMPLES "Build sentry-native examples" OFF)
option(SENTRY_WITH_GPU_INFO "Enable GPU info collection in sentry-native" ON)

# Platform-specific settings
if(WIN32)
    # Build everything on Windows
elseif(UNIX AND NOT APPLE)
    set(BUILD_DIRECTX OFF CACHE BOOL "DirectX not supported on Linux" FORCE)
elseif(APPLE)
    set(BUILD_PYTHON_CUDA OFF CACHE BOOL "CUDA not supported on macOS (no NVIDIA GPUs)" FORCE)
    set(BUILD_CUDA OFF CACHE BOOL "CUDA not supported on macOS (no NVIDIA GPUs)" FORCE)
    set(BUILD_VULKAN OFF CACHE BOOL "Vulkan not supported on macOS in this demo" FORCE)
    set(BUILD_DIRECTX OFF CACHE BOOL "DirectX not supported on macOS" FORCE)
endif()

# Platform validation - prevent unsupported combinations
if(BUILD_CUDA AND NOT (WIN32 OR (UNIX AND NOT APPLE)))
    message(FATAL_ERROR "CUDA demo is only supported on Windows and Linux (requires NVIDIA GPU)")
endif()

if(BUILD_VULKAN AND NOT ((UNIX AND NOT APPLE) OR WIN32))
    message(FATAL_ERROR "Vulkan demo is only supported on Linux and Windows")
endif()

if(BUILD_DIRECTX AND NOT WIN32)
    message(FATAL_ERROR "DirectX demo is only supported on Windows")
endif()

# GPU Info demo is supported on all platforms (Windows, Linux, macOS)
# CUDA demo is supported on Windows and Linux only (requires NVIDIA GPU)

# CUDA setup
if(BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDA REQUIRED)
endif()

# Vulkan setup
if(BUILD_VULKAN)
    find_package(Vulkan REQUIRED)
endif()

include(FetchContent)

FetchContent_Declare(
  sentry
  GIT_REPOSITORY https://github.com/getsentry/sentry-native.git
  GIT_TAG        native-gpu-info
)

FetchContent_MakeAvailable(sentry)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set install prefix to local install directory  
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install CACHE PATH "Installation directory" FORCE)

# Ensure comprehensive debug symbols are always included for better crash reporting
if(NOT MSVC)
    # For GCC/Clang, include comprehensive debug symbols with line info and inlining info
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -gdwarf-4 -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -gdwarf-4 -fno-omit-frame-pointer")
    
    # On macOS, also generate dSYM files for better debugging
    if(APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gfull")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gfull")
        set(CMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT "dwarf-with-dsym")
        set(CMAKE_XCODE_ATTRIBUTE_DWARF_DSYM_FOLDER_PATH "${CMAKE_BINARY_DIR}")
    endif()
    
    # Enable debug info for all build types
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g3 -gdwarf-4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g3 -gdwarf-4")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g3 -gdwarf-4")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g3 -gdwarf-4")
    set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} -g3 -gdwarf-4")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -g3 -gdwarf-4")
    
    # Include CUDA debug symbols if CUDA is enabled
    if(BUILD_CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -lineinfo")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -g -lineinfo")
    endif()
else()
    # For MSVC, ensure comprehensive debug info is generated for all build types
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /DEBUG:FULL")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zi /DEBUG:FULL")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi /DEBUG:FULL")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi /DEBUG:FULL")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Zi /DEBUG:FULL")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /Zi /DEBUG:FULL")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /Zi /DEBUG:FULL")
    set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} /Zi /DEBUG:FULL")
    
    # Ensure linker generates debug info for all build types
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG:FULL")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /DEBUG:FULL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG:FULL /OPT:REF /OPT:ICF")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG:FULL /OPT:REF /OPT:ICF")
endif()

add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()