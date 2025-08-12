#!/bin/bash

set -e

echo "Building Sentry Graphics Demo for Linux/macOS"

if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake."
    exit 1
fi

BUILD_DIR="build"
BUILD_TYPE="${1:-Debug}"
BUILD_CUDA="${BUILD_CUDA:-ON}"
BUILD_VULKAN="${BUILD_VULKAN:-ON}"
BUILD_PYTHON_CUDA="${BUILD_PYTHON_CUDA:-ON}"
BUILD_GPU_INFO="${BUILD_GPU_INFO:-ON}"

# Check for CUDA toolkit
if [[ "$BUILD_CUDA" == "ON" ]]; then
    if ! command -v nvcc &> /dev/null; then
        echo "Warning: CUDA toolkit not found. Disabling CUDA build."
        echo "Install CUDA toolkit and ensure nvcc is in PATH to build CUDA demo."
        BUILD_CUDA="OFF"
    fi
fi

# Check for Vulkan SDK
if [[ "$BUILD_VULKAN" == "ON" ]]; then
    if ! pkg-config --exists vulkan 2>/dev/null && [[ -z "$VULKAN_SDK" ]]; then
        echo "Warning: Vulkan SDK not found. Disabling Vulkan build."
        echo "Install Vulkan SDK or set VULKAN_SDK environment variable to build Vulkan demo."
        BUILD_VULKAN="OFF"
    fi
fi

# Check for Python 3
if [[ "$BUILD_PYTHON_CUDA" == "ON" ]]; then
    if ! command -v python3 &> /dev/null; then
        echo "Warning: Python 3 not found. Disabling Python CUDA build."
        echo "Install Python 3 to build Python CUDA demo."
        BUILD_PYTHON_CUDA="OFF"
    fi
fi

echo "Creating build directory: $BUILD_DIR"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

echo "Running CMake configuration..."
echo "  BUILD_CUDA: $BUILD_CUDA"
echo "  BUILD_VULKAN: $BUILD_VULKAN"
echo "  BUILD_PYTHON_CUDA: $BUILD_PYTHON_CUDA"
echo "  BUILD_GPU_INFO: $BUILD_GPU_INFO"
echo "  BUILD_DIRECTX: OFF (not supported on Linux/macOS)"

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DBUILD_TESTING=ON \
      -DBUILD_CUDA=$BUILD_CUDA \
      -DBUILD_VULKAN=$BUILD_VULKAN \
      -DBUILD_DIRECTX=OFF \
      -DBUILD_PYTHON_CUDA=$BUILD_PYTHON_CUDA \
      -DBUILD_GPU_INFO=$BUILD_GPU_INFO \
      ..

echo "Building project..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Installing project..."
cmake --install . --config $BUILD_TYPE

echo ""
echo "Build and install completed successfully!"

# Upload debug symbols to Sentry if environment variables are set
if [[ -n "$SENTRY_ORG" && -n "$SENTRY_PROJECT" && -n "$SENTRY_AUTH_TOKEN" ]]; then
    echo ""
    echo "Uploading debug symbols to Sentry..."
    
    SENTRY_CLI_PATH="./sentry-cli"
    
    # Download sentry-cli if not present
    if [[ ! -f "$SENTRY_CLI_PATH" ]]; then
        echo "Downloading sentry-cli..."
        
        # Detect OS and architecture
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        ARCH=$(uname -m)
        
        case $ARCH in
            x86_64) ARCH="x86_64" ;;
            arm64|aarch64) ARCH="aarch64" ;;
            *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        
        case $OS in
            linux) PLATFORM="Linux-$ARCH" ;;
            darwin) PLATFORM="Darwin-universal" ;;
            *) echo "Unsupported OS: $OS"; exit 1 ;;
        esac
        
        DOWNLOAD_URL="https://github.com/getsentry/sentry-cli/releases/latest/download/sentry-cli-$PLATFORM"
        
        if command -v curl &> /dev/null; then
            curl -L "$DOWNLOAD_URL" -o "$SENTRY_CLI_PATH" --silent --show-error
        elif command -v wget &> /dev/null; then
            wget "$DOWNLOAD_URL" -O "$SENTRY_CLI_PATH" -q
        else
            echo "Error: Neither curl nor wget found. Cannot download sentry-cli."
            echo "Please install curl or wget, or manually install sentry-cli."
            exit 1
        fi
        
        if [[ $? -eq 0 ]]; then
            chmod +x "$SENTRY_CLI_PATH"
            echo "sentry-cli downloaded successfully"
        else
            echo "Error: Failed to download sentry-cli"
            exit 1
        fi
    fi
    
    # Upload debug symbols for our demo binaries only (not crashpad_handler)
    UPLOAD_FILES=""
    
    if [[ "$BUILD_CUDA" == "ON" && -f "../install/bin/cuda_crash_demo" ]]; then
        UPLOAD_FILES="$UPLOAD_FILES ../install/bin/cuda_crash_demo"
    fi
    
    if [[ "$BUILD_VULKAN" == "ON" && -f "../install/bin/vulkan_crash_demo" ]]; then
        UPLOAD_FILES="$UPLOAD_FILES ../install/bin/vulkan_crash_demo"
    fi
    
    if [[ "$BUILD_GPU_INFO" == "ON" && -f "../install/bin/gpu_info_demo" ]]; then
        UPLOAD_FILES="$UPLOAD_FILES ../install/bin/gpu_info_demo"
    fi
    
    if [[ -n "$UPLOAD_FILES" ]]; then
        "$SENTRY_CLI_PATH" debug-files upload $UPLOAD_FILES
        if [[ $? -eq 0 ]]; then
            echo "Debug symbols uploaded successfully to Sentry!"
        else
            echo "Warning: Failed to upload debug symbols to Sentry"
        fi
    else
        echo "No demo binaries found to upload"
    fi
else
    echo ""
    echo "Skipping debug symbol upload (set SENTRY_ORG, SENTRY_PROJECT, and SENTRY_AUTH_TOKEN to enable)"
fi
echo ""
echo "Available executables:"

if [[ "$BUILD_CUDA" == "ON" ]]; then
    echo "  CUDA Demo: install/bin/cuda_crash_demo"
fi

if [[ "$BUILD_VULKAN" == "ON" ]]; then
    echo "  Vulkan Demo: install/bin/vulkan_crash_demo"
fi

if [[ "$BUILD_PYTHON_CUDA" == "ON" ]]; then
    echo "  Python CUDA Demo: $BUILD_DIR/src/python_cuda/"
fi

if [[ "$BUILD_GPU_INFO" == "ON" ]]; then
    echo "  GPU Info Demo: install/bin/gpu_info_demo"
fi

echo ""
echo "Usage:"
echo "  export SENTRY_DSN='your-sentry-dsn-here'"

if [[ "$BUILD_CUDA" == "ON" ]]; then
    echo ""
    echo "CUDA Demo:"
    echo "  cd install/bin && ./cuda_crash_demo [test_type]"
    echo "  Available tests: divide_by_zero, out_of_bounds, null_pointer, infinite_loop"
fi

if [[ "$BUILD_VULKAN" == "ON" ]]; then
    echo ""
    echo "Vulkan Demo:"
    echo "  cd install/bin && ./vulkan_crash_demo [test_type]"
    echo "  Available tests: invalid_buffer_access, invalid_command_buffer, out_of_bounds_descriptor, invalid_render_pass, device_lost_simulation"
fi

if [[ "$BUILD_PYTHON_CUDA" == "ON" ]]; then
    echo ""
    echo "Python CUDA Demo:"
    echo "  cd src/python_cuda"
    echo "  python3 setup.py                     # Setup dependencies"
    echo "  python3 test_installation.py         # Test installation"
    echo "  python3 cuda_crash_demo.py [test_type]"
    echo "  Available tests: cupy_memory_exhaustion, pycuda_driver_crash, mixed_library_conflicts, all"
fi

if [[ "$BUILD_GPU_INFO" == "ON" ]]; then
    echo ""
    echo "GPU Info Demo:"
    echo "  cd install/bin && ./gpu_info_demo [test_type]"
    echo "  Available tests: basic, warning, error, exception, crash, all"
fi

echo ""
echo "Run without arguments to execute all tests for each demo"