# Sentry GPU Demos

A comprehensive demonstration project showing how to integrate [Sentry Native SDK](https://github.com/getsentry/sentry-native) with multiple GPU Frameworks and APIs for comprehensive error reporting and crash monitoring across different platforms.

## Features

- **Multi-API Support**: CUDA, Vulkan, DirectX, Python CUDA, and GPU info collection demos
- **Platform-Specific**: CUDA (Windows/Linux), Vulkan (Windows/Linux), DirectX (Windows), GPU Info (all platforms)  
- **Sentry Integration**: Automatic error reporting with detailed GPU context
- **GPU Info Collection**: Dedicated demo for GPU hardware information gathering
- **Python Integration**: Python CUDA demo with advanced library conflict scenarios
- **Comprehensive Testing**: Unit tests for all GPU APIs
- **Multiple Build Systems**: CMake, Make, and platform-specific scripts
- **GPU Context Reporting**: Uses Sentry's native-gpu-info branch for enhanced GPU diagnostics
- **Automatic Installation**: Installs all binaries and dependencies to a single location

## Prerequisites

### System Requirements
- CMake 3.18+
- C++17 compatible compiler  
- Git

### Graphics API Requirements

#### CUDA (Windows/Linux Only)
- CUDA Toolkit 11.0+ (with `nvcc` in PATH)
- NVIDIA GPU with compute capability 5.0+
- Not supported on macOS (no NVIDIA GPU support)

#### Vulkan (Windows/Linux)
- Vulkan SDK 1.2+
- Vulkan-compatible GPU and drivers

#### DirectX (Windows Only)
- Windows SDK 10+
- DirectX 11 compatible GPU and drivers
- Visual Studio with Windows development tools

#### Python CUDA (Windows/Linux Only)
- Python 3.6+
- CUDA Toolkit (same as CUDA requirements)
- Python packages: CuPy, PyCUDA (automatically installed)
- Not supported on macOS (no NVIDIA GPU support)

#### GPU Info Demo (Cross-Platform)
- No additional requirements beyond base system
- Uses platform-native APIs: DirectX (Windows), Metal (macOS), OpenGL (Linux)

### Platform-Specific Setup

#### Linux (Ubuntu/Debian)
```bash
# Base requirements
sudo apt update
sudo apt install build-essential cmake git

# CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit

# Vulkan SDK
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
sudo apt update
sudo apt install vulkan-sdk
```

#### Windows
- Visual Studio 2019+ with C++ and Windows SDK support
- CUDA Toolkit from NVIDIA website (for CUDA demos)
- Vulkan SDK from LunarG (for Vulkan demos)
- CMake (via Visual Studio Installer or standalone)

## Quick Start

### 1. Clone and Build

#### Linux/macOS
```bash
git clone <your-repo-url>
cd nvidia-sentry-test
./build.sh
```

#### Windows
```cmd
git clone <your-repo-url>
cd nvidia-sentry-test
build.bat
```

The build system automatically detects available graphics APIs and builds only the supported demos for your platform. All binaries and dependencies are automatically installed to the `install/` directory.

### 2. Set Up Sentry

1. Create a new project at [sentry.io](https://sentry.io)
2. Copy your project's DSN
3. Set the environment variable:

```bash
# Linux/macOS
export SENTRY_DSN="your-dsn-here"

# Optional: For debug symbol upload (enables symbolicated stack traces)
export SENTRY_ORG="your-sentry-organization"
export SENTRY_PROJECT="your-project-name"
export SENTRY_AUTH_TOKEN="your-auth-token"

# Windows
set SENTRY_DSN=your-dsn-here

REM Optional: For debug symbol upload
set SENTRY_ORG=your-sentry-organization
set SENTRY_PROJECT=your-project-name
set SENTRY_AUTH_TOKEN=your-auth-token
```

**Note**: To enable automatic debug symbol upload during build:
1. Set the three Sentry environment variables above
2. The build scripts will automatically download and use sentry-cli to upload debug symbols
3. Requires `curl`/`wget` (Linux/macOS) or PowerShell (Windows) for downloading sentry-cli

### 3. Run the Demos

All executables are installed to the `install/bin/` directory after building.

#### CUDA Demo (Cross-Platform)
```bash
# Linux/macOS
cd install/bin && ./cuda_crash_demo

# Run specific crash type
cd install/bin && ./cuda_crash_demo divide_by_zero
cd install/bin && ./cuda_crash_demo out_of_bounds
cd install/bin && ./cuda_crash_demo null_pointer
cd install/bin && ./cuda_crash_demo infinite_loop

# Windows
cd install\bin && .\cuda_crash_demo.exe [test_type]
```

#### Vulkan Demo (Windows/Linux)
```bash
# Linux - Run all Vulkan crash scenarios
cd install/bin && ./vulkan_crash_demo

# Run specific crash type
cd install/bin && ./vulkan_crash_demo invalid_buffer_access
cd install/bin && ./vulkan_crash_demo invalid_command_buffer
cd install/bin && ./vulkan_crash_demo out_of_bounds_descriptor
cd install/bin && ./vulkan_crash_demo invalid_render_pass
cd install/bin && ./vulkan_crash_demo device_lost_simulation
```

```cmd
REM Windows - Run all Vulkan crash scenarios
cd install\bin && .\vulkan_crash_demo.exe

REM Run specific crash type
cd install\bin && .\vulkan_crash_demo.exe invalid_buffer_access
cd install\bin && .\vulkan_crash_demo.exe invalid_command_buffer
cd install\bin && .\vulkan_crash_demo.exe out_of_bounds_descriptor
cd install\bin && .\vulkan_crash_demo.exe invalid_render_pass
cd install\bin && .\vulkan_crash_demo.exe device_lost_simulation
```

#### DirectX Demo (Windows Only)
```cmd
REM Run all DirectX crash scenarios
cd install\bin && .\directx_crash_demo.exe

REM Run specific crash type
cd install\bin && .\directx_crash_demo.exe invalid_buffer_access
cd install\bin && .\directx_crash_demo.exe invalid_shader_resource
cd install\bin && .\directx_crash_demo.exe device_removed_simulation
cd install\bin && .\directx_crash_demo.exe invalid_render_target
cd install\bin && .\directx_crash_demo.exe out_of_bounds_vertex_buffer
```

#### GPU Info Demo (Cross-Platform)
```bash
# Linux/macOS - Run all info collection tests
cd install/bin && ./gpu_info_demo

# Run specific test types
cd install/bin && ./gpu_info_demo basic
cd install/bin && ./gpu_info_demo warning
cd install/bin && ./gpu_info_demo error
cd install/bin && ./gpu_info_demo exception
cd install/bin && ./gpu_info_demo crash

# Windows
cd install\bin && .\gpu_info_demo.exe [test_type]
```

#### Python CUDA Demo (Cross-Platform)
```bash
# Setup dependencies (run once)
cd src/python_cuda
python3 setup.py

# Test installation
python3 test_installation.py

# Run crash scenarios
python3 cuda_crash_demo.py
python3 cuda_crash_demo.py cupy_memory_exhaustion
python3 cuda_crash_demo.py pycuda_driver_crash
python3 cuda_crash_demo.py mixed_library_conflicts
python3 cuda_crash_demo.py all
```

## Build Options

### Using CMake Directly
```bash
mkdir build && cd build

# Configure with specific APIs
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DBUILD_CUDA=ON \
      -DBUILD_VULKAN=ON \
      -DBUILD_DIRECTX=OFF \
      -DBUILD_PYTHON_CUDA=ON \
      -DBUILD_GPU_INFO=ON \
      ..

make -j$(nproc)

# Install to local install/ directory
cmake --install . --config Release
```

### Using Makefile (Linux/macOS)
```bash
make                          # Build all available APIs
make cuda                     # Build only CUDA demo
make vulkan                   # Build only Vulkan demo  
make BUILD_TYPE=Debug         # Debug build
make test                     # Build and run tests
make clean                    # Clean build and install directories
make install                  # Build and install everything
```

### Windows Visual Studio
```cmd
mkdir build && cd build
cmake -G "Visual Studio 16 2019" -A x64 -DBUILD_CUDA=ON -DBUILD_DIRECTX=ON ..
cmake --build . --config Release
```

## Crash Scenarios by API

### CUDA Crash Scenarios

#### 1. Divide by Zero (`divide_by_zero`)
- **Description**: Attempts division by zero in CUDA kernel
- **Expected Result**: CUDA runtime error with detailed context

#### 2. Out of Bounds Access (`out_of_bounds`)
- **Description**: Accesses memory outside allocated buffer  
- **Expected Result**: Memory access violation with stack trace

#### 3. Null Pointer Dereference (`null_pointer`)
- **Description**: Dereferences null pointer in kernel
- **Expected Result**: Segmentation fault with CUDA context

#### 4. Infinite Loop (`infinite_loop`)
- **Description**: Creates infinite loop causing kernel timeout
- **Expected Result**: Kernel execution timeout error

### Vulkan Crash Scenarios

#### 1. Invalid Buffer Access (`invalid_buffer_access`)
- **Description**: Attempts to bind invalid buffer to graphics pipeline
- **Expected Result**: Vulkan validation layer error

#### 2. Invalid Command Buffer (`invalid_command_buffer`)
- **Description**: Submits null command buffer to queue
- **Expected Result**: VK_ERROR_INVALID_HANDLE

#### 3. Out of Bounds Descriptor (`out_of_bounds_descriptor`) 
- **Description**: Creates pipeline with invalid descriptor set layout
- **Expected Result**: VK_ERROR_INVALID_DESCRIPTOR_SET_LAYOUT

#### 4. Invalid Render Pass (`invalid_render_pass`)
- **Description**: Creates framebuffer with null render pass
- **Expected Result**: VK_ERROR_INVALID_RENDER_PASS

#### 5. Device Lost Simulation (`device_lost_simulation`)
- **Description**: Attempts to allocate excessive memory to trigger device loss
- **Expected Result**: VK_ERROR_DEVICE_LOST

### DirectX Crash Scenarios

#### 1. Invalid Buffer Access (`invalid_buffer_access`)
- **Description**: Maps null buffer for writing
- **Expected Result**: HRESULT error with invalid buffer handle

#### 2. Invalid Shader Resource (`invalid_shader_resource`)
- **Description**: Binds null shader resource view to pipeline
- **Expected Result**: Potential device removal or validation error

#### 3. Device Removed Simulation (`device_removed_simulation`)
- **Description**: Allocates excessive GPU memory to trigger device removal
- **Expected Result**: DXGI_ERROR_DEVICE_REMOVED

#### 4. Invalid Render Target (`invalid_render_target`)
- **Description**: Clears null render target view
- **Expected Result**: D3D11 validation error

#### 5. Out of Bounds Vertex Buffer (`out_of_bounds_vertex_buffer`)
- **Description**: Draws more vertices than exist in buffer
- **Expected Result**: Potential device removal or rendering artifacts

### GPU Info Demo Scenarios

#### 1. Basic Info Collection (`basic`)
- **Description**: Collects and reports basic GPU hardware information
- **Expected Result**: Sentry event with GPU details in device context

#### 2. Warning Level Event (`warning`)
- **Description**: Sends a warning-level event with GPU context
- **Expected Result**: Warning-level Sentry event with hardware info

#### 3. Error Level Event (`error`)
- **Description**: Sends an error-level event with GPU context
- **Expected Result**: Error-level Sentry event with hardware info

#### 4. Exception Handling (`exception`)
- **Description**: Triggers and handles C++ exception with GPU context
- **Expected Result**: Exception event with stack trace and GPU info

#### 5. Crash Test (`crash`)
- **Description**: Triggers an unhandled segmentation fault for crashpad to catch
- **Expected Result**: Crash report with stack trace and GPU context information

#### 6. All Tests (`all`)
- **Description**: Runs all GPU info collection scenarios sequentially
- **Expected Result**: Multiple Sentry events with different severity levels and a crash report

### Python CUDA Demo Scenarios

#### 1. CuPy Memory Exhaustion (`cupy_memory_exhaustion`)
- **Description**: Attempts to allocate excessive GPU memory using CuPy
- **Expected Result**: CUDA out of memory error via CuPy

#### 2. PyCUDA Driver Crash (`pycuda_driver_crash`)
- **Description**: Attempts invalid operations through PyCUDA driver API
- **Expected Result**: CUDA driver error through PyCUDA

#### 3. Mixed Library Conflicts (`mixed_library_conflicts`)
- **Description**: Creates conflicts between CuPy and PyCUDA contexts
- **Expected Result**: Context conflicts and library interaction errors

#### 4. All Python Tests (`all`)
- **Description**: Runs all Python CUDA crash scenarios
- **Expected Result**: Multiple Python crash reports with CUDA context

## Project Structure

```
nvidia-sentry-test/
├── CMakeLists.txt              # Main CMake configuration
├── Makefile                   # Cross-platform Makefile
├── build.sh                  # Linux/macOS build script  
├── build.bat                 # Windows build script
├── README.md                 # This file
├── .gitignore                # Git ignore rules
├── build/                    # Build directory (created during build)
├── install/                  # Installation directory (created during install)
│   └── bin/                  # All executable binaries and dependencies
│       ├── cuda_crash_demo   # CUDA demo executable
│       ├── vulkan_crash_demo # Vulkan demo executable  
│       ├── directx_crash_demo.exe # DirectX demo executable (Windows)
│       ├── gpu_info_demo     # GPU info demo executable
│       └── crashpad_handler  # Sentry crash handler
├── src/                      # Source code for all graphics APIs
│   ├── CMakeLists.txt        # Main source CMake config
│   ├── cuda/                 # CUDA demo (cross-platform)
│   │   ├── CMakeLists.txt    
│   │   ├── main.cpp          # CUDA demo with Sentry integration
│   │   ├── cuda_kernels.cu   # CUDA kernel implementations
│   │   └── cuda_kernels.h    # CUDA kernel headers
│   ├── vulkan/               # Vulkan demo (Linux only)
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp          # Vulkan demo with Sentry integration  
│   │   ├── vulkan_renderer.cpp # Vulkan renderer implementation
│   │   └── vulkan_renderer.h   # Vulkan renderer headers
│   ├── directx/              # DirectX demo (Windows only)
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp          # DirectX demo with Sentry integration
│   │   ├── d3d11_renderer.cpp # DirectX 11 renderer implementation
│   │   └── d3d11_renderer.h   # DirectX 11 renderer headers
│   ├── gpu_info/             # GPU info collection demo (cross-platform)
│   │   ├── CMakeLists.txt
│   │   ├── main.cpp          # GPU info demo with Sentry integration
│   │   └── README.md         # GPU info demo documentation
│   └── python_cuda/          # Python CUDA demo (cross-platform)
│       ├── CMakeLists.txt
│       ├── cuda_crash_demo.py # Main Python CUDA crash scenarios
│       ├── setup.py          # Python dependencies setup
│       ├── test_installation.py # Installation verification
│       ├── requirements.txt  # Python package requirements
│       └── README.md         # Python CUDA demo documentation
└── tests/                    # Unit tests for all APIs
    ├── CMakeLists.txt        # Test configuration
    ├── test_cuda_sentry.cpp  # CUDA tests
    ├── test_vulkan_sentry.cpp # Vulkan tests  
    └── test_directx_sentry.cpp # DirectX tests
```

## Configuration

### Sentry Configuration
All demos support Sentry configuration through environment variables:

- `SENTRY_DSN`: Your Sentry project DSN (required for reporting)
- `SENTRY_ENVIRONMENT`: Environment tag (e.g., "development", "production")  
- `SENTRY_RELEASE`: Release version for tracking

#### Debug Symbol Upload (Optional)
For symbolicated stack traces in crash reports, set these additional variables:

- `SENTRY_ORG`: Your Sentry organization slug
- `SENTRY_PROJECT`: Your Sentry project slug  
- `SENTRY_AUTH_TOKEN`: Sentry auth token with `project:releases` scope

When these are set, the build scripts will automatically download and use sentry-cli to upload debug symbols. This enables:
- Function names in stack traces instead of memory addresses
- Source file names and line numbers in crash reports  
- Better debugging information for GPU-related crashes
- No manual sentry-cli installation required

### Graphics API Configuration

#### CUDA Configuration
- Automatically detects available CUDA devices
- Supports CUDA architectures from compute capability 5.0+
- Memory allocation size can be modified in `src/cuda/main.cpp`

#### Vulkan Configuration  
- Automatically detects Vulkan-compatible devices
- Uses validation layers in debug builds for enhanced error detection
- Requires Vulkan SDK installation (Windows: LunarG Vulkan SDK, Linux: vulkan-sdk package)

#### DirectX Configuration
- Automatically detects DirectX 11 compatible devices
- Uses D3D11 debug layer in debug builds
- Requires Windows SDK with DirectX support

## Testing

### Run Unit Tests
```bash
# After building with BUILD_TESTING=ON
cd build
ctest --output-on-failure

# Or using Makefile (Linux/macOS)
make test

# Run specific API tests
ctest -R CudaSentryTests
ctest -R VulkanSentryTests  
ctest -R DirectXSentryTests
```

### Manual Testing

#### General Testing Steps
1. Run demos without `SENTRY_DSN` to test local error handling
2. Set `SENTRY_DSN` and run specific crash scenarios  
3. Check your Sentry dashboard for reported errors with graphics context

#### API-Specific Testing
- **CUDA**: Test on systems with different NVIDIA GPU generations (Windows/Linux)
- **Vulkan**: Test with different Vulkan drivers (NVIDIA, AMD, Intel) on Windows and Linux
- **DirectX**: Test on systems with different DirectX versions (Windows only)

## Troubleshooting

### Common Issues

#### Graphics API Not Available
- **CUDA**: Ensure CUDA Toolkit is installed and `nvcc` is in PATH (`nvcc --version`)
- **Vulkan**: Install Vulkan SDK and verify with `vulkaninfo` (Windows: LunarG installer, Linux: vulkan-sdk package)
- **DirectX**: Ensure Windows SDK with DirectX support is installed

#### No Graphics Devices Detected
- **CUDA**: Check NVIDIA driver with `nvidia-smi` 
- **Vulkan**: Verify GPU support with `vkcube` or `vulkaninfo`
- **DirectX**: Check DirectX diagnostics with `dxdiag`

#### "Sentry initialization failed" or "invalid handler_path"
- Verify `SENTRY_DSN` format and validity
- Ensure you ran the build with installation: `./build.sh` or `make install`
- Check that `crashpad_handler` exists in `install/bin/` directory
- Run executables from `install/bin/` directory (where crashpad_handler is located)
- Check network connectivity to Sentry
- Review Sentry project settings

### Build Issues

#### Missing Dependencies (Linux)
```bash
# Base development tools
sudo apt install build-essential cmake git

# Graphics API dependencies
sudo apt install cuda-toolkit-11-8        # CUDA
sudo apt install vulkan-sdk               # Vulkan
sudo apt install libvulkan-dev vulkan-utils # Additional Vulkan tools
```

#### Missing Dependencies (Windows)
- Install Visual Studio with C++ workload and Windows SDK
- Install CUDA Toolkit from NVIDIA (for CUDA demos)
- Install Vulkan SDK from LunarG (for Vulkan demos)
- Ensure Windows SDK includes DirectX headers (for DirectX demos)

#### CMake Configuration Issues
- Minimum CMake version: 3.18
- For Vulkan: Set `VULKAN_SDK` environment variable if not auto-detected
- For CUDA: Ensure CUDA_PATH environment variable is set

#### Platform-Specific Build Issues

##### Linux/macOS
```bash
# Fix pkg-config issues for Vulkan
export PKG_CONFIG_PATH=$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH

# Fix CUDA path issues
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

##### Windows
```cmd
REM Fix CUDA path issues
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set PATH=%CUDA_PATH%\bin;%PATH%

REM Fix DirectX issues
set WindowsSdkDir="C:\Program Files (x86)\Windows Kits\10"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is provided as a demonstration and learning resource. Please refer to individual component licenses:
- Sentry Native SDK: MIT License
- CUDA samples: NVIDIA License
- Vulkan samples: Apache 2.0 License
- DirectX samples: Microsoft License

## Resources

### Documentation
- [Sentry Native SDK Documentation](https://docs.sentry.io/platforms/native/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Vulkan Specification](https://www.khronos.org/vulkan/)
- [DirectX 11 Documentation](https://docs.microsoft.com/en-us/windows/win32/direct3d11/atoc-dx-graphics-direct3d-11)
- [CMake Documentation](https://cmake.org/documentation/)

### SDKs and Tools
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Vulkan SDK](https://vulkan.lunarg.com/)
- [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/)
- [Sentry](https://sentry.io/)

### Graphics APIs
- [CUDA Zone](https://developer.nvidia.com/cuda-zone)
- [Vulkan Developer Portal](https://www.khronos.org/developers/vulkan/)
- [DirectX Developer Center](https://devblogs.microsoft.com/directx/)