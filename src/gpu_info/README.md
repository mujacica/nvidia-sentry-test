# GPU Info Collection Demo

A cross-platform demonstration showing GPU information collection in Sentry events using the sentry-native SDK with GPU info capabilities.

## Purpose

This demo verifies that GPU information is properly collected and included in Sentry events across different platforms (Windows, macOS, Linux). It sends various types of events to Sentry and allows you to inspect them to confirm GPU context is attached.

## Features

- **Cross-Platform**: Works on Windows, macOS, and Linux
- **GPU Detection**: Platform-specific GPU enumeration and information collection
- **Multiple Event Types**: Tests info, warning, error, and exception events
- **Sentry Integration**: Uses sentry-native with GPU info collection enabled
- **Verification Tool**: Easy way to confirm GPU info is working in your Sentry project

## Platform-Specific GPU Detection

### Windows
- **DirectX/DXGI**: Enumerates GPU devices using DXGI factory
- **Information Collected**: GPU names, dedicated video memory, adapter descriptions
- **Libraries**: d3d11.dll, dxgi.dll

### macOS  
- **Metal**: Uses Metal framework for GPU device enumeration
- **Information Collected**: GPU device names, recommended working set size
- **Frameworks**: Metal.framework, CoreFoundation.framework

### Linux
- **OpenGL**: Basic GPU detection via OpenGL (when available)
- **System Info**: Collects system information and attempts GPU detection
- **Fallback**: Graceful degradation when GPU APIs are not available

## GPU Info in Sentry Events

When GPU info collection is enabled (via `SENTRY_WITH_GPU_INFO=ON`), Sentry automatically collects:

- **GPU Device Information**: Device names, vendor IDs, device IDs
- **Memory Information**: Total GPU memory, available memory
- **Driver Information**: Driver versions, API levels
- **Platform Context**: Operating system GPU subsystem details

This information appears in Sentry events under:
- **Device Context**: Hardware information including GPU details
- **Contexts Section**: Additional GPU-specific context data
- **Tags**: Automatically tagged with GPU-related information

## Prerequisites

- CMake 3.18+
- C++17 compatible compiler
- Sentry DSN (from your Sentry project)
- Platform-specific requirements:
  - **Windows**: Windows SDK with DirectX support
  - **macOS**: Xcode with Metal framework
  - **Linux**: OpenGL development headers (optional)

## Building

### Using Main Build System

```bash
# Enable GPU info demo in main CMake build
cmake -DBUILD_GPU_INFO=ON -DSENTRY_WITH_GPU_INFO=ON ..
make

# Or use build scripts
./build.sh  # Linux/macOS
build.bat   # Windows
```

### Standalone Build

```bash
cd src/gpu_info
mkdir build && cd build

# Configure with GPU info enabled
cmake -DSENTRY_WITH_GPU_INFO=ON ..

# Build
cmake --build .
```

## Usage

### Environment Setup

```bash
# Set your Sentry DSN (required)
export SENTRY_DSN="https://your-key@sentry.io/your-project-id"

# Optional: Set environment
export SENTRY_ENVIRONMENT="development"
```

### Running Tests

```bash
# Run basic GPU info test
./gpu_info_demo

# Run specific test type
./gpu_info_demo basic      # Basic info event
./gpu_info_demo warning    # Warning level event  
./gpu_info_demo error      # Error level event
./gpu_info_demo exception  # Exception with stack trace
./gpu_info_demo all        # All test types

# Windows
gpu_info_demo.exe [test_type]
```

## Expected Output

### Console Output
```
Sentry GPU Info Collection Demo
===============================

Sentry initialized successfully with GPU info collection enabled
DSN: https://your-key@sentry.io/your-project-id

=== GPU Information Collection Demo ===
Platform: Linux
Available GPU devices (via OpenGL):
  [0] NVIDIA GeForce RTX 4090 (8192 MB VRAM)

Running test type: basic

Sending Sentry event: basic_info
Event sent with ID: 12345678-1234-5678-9abc-def012345678

All events sent! Check your Sentry dashboard to verify GPU info is included.
GPU information should appear in the 'Device' or 'Contexts' section of each event.

Demo completed successfully!
```

### Sentry Dashboard

In your Sentry project, events should show:

**Device Context:**
```json
{
  "name": "My Computer",
  "family": "desktop",
  "model": "Custom Build",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4090",
    "vendor": "NVIDIA Corporation", 
    "memory_size": 25757220864,
    "api_type": "DirectX 12",
    "version": "531.68"
  }
}
```

**Event Tags:**
- `component: gpu_info`
- `platform: windows/linux/macos`  
- `test_type: basic/warning/error/exception`

## Verification Checklist

To verify GPU info collection is working:

1. **Run the demo**: Execute `./gpu_info_demo all`
2. **Check Sentry events**: Look for events in your Sentry dashboard
3. **Inspect device context**: Expand the "Device" section in event details
4. **Look for GPU data**: Verify GPU name, memory, and driver info are present
5. **Check multiple events**: Different event types should all include GPU info

## Troubleshooting

### "SENTRY_DSN not set"
```bash
export SENTRY_DSN="https://your-key@sentry.io/your-project-id"
```

### "Failed to initialize Sentry"
- Verify DSN format is correct
- Check network connectivity to sentry.io
- Ensure sentry-native was built with GPU info support

### "No GPU information in events"
- Verify `SENTRY_WITH_GPU_INFO=ON` was used during CMake configuration
- Check that sentry-native was built from the `native-gpu-info` branch
- Ensure platform-specific GPU libraries are available

### Platform-Specific Issues

#### Windows
- Install Windows SDK with DirectX support
- Ensure Visual Studio is properly configured

#### macOS  
- Install Xcode with Metal framework support
- May require code signing for system GPU access

#### Linux
- Install OpenGL development headers: `sudo apt install libgl1-mesa-dev`
- For NVIDIA: Install proprietary drivers for full GPU info

## Integration with Main Project

This GPU info demo complements other crash demos by:

1. **Baseline Testing**: Verifies GPU info collection works before crash testing
2. **Event Validation**: Confirms Sentry integration is working properly  
3. **Platform Testing**: Tests GPU detection across all supported platforms
4. **Debug Tool**: Helps troubleshoot GPU info issues in other demos

## Development

### Adding Platform Support

To add support for additional platforms:

1. Add platform detection in `main.cpp`
2. Include platform-specific GPU API headers
3. Implement GPU enumeration for the platform
4. Update CMakeLists.txt with required libraries
5. Test GPU info appears in Sentry events

### Extending GPU Detection

To enhance GPU detection:

1. Add more detailed GPU queries (compute units, clock speeds, etc.)
2. Include multiple GPU API support per platform
3. Add GPU memory usage monitoring
4. Include GPU driver/runtime version detection

## Resources

- [Sentry Native SDK Documentation](https://docs.sentry.io/platforms/native/)
- [DirectX Graphics Programming Guide](https://docs.microsoft.com/en-us/windows/win32/direct3d)
- [Metal Programming Guide](https://developer.apple.com/metal/)
- [OpenGL Documentation](https://www.opengl.org/documentation/)