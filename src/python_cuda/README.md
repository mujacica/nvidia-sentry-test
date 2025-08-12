# Python CUDA Crash Demo

A comprehensive Python demonstration showing how to integrate Sentry error reporting with CUDA applications using popular Python CUDA libraries (CuPy and PyCUDA).

## Features

- **Multiple CUDA Libraries**: Supports both CuPy and PyCUDA
- **Comprehensive Crash Scenarios**: 7 different crash types targeting various CUDA subsystems
- **Advanced Sentry Integration**: Detailed GPU context, memory info, and crash classification
- **Cross-Library Testing**: Tests conflicts between different CUDA Python libraries
- **Easy Setup**: Automated dependency installation and verification

## Crash Scenarios

### CuPy Crash Scenarios

#### 1. **Memory Exhaustion** (`cupy_memory_exhaustion`)
- **Description**: Attempts to allocate excessive GPU memory (1000+ GB)
- **Target**: GPU memory allocator and driver memory management
- **Expected Result**: `cupy.cuda.memory.MemoryError` or system crash

#### 2. **Invalid Kernel Launch** (`cupy_invalid_kernel`) 
- **Description**: Custom kernel with divide-by-zero and null pointer access
- **Target**: CUDA kernel execution and error handling
- **Expected Result**: Kernel execution errors or GPU hang

#### 3. **Device Memory Corruption** (`cupy_memory_corruption`)
- **Description**: Kernel that writes to invalid memory addresses
- **Target**: GPU memory protection and driver stability
- **Expected Result**: Memory access violations or driver crash

#### 4. **Concurrent Stream Destruction** (`cupy_stream_destruction`)
- **Description**: Destroys CUDA streams while kernels are executing
- **Target**: CUDA stream management and synchronization
- **Expected Result**: Stream synchronization errors or hangs

### PyCUDA Crash Scenarios

#### 5. **Driver Crash** (`pycuda_driver_crash`)
- **Description**: Infinite loop kernel with excessive memory allocation
- **Target**: CUDA driver timeout mechanisms and recovery
- **Expected Result**: Driver reset or system freeze

#### 6. **Context Corruption** (`pycuda_context_corruption`)
- **Description**: Multiple contexts with improper cleanup and memory corruption
- **Target**: CUDA context lifecycle management
- **Expected Result**: Context errors or driver instability

### Mixed Library Scenarios

#### 7. **Library Conflicts** (`mixed_library_conflicts`)
- **Description**: Mixing CuPy and PyCUDA memory management inappropriately
- **Target**: Inter-library compatibility and memory ownership
- **Expected Result**: Memory management errors or corruption

## Installation

### Prerequisites

- Python 3.8+
- CUDA Toolkit 11.0+ (with `nvcc` in PATH)
- NVIDIA GPU with compute capability 3.5+
- pip package manager

### Quick Setup

```bash
cd src/python_cuda

# Run automated setup
python setup.py

# Verify installation
python test_installation.py
```

### Manual Installation

```bash
# Install base requirements
pip install sentry-sdk>=1.40.0 numpy>=1.24.0 psutil>=5.9.0

# Install CUDA libraries (choose based on your CUDA version)
pip install cupy-cuda12x  # For CUDA 12.x
# OR
pip install cupy-cuda11x  # For CUDA 11.x

# Install PyCUDA
pip install pycuda
```

## Usage

### Environment Setup

```bash
# Set your Sentry DSN (required for error reporting)
export SENTRY_DSN="your-sentry-dsn-here"

# Optional: Set environment
export SENTRY_ENVIRONMENT="development"
```

### Running Tests

```bash
# List all available tests
python cuda_crash_demo.py --list-tests

# Run all crash scenarios
python cuda_crash_demo.py all

# Run specific crash scenario
python cuda_crash_demo.py cupy_memory_exhaustion
python cuda_crash_demo.py pycuda_driver_crash
python cuda_crash_demo.py mixed_library_conflicts

# Verify installation
python test_installation.py
```

### Available Test Types

| Test Name | Library | Risk Level | Description |
|-----------|---------|------------|-------------|
| `cupy_memory_exhaustion` | CuPy | ⚠️ Moderate | GPU memory exhaustion |
| `cupy_invalid_kernel` | CuPy | 🔥 High | Invalid kernel operations |
| `cupy_memory_corruption` | CuPy | 🔥 High | Device memory corruption |
| `cupy_stream_destruction` | CuPy | ⚠️ Moderate | Stream lifecycle violations |
| `pycuda_driver_crash` | PyCUDA | ☢️ Extreme | Driver-level crashes |
| `pycuda_context_corruption` | PyCUDA | 🔥 High | Context management errors |
| `mixed_library_conflicts` | Both | 🔥 High | Cross-library incompatibilities |

## Sentry Integration

### Automatic Context Collection

The demo automatically collects and reports:

- **GPU Information**: Device name, compute capability, memory info
- **Library Versions**: CuPy, PyCUDA, NumPy versions
- **CUDA Environment**: Runtime version, driver version, device count
- **Memory State**: Available/total GPU memory at crash time
- **Crash Context**: Error type, stack traces, operation details

### Custom Error Tags

Each crash type is tagged for easy filtering:

```python
# Example Sentry event tags
{
    "component": "python_cuda",
    "language": "python", 
    "crash_type": "cupy_memory_exhaustion",
    "cupy_available": true,
    "pycuda_available": true
}
```

### Example Sentry Event

```json
{
    "message": "CuPy memory exhaustion crash",
    "level": "error",
    "tags": {
        "crash_type": "cupy_memory_exhaustion",
        "component": "python_cuda"
    },
    "extra": {
        "cupy_version": "12.3.0",
        "cuda_runtime_version": 12030,
        "gpu_device_name": "NVIDIA GeForce RTX 4090",
        "gpu_memory_total": 25757220864,
        "arrays_allocated": 42,
        "total_memory_attempted": 107374182400
    }
}
```

## Safety Warnings ⚠️

These tests are designed to crash CUDA drivers and may:

- **Hang the GPU** requiring driver restart
- **Crash the system** requiring reboot  
- **Corrupt GPU state** requiring power cycle
- **Cause data loss** if unsaved work is open

**Always save your work before running these tests!**

## Troubleshooting

### Common Issues

#### "No module named 'cupy'"
- Install CuPy: `pip install cupy-cuda12x`
- Verify CUDA installation: `nvcc --version`

#### "PyCUDA installation failed"
- Install build tools: `pip install wheel setuptools`
- On Linux: `apt install build-essential python3-dev`
- On Windows: Install Visual Studio with C++ tools

#### "CUDA not found"
- Ensure CUDA Toolkit is installed
- Add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`
- Verify with: `nvcc --version`

#### "Sentry not reporting errors"  
- Check SENTRY_DSN is set: `echo $SENTRY_DSN`
- Verify DSN format: `https://key@sentry.io/project`
- Check network connectivity to sentry.io

### Platform-Specific Notes

#### Linux
```bash
# Install CUDA toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Install development headers
sudo apt install python3-dev build-essential
```

#### Windows
- Install CUDA Toolkit from NVIDIA website
- Install Visual Studio with C++ support
- Use Command Prompt or PowerShell (not WSL) for better compatibility

#### macOS
- CUDA not supported on recent macOS versions
- Use CPU-only mode for testing (limited functionality)

## Integration with Main Project

This Python demo complements the C++/CUDA demos by providing:

1. **Higher-Level API Testing**: Tests Python CUDA libraries vs raw CUDA
2. **Different Crash Vectors**: Python-specific memory management issues
3. **Library Interaction Bugs**: Cross-library compatibility problems
4. **Rapid Prototyping**: Easier to modify and extend test scenarios

## Development

### Adding New Crash Scenarios

1. Add test method to `PythonCudaCrashDemo` class
2. Update `available_tests` dictionary in `main()`
3. Add argument handling in main function
4. Include Sentry error reporting with appropriate context

### Example New Test

```python
def test_my_new_crash(self):
    """Test description."""
    print("\n--- Testing My New Crash ---")
    
    try:
        # Crash scenario implementation
        pass
        
    except Exception as e:
        self.report_cuda_error(e, "my_new_crash", {
            "custom_context": "additional_info"
        })
```

## Resources

- [CuPy Documentation](https://cupy.dev/)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [Sentry Python SDK](https://docs.sentry.io/platforms/python/)
- [CUDA Python Documentation](https://nvidia.github.io/cuda-python/)