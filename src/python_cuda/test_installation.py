#!/usr/bin/env python3
"""
Test script to verify Python CUDA installation and basic functionality.
"""

import sys
import traceback

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
        
    try:
        import sentry_sdk
        print(f"✓ Sentry SDK available")
    except ImportError as e:
        print(f"✗ Sentry SDK import failed: {e}")
        return False
        
    return True

def test_cupy():
    """Test CuPy installation and basic functionality."""
    print("\nTesting CuPy...")
    
    try:
        import cupy as cp
        print(f"✓ CuPy {cp.__version__}")
        
        # Test basic array creation
        x = cp.array([1, 2, 3, 4, 5])
        y = x * 2
        result = cp.asnumpy(y)
        print(f"✓ CuPy basic operations: {result}")
        
        # Test GPU info
        print(f"✓ GPU Device: {cp.cuda.Device().name.decode()}")
        print(f"✓ CUDA Runtime: {cp.cuda.runtime.runtimeGetVersion()}")
        
        # Test memory info
        meminfo = cp.cuda.Device().mem_info
        print(f"✓ GPU Memory: {meminfo[0] / 1024**3:.1f}GB free / {meminfo[1] / 1024**3:.1f}GB total")
        
        return True
        
    except ImportError as e:
        print(f"✗ CuPy not available: {e}")
        return False
    except Exception as e:
        print(f"✗ CuPy test failed: {e}")
        traceback.print_exc()
        return False

def test_pycuda():
    """Test PyCUDA installation and basic functionality."""
    print("\nTesting PyCUDA...")
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        import numpy as np
        
        print(f"✓ PyCUDA available")
        
        # Test device info
        print(f"✓ Device count: {cuda.Device.count()}")
        if cuda.Device.count() > 0:
            dev = cuda.Device(0)
            print(f"✓ Device 0: {dev.name()}")
            print(f"✓ Compute capability: {dev.compute_capability()}")
            print(f"✓ Total memory: {dev.total_memory() / 1024**3:.1f}GB")
        
        # Test basic kernel compilation and execution
        mod = SourceModule("""
        __global__ void multiply_them(float *dest, float *a, float *b)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            dest[i] = a[i] * b[i];
        }
        """)
        
        multiply_them = mod.get_function("multiply_them")
        
        a = np.random.randn(400).astype(np.float32)
        b = np.random.randn(400).astype(np.float32)
        
        dest = np.zeros_like(a)
        multiply_them(
            cuda.Out(dest), cuda.In(a), cuda.In(b),
            block=(400,1,1), grid=(1,1))
        
        expected = a * b
        if np.allclose(dest, expected):
            print("✓ PyCUDA kernel execution successful")
        else:
            print("✗ PyCUDA kernel execution failed")
            return False
            
        return True
        
    except ImportError as e:
        print(f"✗ PyCUDA not available: {e}")
        return False
    except Exception as e:
        print(f"✗ PyCUDA test failed: {e}")
        traceback.print_exc()
        return False

def test_sentry_integration():
    """Test Sentry integration."""
    print("\nTesting Sentry integration...")
    
    try:
        import sentry_sdk
        import os
        
        dsn = os.getenv('SENTRY_DSN')
        if not dsn:
            print("⚠ SENTRY_DSN not set - Sentry will not report errors")
            print("  Set SENTRY_DSN environment variable to test error reporting")
            return True
            
        # Initialize Sentry
        sentry_sdk.init(dsn=dsn, debug=True)
        print("✓ Sentry initialized successfully")
        
        # Test capture
        try:
            raise Exception("Test exception for Sentry")
        except Exception as e:
            sentry_sdk.capture_exception(e)
            print("✓ Test exception sent to Sentry")
            
        return True
        
    except Exception as e:
        print(f"✗ Sentry test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Python CUDA Installation Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test basic imports
    if not test_basic_imports():
        all_passed = False
    
    # Test CUDA libraries
    cupy_available = test_cupy()
    pycuda_available = test_pycuda()
    
    if not (cupy_available or pycuda_available):
        print("\n⚠ Warning: Neither CuPy nor PyCUDA available")
        print("  The crash demo will run but skip CUDA-specific tests")
        
    # Test Sentry
    if not test_sentry_integration():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! Ready to run crash demos.")
    else:
        print("✗ Some tests failed. Check installation.")
        
    print(f"\nPython: {sys.version}")
    print(f"CuPy available: {cupy_available}")
    print(f"PyCUDA available: {pycuda_available}")

if __name__ == '__main__':
    main()