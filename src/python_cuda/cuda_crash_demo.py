#!/usr/bin/env python3
"""
Sentry Python CUDA Crash Demo

Demonstrates various CUDA crash scenarios using Python CUDA libraries
with comprehensive error reporting to Sentry.
"""

import sys
import os
import traceback
import gc
from typing import Dict, Any, List, Optional
import argparse

import numpy as np
import sentry_sdk
from sentry_sdk import capture_exception, capture_message, configure_scope

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. Some tests will be skipped.")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("Warning: PyCUDA not available. Some tests will be skipped.")


class PythonCudaCrashDemo:
    """Python CUDA crash demonstration with Sentry integration."""
    
    def __init__(self):
        """Initialize the demo with Sentry configuration."""
        self.sentry_initialized = False
        self.initialize_sentry()
        self.setup_gpu_info()
        
    def initialize_sentry(self):
        """Initialize Sentry SDK for Python."""
        dsn = os.getenv('SENTRY_DSN')
        if not dsn:
            print("Warning: SENTRY_DSN not set. Errors will not be reported to Sentry.")
            return
            
        try:
            sentry_sdk.init(
                dsn=dsn,
                release="sentry-python-cuda-demo@1.0.0",
                environment=os.getenv('SENTRY_ENVIRONMENT', 'development'),
                traces_sample_rate=1.0,
                debug=True,
                attach_stacktrace=True,
                send_default_pii=True,
                max_breadcrumbs=100,
                before_send=self._before_send_hook
            )
            
            # Set user context
            with configure_scope() as scope:
                scope.set_user({
                    "id": "python-cuda-demo-user",
                    "username": "demo"
                })
                scope.set_tag("component", "python_cuda")
                scope.set_tag("language", "python")
                
            self.sentry_initialized = True
            print("Sentry initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Sentry: {e}")
    
    def _before_send_hook(self, event, hint):
        """Custom hook to add GPU context to Sentry events."""
        if hasattr(self, 'gpu_info'):
            event.setdefault('extra', {}).update(self.gpu_info)
        return event
    
    def setup_gpu_info(self):
        """Collect GPU information for Sentry context."""
        self.gpu_info = {
            "cupy_available": CUPY_AVAILABLE,
            "pycuda_available": PYCUDA_AVAILABLE,
            "numpy_version": np.__version__,
        }
        
        if CUPY_AVAILABLE:
            try:
                self.gpu_info.update({
                    "cupy_version": cp.__version__,
                    "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                    "gpu_memory_total": cp.cuda.Device().mem_info[1],
                    "gpu_memory_free": cp.cuda.Device().mem_info[0],
                    "gpu_device_count": cp.cuda.runtime.getDeviceCount(),
                    "gpu_device_name": cp.cuda.Device().name.decode(),
                    "gpu_compute_capability": f"{cp.cuda.Device().compute_capability[0]}.{cp.cuda.Device().compute_capability[1]}"
                })
            except Exception as e:
                self.gpu_info["cupy_info_error"] = str(e)
                
        if PYCUDA_AVAILABLE:
            try:
                self.gpu_info.update({
                    "pycuda_version": ".".join(map(str, cuda.get_version())),
                    "pycuda_device_count": cuda.Device.count(),
                })
                if cuda.Device.count() > 0:
                    dev = cuda.Device(0)
                    self.gpu_info.update({
                        "pycuda_device_name": dev.name(),
                        "pycuda_compute_capability": dev.compute_capability(),
                        "pycuda_total_memory": dev.total_memory()
                    })
            except Exception as e:
                self.gpu_info["pycuda_info_error"] = str(e)
    
    def report_cuda_error(self, error: Exception, crash_type: str, extra_context: Optional[Dict] = None):
        """Report CUDA error to Sentry with detailed context."""
        if not self.sentry_initialized:
            print(f"Sentry not initialized. Would report: {crash_type} - {error}")
            return
            
        with configure_scope() as scope:
            scope.set_tag("crash_type", crash_type)
            scope.set_context("crash_info", {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "crash_scenario": crash_type,
                **(extra_context or {})
            })
            
        capture_exception(error)
        print(f"Reported {crash_type} error to Sentry: {error}")

    # CuPy Crash Scenarios
    def test_cupy_memory_exhaustion(self):
        """Test CuPy memory exhaustion crash."""
        if not CUPY_AVAILABLE:
            print("Skipping CuPy memory exhaustion test - CuPy not available")
            return
            
        print("\n--- Testing CuPy Memory Exhaustion ---")
        arrays = []
        
        try:
            # Try to allocate way more memory than available
            for i in range(1000):
                # Allocate 1GB arrays
                size = 1024 * 1024 * 256  # 1GB in float32
                array = cp.zeros(size, dtype=cp.float32)
                arrays.append(array)
                print(f"Allocated array {i+1}: {array.nbytes / 1024**3:.2f} GB")
                
        except Exception as e:
            self.report_cuda_error(e, "cupy_memory_exhaustion", {
                "arrays_allocated": len(arrays),
                "total_memory_attempted": sum(arr.nbytes for arr in arrays),
                "gpu_memory_info": cp.cuda.Device().mem_info if CUPY_AVAILABLE else None
            })
            
    def test_cupy_invalid_kernel_launch(self):
        """Test CuPy invalid kernel launch."""
        if not CUPY_AVAILABLE:
            print("Skipping CuPy invalid kernel test - CuPy not available")
            return
            
        print("\n--- Testing CuPy Invalid Kernel Launch ---")
        
        try:
            # Create arrays
            x = cp.random.random((10000, 10000), dtype=cp.float32)
            
            # Create a custom kernel with intentional errors
            kernel_code = cp.RawKernel(r'''
            extern "C" __global__
            void crash_kernel(float* data, int size) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size) {
                    // Intentional divide by zero
                    data[idx] = data[idx] / 0.0f;
                    
                    // Intentional null pointer dereference
                    float* null_ptr = nullptr;
                    *null_ptr = data[idx];
                    
                    // Out of bounds access
                    data[idx + size * 1000] = 42.0f;
                }
            }
            ''', 'crash_kernel')
            
            # Launch kernel with invalid parameters
            kernel_code((65536,), (1024,), (x, x.size))  # Too many blocks/threads
            cp.cuda.Stream.null.synchronize()
            
        except Exception as e:
            self.report_cuda_error(e, "cupy_invalid_kernel_launch", {
                "array_shape": x.shape if 'x' in locals() else None,
                "array_dtype": str(x.dtype) if 'x' in locals() else None
            })
            
    def test_cupy_device_memory_corruption(self):
        """Test CuPy device memory corruption."""
        if not CUPY_AVAILABLE:
            print("Skipping CuPy memory corruption test - CuPy not available")
            return
            
        print("\n--- Testing CuPy Device Memory Corruption ---")
        
        try:
            # Create array
            x = cp.random.random((1000, 1000), dtype=cp.float32)
            
            # Get raw memory pointer and corrupt it
            ptr = x.data.ptr
            
            # Try to write to invalid memory locations
            kernel_code = cp.RawKernel(r'''
            extern "C" __global__
            void memory_corruption_kernel(float* data, int size) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                // Corrupt memory by writing to random high addresses
                float* corrupt_ptr = (float*)0xDEADBEEF;
                *corrupt_ptr = 42.0f;
                
                // Also corrupt the actual data in invalid ways
                if (idx < size) {
                    data[idx + size * 100] = data[idx];  // Way out of bounds
                }
            }
            ''', 'memory_corruption_kernel')
            
            kernel_code((1000,), (1000,), (x, x.size))
            cp.cuda.Stream.null.synchronize()
            
        except Exception as e:
            self.report_cuda_error(e, "cupy_device_memory_corruption", {
                "memory_pointer": hex(ptr) if 'ptr' in locals() else None,
                "array_size_bytes": x.nbytes if 'x' in locals() else None
            })

    def test_cupy_concurrent_stream_destruction(self):
        """Test CuPy concurrent stream destruction."""
        if not CUPY_AVAILABLE:
            print("Skipping CuPy stream destruction test - CuPy not available")
            return
            
        print("\n--- Testing CuPy Concurrent Stream Destruction ---")
        
        try:
            streams = []
            arrays = []
            
            # Create many streams and launch work
            for i in range(100):
                stream = cp.cuda.Stream()
                streams.append(stream)
                
                with stream:
                    # Launch long-running work
                    x = cp.random.random((5000, 5000), dtype=cp.float32)
                    y = cp.random.random((5000, 5000), dtype=cp.float32)
                    
                    # Chain many operations
                    for j in range(100):
                        x = cp.matmul(x, y)
                        y = cp.sin(x) + cp.cos(y)
                    
                    arrays.append((x, y))
            
            # Immediately destroy all streams while work is running
            for stream in streams:
                del stream
                
            # Force garbage collection
            gc.collect()
            
            # Try to synchronize (should fail)
            cp.cuda.Device().synchronize()
            
        except Exception as e:
            self.report_cuda_error(e, "cupy_concurrent_stream_destruction", {
                "streams_created": len(streams) if 'streams' in locals() else 0,
                "arrays_created": len(arrays) if 'arrays' in locals() else 0
            })

    # PyCUDA Crash Scenarios  
    def test_pycuda_driver_crash(self):
        """Test PyCUDA driver-level crash."""
        if not PYCUDA_AVAILABLE:
            print("Skipping PyCUDA driver crash test - PyCUDA not available")
            return
            
        print("\n--- Testing PyCUDA Driver Crash ---")
        
        try:
            # Create context and allocate memory
            dev = cuda.Device(0)
            ctx = dev.make_context()
            
            # Allocate massive amount of memory
            allocations = []
            for i in range(1000):
                try:
                    # Try to allocate 100MB each
                    mem_gpu = cuda.mem_alloc(100 * 1024 * 1024)
                    allocations.append(mem_gpu)
                except cuda.MemoryError:
                    break
                    
            print(f"Allocated {len(allocations)} memory blocks")
            
            # Create kernel that will crash the driver
            mod = SourceModule("""
            __global__ void driver_crash_kernel(float *dest, float *src, int n)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                
                // Multiple driver-crashing operations
                if (idx < n) {
                    // Infinite loop to hang GPU
                    while(true) {
                        dest[idx] = src[idx] * 2.0f;
                    }
                }
                
                // Also try invalid memory access
                float *invalid_ptr = (float*)0xFFFFFFFFFFFFFFFF;
                *invalid_ptr = 42.0f;
            }
            """)
            
            func = mod.get_function("driver_crash_kernel")
            
            # Launch with maximum grid size
            if allocations:
                func(allocations[0], allocations[1] if len(allocations) > 1 else allocations[0], 
                     np.int32(1000000), block=(1024,1,1), grid=(65535, 65535, 1))
                
            ctx.synchronize()
            ctx.pop()
            
        except Exception as e:
            self.report_cuda_error(e, "pycuda_driver_crash", {
                "allocations_made": len(allocations) if 'allocations' in locals() else 0,
                "device_name": dev.name() if 'dev' in locals() else None
            })

    def test_pycuda_context_corruption(self):
        """Test PyCUDA context corruption."""
        if not PYCUDA_AVAILABLE:
            print("Skipping PyCUDA context corruption test - PyCUDA not available")
            return
            
        print("\n--- Testing PyCUDA Context Corruption ---")
        
        try:
            contexts = []
            
            # Create multiple contexts and corrupt them
            for i in range(10):
                dev = cuda.Device(0)
                ctx = dev.make_context()
                contexts.append(ctx)
                
                # Make context current and do work
                ctx.push()
                
                # Allocate memory in this context
                mem = cuda.mem_alloc(1024 * 1024)
                
                # Create and compile kernel
                mod = SourceModule("""
                __global__ void corrupt_kernel(float *data, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        // Corrupt memory across contexts
                        data[idx * 1000000] = idx;  // Way out of bounds
                    }
                }
                """)
                
                func = mod.get_function("corrupt_kernel")
                func(mem, np.int32(1024), block=(256,1,1), grid=(1000,1,1))
                
                # Don't properly clean up context (leave on stack)
                
            # Now try to destroy contexts while they may have active work
            for ctx in contexts:
                ctx.pop()
                ctx.detach()
                del ctx
                
        except Exception as e:
            self.report_cuda_error(e, "pycuda_context_corruption", {
                "contexts_created": len(contexts) if 'contexts' in locals() else 0
            })

    def test_mixed_library_conflicts(self):
        """Test conflicts between CuPy and PyCUDA."""
        if not (CUPY_AVAILABLE and PYCUDA_AVAILABLE):
            print("Skipping mixed library test - both CuPy and PyCUDA needed")
            return
            
        print("\n--- Testing Mixed Library Conflicts ---")
        
        try:
            # Use CuPy to create arrays
            cupy_array = cp.random.random((1000, 1000), dtype=cp.float32)
            
            # Try to use PyCUDA on CuPy memory (this should fail)
            cupy_ptr = cupy_array.data.ptr
            
            # Create PyCUDA context
            dev = cuda.Device(0)
            ctx = dev.make_context()
            
            # Try to use CuPy pointer in PyCUDA kernel
            mod = SourceModule("""
            __global__ void mixed_library_kernel(float *data, int n) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    data[idx] = data[idx] * 2.0f;
                }
            }
            """)
            
            func = mod.get_function("mixed_library_kernel")
            
            # This should cause issues as we're mixing memory management
            func(cupy_ptr, np.int32(cupy_array.size), 
                 block=(256,1,1), grid=(1000,1,1))
            
            ctx.synchronize()
            ctx.pop()
            
        except Exception as e:
            self.report_cuda_error(e, "mixed_library_conflicts", {
                "cupy_array_shape": cupy_array.shape if 'cupy_array' in locals() else None,
                "cupy_pointer": hex(cupy_ptr) if 'cupy_ptr' in locals() else None
            })

    def run_all_tests(self):
        """Run all available crash tests."""
        print("Starting Python CUDA crash demonstration with Sentry reporting...")
        print(f"CuPy Available: {CUPY_AVAILABLE}")
        print(f"PyCUDA Available: {PYCUDA_AVAILABLE}")
        print()
        
        # CuPy tests
        if CUPY_AVAILABLE:
            self.test_cupy_memory_exhaustion()
            self.test_cupy_invalid_kernel_launch()
            self.test_cupy_device_memory_corruption()
            self.test_cupy_concurrent_stream_destruction()
        
        # PyCUDA tests  
        if PYCUDA_AVAILABLE:
            self.test_pycuda_driver_crash()
            self.test_pycuda_context_corruption()
            
        # Mixed library tests
        if CUPY_AVAILABLE and PYCUDA_AVAILABLE:
            self.test_mixed_library_conflicts()
            
        print("\nAll Python CUDA tests completed. Check Sentry dashboard for reported errors.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Python CUDA Crash Demo with Sentry')
    parser.add_argument('test_type', nargs='?', default='all',
                       help='Type of test to run (default: all)')
    parser.add_argument('--list-tests', action='store_true',
                       help='List available tests')
    
    args = parser.parse_args()
    
    available_tests = {
        'cupy_memory_exhaustion': 'CuPy memory exhaustion crash',
        'cupy_invalid_kernel': 'CuPy invalid kernel launch',
        'cupy_memory_corruption': 'CuPy device memory corruption',  
        'cupy_stream_destruction': 'CuPy concurrent stream destruction',
        'pycuda_driver_crash': 'PyCUDA driver-level crash',
        'pycuda_context_corruption': 'PyCUDA context corruption',
        'mixed_library_conflicts': 'Mixed CuPy/PyCUDA library conflicts',
        'all': 'Run all available tests'
    }
    
    if args.list_tests:
        print("Available tests:")
        for test, description in available_tests.items():
            print(f"  {test}: {description}")
        return
    
    if args.test_type not in available_tests:
        print(f"Unknown test type: {args.test_type}")
        print("Available tests:", ', '.join(available_tests.keys()))
        sys.exit(1)
    
    demo = PythonCudaCrashDemo()
    
    try:
        if args.test_type == 'all':
            demo.run_all_tests()
        elif args.test_type == 'cupy_memory_exhaustion':
            demo.test_cupy_memory_exhaustion()
        elif args.test_type == 'cupy_invalid_kernel':
            demo.test_cupy_invalid_kernel_launch()
        elif args.test_type == 'cupy_memory_corruption':
            demo.test_cupy_device_memory_corruption()
        elif args.test_type == 'cupy_stream_destruction':
            demo.test_cupy_concurrent_stream_destruction()
        elif args.test_type == 'pycuda_driver_crash':
            demo.test_pycuda_driver_crash()
        elif args.test_type == 'pycuda_context_corruption':
            demo.test_pycuda_context_corruption()
        elif args.test_type == 'mixed_library_conflicts':
            demo.test_mixed_library_conflicts()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        if demo.sentry_initialized:
            capture_exception(e)

if __name__ == '__main__':
    main()