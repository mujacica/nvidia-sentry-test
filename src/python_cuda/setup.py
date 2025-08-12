#!/usr/bin/env python3
"""
Setup script for Python CUDA crash demo dependencies.
"""

import subprocess
import sys
import os
import platform

def check_cuda_installation():
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA toolkit found")
            print(result.stdout.split('\n')[3])  # Version line
            return True
        else:
            print("✗ CUDA toolkit not found")
            return False
    except FileNotFoundError:
        print("✗ nvcc not found in PATH")
        return False

def install_requirements():
    """Install Python requirements."""
    print("Installing Python requirements...")
    
    requirements = [
        "sentry-sdk>=1.40.0",
        "numpy>=1.24.0", 
        "psutil>=5.9.0"
    ]
    
    # Install basic requirements first
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    
    # Try to install CUDA libraries
    cuda_available = check_cuda_installation()
    
    if cuda_available:
        print("\nInstalling CUDA Python libraries...")
        
        # Try CuPy installation
        try:
            print("Installing CuPy...")
            # Detect CUDA version and install appropriate CuPy
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda12x"])
            print("✓ CuPy installed successfully")
        except subprocess.CalledProcessError:
            print("✗ CuPy installation failed. Trying fallback...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy"])
                print("✓ CuPy installed with fallback")
            except subprocess.CalledProcessError:
                print("✗ CuPy installation failed completely")
        
        # Try PyCUDA installation  
        try:
            print("Installing PyCUDA...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycuda"])
            print("✓ PyCUDA installed successfully")
        except subprocess.CalledProcessError:
            print("✗ PyCUDA installation failed")
            
    else:
        print("\nWarning: CUDA not detected. CUDA libraries will not be installed.")
        print("You can still run the demo, but CUDA tests will be skipped.")

def main():
    """Main setup function."""
    print("Python CUDA Crash Demo Setup")
    print("=" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()
    
    try:
        install_requirements()
        print("\n✓ Setup completed successfully!")
        print("\nUsage:")
        print("  export SENTRY_DSN='your-sentry-dsn-here'")
        print("  python cuda_crash_demo.py --list-tests")
        print("  python cuda_crash_demo.py all")
        print("  python cuda_crash_demo.py cupy_memory_exhaustion")
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()