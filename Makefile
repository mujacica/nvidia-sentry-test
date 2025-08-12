# Makefile for Sentry GPU Demo

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS := linux
    NPROC := $(shell nproc)
endif
ifeq ($(UNAME_S),Darwin)
    OS := macos
    NPROC := $(shell sysctl -n hw.ncpu)
endif

# Default values
BUILD_TYPE ?= Release
BUILD_DIR ?= build
BUILD_CUDA ?= ON
BUILD_VULKAN ?= ON
BUILD_DIRECTX := OFF
CUDA_ARCH ?= sm_50

# Platform-specific API availability
ifeq ($(OS),linux)
    BUILD_DIRECTX := OFF
endif
ifeq ($(OS),macos)
    BUILD_VULKAN := OFF
    BUILD_DIRECTX := OFF
	BUILD_CUDA := OFF
endif

# CUDA toolkit detection
ifeq ($(BUILD_CUDA),ON)
    NVCC := $(shell which nvcc 2>/dev/null)
    ifndef NVCC
        $(warning CUDA toolkit not found. Disabling CUDA build.)
        BUILD_CUDA := OFF
    endif
endif

# Vulkan SDK detection
ifeq ($(BUILD_VULKAN),ON)
    VULKAN_PKG := $(shell pkg-config --exists vulkan 2>/dev/null && echo "found")
    ifndef VULKAN_PKG
        ifdef VULKAN_SDK
            VULKAN_PKG := found
        endif
    endif
    ifndef VULKAN_PKG
        $(warning Vulkan SDK not found. Disabling Vulkan build.)
        BUILD_VULKAN := OFF
    endif
endif

# CMake detection
CMAKE := $(shell which cmake 2>/dev/null)
ifndef CMAKE
$(error CMake not found. Please install CMake)
endif

.PHONY: all build clean test install help cuda vulkan directx

all: build

help:
	@echo "Sentry Graphics Demo - Available targets:"
	@echo "  build      - Build all enabled graphics API demos (default)"
	@echo "  cuda       - Build only CUDA demo"
	@echo "  vulkan     - Build only Vulkan demo"
	@echo "  test       - Build and run tests"
	@echo "  clean      - Clean build directory"
	@echo "  install    - Install the built executables"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Configuration options:"
	@echo "  BUILD_TYPE=Release|Debug  - Build configuration (default: Release)"
	@echo "  BUILD_DIR=<dir>          - Build directory (default: build)"
	@echo "  BUILD_CUDA=ON|OFF        - Enable CUDA demo (default: ON)"
	@echo "  BUILD_VULKAN=ON|OFF      - Enable Vulkan demo (default: ON on Linux)"
	@echo "  CUDA_ARCH=<arch>         - CUDA architecture (default: sm_50)"
	@echo ""
	@echo "Current configuration:"
	@echo "  OS: $(OS)"
	@echo "  BUILD_CUDA: $(BUILD_CUDA)"
	@echo "  BUILD_VULKAN: $(BUILD_VULKAN)"
	@echo "  BUILD_DIRECTX: $(BUILD_DIRECTX)"
	@echo ""
	@echo "Usage example:"
	@echo "  make BUILD_TYPE=Debug"
	@echo "  make cuda"
	@echo "  make test"

build: $(BUILD_DIR)/Makefile
	@echo "Building project..."
	@cd $(BUILD_DIR) && $(MAKE) -j$(NPROC)
	@echo "Build completed successfully!"
	@echo ""
	@echo "Available executables:"
ifeq ($(BUILD_CUDA),ON)
	@echo "  CUDA Demo: $(BUILD_DIR)/src/cuda/cuda_crash_demo"
endif
ifeq ($(BUILD_VULKAN),ON)
	@echo "  Vulkan Demo: $(BUILD_DIR)/src/vulkan/vulkan_crash_demo"
endif

$(BUILD_DIR)/Makefile:
	@echo "Creating build directory and configuring..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_TESTING=ON \
		-DBUILD_CUDA=$(BUILD_CUDA) \
		-DBUILD_VULKAN=$(BUILD_VULKAN) \
		-DBUILD_DIRECTX=$(BUILD_DIRECTX) \
		-DCUDA_ARCH=$(CUDA_ARCH) \
		..

cuda: BUILD_VULKAN := OFF
cuda: BUILD_DIRECTX := OFF
cuda: build

vulkan: BUILD_CUDA := OFF
vulkan: BUILD_DIRECTX := OFF
vulkan: build

test: build
	@echo "Running tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure

clean:
	@echo "Cleaning build and install directories..."
	@rm -rf $(BUILD_DIR)
	@rm -rf install
	@echo "Clean completed."

install: build
	@echo "Installing..."
	@cd $(BUILD_DIR) && $(MAKE) install

format:
	@echo "Formatting code..."
	@find src tests -name "*.cpp" -o -name "*.cu" -o -name "*.h" | xargs clang-format -i
