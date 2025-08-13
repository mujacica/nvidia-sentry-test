@echo off
setlocal enabledelayedexpansion

echo Building Sentry Graphics Demo for Windows

where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: CMake not found. Please install CMake.
    exit /b 1
)

set BUILD_DIR=build
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Debug

if "%BUILD_CUDA%"=="" set BUILD_CUDA=ON
if "%BUILD_VULKAN%"=="" set BUILD_VULKAN=ON
if "%BUILD_DIRECTX%"=="" set BUILD_DIRECTX=ON
if "%BUILD_PYTHON_CUDA%"=="" set BUILD_PYTHON_CUDA=ON
if "%BUILD_GPU_INFO%"=="" set BUILD_GPU_INFO=ON

REM Check for CUDA toolkit
if "%BUILD_CUDA%"=="ON" (
    where nvcc >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo Warning: CUDA toolkit not found. Disabling CUDA build.
        echo Install CUDA toolkit and ensure nvcc is in PATH to build CUDA demo.
        set BUILD_CUDA=OFF
    )
)

REM Check for Vulkan SDK
if "%BUILD_VULKAN%"=="ON" (
    if defined VULKAN_SDK (
        if not exist "%VULKAN_SDK%\Include\vulkan\vulkan.h" (
            echo Warning: Vulkan SDK installation appears incomplete. Disabling Vulkan build.
            echo Please reinstall Vulkan SDK from https://vulkan.lunarg.com/
            set BUILD_VULKAN=OFF
        )
    ) else (
        echo Warning: Vulkan SDK not found. Disabling Vulkan build.
        echo Install Vulkan SDK from https://vulkan.lunarg.com/ and ensure VULKAN_SDK environment variable is set.
        set BUILD_VULKAN=OFF
    )
)


REM Check for Python 3
if "%BUILD_PYTHON_CUDA%"=="ON" (
    where python >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        where python3 >nul 2>&1
        if !ERRORLEVEL! neq 0 (
            echo Warning: Python 3 not found. Disabling Python CUDA build.
            echo Install Python 3 to build Python CUDA demo.
            set BUILD_PYTHON_CUDA=OFF
        )
    )
)

echo Creating build directory: %BUILD_DIR%
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

echo Running CMake configuration...
echo   BUILD_CUDA: %BUILD_CUDA%
echo   BUILD_VULKAN: %BUILD_VULKAN%
echo   BUILD_DIRECTX: %BUILD_DIRECTX%
echo   BUILD_PYTHON_CUDA: %BUILD_PYTHON_CUDA%
echo   BUILD_GPU_INFO: %BUILD_GPU_INFO%

cmake -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
      -DBUILD_TESTING=ON ^
      -DBUILD_CUDA=%BUILD_CUDA% ^
      -DBUILD_VULKAN=%BUILD_VULKAN% ^
      -DBUILD_DIRECTX=%BUILD_DIRECTX% ^
      -DBUILD_PYTHON_CUDA=%BUILD_PYTHON_CUDA% ^
      -DBUILD_GPU_INFO=%BUILD_GPU_INFO% ^
      -G "Visual Studio 17 2022" -A x64 ..

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

echo Building project...
cmake --build . --config %BUILD_TYPE% -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo Installing project...
cmake --install . --config %BUILD_TYPE%

echo.
echo Build and install completed successfully!

if not "%SENTRY_ORG%"=="" if not "%SENTRY_PROJECT%"=="" if not "%SENTRY_AUTH_TOKEN%"=="" (
    echo.
    echo Uploading debug symbols to Sentry...
    
    set SENTRY_CLI_PATH=%CD%\sentry-cli.exe
    
    REM Download sentry-cli if not present
    if not exist "!SENTRY_CLI_PATH!" (
        echo Downloading sentry-cli...
        set DOWNLOAD_URL=https://github.com/getsentry/sentry-cli/releases/latest/download/sentry-cli-Windows-x86_64.exe
        
        REM Try to download using PowerShell
        powershell -Command "try { Invoke-WebRequest -Uri '!DOWNLOAD_URL!' -OutFile '!SENTRY_CLI_PATH!' -UseBasicParsing } catch { exit 1 }" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo sentry-cli downloaded successfully
        )
    )
    
    REM Upload debug symbols for our demo binaries only (not crashpad_handler)
    set UPLOAD_FILES=
    
    if "%BUILD_CUDA%"=="ON" if exist "Debug\cuda_crash_demo.exe" (
        set UPLOAD_FILES=!UPLOAD_FILES! Debug\cuda_crash_demo.exe Debug\cuda_crash_demo.pdb Debug\cuda_crash_demo.exp Debug\cuda_crash_demo.lib
    )
    
    if "%BUILD_VULKAN%"=="ON" if exist "Debug\vulkan_crash_demo.exe" (
        set UPLOAD_FILES=!UPLOAD_FILES! Debug\vulkan_crash_demo.exe Debug\vulkan_crash_demo.pdb
    )
    
    if "%BUILD_DIRECTX%"=="ON" if exist "Debug\directx_crash_demo.exe" (
        set UPLOAD_FILES=!UPLOAD_FILES! Debug\directx_crash_demo.exe Debug\directx_crash_demo.pdb
    )
    
    if "%BUILD_GPU_INFO%"=="ON" if exist "Debug\gpu_info_demo.exe" (
        set UPLOAD_FILES=!UPLOAD_FILES! Debug\gpu_info_demo.exe Debug\gpu_info_demo.pdb
    )
    
    echo Debug: UPLOAD_FILES=[!UPLOAD_FILES!]
    if defined UPLOAD_FILES (
        REM Upload with proper source mapping and paths
        "!SENTRY_CLI_PATH!" debug-files upload --include-sources --log-level info!UPLOAD_FILES!
        if !ERRORLEVEL! equ 0 (
            echo Debug symbols and source files uploaded successfully to Sentry!
        ) else (
            echo Warning: Failed to upload debug symbols and source files to Sentry
        )
    ) else (
        echo No demo binaries found to upload
    )
) else (
    echo.
    echo Skipping debug symbol upload ^(set SENTRY_ORG, SENTRY_PROJECT, and SENTRY_AUTH_TOKEN to enable^)
)
echo.
echo Available executables:

if "%BUILD_CUDA%"=="ON" (
    echo   CUDA Demo: install\bin\cuda_crash_demo.exe
)

if "%BUILD_VULKAN%"=="ON" (
    echo   Vulkan Demo: install\bin\vulkan_crash_demo.exe
)

if "%BUILD_DIRECTX%"=="ON" (
    echo   DirectX Demo: install\bin\directx_crash_demo.exe
)

if "%BUILD_PYTHON_CUDA%"=="ON" (
    echo   Python CUDA Demo: %BUILD_DIR%\src\python_cuda\
)

if "%BUILD_GPU_INFO%"=="ON" (
    echo   GPU Info Demo: install\bin\gpu_info_demo.exe
)

echo.
echo Usage:
echo   set SENTRY_DSN=your-sentry-dsn-here

if "%BUILD_CUDA%"=="ON" (
    echo.
    echo CUDA Demo:
    echo   cd install\bin ^&^& .\cuda_crash_demo.exe [test_type]
    echo   Available tests: divide_by_zero, out_of_bounds, null_pointer, infinite_loop
)

if "%BUILD_VULKAN%"=="ON" (
    echo.
    echo Vulkan Demo:
    echo   cd install\bin ^&^& .\vulkan_crash_demo.exe [test_type]
    echo   Available tests: invalid_buffer_access, invalid_descriptor_set, device_lost_simulation, invalid_pipeline, out_of_bounds_memory
)

if "%BUILD_DIRECTX%"=="ON" (
    echo.
    echo DirectX Demo:
    echo   cd install\bin ^&^& .\directx_crash_demo.exe [test_type]
    echo   Available tests: invalid_buffer_access, invalid_shader_resource, device_removed_simulation, invalid_render_target, out_of_bounds_vertex_buffer
)

if "%BUILD_PYTHON_CUDA%"=="ON" (
    echo.
    echo Python CUDA Demo:
    echo   cd src\python_cuda
    echo   python setup.py                     # Setup dependencies
    echo   python test_installation.py         # Test installation
    echo   python cuda_crash_demo.py [test_type]
    echo   Available tests: cupy_memory_exhaustion, pycuda_driver_crash, mixed_library_conflicts, all
)

if "%BUILD_GPU_INFO%"=="ON" (
    echo.
    echo GPU Info Demo:
    echo   cd install\bin ^&^& .\gpu_info_demo.exe [test_type]
    echo   Available tests: basic, warning, error, exception, crash, all
)

echo.
echo Run without arguments to execute all tests for each demo

endlocal