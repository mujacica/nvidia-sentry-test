# Python CUDA Demo CMake configuration

# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Custom target to set up Python environment and dependencies
add_custom_target(python_cuda_setup
    COMMAND ${Python3_EXECUTABLE} setup.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Setting up Python CUDA demo dependencies"
)

# Custom target to test Python installation
add_custom_target(python_cuda_test
    COMMAND ${Python3_EXECUTABLE} test_installation.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS python_cuda_setup
    COMMENT "Testing Python CUDA installation"
)

# Custom target to run Python CUDA crash demo
add_custom_target(python_cuda_demo
    COMMAND ${Python3_EXECUTABLE} cuda_crash_demo.py all
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS python_cuda_setup
    COMMENT "Running Python CUDA crash demo"
)

# Install Python files
install(FILES 
    cuda_crash_demo.py
    setup.py
    test_installation.py
    requirements.txt
    README.md
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python_cuda
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Print helpful information
message(STATUS "Python CUDA demo configured")
message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
message(STATUS "  To setup: make python_cuda_setup")
message(STATUS "  To test: make python_cuda_test") 
message(STATUS "  To run: make python_cuda_demo")