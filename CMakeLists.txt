cmake_minimum_required(VERSION 3.30)
project(RenderingEngine CUDA CXX)

# Set the C++ standard and CUDA standard (ensure CUDA version supports C++20)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Define the CUDA architectures you're targeting. You can adjust these as necessary.
set(CMAKE_CUDA_ARCHITECTURES 86)

# Enable separable compilation (useful for managing multiple .cu files)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Add the executable (if using WIN32, it'll link against user32 and gdi32 automatically)
add_executable(RenderingEngine WIN32 main.cpp
        matrix/matrix.cuh
        vector/vector.cuh
        rendering/rendering.cu
        rendering/rendering.cuh
        vector/vector.cu
        matrix/matrix.cu
        tracing/tracing.cu
        tracing/tracing.cuh
)

# Link libraries for a Windows application (user32, gdi32 are common)
target_link_libraries(RenderingEngine PRIVATE user32 gdi32)

# Set the properties of the target (ensure CUDA code is compiled correctly)
set_target_properties(RenderingEngine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;86"
)