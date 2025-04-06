#include "rendering.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "../tracing/tracing.cuh"

static Camera* d_camera;

void update_camera() {
    cudaMemcpy(d_camera, &h_camera, sizeof(Camera), cudaMemcpyHostToDevice);
}

void update_objects() {
    update_camera();
}


void prepare_objects() {
    cudaMalloc(&d_camera, sizeof(Camera));
    cudaError_t error = cudaMemcpyToSymbol(
        camera,
        &d_camera,
        sizeof(Camera*)
        );
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    auto* _spheres = (Sphere*)malloc(sizeof(Sphere) * 3);
    _spheres[0].center = vec3f { 5, 0, 0 };
    _spheres[0].material = {vec3f { 1, 0, 0 }, {}, 0};
    _spheres[0].radius = 1;

    _spheres[1].center = vec3f { 2, 3, 0 };
    _spheres[1].material = {vec3f { 0, 0, 1 }, {}, 0};
    _spheres[1].radius = 1;

    _spheres[2].center = vec3f { 4, -4, 4 };
    _spheres[2].material = {vec3f { 0, 0, 0 }, {1, 1, 1}, 1};
    _spheres[2].radius = 3;

    Sphere* d_spheres;
    cudaMalloc(&d_spheres, sizeof(Sphere) * 3);
    cudaMemcpy(d_spheres, _spheres, sizeof(Sphere) * 3, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(spheres, &d_spheres, sizeof(Sphere*));

    free(_spheres);
    //cudaFree(d_spheres);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

__device__ uint32_t vec_to_rgb(const vec3f rgb) {
    const float x = min(max(rgb.x, 0.0f), 1.f);
    const float y = min(max(rgb.y, 0.0f), 1.f);
    const float z = min(max(rgb.z, 0.0f), 1.f);

    const int _x = static_cast<int>(x * 0xFF);
    const int _y = static_cast<int>(y * 0xFF);
    const int _z = static_cast<int>(z * 0xFF);

    return _z | (_y << 8) | (_x << 16) | (0xFF << 24);
}

__global__ void build(uint32_t* buffer, int width, int height) {
    // Calculate the 2D thread position within the grid
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // X index of thread
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // Y index of thread

    // Calculate the position in the 1D buffer (linearized index)
    unsigned int pos = y * width + x;

    // Check if the position is within bounds to avoid out-of-bounds access
    if (x < width && y < height) {
        float aspect_ratio = (float)width / (float)height;

        float _x = (float)x / (float)width * 2.0f - 1.0f;
        float _y = (float)y / (float)height * 2.0f - 1.0f;
        _x *= aspect_ratio;

        vec3f rgb = frag(
            {_x, _y},
            {(int)x, (int)y},
            {width, height}
        );
        buffer[pos] = vec_to_rgb(rgb);  // Set the pixel value to white (or any value)
    }
}

void render(Frame* frame) {
    uint32_t* buffer = nullptr;
    size_t pitch;
    cudaMallocPitch(&buffer, &pitch, frame->width * sizeof(uint32_t), frame->height);

    dim3 block_size(64, 64);  // 16x16 block size
    dim3 grid_size((frame->width + block_size.x - 1) / block_size.x,
                   (frame->height + block_size.y - 1) / block_size.y);

    build<<<block_size, grid_size>>>(buffer, frame->width, frame->height);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaMemcpy(
        frame->pixels,
        buffer,
        (frame->width * frame->height) * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
        );

    cudaFree(buffer);
}
