#include "rendering.cuh"
#include <iostream>
#include "vector.hpp"
#include "matrix.hpp"

struct Camera {
    Matrix4x4 local_matrix;
    Matrix4x4 inv_local_matrix;
};

struct Sphere {
    vec3f center;
    float radius;
};

struct Ray {
    vec3f origin;
    vec3f direction;
};

__device__ Camera* camera;
__device__ Sphere* spheres;

void prepare_objects() {
    Camera _camera;
    _camera.local_matrix = Matrix4x4(
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0}
        );
    _camera.inv_local_matrix = _camera.local_matrix.inverse();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << _camera.inv_local_matrix.m[i][j];
        }
        std::cout << std::endl;
    }
    auto x = _camera.inv_local_matrix.x();
    std::cout << x.x << " " << x.y << " " << x.z << std::endl;
    // Allocate memory for Camera object on the device
    Camera* d_camera;
    cudaMalloc(&d_camera, sizeof(Camera));  // Allocate memory for Camera on the device
    // Copy Camera data from host to device
    cudaMemcpy(d_camera, &_camera, sizeof(Camera), cudaMemcpyHostToDevice);
    // Optionally copy the device pointer to the device global symbol
    cudaMemcpyToSymbol(camera, &d_camera, sizeof(Camera*));

    Sphere* _spheres = (Sphere*)malloc(sizeof(Sphere) * 2);
    _spheres[0].center = vec3f { 5, 0, 0 };
    _spheres[0].radius = 1;

    _spheres[1].center = vec3f { 2, 3, 0 };
    _spheres[1].radius = 1;

    Sphere* d_spheres;
    cudaMalloc(&d_spheres, sizeof(Sphere) * 2);
    cudaMemcpy(d_spheres, _spheres, sizeof(Sphere) * 2, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(spheres, &d_spheres, sizeof(Sphere*));

    free(_spheres);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

__device__ bool ray_intersection(Ray ray) {
    for (int i = 0; i < 2; i++) {
        auto sphere = spheres[i];

        vec3f C = camera->inv_local_matrix * sphere.center;
        float a = ray.direction.dot(ray.direction);
        float b = 2 * ray.direction.dot(ray.origin - C);
        float c = (ray.origin - C).dot(ray.origin - C) - sphere.radius * sphere.radius;

        float determinant = b*b - 4 * a * c;

        if (determinant >= 0)
            return true;
    }
    return false;
}

__device__ vec3f frag(vec2f uv, vec2i size) {
    vec3f forward = {1, 0, 0};

    vec3f up = {0, 0, 1};
    vec3f right = {0, 1, 0};

    vec3f dir = (forward + right * uv.x + up * uv.y).normalize();

    Ray ray{};
    ray.origin = {0,0,0};
    ray.direction = dir;

    bool hit = ray_intersection(ray);

    return hit ? vec3f {1, 1, 1} : vec3f {0, 0, 0};
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
            {width, height}
        );
        buffer[pos] = vec_to_rgb(rgb);  // Set the pixel value to white (or any value)
    }
}

void render(Frame* frame) {
    uint32_t* buffer = nullptr;
    size_t pitch;
    cudaMallocPitch(&buffer, &pitch, frame->width * sizeof(uint32_t), frame->height * sizeof(uint32_t));

    dim3 block_size(16, 16);  // 16x16 block size
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
