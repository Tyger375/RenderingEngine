#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cuda_runtime.h>

struct vec2i {
    int x, y;
};

struct vec2f {
    float x, y;

    __device__ __host__ vec2f operator+(vec2f s) const;

    __device__ __host__ vec2f operator-(vec2f s) const;

    __device__ __host__ vec2f operator*(float s) const;

    __device__ __host__ vec2f operator/(float s) const;

    __device__ __host__ float dot(vec2f s) const;

    __device__ __host__ float magnitude() const;

    __device__ __host__ vec2f normalize() const;
};

struct vec3f {
    float x, y, z;

    __device__ __host__ vec3f operator+(vec3f s) const;

    __device__ __host__ vec3f operator-(vec3f s) const;

    __device__ __host__ vec3f operator*(float s) const;
    __device__ __host__ vec3f operator*(vec3f s) const;

    __device__ __host__ vec3f operator/(float s) const;

    __device__ __host__ float dot(vec3f s) const;

    __device__ __host__ float magnitude() const;

    __device__ __host__ vec3f normalize() const;

    __device__ __host__ vec3f cross(vec3f s) const;
};

struct vec4f {
    float x, y, z, w;

    __device__ __host__ float dot(vec4f s) const;
};

#endif //VECTOR_HPP
