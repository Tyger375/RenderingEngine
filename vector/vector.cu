#include "vector.cuh"

__device__ __host__ vec3f vec3f::operator+(vec3f s) const {
    return vec3f{ x + s.x, y + s.y, z + s.z };
}

__device__ __host__ vec3f vec3f::operator-(vec3f s) const {
    return vec3f{ x - s.x, y - s.y, z - s.z };
}

__device__ __host__ vec3f vec3f::operator*(const float s) const {
    return vec3f{ x * s, y * s, z * s };
}

__device__ __host__ vec3f vec3f::operator/(const float s) const {
    return vec3f{ x / s, y / s, z / s };
}

__device__ __host__ float vec3f::dot(const vec3f s) const {
    return x * s.x + y * s.y + z * s.z;
}

__device__ __host__ float vec3f::magnitude() const {
    return std::sqrt(dot(*this));
}

__device__ __host__ vec3f vec3f::normalize() const {
    return *this / this->magnitude();
}

__device__ __host__ vec3f vec3f::cross(const vec3f s) const {
    return vec3f {
        y * s.z - z * s.y,
        z * s.x - x * s.z,
        x * s.y - y * s.x
    };
}

__device__ __host__ float vec4f::dot(const vec4f s) const {
    return x * s.x + y * s.y + z * s.z + w * s.w;
}