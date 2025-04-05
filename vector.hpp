#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cmath>

struct vec2i {
    int x, y;
};

struct vec2f {
    float x, y;
};

struct vec3f {
    float x, y, z;

    __device__ __host__ vec3f operator+(vec3f s) const {
        return vec3f{ x + s.x, y + s.y, z + s.z };
    }

    __device__ __host__ vec3f operator-(vec3f s) const {
        return vec3f{ x - s.x, y - s.y, z - s.z };
    }

    __device__ __host__ vec3f operator*(const float s) const {
        return vec3f{ x * s, y * s, z * s };
    }

    __device__ __host__ vec3f operator/(const float s) const {
        return vec3f{ x / s, y / s, z / s };
    }

    __device__ __host__ float dot(const vec3f s) const {
        return x * s.x + y * s.y + z * s.z;
    }

    __device__ __host__ float magnitude() const {
        return std::sqrt(dot(*this));
    }

    __device__ __host__ vec3f normalize() const {
        return *this / this->magnitude();
    }

    __device__ __host__ vec3f cross(const vec3f s) const {
        return vec3f {
            y * s.z - z * s.y,
            z * s.x - x * s.z,
            x * s.y - y * s.x
        };
    }
};

struct vec4f {
    float x, y, z, w;

    __device__ __host__ float dot(const vec4f s) const {
        return x * s.x + y * s.y + z * s.z + w * s.w;
    }
};

#endif //VECTOR_HPP
