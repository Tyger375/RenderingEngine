#ifndef MATRIX_H
#define MATRIX_H
#include "../vector/vector.cuh"
#include <cuda_runtime.h>

struct Matrix2x2 {
    float m[2][2]{};

    Matrix2x2(float m1, float m2, float m3, float m4) {
        m[0][0] = m1;
        m[0][1] = m2;
        m[1][0] = m3;
        m[1][1] = m4;
    }

    float determinant() const {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }
};

struct Matrix3x3 {
    float m[3][3]{};

    Matrix3x3() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = 0;
            }
        }
    }

    Matrix3x3(
        float m1, float m2, float m3,
        float m4, float m5, float m6,
        float m7, float m8, float m9
        ) {
        m[0][0] = m1;
        m[0][1] = m2;
        m[0][2] = m3;
        m[1][0] = m4;
        m[1][1] = m5;
        m[1][2] = m6;
        m[2][0] = m7;
        m[2][1] = m8;
        m[2][2] = m9;
    }

    float determinant() const {
        auto first = Matrix2x2(
            m[1][1], m[1][2],
            m[2][1], m[2][2]
            );
        auto second = Matrix2x2(
            m[1][0], m[1][2],
            m[2][0], m[2][2]
            );
        auto third = Matrix2x2(
            m[1][0], m[1][1],
            m[2][0], m[2][1]
            );

        return first.determinant() * m[0][0]
            - second.determinant() * m[0][1]
            + third.determinant() * m[0][2];
    }
};

struct Matrix4x4 {
    float m[4][4]{};

    __device__ __host__ vec3f x() const;
    __device__ __host__ vec3f y() const;
    __device__ __host__ vec3f z() const;
    __device__ __host__ vec3f origin() const;

    Matrix4x4();
    Matrix4x4(vec3f x, vec3f y, vec3f z, vec3f origin);
    Matrix4x4(
        const float row1[4],
        const float row2[4],
        const float row3[4],
        const float row4[4]
        );

    float determinant() const;

    Matrix3x3 minor(int i, int j) const;

    Matrix4x4 adjugate() const;

    Matrix4x4 inverse() const;

    Matrix4x4 transpose() const;

    Matrix4x4 operator*(float scalar) const;

    Matrix4x4 operator/(float scalar) const;

    __device__ __host__ vec4f operator*(vec4f vec) const;

    __device__ __host__ vec3f operator*(vec3f vec) const;

    Matrix4x4 operator*(const Matrix4x4& m2) const;

    void translate_origin(vec3f add);

    static Matrix4x4 rotation_x(float angle);
};

#endif //MATRIX_H
