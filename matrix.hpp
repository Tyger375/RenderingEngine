#ifndef MATRIX_H
#define MATRIX_H
#include <cmath>
#include "vector.hpp"

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

    __device__ __host__ vec3f x() const {
        return {m[0][0], m[1][0],  m[2][0] };
    }

    __device__ __host__ vec3f y() const {
        return {m[0][1], m[1][1],  m[2][1] };
    }

    __device__ __host__ vec3f z() const {
        return {m[0][2], m[1][2],  m[2][2] };
    }

    __device__ __host__ vec3f origin() const {
        return {m[0][3], m[1][3],  m[2][3] };
    }

    Matrix4x4() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] = 0;
            }
        }
    }

    Matrix4x4(vec3f x, vec3f y, vec3f z, vec3f origin) {
        const float row1[4] = { x.x, x.y, x.z, 0 };
        const float row2[4] = { y.x, y.y, y.z, 0 };
        const float row3[4] = { z.x, z.y, z.z, 0 };
        const float row4[4] = { origin.x, origin.y, origin.z, 1 };
        *this = Matrix4x4(row1, row2, row3, row4).transpose();
    }

    Matrix4x4(
        const float row1[4],
        const float row2[4],
        const float row3[4],
        const float row4[4]
        ) {
        for (int i = 0; i < 4; i++) {
            m[0][i] = row1[i];
        }
        for (int i = 0; i < 4; i++) {
            m[1][i] = row2[i];
        }
        for (int i = 0; i < 4; i++) {
            m[2][i] = row3[i];
        }
        for (int i = 0; i < 4; i++) {
            m[3][i] = row4[i];
        }
    }

    float determinant() const {
        auto first = Matrix3x3(
            m[1][1], m[1][2], m[1][3],
            m[2][1], m[2][2], m[2][3],
            m[3][1], m[3][2], m[3][3]
            );
        auto second = Matrix3x3(
            m[1][0], m[1][2], m[1][3],
            m[2][0], m[2][2], m[2][3],
            m[3][0], m[3][2], m[3][3]
            );
        auto third = Matrix3x3(
            m[1][0], m[1][1], m[1][3],
            m[2][0], m[2][1], m[2][3],
            m[3][0], m[3][1], m[3][3]
            );
        auto fourth = Matrix3x3(
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2],
            m[3][0], m[3][1], m[3][2]
            );
        return first.determinant() * m[0][0]
            - second.determinant() * m[0][1]
            + third.determinant() * m[0][2]
            - fourth.determinant() * m[0][3];
    }

    Matrix3x3 minor(int i, int j) const {
        auto matrix = Matrix3x3();

        int minorRow = 0;
        for (int row = 0; row < 4; ++row) {
            if (row == i) continue;
            int minorCol = 0;
            for (int col = 0; col < 4; ++col) {
                if (col == j) continue;
                matrix.m[minorCol][minorRow] = m[col][row];
                ++minorCol;
            }
            ++minorRow;
        }

        return matrix;
    }

    Matrix4x4 adjugate() const {
        Matrix4x4 matrix = Matrix4x4();
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                auto m = minor(i, j);
                float c = static_cast<float>(std::pow(-1, (i + 1) + (j + 1))) * m.determinant();
                matrix.m[i][j] = c;
            }
        }

        return matrix;
    }

    Matrix4x4 inverse() const {
        if (this->determinant() == 0) return *this;

        return adjugate() / determinant();
    }

    Matrix4x4 transpose() const {
        Matrix4x4 matrix = Matrix4x4();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                matrix.m[i][j] = m[j][i];
            }
        }
        return matrix;
    }

    Matrix4x4 operator*(float scalar) {
        Matrix4x4 result;

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.m[i][j] = m[i][j] * scalar;
            }
        }

        return result;
    }

    Matrix4x4 operator/(float scalar) {
        return *this * (1 / scalar);
    }

    __device__ __host__ vec4f operator*(vec4f vec) const {
        vec4f result;

        vec4f row1 = {
            m[0][0],
            m[0][1],
            m[0][2],
            m[0][3]
        };
        result.x = row1.dot(vec);

        vec4f row2 = {
            m[1][0],
            m[1][1],
            m[1][2],
            m[1][3]
        };
        result.y = row2.dot(vec);

        vec4f row3 = {
            m[2][0],
            m[2][1],
            m[2][2],
            m[2][3]
        };
        result.z = row3.dot(vec);

        vec4f row4 = {
            m[3][0],
            m[3][1],
            m[3][2],
            m[3][3]
        };
        result.w = row4.dot(vec);

        return result;
    }

    __device__ __host__ vec3f operator*(vec3f vec) const {
        vec4f result = *this * vec4f {vec.x, vec.y, vec.z, 1};
        return vec3f {result.x, result.y, result.z};
    }
};

#endif //MATRIX_H
