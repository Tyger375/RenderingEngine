#include "matrix.cuh"
#include "../vector/vector.cuh"

__device__ __host__ vec3f Matrix4x4::x() const {
    return vec3f{m[0][0], m[1][0], m[2][0]};
}

__device__ __host__ vec3f Matrix4x4::y() const {
    return vec3f{m[0][1], m[1][1], m[2][1]};
}

__device__ __host__ vec3f Matrix4x4::z() const {
    return vec3f{m[0][2], m[1][2], m[2][2]};
}

__device__ __host__ vec3f Matrix4x4::origin() const {
    return vec3f{m[0][3], m[1][3], m[2][3]};
}

Matrix4x4::Matrix4x4() {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m[i][j] = 0;
        }
    }
}

Matrix4x4::Matrix4x4(vec3f x, vec3f y, vec3f z, vec3f origin) {
    const float row1[4] = {x.x, x.y, x.z, 0};
    const float row2[4] = {y.x, y.y, y.z, 0};
    const float row3[4] = {z.x, z.y, z.z, 0};
    const float row4[4] = {origin.x, origin.y, origin.z, 1};
    *this = Matrix4x4(row1, row2, row3, row4).transpose();
}

Matrix4x4::Matrix4x4(
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

float Matrix4x4::determinant() const {
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

Matrix3x3 Matrix4x4::minor(int i, int j) const {
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

Matrix4x4 Matrix4x4::adjugate() const {
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

Matrix4x4 Matrix4x4::inverse() const {
    if (this->determinant() == 0) return *this;

    return adjugate() / determinant();
}

Matrix4x4 Matrix4x4::transpose() const {
    Matrix4x4 matrix = Matrix4x4();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matrix.m[i][j] = m[j][i];
        }
    }
    return matrix;
}

Matrix4x4 Matrix4x4::operator*(float scalar) const {
    Matrix4x4 result;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.m[i][j] = m[i][j] * scalar;
        }
    }

    return result;
}

Matrix4x4 Matrix4x4::operator/(float scalar) const {
    return *this * (1 / scalar);
}

__device__ __host__ vec4f Matrix4x4::operator*(vec4f vec) const {
    vec4f result{};

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

__device__ __host__ vec3f Matrix4x4::operator*(vec3f vec) const {
    vec4f result = *this * vec4f{vec.x, vec.y, vec.z, 1};
    return vec3f{result.x, result.y, result.z};
}

Matrix4x4 Matrix4x4::operator*(const Matrix4x4 &m2) const {
    Matrix4x4 result;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result.m[i][j] += this->m[i][k] * m2.m[k][j];
            }
        }
    }

    return result;
}

void Matrix4x4::translate_origin(const vec3f add) {
    m[0][3] += add.x;
    m[1][3] += add.y;
    m[2][3] += add.z;
}

Matrix4x4 Matrix4x4::rotation_x(float angle) {
    Matrix4x4 mat;
    mat.m[0][0] = 1.0f;
    mat.m[0][1] = 0.0f;
    mat.m[0][2] = 0.0f;
    mat.m[0][3] = 0.0f;

    mat.m[1][0] = 0.0f;
    mat.m[1][1] = cos(angle);
    mat.m[1][2] = -sin(angle);
    mat.m[1][3] = 0.0f;

    mat.m[2][0] = 0.0f;
    mat.m[2][1] = sin(angle);
    mat.m[2][2] = cos(angle);
    mat.m[2][3] = 0.0f;

    mat.m[3][0] = 0.0f;
    mat.m[3][1] = 0.0f;
    mat.m[3][2] = 0.0f;
    mat.m[3][3] = 1.0f;

    return mat;
}

Matrix4x4 Matrix4x4::rotation_y(float alpha) {
    Matrix4x4 mat;
    mat.m[0][0] = cos(alpha);
    mat.m[0][1] = 0.0f;
    mat.m[0][2] = sin(alpha);
    mat.m[0][3] = 0.0f;

    mat.m[1][0] = 0.0f;
    mat.m[1][1] = 1.0f;
    mat.m[1][2] = 0.0f;
    mat.m[1][3] = 0.0f;

    mat.m[2][0] = -sin(alpha);
    mat.m[2][1] = 0.0f;
    mat.m[2][2] = cos(alpha);
    mat.m[2][3] = 0.0f;

    mat.m[3][0] = 0.0f;
    mat.m[3][1] = 0.0f;
    mat.m[3][2] = 0.0f;
    mat.m[3][3] = 1.0f;

    return mat;
}

Matrix4x4 Matrix4x4::rotation_z(float alpha) {
    Matrix4x4 mat;
    mat.m[0][0] = cos(alpha);
    mat.m[0][1] = -sin(alpha);
    mat.m[0][2] = 0.0f;
    mat.m[0][3] = 0.0f;

    mat.m[1][0] = sin(alpha);
    mat.m[1][1] = cos(alpha);
    mat.m[1][2] = 0.0f;
    mat.m[1][3] = 0.0f;

    mat.m[2][0] = 0.0f;
    mat.m[2][1] = 0.0f;
    mat.m[2][2] = 1.0f;
    mat.m[2][3] = 0.0f;

    mat.m[3][0] = 0.0f;
    mat.m[3][1] = 0.0f;
    mat.m[3][2] = 0.0f;
    mat.m[3][3] = 1.0f;

    return mat;
}