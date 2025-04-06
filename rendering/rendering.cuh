#ifndef TEST_CUH
#define TEST_CUH
#include "../vector/vector.cuh"
#include "../matrix/matrix.cuh"
#define M_PI           3.14159265358979323846

struct Frame {
    int width;
    int height;
    uint32_t* pixels;
};

struct Camera {
    Matrix4x4 local_matrix;
    Matrix4x4 inv_local_matrix;
};

struct Material {
    vec3f color;
    vec3f emission_color;
    float emission_strength;
};

struct Sphere {
    vec3f center;
    float radius;
    Material material;
};

extern Camera h_camera;

__device__ inline Camera* camera;
__device__ inline Sphere* spheres;

constexpr int DISPLAY_WIDTH = 600;
constexpr int DISPLAY_HEIGHT = 400;

void render(struct Frame*);
void prepare_objects();
void update_objects();


#endif //TEST_CUH
