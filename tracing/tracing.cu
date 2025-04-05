#include "tracing.cuh"
#include "../rendering/rendering.cuh"

struct Ray {
    vec3f origin;
    vec3f direction;
};

struct HitInfo {
    bool hit;
    vec3f point;
    vec3f normal;
    float distance;
};

__device__ HitInfo ray_intersection(Ray ray) {
    HitInfo hit_info{false, {}, {}, 0.f};

    for (int i = 0; i < 2; i++) {
        auto sphere = spheres[i];

        vec3f C = camera->inv_local_matrix * sphere.center;
        float a = ray.direction.dot(ray.direction);
        float b = 2 * ray.direction.dot(ray.origin - C);
        float c = (ray.origin - C).dot(ray.origin - C) - sphere.radius * sphere.radius;

        float determinant = b*b - 4 * a * c;

        if (determinant >= 0) {
            float distance = (-b - sqrt(determinant)) / (2.0*a);
            if (distance < 0) continue;

            if (distance < hit_info.distance || !hit_info.hit) {
                const vec3f point = ray.origin + ray.direction * distance;
                const vec3f normal = (sphere.center - point).normalize();
                hit_info = {true, point, normal, distance};
            }
        }
    }
    return hit_info;
}

__device__ vec3f frag(vec2f uv, vec2i size) {
    vec3f forward = {1, 0, 0};

    vec3f up = {0, 0, 1};
    vec3f right = {0, 1, 0};

    vec3f dir = (forward + right * uv.x + up * uv.y).normalize();

    Ray ray{};
    ray.origin = {0,0,0};
    ray.direction = dir;

    const HitInfo hit = ray_intersection(ray);

    //return camera->local_matrix.x();
    return hit.hit ? hit.normal : vec3f {0, 0, 0};
}