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
    Material material;
};

struct TracingResult {
    vec3f color;
};

__device__ double rng_value(uint32_t& state) {
    state = state * 747796405 + 2891336453;
    uint32_t result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    result = (result >> 22) ^ result;
    return result / 4294967296.0;
}

__device__ double rng_value_normal_distribution(uint32_t& state) {
    const double theta = 2 * M_PI * rng_value(state);
    const double rho = sqrt(-2 * log(rng_value(state)));
    return rho * cos(theta);
}

__device__ vec3f rng_direction(uint32_t& state) {
    auto x = (float)rng_value_normal_distribution(state);
    auto y = (float)rng_value_normal_distribution(state);
    auto z = (float)rng_value_normal_distribution(state);
    return vec3f {x, y, z}.normalize();
}

__device__ HitInfo ray_intersection(Ray ray) {
    HitInfo hit_info{false, {}, {}, 0.f, {}};

    for (int i = 0; i < 3; i++) {
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
                const vec3f normal = (point - C).normalize();
                hit_info = {true, point, normal, distance, sphere.material};
            }
        }
    }
    return hit_info;
}

__device__ TracingResult tracing(Ray ray, uint32_t& rng_state) {
    constexpr int BOUNCES_LIMIT = 10;

    vec3f incoming_light = vec3f{0, 0, 0};
    vec3f color = vec3f{1.f, 1.f, 1.f};

    for (int i = 0; i < BOUNCES_LIMIT; i++) {
        HitInfo hit_info = ray_intersection(ray);

        if (hit_info.hit) {
            vec3f emitted_light = hit_info.material.emission_color * hit_info.material.emission_strength;
            incoming_light = incoming_light + emitted_light * color;
            color = color * hit_info.material.color;
            ray.origin = hit_info.point;
            vec3f dir = rng_direction(rng_state);
            if (dir.dot(hit_info.normal) < 0)
                dir = dir * -1;
            ray.direction = dir;
        } else {
            vec3f sky_color;
            float dot = ray.direction.dot(camera->local_matrix.z());
            if (dot < 0)
                sky_color = vec3f{99, 69, 24}.normalize() * (-dot * 0.5f);
            else
                sky_color = vec3f{122, 173, 255}.normalize() * 2 * dot;
            color = color * sky_color;
            incoming_light = incoming_light + color;
            break;
        }
    }

    return TracingResult {incoming_light};
}

__device__ vec3f frag(vec2f uv, vec2i i, vec2i size) {
    uint32_t pixelIndex = i.x + i.y * size.x;

    uint32_t rng_state = pixelIndex;

    vec3f forward = {1, 0, 0};

    vec3f up = {0, 0, 1};
    vec3f right = {0, 1, 0};

    vec3f dir = (forward + right * uv.x + up * uv.y).normalize();

    Ray ray{};
    ray.origin = {0,0,0};
    ray.direction = dir;

    vec3f color = vec3f{0, 0, 0};
    constexpr int iterations = 50;

    for (int j = 0; j < iterations; j++) {
        color = color + tracing(ray, rng_state).color;
    }

    return color / (float)iterations;

    //return camera->local_matrix.x();
    //return hit.hit ? hit.normal : vec3f {0, 0, 0};
}