#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Kd-treeConverter.h"
#include "cuCommonDefs.cuh"
//필요한 커널만 호출

// File: mini_renderer.cu
struct Camera {
    float3 origin;
    float3 lookat;
    float3 up;
    float fovY;
};

__device__ Ray generateRayFromCamera(const Camera& cam, float u, float v) {
    float3 forward = normalize(cam.lookat - cam.origin);
    float3 right = normalize(cross(forward, cam.up));
    float3 trueUp = cross(right, forward);

    float aspect = 1.0f; // 가정
    float scale = tanf(cam.fovY * 0.5f * 3.14159265f / 180.0f);

    float3 imagePlaneDir = normalize(forward +
        (2 * u - 1) * aspect * scale * right +
        (2 * v - 1) * scale * trueUp);

    Ray ray;
    ray.origin = cam.origin;
    ray.direction = imagePlaneDir;
    return ray;
}

__global__ void render_kernel(
    uchar4* output, int width, int height,
    Camera cam,
    CompositeObject obj)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    Ray ray = generateRayFromCamera(cam, u, v);

    Hit hit;
    bool isHit = traverseKdTreeForRayTracing(
        ray, obj.kd_tree_root, obj.triangle_list,
        obj.extended_vertex_list, hit, 0.001f, 1e30f);

    if (isHit)
        output[y * width + x] = make_uchar4(255, 255, 255, 255);
    else
        output[y * width + x] = make_uchar4(20, 20, 20, 255);
}
