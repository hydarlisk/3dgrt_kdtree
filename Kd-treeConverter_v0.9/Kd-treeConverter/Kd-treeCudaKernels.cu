#include "Kd-treeCudaKernels.h"
#include <cuda_runtime.h>
#include <vector> // C++ Wrapper 함수에서만 사용

#if SAH_OPACITY == 6

// ======================================================================
// 1. CUDA __device__ 헬퍼 함수
// CPU의 C++ 함수들을 GPU 스레드에서 실행 가능하도록 변환한 버전입니다.
// ======================================================================

// 최대 폴리곤 정점 수. 클리핑 결과가 이보다 많아지면 잘립니다.
#define MAX_POLY_VERTS 10

enum ClipAxis { _AXIS_X = 0, _AXIS_Y, _AXIS_Z };
enum ClipSide { _SIDE_MIN = 0, _SIDE_MAX };

__forceinline__ __device__ double device_dMyVecLength(const double* v) {
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

__forceinline__ __device__ void device_dMyVecCrossProduct(const double* v1, const double* v2, double* v) {
    v[0] = ((v1[1] * v2[2]) - (v1[2] * v2[1]));
    v[1] = -((v1[0] * v2[2]) - (v1[2] * v2[0]));
    v[2] = ((v1[0] * v2[1]) - (v1[1] * v2[0]));
}

__forceinline__ __device__ bool is_inside_device(float coord, float boundary, ClipSide side) {
    if (side == _SIDE_MIN) return coord >= boundary;
    return coord <= boundary;
}

__device__ double calculate_polygon_area_device(const ExtendedVertex* polygon, int vertex_count) {
    if (vertex_count < 3) {
        return 0.0;
    }

    double total_area = 0.0;
    const ExtendedVertex v0 = polygon[0];

    for (int i = 1; i < vertex_count - 1; ++i) {
        const ExtendedVertex v1 = polygon[i];
        const ExtendedVertex v2 = polygon[i + 1];

        double edge1[3] = { (double)v1.vertex[0] - v0.vertex[0], (double)v1.vertex[1] - v0.vertex[1], (double)v1.vertex[2] - v0.vertex[2] };
        double edge2[3] = { (double)v2.vertex[0] - v0.vertex[0], (double)v2.vertex[1] - v0.vertex[1], (double)v2.vertex[2] - v0.vertex[2] };

        double cross_product[3];
        device_dMyVecCrossProduct(edge1, edge2, cross_product);
        total_area += device_dMyVecLength(cross_product);
    }
    return total_area / 2.0;
}

__device__ void clip_polygon_against_plane_device(
    const ExtendedVertex* in_polygon, int in_count,
    ExtendedVertex* out_polygon, int* out_count,
    float boundary, ClipAxis axis, ClipSide side)
{
    *out_count = 0;
    if (in_count == 0) return;

    for (int i = 0; i < in_count; ++i) {
        if (*out_count >= MAX_POLY_VERTS) return; // 버퍼 초과 방지

        const ExtendedVertex p1 = in_polygon[i];
        const ExtendedVertex p2 = in_polygon[(i + 1) % in_count];

        bool p1_inside = is_inside_device(p1.vertex[axis], boundary, side);
        bool p2_inside = is_inside_device(p2.vertex[axis], boundary, side);

        if (p1_inside) {
            out_polygon[(*out_count)++] = p1;
        }

        if (p1_inside != p2_inside) {
            if (*out_count >= MAX_POLY_VERTS) return;
            float diff = p2.vertex[axis] - p1.vertex[axis];
            if (fabsf(diff) > 1e-6f) {
                float t = (boundary - p1.vertex[axis]) / diff;
                ExtendedVertex intersection;
                for (int j = 0; j < 3; ++j) {
                    intersection.vertex[j] = p1.vertex[j] + t * (p2.vertex[j] - p1.vertex[j]);
                }
                intersection.material_ID = p1.material_ID;
                out_polygon[(*out_count)++] = intersection;
            }
        }
    }
}

__device__ void clip_triangle_against_AABB_device(
    const ExtendedVertex triangle[3], const BoundingBox& bbox,
    ExtendedVertex* out_polygon, int* out_count)
{
    ExtendedVertex pg_buffers[2][MAX_POLY_VERTS];
    int pg_counts[2] = { 3, 0 };

    // 초기 삼각형 복사
    pg_buffers[0][0] = triangle[0];
    pg_buffers[0][1] = triangle[1];
    pg_buffers[0][2] = triangle[2];

    int current = 0;

    clip_polygon_against_plane_device(pg_buffers[current], pg_counts[current], pg_buffers[1 - current], &pg_counts[1 - current], bbox.min[0], _AXIS_X, _SIDE_MIN); current = 1 - current;
    clip_polygon_against_plane_device(pg_buffers[current], pg_counts[current], pg_buffers[1 - current], &pg_counts[1 - current], bbox.max[0], _AXIS_X, _SIDE_MAX); current = 1 - current;
    clip_polygon_against_plane_device(pg_buffers[current], pg_counts[current], pg_buffers[1 - current], &pg_counts[1 - current], bbox.min[1], _AXIS_Y, _SIDE_MIN); current = 1 - current;
    clip_polygon_against_plane_device(pg_buffers[current], pg_counts[current], pg_buffers[1 - current], &pg_counts[1 - current], bbox.max[1], _AXIS_Y, _SIDE_MAX); current = 1 - current;
    clip_polygon_against_plane_device(pg_buffers[current], pg_counts[current], pg_buffers[1 - current], &pg_counts[1 - current], bbox.min[2], _AXIS_Z, _SIDE_MIN); current = 1 - current;
    clip_polygon_against_plane_device(pg_buffers[current], pg_counts[current], pg_buffers[1 - current], &pg_counts[1 - current], bbox.max[2], _AXIS_Z, _SIDE_MAX); current = 1 - current;

    *out_count = pg_counts[current];
    for (int i = 0; i < *out_count; ++i) {
        out_polygon[i] = pg_buffers[current][i];
    }
}

// ======================================================================
// 2. 메인 CUDA 커널 (__global__ 함수)
// ======================================================================

__global__ void clip_triangles_kernel(
    const TriangleList* d_triangles,
    int num_triangles,
    BoundingBox left_bbox,
    BoundingBox right_bbox,
    double* d_contrib_L_results,
    double* d_contrib_R_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_triangles) {
        TriangleList tri = d_triangles[idx];

        ExtendedVertex clipped_poly[MAX_POLY_VERTS];
        int clipped_poly_size = 0;

        // 왼쪽 BBox에 대한 클리핑 및 면적 계산
        clip_triangle_against_AABB_device(tri.point, left_bbox, clipped_poly, &clipped_poly_size);
        if (clipped_poly_size >= 3) {
            d_contrib_L_results[idx] = tri.opacity * calculate_polygon_area_device(clipped_poly, clipped_poly_size);
        }
        else {
            d_contrib_L_results[idx] = 0.0;
        }

        // 오른쪽 BBox에 대한 클리핑 및 면적 계산
        clip_triangle_against_AABB_device(tri.point, right_bbox, clipped_poly, &clipped_poly_size);
        if (clipped_poly_size >= 3) {
            d_contrib_R_results[idx] = tri.opacity * calculate_polygon_area_device(clipped_poly, clipped_poly_size);
        }
        else {
            d_contrib_R_results[idx] = 0.0;
        }
    }
}

// ======================================================================
// 3. C++ 인터페이스 함수 구현 (Wrapper)
// ======================================================================

CudaClipResult calculate_clipped_contributions_cuda(
    const std::vector<const TriangleList*>& active_triangles,
    const BoundingBox& left_bbox,
    const BoundingBox& right_bbox
) {
    int num_triangles = active_triangles.size();
    if (num_triangles == 0) {
        return { 0.0, 0.0 };
    }

    std::vector<TriangleList> h_triangles_data;
    h_triangles_data.reserve(num_triangles);
    for (const auto* tri_ptr : active_triangles) {
        h_triangles_data.push_back(*tri_ptr);
    }

    TriangleList* d_triangles;
    double* d_contrib_L_results;
    double* d_contrib_R_results;
    cudaMalloc(&d_triangles, num_triangles * sizeof(TriangleList));
    cudaMalloc(&d_contrib_L_results, num_triangles * sizeof(double));
    cudaMalloc(&d_contrib_R_results, num_triangles * sizeof(double));

    cudaMemcpy(d_triangles, h_triangles_data.data(), num_triangles * sizeof(TriangleList), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (num_triangles + threads_per_block - 1) / threads_per_block;
    clip_triangles_kernel << <blocks_per_grid, threads_per_block >> > (
        d_triangles, num_triangles, left_bbox, right_bbox, d_contrib_L_results, d_contrib_R_results
        );

    // 에러 체크 (디버깅 시 유용)
    // cudaDeviceSynchronize(); 
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    // }

    std::vector<double> h_contrib_L(num_triangles);
    std::vector<double> h_contrib_R(num_triangles);
    cudaMemcpy(h_contrib_L.data(), d_contrib_L_results, num_triangles * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_contrib_R.data(), d_contrib_R_results, num_triangles * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_triangles);
    cudaFree(d_contrib_L_results);
    cudaFree(d_contrib_R_results);

    CudaClipResult final_result = { 0.0, 0.0 };
    for (int i = 0; i < num_triangles; ++i) {
        final_result.total_contrib_L += h_contrib_L[i];
        final_result.total_contrib_R += h_contrib_R[i];
    }

    return final_result;
}

#endif