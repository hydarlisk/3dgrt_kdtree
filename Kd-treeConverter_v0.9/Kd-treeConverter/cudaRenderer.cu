#include <GL/glew.h>
#include <GL/freeglut.h>
#include <curand_kernel.h>
#include "OpenGLStuffs.h"
#include "CudaRenderer.h"
//#include "SGRTx2Lib/cudaRenderPipeline.h"
#include "SGRTx2Lib/cuda_math.h"


#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include <texture_indirect_functions.h>
#include <vector_types.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// =================================================================================
// 1. CUDA 커널 및 디바이스 헬퍼 함수/구조체 (SGRT 파일들에서 필요한 부분만 추출)
// =================================================================================
#define M_PI 3.14159265358979323846f
#define RAY_START_EPSILON 1e-4f
#define BARYCENTRY_EPSILON 1e-7f

// --- Device-side Data Structures ---
struct cuRay {
    float3 pos;
    float3 dir;
    __device__ float2 get_dir_pos(const unsigned axis) const {
        if (axis == 0) return make_float2(pos.x, dir.x);
        if (axis == 1) return make_float2(pos.y, dir.y);
        return make_float2(pos.z, dir.z);
    }
};

struct cuIntersectionCheck {
    float tHit;
    float beta, gamma;
    int triIndex;
    int objectIndex;
    int hitCount;
    __device__ void init() { tHit = FLT_MAX; triIndex = -1; objectIndex = 0; hitCount = 0; }
    __device__ bool isHit() const { return triIndex != -1; }
};

struct cuIntersectionPoint {
    float3 pos, dir, normal;
    float3 colorWeight;
    __device__ void init() { colorWeight = make_float3(1.0f, 1.0f, 1.0f); }
};

struct cuObjectMaterial {
    float3 ambient_emission;
    float3 diffuse;
    float3 specular;
    float reflection;
    float transparency;
    float roughness;
    float refractionIndex;
};

// Kd-tree 노드 (GKDTreeNode.h에서 추출)
typedef uint2 kdtreeNode;
#define IS_LEAF(node)               (((node).x & 3) == 3)
#define SPLIT_AXIS(node)            ( (node).x & 3)
#define FIRST_CHILD_OFFSET(node)    ( (node).x >> 3)
#define SPLIT_POS(node)             (*(float *)&((node).y))
#define OBJECT_SIZE(node)           ( (node).x >> 3)
#define OBJECTLIST_OFFSET(node)     ( (node).y)

// 스택 (cudaRenderPipelineCommonKernel.cu에서 추출)
#define SHORT_STACK_DEPTH 12
typedef struct { unsigned nodeID; float tMax; } cu_traceState;
extern __shared__ cu_traceState smemBuffer[SHORT_STACK_DEPTH * 256];

struct shortStack {
    unsigned _top, quant, baseOffset;
    __device__ void init(const unsigned smem_baseOffset) {
        baseOffset = smem_baseOffset * SHORT_STACK_DEPTH;
        _top = SHORT_STACK_DEPTH - 1;
        quant = 0;
    }
    __device__ cu_traceState top() { return smemBuffer[baseOffset + _top]; }
    __device__ void push(unsigned id, float t_max) {
        if (++_top == SHORT_STACK_DEPTH) _top = 0;
        quant = min(quant + 1, SHORT_STACK_DEPTH);
        smemBuffer[baseOffset + _top].nodeID = id;
        smemBuffer[baseOffset + _top].tMax = t_max;
    }
    __device__ bool empty() { return quant == 0; }
    __device__ void pop() {
        if (_top == 0) _top = SHORT_STACK_DEPTH;
        --_top; --quant;
    }
};

// --- Device-side Helper Functions ---

__device__ float3 reflection(float3 I, float3 N) {
    return I - 2.0f * N * dot(I, N);
}

__device__ float3 refraction(float3 I, float3 N, float eta) {
    float dotNI = dot(N, I);
    float k = 1.0f - eta * eta * (1.0f - dotNI * dotNI);
    if (k < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    return eta * I - (eta * dotNI + sqrtf(k)) * N;
}

// =================================================================================
// 2. 텍스춰 및 상수 메모리 선언
// =================================================================================
texture<uint2, 1, cudaReadModeElementType> inKdTreeNodeTex;
texture<uint, 1, cudaReadModeElementType> inObjectOffsetListTex;
texture<float4, 1, cudaReadModeElementType> inTriAccelTex;

struct SceneInfo { int resX, resY; };
struct CameraInfo { float3 eye, u, v, startPoint; float stepX, stepY; };

__constant__ SceneInfo g_SceneInfo;
__constant__ CameraInfo g_CameraInfo;
__constant__ float3 g_SceneBBoxMin;
__constant__ float3 g_SceneBBoxMax;
__device__ Gaussian* g_d_gaussians;

// =================================================================================
// 3. CUDA 커널 코드 (사용자 제공 커널)
// =================================================================================

__device__ bool BoundsRayIntersect(const float3& bmin, const float3& bmax, const cuRay& ray, float& tmin, float& tmax) {
    float3 invD = 1.0f / ray.dir;
    float3 t0s = (bmin - ray.pos) * invD;
    float3 t1s = (bmax - ray.pos) * invD;
    //float3 tsmaller = fminf(t0s, t1s);
    float3 tsmaller = min(t0s, t1s);
    //float3 tsmaller = make_float3(fminf(t0s.x, t1s.x), fminf(t0s.y, t1s.y), fminf(t0s.z, t1s.z));
    //float3 tbigger = fmaxf(t0s, t1s);
    float3 tbigger = max(t0s, t1s);
    //float3 tbigger = make_float3(fmaxf(t0s.x, t1s.x), fmaxf(t0s.y, t1s.y), fmaxf(t0s.z, t1s.z));
    tmin = fmaxf(tmin, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
    tmax = fminf(tmax, fminf(tbigger.x, fminf(tbigger.y, tbigger.z)));
    return ((tmin < tmax) & (tmax >= 0.f));
}

__device__ void singlePassIntersectRoutine(const cuRay& ray, int id, cuIntersectionCheck& hit, float t_near, float t_far) {
    float4 d0 = tex1Dfetch(inTriAccelTex, id * 4 + 0);
    unsigned int packed_flags = __float_as_uint(d0.w);
    unsigned int k = packed_flags & 0x3;
    float n_u = d0.x, n_v = d0.y, n_d = d0.z;

    float3 p_pos = ray.pos, p_dir = ray.dir;
    /*if (k == 1) {
        p_pos = make_float3(ray.pos.y, ray.pos.z, ray.pos.x);
        p_dir = make_float3(ray.dir.y, ray.dir.z, ray.dir.x);
    }
    else if (k == 2) {
        p_pos = make_float3(ray.pos.z, ray.pos.x, ray.pos.y);
        p_dir = make_float3(ray.dir.z, ray.dir.x, ray.dir.y);
    }*/
    //CPU 투영 방식과 일치하오록 수정
    if (k == 0) {       //YZ 평면에 투영: (y, z, x) 순서로 변경
        p_pos = make_float3(ray.pos.y, ray.pos.z, ray.pos.x);
        p_dir = make_float3(ray.dir.y, ray.dir.z, ray.dir.x);
    }
    else if (k == 1) {  //ZX 평면에 투영: (z, x, y) 순서로 변경
        p_pos = make_float3(ray.pos.z, ray.pos.x, ray.pos.y);
        p_dir = make_float3(ray.dir.z, ray.dir.x, ray.dir.y);
    }

    float den = p_dir.z + n_u * p_dir.x + n_v * p_dir.y;
    if (fabsf(den) < 1e-8f) return;
    float t = (n_d - (p_pos.z + n_u * p_pos.x + n_v * p_pos.y)) / den;

    //if (t >= hit.tHit || t <= t_near || t >= t_far) return;
    if (t <= t_near || t >= t_far) return;

    float4 d1 = tex1Dfetch(inTriAccelTex, id * 4 + 1);
    float4 d2 = tex1Dfetch(inTriAccelTex, id * 4 + 2);

    float u_coord = p_pos.x + t * p_dir.x;
    float v_coord = p_pos.y + t * p_dir.y;

    float beta = u_coord * d1.x + v_coord * d1.y + d1.z;
    float gamma = u_coord * d2.x + v_coord * d2.y + d2.z;

    if (beta >= -BARYCENTRY_EPSILON && gamma >= -BARYCENTRY_EPSILON && (beta + gamma) <= 1.0f + BARYCENTRY_EPSILON) {
        hit.hitCount++; // 유효한 충돌이므로 카운터를 1 증가

        // 가장 가까운 충돌점 정보는 계속 갱신
        if (t < hit.tHit) {
            hit.tHit = t;
            hit.beta = beta;
            hit.gamma = gamma;
            hit.triIndex = id;
        }
    }

    //if ((beta < 0.f - BARYCENTRY_EPSILON) | (gamma < 0.f - BARYCENTRY_EPSILON) | ((1.0f - beta - gamma) < 0.0f - BARYCENTRY_EPSILON)) {
    //    return;
    //}

    //hit.tHit = t;
    //hit.beta = beta;
    //hit.gamma = gamma;
    //hit.triIndex = id;
}

__device__ void singlePassIntersect(cuRay& currRay, cuIntersectionCheck& intersectionCheck) {
    float t_near = RAY_START_EPSILON, t_far = FLT_MAX;
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, currRay, t_near, t_far)) {
        shortStack stack;
        stack.init(threadIdx.x + threadIdx.y * blockDim.x);
        kdtreeNode node = tex1Dfetch(inKdTreeNodeTex, 0);
        while (true) {
            while (!IS_LEAF(node)) {
                unsigned axis = SPLIT_AXIS(node);
                float split_pos = SPLIT_POS(node);
                float dir_axis = (&(currRay.dir.x))[axis];
                if (fabsf(dir_axis) < 1e-8f) { // 광선이 축과 평행한 경우
                    node = tex1Dfetch(inKdTreeNodeTex, FIRST_CHILD_OFFSET(node) + 1); // 임의로 한쪽으로 보냄
                    continue;
                }
                float pos_axis = (&(currRay.pos.x))[axis];
                float t_split = (split_pos - pos_axis) / dir_axis;
                unsigned near_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 0 : 1);
                unsigned far_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 1 : 0);
                if (t_split > t_far || t_split < 0) {
                    node = tex1Dfetch(inKdTreeNodeTex, near_child);
                }
                else if (t_split < t_near) {
                    node = tex1Dfetch(inKdTreeNodeTex, far_child);
                }
                else {
                    stack.push(far_child, t_far);
                    node = tex1Dfetch(inKdTreeNodeTex, near_child);
                    t_far = t_split;
                }
            }
            unsigned offset = OBJECTLIST_OFFSET(node);
            unsigned count = OBJECT_SIZE(node);
            for (unsigned i = 0; i < count; ++i) {
                unsigned tri_idx = tex1Dfetch(inObjectOffsetListTex, offset + i);
                singlePassIntersectRoutine(currRay, tri_idx, intersectionCheck, t_near, t_far);
            }
            if (/*intersectionCheck.tHit < t_far || */stack.empty()) break;
            cu_traceState next = stack.top(); stack.pop();
            t_near = t_far; t_far = next.tMax;
            node = tex1Dfetch(inKdTreeNodeTex, next.nodeID);
        }
    }
}

__global__ void renderKernelObj(float* pFrameBuffer, int* maxhit) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    float sx = (float)x + 0.5f, sy = (float)y + 0.5f;
    float3 dir = g_CameraInfo.startPoint + g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;

    cuRay ray = { g_CameraInfo.eye, normalize(dir - g_CameraInfo.eye) };
    cuIntersectionCheck hit;
    hit.init();
    //hit.tHit = FLT_MAX; hit.triIndex = -1;

    singlePassIntersect(ray, hit);

    float3 finalColor = make_float3(0.2f, 0.3f, 0.4f); // 배경색
    if (hit.triIndex != -1) {
        float4 N_packed = tex1Dfetch(inTriAccelTex, hit.triIndex * 4 + 3);
        float3 N = normalize(make_float3(N_packed.x, N_packed.y, N_packed.z));
        float lambert = fmaxf(0.0f, dot(N, normalize(make_float3(1, -1, -1))));
        finalColor = make_float3(0.9f, 0.8f, 0.7f) * lambert + make_float3(0.1f, 0.1f, 0.1f);

        atomicMax(maxhit, hit.hitCount);
        //if (*maxhit == hit.hitCount && hit.hitCount != 0) printf("(in kernel) maxhit:%d\n", *maxhit);
        //printf("(kernel)maxhit:%d", *maxhit);
        float red_factor = fminf(1.0f, (float)hit.hitCount / 20.f);
        finalColor += make_float3(red_factor * 0.8f, 0.0f, 0.0f);
    }

    int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
    pFrameBuffer[idx + 0] = finalColor.x;
    pFrameBuffer[idx + 1] = finalColor.y;
    pFrameBuffer[idx + 2] = finalColor.z;
}

// =================================================================================
// 4. Host-Side Public Render Function
// =================================================================================

bool initCuda() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CUDA Init] No CUDA devices found." << std::endl;
        return false;
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Init] Failed to set device 0." << std::endl;
        return false;
    }
    std::cout << "[CUDA Init] CUDA device initialized successfully." << std::endl;
    return true;
}

void renderObjWithCuda(const CompositeObject& object, const Camera& camera, int width, int height, float*& out_framebuffer, bool& is_done) {
    is_done = false;
    //int num_gpus;
    //cudaGetDeviceCount(&num_gpus);
    //printf("numgpu:%d\n", num_gpus);
    //cudaSetDevice(0);
    //std::cout << "--- Minimal CUDA Renderer Started ---" << std::endl;

    KdTree* kdTree = object.kd_tree;
    if (!kdTree || object.n_triangles == 0) {
        std::cerr << "[CUDA Error] Object or Kd-tree is empty." << std::endl;
        return;
    }

    //printf("0. exist KD-Tree\n");

    //printf("[DEBUG] object.n_triangles = %d\n", object.n_triangles);
    if (kdTree == nullptr) {
        printf("[FATAL] kdTree == nullptr\n");
        return;
    }
    if (kdTree->tri_accel_list == nullptr) {
        printf("[FATAL] tri_accel_list == nullptr\n");
        return;
    }
    if (kdTree->tri_accel_list + object.n_triangles <= kdTree->tri_accel_list) {
        printf("[FATAL] tri_accel_list too small or corrupt pointer\n");
        return;
    }

    // 1. 데이터 패킹 (Host)
    // TriAccel -> float4[4] (n_u, n_v, n_d, k | b_nu, b_nv, b_d, idx | c_nu, c_nv, c_d, matID | N.x, N.y, N.z, pad)
    std::vector<float4> h_triangles(object.n_triangles * 4);
    //float4* h_triangles = (float4*)malloc(sizeof(float4) * (object.n_triangles * 4));
    for (int i = 0; i < object.n_triangles; ++i) {
        const TriAccel& src = kdTree->tri_accel_list[i];
        h_triangles[i * 4 + 0] = make_float4(src.n_u, src.n_v, src.n_d, uint_as_float_H(src.k));
        h_triangles[i * 4 + 1] = make_float4(src.b_nu, src.b_nv, src.b_d, int_as_float_H(src.indexInObject));
        h_triangles[i * 4 + 2] = make_float4(src.c_nu, src.c_nv, src.c_d, int_as_float_H(src.material_ID));
        h_triangles[i * 4 + 3] = make_float4(src.N[0], src.N[1], src.N[2], 0.0f);
    }
    //printf("1. Data packing done\n");

    // 2. GPU 메모리 할당 및 데이터 전송
    cudaError_t err;
    kdtreeNode* d_kdtree_nodes;
    unsigned int* d_tri_offsets;
    float4* d_tri_accel;

    //printf("kdtree node count: %d\n", kdTree->tree_node_count);
    cudaChannelFormatDesc node_desc = cudaCreateChannelDesc<uint2>();

    cudaChannelFormatDesc offset_desc = cudaCreateChannelDesc<unsigned int>();

    cudaChannelFormatDesc tri_desc = cudaCreateChannelDesc<float4>();

    size_t node_size = kdTree->tree_node_count * sizeof(kdtreeNode);
    CUDA_CHECK(cudaMalloc(&d_kdtree_nodes, node_size));
    CUDA_CHECK(cudaMemcpy(d_kdtree_nodes, kdTree->tree, node_size, cudaMemcpyHostToDevice));

    size_t offset_size = kdTree->tri_offset_count * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_tri_offsets, offset_size));
    CUDA_CHECK(cudaMemcpy(d_tri_offsets, kdTree->tri_offset_list, offset_size, cudaMemcpyHostToDevice));

    size_t accel_size = h_triangles.size() * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&d_tri_accel, accel_size));
    CUDA_CHECK(cudaMemcpy(d_tri_accel, h_triangles.data(), accel_size, cudaMemcpyHostToDevice));


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] GPU memcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("2. gpu memcpy done\n");
    
    // 3. 텍스처 바인딩
    CUDA_CHECK(cudaBindTexture(0, &inKdTreeNodeTex, d_kdtree_nodes, &node_desc, node_size));
    CUDA_CHECK(cudaBindTexture(0, &inObjectOffsetListTex, d_tri_offsets, &offset_desc, offset_size));
    CUDA_CHECK(cudaBindTexture(0, &inTriAccelTex, d_tri_accel, &tri_desc, accel_size));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] texture bind failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("3. texture Bind done\n");
    
    // 4. 상수 메모리 설정
    SceneInfo h_scene_info = { width, height };
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneInfo, &h_scene_info, sizeof(SceneInfo)));

    CameraInfo h_camera_info;
    h_camera_info.eye = make_float3(camera.pos[0], camera.pos[1], camera.pos[2]);
    h_camera_info.u = make_float3(camera.uaxis[0], camera.uaxis[1], camera.uaxis[2]);
    h_camera_info.v = make_float3(camera.vaxis[0], camera.vaxis[1], camera.vaxis[2]);
    float3 n_axis = make_float3(camera.naxis[0], camera.naxis[1], camera.naxis[2]);

    float fov_rad = camera.fovy * (M_PI / 180.0f);
    float plane_height = 2.0f * camera.near_c * tanf(fov_rad * 0.5f);
    float plane_width = plane_height * camera.aspect;
    h_camera_info.stepX = plane_width / width;
    h_camera_info.stepY = plane_height / height;
    h_camera_info.startPoint = h_camera_info.eye - n_axis * camera.near_c
        - h_camera_info.u * (plane_width * 0.5f)
        + h_camera_info.v * (plane_height * 0.5f);
    CUDA_CHECK(cudaMemcpyToSymbol(g_CameraInfo, &h_camera_info, sizeof(CameraInfo)));

    float3 h_bbox_min = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    float3 h_bbox_max = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMin, &h_bbox_min, sizeof(float3)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMax, &h_bbox_max, sizeof(float3)));

    //cuObjectMaterial h_material;
    //h_material.ambient_emission = make_float3(0.1f, 0.1f, 0.1f);
    //h_material.diffuse = make_float3(0.8f, 0.7f, 0.6f);
    //h_material.specular = make_float3(0.2f, 0.2f, 0.2f);
    //h_material.reflection = 0.05f;
    //h_material.transparency = 0.0f;
    //h_material.roughness = 32.0f;
    //h_material.refractionIndex = 1.0f;
    //CUDA_CHECK(cudaMemcpyToSymbol(g_materials, &h_material, sizeof(cuObjectMaterial)));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] const memory set failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("4. const memory set done\n");

    // 5. 커널 실행
    float* d_framebuffer;
    size_t framebuffer_size = width * height * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_framebuffer, framebuffer_size));
    CUDA_CHECK(cudaMemset(d_framebuffer, 0, framebuffer_size));

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    size_t shared_mem_size = threads.x * threads.y * SHORT_STACK_DEPTH * sizeof(cu_traceState);

    int* d_maxhit;
    CUDA_CHECK(cudaMalloc((void**)&d_maxhit, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_maxhit, 0, sizeof(int)));

    //singlePassRayTracingKernel_ShadowOff <<< blocks, threads, shared_mem_size >>> (d_framebuffer, 3);
    renderKernelObj <<< blocks, threads, shared_mem_size >>> (d_framebuffer, d_maxhit);
    CUDA_CHECK(cudaGetLastError());        // launch 실패 확인
    CUDA_CHECK(cudaDeviceSynchronize()); // 실행 중 오류 확인

    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_maxhit, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_maxhit));
    printf("af maxHIT count: %d\n", h_count);
    //free(h_count);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl; 
    }

    //printf("5. kernel launch done\n");

    // 6. 결과 복사 및 메모리 해제
    if (out_framebuffer) delete[] out_framebuffer;
    out_framebuffer = new float[width * height * 3];
    CUDA_CHECK(cudaMemcpy(out_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost));
    is_done = true;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    //free(h_triangles);

    //cudaFreeArray(d_kdtree_nodes);
    //cudaFreeArray(d_tri_offsets);
    //cudaFreeArray(d_tri_accel);
    cudaFree(d_kdtree_nodes);
    cudaFree(d_tri_offsets);
    cudaFree(d_tri_accel);
    cudaFree(d_framebuffer);
    cudaUnbindTexture(inKdTreeNodeTex);
    cudaUnbindTexture(inObjectOffsetListTex);
    cudaUnbindTexture(inTriAccelTex);
    //printf("6. free done\n");
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] free failed: " << cudaGetErrorString(err) << std::endl;
    }
    //std::cout << "--- Minimal CUDA Renderer Finished ---" << std::endl;
}

//-----------------------------------------------------------------------
// Gaussian Render
//-----------------------------------------------------------------------
struct HitRecord {
    float t;
    int triIndex;
};

__device__ void sortHits(HitRecord* hits, int count) {
    //printf("count: %d\n", count);
    for (int i = 1; i < count; i++) {
        HitRecord key = hits[i];
        int j = i - 1;
        while (j >= 0 && hits[j].t > key.t) {
            hits[j + 1] = hits[j];
            j = j - 1;
        }
        hits[j + 1] = key;
    }
}

__device__ void singlePassIntersectRoutineGaussian(const cuRay& ray, int id, cuIntersectionCheck& hit, float t_near, float t_far, HitRecord* hits) {
    if (hit.hitCount >= MAX_HITS) return;

    float4 d0 = tex1Dfetch(inTriAccelTex, id * 4 + 0);
    unsigned int packed_flags = __float_as_uint(d0.w);
    unsigned int k = packed_flags & 0x3;
    float n_u = d0.x, n_v = d0.y, n_d = d0.z;

    float3 p_pos = ray.pos, p_dir = ray.dir;
    //CPU 투영 방식과 일치하오록 수정
    if (k == 0) {       //YZ 평면에 투영: (y, z, x) 순서로 변경
        p_pos = make_float3(ray.pos.y, ray.pos.z, ray.pos.x);
        p_dir = make_float3(ray.dir.y, ray.dir.z, ray.dir.x);
    }
    else if (k == 1) {  //ZX 평면에 투영: (z, x, y) 순서로 변경
        p_pos = make_float3(ray.pos.z, ray.pos.x, ray.pos.y);
        p_dir = make_float3(ray.dir.z, ray.dir.x, ray.dir.y);
    }

    float den = p_dir.z + n_u * p_dir.x + n_v * p_dir.y;
    if (fabsf(den) < 1e-8f) return;
    float t = (n_d - (p_pos.z + n_u * p_pos.x + n_v * p_pos.y)) / den;

    //if (t >= hit.tHit || t <= t_near || t >= t_far) return;
    if (t <= t_near || t >= t_far) return;

    float4 d1 = tex1Dfetch(inTriAccelTex, id * 4 + 1);
    float4 d2 = tex1Dfetch(inTriAccelTex, id * 4 + 2);

    float u_coord = p_pos.x + t * p_dir.x;
    float v_coord = p_pos.y + t * p_dir.y;

    float beta = u_coord * d1.x + v_coord * d1.y + d1.z;
    float gamma = u_coord * d2.x + v_coord * d2.y + d2.z;

    if (beta >= -BARYCENTRY_EPSILON && gamma >= -BARYCENTRY_EPSILON && (beta + gamma) <= 1.0f + BARYCENTRY_EPSILON) {
        float4 N_packed = tex1Dfetch(inTriAccelTex, id * 4 + 3);
        float3 N = make_float3(N_packed.x, N_packed.y, N_packed.z);

        // 2. 법선 벡터와 광선 방향의 내적(dot product)을 계산합니다.
        //    내적 값이 0보다 크면 광선이 삼각형의 뒷면에 부딪혔다는 의미입니다.
        if (dot(N, ray.dir) > 0.0f) {
            return; // 뒷면이므로 이 충돌을 무시하고 함수를 즉시 종료합니다.
        }
        
        hits[hit.hitCount].t = t;
        hits[hit.hitCount].triIndex = id;
        hit.hitCount++; // 유효한 충돌이므로 카운터를 1 증가

        //// 가장 가까운 충돌점 정보는 계속 갱신
        //if (t < hit.tHit) {
        //    hit.tHit = t;
        //    hit.beta = beta;
        //    hit.gamma = gamma;
        //    hit.triIndex = id;
        //}
    }
}

__device__ void singlePassIntersectGaussian(
    cuRay& currRay,
    cuIntersectionCheck& intersectionCheck,
    HitRecord* hits
) {
    float t_near = RAY_START_EPSILON, t_far = FLT_MAX;
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, currRay, t_near, t_far)) {
        shortStack stack;
        stack.init(threadIdx.x + threadIdx.y * blockDim.x);
        kdtreeNode node = tex1Dfetch(inKdTreeNodeTex, 0);
        while (true) {
            if (intersectionCheck.hitCount >= MAX_HITS) break;
            while (!IS_LEAF(node)) {
                unsigned axis = SPLIT_AXIS(node);
                float split_pos = SPLIT_POS(node);
                float dir_axis = (&(currRay.dir.x))[axis];
                if (fabsf(dir_axis) < 1e-8f) { // 광선이 축과 평행한 경우
                    node = tex1Dfetch(inKdTreeNodeTex, FIRST_CHILD_OFFSET(node) + 1); // 임의로 한쪽으로 보냄
                    continue;
                }
                float pos_axis = (&(currRay.pos.x))[axis];
                float t_split = (split_pos - pos_axis) / dir_axis;
                unsigned near_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 0 : 1);
                unsigned far_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 1 : 0);
                if (t_split > t_far || t_split < 0) {
                    node = tex1Dfetch(inKdTreeNodeTex, near_child);
                }
                else if (t_split < t_near) {
                    node = tex1Dfetch(inKdTreeNodeTex, far_child);
                }
                else {
                    stack.push(far_child, t_far);
                    node = tex1Dfetch(inKdTreeNodeTex, near_child);
                    t_far = t_split;
                }
            }
            unsigned offset = OBJECTLIST_OFFSET(node);
            unsigned count = OBJECT_SIZE(node);
            for (unsigned i = 0; i < count; ++i) {
                unsigned tri_idx = tex1Dfetch(inObjectOffsetListTex, offset + i);
                singlePassIntersectRoutineGaussian(currRay, tri_idx, intersectionCheck, t_near, t_far, hits);
            }
            if (stack.empty()) break;
            cu_traceState next = stack.top(); stack.pop();
            t_near = t_far; t_far = next.tMax;
            node = tex1Dfetch(inKdTreeNodeTex, next.nodeID);
        }
    }
}

__device__ __forceinline__ float3 eval_sh_final(
    const int degree,
    const float3& view_dir,
    const Gaussian& g
) {
    // 0차 SH: 기본 색상
    const float SH_C0 = 0.2820947917f;
    float3 result = make_float3(g.f_dc[0], g.f_dc[1], g.f_dc[2]);

    if (degree > 0) {
        // 1차 SH
        const float SH_C1 = 0.4886025119f;
        float x = view_dir.x, y = view_dir.y, z = view_dir.z;
        result.x += SH_C1 * (-y * g.f_rest[0 * 3 + 0] + z * g.f_rest[1 * 3 + 0] - x * g.f_rest[2 * 3 + 0]);
        result.y += SH_C1 * (-y * g.f_rest[0 * 3 + 1] + z * g.f_rest[1 * 3 + 1] - x * g.f_rest[2 * 3 + 1]);
        result.z += SH_C1 * (-y * g.f_rest[0 * 3 + 2] + z * g.f_rest[1 * 3 + 2] - x * g.f_rest[2 * 3 + 2]);

        if (degree > 1) {
            // 2차 SH
            const float SH_C2_0 = 1.0925484306f, SH_C2_1 = -1.0925484306f, SH_C2_2 = 0.3153915652f, SH_C2_3 = -1.0925484306f, SH_C2_4 = 0.5462742153f;
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result.x += SH_C2_0 * xy * g.f_rest[3 * 3 + 0] + SH_C2_1 * yz * g.f_rest[4 * 3 + 0] + SH_C2_2 * (2.f * zz - xx - yy) * g.f_rest[5 * 3 + 0] + SH_C2_3 * xz * g.f_rest[6 * 3 + 0] + SH_C2_4 * (xx - yy) * g.f_rest[7 * 3 + 0];
            result.y += SH_C2_0 * xy * g.f_rest[3 * 3 + 1] + SH_C2_1 * yz * g.f_rest[4 * 3 + 1] + SH_C2_2 * (2.f * zz - xx - yy) * g.f_rest[5 * 3 + 1] + SH_C2_3 * xz * g.f_rest[6 * 3 + 1] + SH_C2_4 * (xx - yy) * g.f_rest[7 * 3 + 1];
            result.z += SH_C2_0 * xy * g.f_rest[3 * 3 + 2] + SH_C2_1 * yz * g.f_rest[4 * 3 + 2] + SH_C2_2 * (2.f * zz - xx - yy) * g.f_rest[5 * 3 + 2] + SH_C2_3 * xz * g.f_rest[6 * 3 + 2] + SH_C2_4 * (xx - yy) * g.f_rest[7 * 3 + 2];

            if (degree > 2) {
                // 3차 SH
                const float SH_C3_0 = -0.5900435899f, SH_C3_1 = 2.8906114426f, SH_C3_2 = -0.4570457996f, SH_C3_3 = 0.3731763326f, SH_C3_4 = -0.4570457996f, SH_C3_5 = 1.4453057213f, SH_C3_6 = -0.5900435899f;
                result.x += SH_C3_0 * y * (3 * xx - yy) * g.f_rest[8 * 3 + 0] + SH_C3_1 * xy * z * g.f_rest[9 * 3 + 0] + SH_C3_2 * y * (4 * zz - xx - yy) * g.f_rest[10 * 3 + 0] + SH_C3_3 * z * (2 * zz - 3 * xx - 3 * yy) * g.f_rest[11 * 3 + 0] + SH_C3_4 * x * (4 * zz - xx - yy) * g.f_rest[12 * 3 + 0] + SH_C3_5 * z * (xx - yy) * g.f_rest[13 * 3 + 0] + SH_C3_6 * x * (xx - 3 * yy) * g.f_rest[14 * 3 + 0];
                result.y += SH_C3_0 * y * (3 * xx - yy) * g.f_rest[8 * 3 + 1] + SH_C3_1 * xy * z * g.f_rest[9 * 3 + 1] + SH_C3_2 * y * (4 * zz - xx - yy) * g.f_rest[10 * 3 + 1] + SH_C3_3 * z * (2 * zz - 3 * xx - 3 * yy) * g.f_rest[11 * 3 + 1] + SH_C3_4 * x * (4 * zz - xx - yy) * g.f_rest[12 * 3 + 1] + SH_C3_5 * z * (xx - yy) * g.f_rest[13 * 3 + 1] + SH_C3_6 * x * (xx - 3 * yy) * g.f_rest[14 * 3 + 1];
                result.z += SH_C3_0 * y * (3 * xx - yy) * g.f_rest[8 * 3 + 2] + SH_C3_1 * xy * z * g.f_rest[9 * 3 + 2] + SH_C3_2 * y * (4 * zz - xx - yy) * g.f_rest[10 * 3 + 2] + SH_C3_3 * z * (2 * zz - 3 * xx - 3 * yy) * g.f_rest[11 * 3 + 2] + SH_C3_4 * x * (4 * zz - xx - yy) * g.f_rest[12 * 3 + 2] + SH_C3_5 * z * (xx - yy) * g.f_rest[13 * 3 + 2] + SH_C3_6 * x * (xx - 3 * yy) * g.f_rest[14 * 3 + 2];
            }
        }
    }
    // 원본 3dgrut과 동일하게, 최종적으로 0.5를 더해 [0,1] 범위로 이동
    result.x = result.x * SH_C0 + 0.5f;
    result.y = result.y * SH_C0 + 0.5f;
    result.z = result.z * SH_C0 + 0.5f;
    return result;
}

__global__ void renderKernelGaussian(float* pFrameBuffer
    ,int* maxhit, int* hcount
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    float sx = (float)x + 0.5f, sy = (float)y + 0.5f;
#if SUPER_SAMPLING
const int samples_per_pixel = 16; // 픽셀당 샘플 수 (4, 9, 16 등 제곱수 사용)
float3 final_color = make_float3(0.0f, 0.0f, 0.0f);

unsigned int seed = y * g_SceneInfo.resX + x;
curandState rand_state;
curand_init(seed, 0, 0, &rand_state);

for (int s = 0; s < samples_per_pixel; ++s) {
    // 픽셀 내에서 약간의 랜덤한 오프셋을 줍니다.
    float u_offset = curand_uniform(&rand_state);
    float v_offset = curand_uniform(&rand_state);

    sx = (float)x + u_offset;
    sy = (float)y + v_offset;
#endif

    float3 dir = g_CameraInfo.startPoint + g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;

    cuRay ray = { g_CameraInfo.eye, normalize(dir - g_CameraInfo.eye) };
    cuIntersectionCheck hit;
    hit.init();
    HitRecord hits[MAX_HITS];

    singlePassIntersectGaussian(ray, hit, hits);
    if (hit.hitCount == 0) {
        int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
        //pFrameBuffer[idx + 0] = 0.2f; // 배경색 R
        //pFrameBuffer[idx + 1] = 0.3f; // 배경색 G
        //pFrameBuffer[idx + 2] = 0.4f; // 배경색 B
        pFrameBuffer[idx + 0] = 0.0f; // 배경색 R
        pFrameBuffer[idx + 1] = 0.0f; // 배경색 G
        pFrameBuffer[idx + 2] = 0.0f; // 배경색 B
        return;
    }

    // sort
    sortHits(hits, hit.hitCount);

    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    float accumulated_opacity = 0.0f;

    for (int i = 0; i < hit.hitCount; ++i) {
        // TriAccel 텍스처에서 material_ID (가우시안 ID)를 가져옵니다.
        float4 d2 = tex1Dfetch(inTriAccelTex, hits[i].triIndex * 4 + 2);
        int gaussianID = __float_as_int(d2.w);

        Gaussian g = g_d_gaussians[gaussianID];

        float sample_opacity = 1.0f / (1.0f + expf(-g.opacity)); // Sigmoid
        //float3 dir = normalize(ray.pos - make_float3(g.pos[0], g.pos[1], g.pos[2]));
        //float3 sample_color = make_float3(
        //    0.5f + 0.5f * g.f_dc[0], // SH DC 계수는 -0.5~0.5 범위일 수 있으므로 0~1로 변환
        //    0.5f + 0.5f * g.f_dc[1],
        //    0.5f + 0.5f * g.f_dc[2]
        //);
        float3 dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - ray.pos);
        float3 sample_color = eval_sh_final(3, dir, g);

        accumulated_color += sample_color * sample_opacity * (1.0f - accumulated_opacity);
        accumulated_opacity += sample_opacity * (1.0f - accumulated_opacity);

        if (accumulated_opacity > 0.97f) {
            //printf("hitCount %d | ao: %f\n", hit.hitCount, accumulated_opacity);
            break;
        }
    }
    //if(accumulated_opacity<0.95f)
    //printf("hitCount %d | ao: %f\n", hit.hitCount, accumulated_opacity);

    // 5. 최종 색상 계산 및 프레임버퍼에 쓰기
    //float3 background_color = make_float3(0.2f, 0.3f, 0.4f);
    float3 background_color = make_float3(0.0f, 0.0f, 0.0f);
#if SUPER_SAMPLING
    final_color += accumulated_color + background_color * (1.0f - accumulated_opacity);
}
// 모든 샘플의 색상 값을 평균냅니다.
final_color /= samples_per_pixel;
#endif
    float3 final_color = accumulated_color + background_color * (1.0f - accumulated_opacity);
    //float3 final_color = accumulated_color / accumulated_opacity;

    int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
    pFrameBuffer[idx + 0] = final_color.x;
    pFrameBuffer[idx + 1] = final_color.y;
    pFrameBuffer[idx + 2] = final_color.z;

    if (hit.hitCount > 0) {
        //if(accumulated_opacity<0.95f) printf("hitCount %d | ao: %f\n", hit.hitCount, accumulated_opacity);
        atomicAdd(maxhit, hit.hitCount);
        atomicAdd(hcount, 1);
    }
}

void renderGaussianWithCuda(const CompositeObject& object, const std::vector<Gaussian>& gaussians, const Camera& camera, int width, int height, float*& out_framebuffer, bool& is_done) {
    is_done = false;
    //std::cout << "--- Minimal CUDA Renderer Started ---" << std::endl;
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaEventRecord(start);

    KdTree* kdTree = object.kd_tree;
    if (!kdTree || object.n_triangles == 0) {
        std::cerr << "[CUDA Error] Object or Kd-tree is empty." << std::endl;
        return;
    }
    if (!kdTree || gaussians.empty()) {
        std::cerr << "[CUDA Error] Kd-tree or Gaussian data is empty." << std::endl;
        return;
    }

    //printf("0. exist KD-Tree\n");

    //printf("[DEBUG] object.n_triangles = %d\n", object.n_triangles);
    if (kdTree == nullptr) {
        printf("[FATAL] kdTree == nullptr\n");
        return;
    }
    if (kdTree->tri_accel_list == nullptr) {
        printf("[FATAL] tri_accel_list == nullptr\n");
        return;
    }
    if (kdTree->tri_accel_list + object.n_triangles <= kdTree->tri_accel_list) {
        printf("[FATAL] tri_accel_list too small or corrupt pointer\n");
        return;
    }

    // 1. 데이터 패킹 (Host)
    // TriAccel -> float4[4] (n_u, n_v, n_d, k | b_nu, b_nv, b_d, idx | c_nu, c_nv, c_d, matID | N.x, N.y, N.z, pad)
    std::vector<float4> h_triangles(object.n_triangles * 4);
    //float4* h_triangles = (float4*)malloc(sizeof(float4) * (object.n_triangles * 4));
    for (int i = 0; i < object.n_triangles; ++i) {
        const TriAccel& src = kdTree->tri_accel_list[i];
        h_triangles[i * 4 + 0] = make_float4(src.n_u, src.n_v, src.n_d, uint_as_float_H(src.k));
        h_triangles[i * 4 + 1] = make_float4(src.b_nu, src.b_nv, src.b_d, int_as_float_H(src.indexInObject));
        h_triangles[i * 4 + 2] = make_float4(src.c_nu, src.c_nv, src.c_d, int_as_float_H(src.material_ID));
        h_triangles[i * 4 + 3] = make_float4(src.N[0], src.N[1], src.N[2], 0.0f);
    }
    //printf("1. Data packing done\n");

    // 2. GPU 메모리 할당 및 데이터 전송
    cudaError_t err;
    kdtreeNode* d_kdtree_nodes;
    unsigned int* d_tri_offsets;
    float4* d_tri_accel;
    Gaussian* d_gaussians_ptr;

    //printf("kdtree node count: %d\n", kdTree->tree_node_count);
    cudaChannelFormatDesc node_desc = cudaCreateChannelDesc<uint2>();

    cudaChannelFormatDesc offset_desc = cudaCreateChannelDesc<unsigned int>();

    cudaChannelFormatDesc tri_desc = cudaCreateChannelDesc<float4>();

    size_t node_size = kdTree->tree_node_count * sizeof(kdtreeNode);
    CUDA_CHECK(cudaMalloc(&d_kdtree_nodes, node_size));
    CUDA_CHECK(cudaMemcpy(d_kdtree_nodes, kdTree->tree, node_size, cudaMemcpyHostToDevice));

    size_t offset_size = kdTree->tri_offset_count * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_tri_offsets, offset_size));
    CUDA_CHECK(cudaMemcpy(d_tri_offsets, kdTree->tri_offset_list, offset_size, cudaMemcpyHostToDevice));

    size_t accel_size = h_triangles.size() * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&d_tri_accel, accel_size));
    CUDA_CHECK(cudaMemcpy(d_tri_accel, h_triangles.data(), accel_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_gaussians_ptr, gaussians.size() * sizeof(Gaussian)));
    CUDA_CHECK(cudaMemcpy(d_gaussians_ptr, gaussians.data(), gaussians.size() * sizeof(Gaussian), cudaMemcpyHostToDevice));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] GPU memcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("2. gpu memcpy done\n");

    // 3. 텍스처 바인딩
    CUDA_CHECK(cudaBindTexture(0, &inKdTreeNodeTex, d_kdtree_nodes, &node_desc, node_size));
    CUDA_CHECK(cudaBindTexture(0, &inObjectOffsetListTex, d_tri_offsets, &offset_desc, offset_size));
    CUDA_CHECK(cudaBindTexture(0, &inTriAccelTex, d_tri_accel, &tri_desc, accel_size));
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] texture bind failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("3. texture Bind done\n");

    // 4. 상수 메모리 설정
    SceneInfo h_scene_info = { width, height };
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneInfo, &h_scene_info, sizeof(SceneInfo)));

    CameraInfo h_camera_info;
    h_camera_info.eye = make_float3(camera.pos[0], camera.pos[1], camera.pos[2]);
    h_camera_info.u = make_float3(camera.uaxis[0], camera.uaxis[1], camera.uaxis[2]);
    h_camera_info.v = make_float3(camera.vaxis[0], camera.vaxis[1], camera.vaxis[2]);
    float3 n_axis = make_float3(camera.naxis[0], camera.naxis[1], camera.naxis[2]);

    float fov_rad = camera.fovy * (M_PI / 180.0f);
    float plane_height = 2.0f * camera.near_c * tanf(fov_rad * 0.5f);
    float plane_width = plane_height * camera.aspect;
    h_camera_info.stepX = plane_width / width;
    h_camera_info.stepY = plane_height / height;
    h_camera_info.startPoint = h_camera_info.eye - n_axis * camera.near_c
        - h_camera_info.u * (plane_width * 0.5f)
        + h_camera_info.v * (plane_height * 0.5f);
    CUDA_CHECK(cudaMemcpyToSymbol(g_CameraInfo, &h_camera_info, sizeof(CameraInfo)));

    float3 h_bbox_min = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    float3 h_bbox_max = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMin, &h_bbox_min, sizeof(float3)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMax, &h_bbox_max, sizeof(float3)));

    CUDA_CHECK(cudaMemcpyToSymbol(g_d_gaussians, &d_gaussians_ptr, sizeof(Gaussian*)));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] const memory set failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("4. const memory set done\n");

    // 5. 커널 실행
    float* d_framebuffer;
    size_t framebuffer_size = width * height * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_framebuffer, framebuffer_size));
    CUDA_CHECK(cudaMemset(d_framebuffer, 0, framebuffer_size));

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    size_t shared_mem_size = threads.x * threads.y * SHORT_STACK_DEPTH * sizeof(cu_traceState);

    int* d_maxhit, *hitcount;
    CUDA_CHECK(cudaMalloc((void**)&d_maxhit, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_maxhit, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&hitcount, sizeof(int)));
    CUDA_CHECK(cudaMemset(hitcount, 0, sizeof(int)));

    renderKernelGaussian << < blocks, threads, shared_mem_size >> > (d_framebuffer, d_maxhit, hitcount);
    CUDA_CHECK(cudaGetLastError());        // launch 실패 확인
    CUDA_CHECK(cudaDeviceSynchronize()); // 실행 중 오류 확인

    int h_maxhit = 0, h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_maxhit, d_maxhit, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_count, hitcount, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_maxhit));
    CUDA_CHECK(cudaFree(hitcount));
    //printf("af maxHIT sum, count, avg: %d, %d, %f\n", h_maxhit, h_count, (float)(h_maxhit/h_count));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    //printf("5. kernel launch done\n");

    // 6. 결과 복사 및 메모리 해제
    if (out_framebuffer) delete[] out_framebuffer;
    out_framebuffer = new float[width * height * 3];
    CUDA_CHECK(cudaMemcpy(out_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost));
    is_done = true;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_kdtree_nodes);
    cudaFree(d_tri_offsets);
    cudaFree(d_tri_accel);
    cudaFree(d_gaussians_ptr);
    cudaFree(d_framebuffer);
    cudaUnbindTexture(inKdTreeNodeTex);
    cudaUnbindTexture(inObjectOffsetListTex);
    cudaUnbindTexture(inTriAccelTex);
    //printf("6. free done\n");
    //for fps check
    //cudaEventRecord(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //float frame_time_sec = milliseconds / 1000.0f;
    //float current_fps = 1.0f / frame_time_sec;
    //printf("Frame Time: %.2f ms, FPS: %.2f\n", milliseconds, current_fps);
    //// g_fps = current_fps; // 직접 접근은 불가, Host 함수에서 처리
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] free failed: " << cudaGetErrorString(err) << std::endl;
    }
    //std::cout << "--- Minimal CUDA Renderer Finished ---" << std::endl;
}