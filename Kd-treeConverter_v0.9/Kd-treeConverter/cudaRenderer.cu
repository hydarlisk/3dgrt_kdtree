#include <GL/glew.h>
#include <GL/freeglut.h>
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

extern std::vector<Gaussian> g_gaussians;
extern KdTreeNode* g_pKdTree_Node_Array;
extern unsigned int* g_pKdTree_TriOffset_Array;
extern unsigned int  g_iKdTree_Node_Count;
extern unsigned int  g_iKdTree_TriOffset_Count;

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

    if (t >= hit.tHit || t <= t_near || t >= t_far) return;

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


__global__ void renderKernel(float* pFrameBuffer, int* maxhit) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    float sx = (float)x + 0.5f, sy = (float)y + 0.5f;
    float3 dir = g_CameraInfo.startPoint + g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;

    cuRay ray = { g_CameraInfo.eye, normalize(dir - g_CameraInfo.eye) };
    cuIntersectionCheck hit;
    hit.tHit = FLT_MAX; hit.triIndex = -1;

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
        float red_factor = fminf(1.0f, (float)hit.hitCount / 1127192737.f);
        finalColor += make_float3(red_factor * 0.8f, 0.0f, 0.0f);
    }

    int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
    pFrameBuffer[idx + 0] = finalColor.x;
    pFrameBuffer[idx + 1] = finalColor.y;
    pFrameBuffer[idx + 2] = finalColor.z;
}

// cudaRenderer.cu

// GPU에 복사된 원본 가우시안 데이터에 접근하기 위한 전역 포인터
__device__ FullPLYVertex* d_gaussians_ptr;

// find_next_hit: min_t 이후의 가장 가까운 충돌을 찾는 Kd-tree 순회 함수
// (기존 singlePassIntersect 로직을 수정하여 구현)
__device__ cuIntersectionCheck find_next_hit(const cuRay& ray, float min_t) {
    // ... Kd-tree 순회 로직 ...
    // 삼각형 교차 판정 시:
    // if (t > min_t && t < hit.tHit) { ... } // 이 조건으로 수정
    // ...
}

__global__ void renderKernelGaussian(float* pFrameBuffer/*, ... d_gaussians_ptr를 상수로 복사했다면 인자 불필요 */) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    // 1. 레이 생성 및 누적 변수 초기화
    cuRay ray = { /* ... 카메라 정보로 레이 생성 ... */ };
    float3 accum_color = make_float3(0.0f, 0.0f, 0.0f);
    float  accum_alpha = 0.0f;
    float  min_t_for_next_search = 0.0f; // 광선의 시작점

    // 2. 누적 렌더링 루프
    const int MAX_STEPS = 64; // 최대 충돌 횟수 제한
    for (int step = 0; step < MAX_STEPS && accum_alpha < 0.99f; ++step) {

        // 3. 다음으로 가장 가까운 20면체 삼각형 찾기
        cuIntersectionCheck hit = find_next_hit(ray, min_t_for_next_search);

        if (!hit.isHit()) { // 더 이상 충돌하는 삼각형이 없으면 종료
            break;
        }

        // 4. 충돌한 삼각형의 material_ID로부터 원본 가우시안 인덱스 가져오기
        // TriAccel 텍스처에서 material_ID를 가져오는 방법 필요 (기존 코드 참조)
        float4 d2 = tex1Dfetch(inTriAccelTex, hit.triIndex * 4 + 2);
        int gaussian_index = __float_as_int(d2.w);

        // 5. 원본 가우시안 데이터 가져오기
        FullPLYVertex g = d_gaussians_ptr[gaussian_index];

        // 6. 기여도 계산 (가장 간단한 버전)
        // (f_dc는 0~1 정규화 필요할 수 있음)
        float3 contrib_color = make_float3(g.f_dc[0], g.f_dc[1], g.f_dc[2]);
        float  contrib_alpha = g.opacity;

        // 7. Front-to-back "Over" 연산자로 블렌딩
        accum_color += contrib_color * contrib_alpha * (1.0f - accum_alpha);
        accum_alpha += contrib_alpha * (1.0f - accum_alpha);

        // 8. 다음 탐색을 위해 시작점 업데이트
        min_t_for_next_search = hit.tHit;
    }

    // 10. 배경색과 최종 혼합하여 프레임버퍼에 쓰기
    float3 background_color = make_float3(0.2f, 0.3f, 0.4f);
    float3 final_color = accum_color + background_color * (1.0f - accum_alpha);

    int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
    pFrameBuffer[idx + 0] = final_color.x;
    pFrameBuffer[idx + 1] = final_color.y;
    pFrameBuffer[idx + 2] = final_color.z;
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
    std::cout << "--- Minimal CUDA Renderer Started ---" << std::endl;

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
    renderKernel <<< blocks, threads, shared_mem_size >>> (d_framebuffer, d_maxhit);
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
    std::cout << "--- Minimal CUDA Renderer Finished ---" << std::endl;
}

//-----------------------------------------------------------------------
// Gaussian Render
//-----------------------------------------------------------------------
// 순회 결과를 담을 간단한 구조체
struct TraversalResult {
    unsigned int candidate_gaussians[64]; // 후보 가우시안 인덱스를 담을 배열
    int count;                            // 찾은 후보의 개수
};


// 가우시안 Kd-tree를 순회하여 광선 경로상의 가우시안 인덱스를 수집하는 함수
__device__ void traverseGaussianKdTree(const cuRay& ray, TraversalResult& result, const KdTreeNode* nodes, const unsigned int* offsets) {
    result.count = 0;

    float t_near = 0.0f, t_far = FLT_MAX;
    if (!BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, ray, t_near, t_far)) {
        return; // 광선이 씬 바운딩 박스를 통과하지 않으면 바로 종료
    }

    shortStack stack;
    stack.init(threadIdx.x + threadIdx.y * blockDim.x);

    const KdTreeNode* currentNode = &nodes[0]; // 루트 노드에서 시작

    while (true) {
        // 리프 노드를 만날 때까지 트리를 타고 내려감
        while (!IS_LEAF(*currentNode)) {
            unsigned axis = SPLIT_AXIS(*currentNode);
            float split_pos = SPLIT_POS(*currentNode);
            float dir_axis = (&(ray.dir.x))[axis];
            float pos_axis = (&(ray.pos.x))[axis];

            // ... (이 부분은 기존 singlePassIntersect의 순회 로직과 거의 동일) ...
            float t_split = (split_pos - pos_axis) / dir_axis;
            unsigned near_child_offset = FIRST_CHILD_OFFSET(*currentNode) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 0 : 1);
            unsigned far_child_offset = FIRST_CHILD_OFFSET(*currentNode) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 1 : 0);

            if (t_split > t_far || t_split < 0) {
                currentNode = &nodes[near_child_offset];
            }
            else if (t_split < t_near) {
                currentNode = &nodes[far_child_offset];
            }
            else {
                stack.push(far_child_offset, t_far);
                currentNode = &nodes[near_child_offset];
                t_far = t_split;
            }
        }

        // 리프 노드에 도달하면, 포함된 가우시안 인덱스들을 결과 배열에 추가
        unsigned int offset = OBJECTLIST_OFFSET(*currentNode);
        unsigned int count = OBJECT_SIZE(*currentNode);
        for (int i = 0; i < count; ++i) {
            if (result.count < 64) { // 배열 오버플로우 방지
                result.candidate_gaussians[result.count++] = offsets[offset + i];
            }
        }

        // 스택이 비었으면 순회 종료
        if (stack.empty()) break;

        // 스택에서 다음 노드를 가져와 계속 순회
        cu_traceState next = stack.top(); stack.pop();
        t_near = t_far;
        t_far = next.tMax;
        currentNode = &nodes[next.nodeID];
    }
}


// 새로운 메인 커널
__global__ void gaussianRayTraceKernel(float* pFrameBuffer, const KdTreeNode* kdtree_nodes, const unsigned int* kdtree_offsets, const Gaussian* gaussians) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    // 1. 광선 생성 (기존과 동일)
    float sx = (float)x + 0.5f, sy = (float)y + 0.5f;
    float3 dir = g_CameraInfo.startPoint + g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
    cuRay ray = { g_CameraInfo.eye, normalize(dir - g_CameraInfo.eye) };

    // 2. Kd-tree 순회하여 후보 가우시안 수집
    TraversalResult result;
    traverseGaussianKdTree(ray, result, kdtree_nodes, kdtree_offsets);

    // 3. 디버깅: 찾은 후보 가우시안의 '개수'를 명암으로 시각화
    float intensity = fminf(1.0f, (float)result.count / 20.0f); // 20개 이상이면 흰색
    float3 finalColor = make_float3(intensity, intensity, intensity);

    // 4. 프레임버퍼에 쓰기
    int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
    pFrameBuffer[idx + 0] = finalColor.x;
    pFrameBuffer[idx + 1] = finalColor.y;
    pFrameBuffer[idx + 2] = finalColor.z;
}

void renderGaussiansWithCuda(const CompositeObject& object, const Camera& camera, int width, int height, float*& out_framebuffer, bool& is_done) {
    is_done = false;
    if (g_gaussians.empty() || g_pKdTree_Node_Array == nullptr) {
        std::cerr << "[CUDA Error] Gaussian data or Kd-tree not ready." << std::endl;
        return;
    }

    std::cout << "--- Gaussian Ray Tracing Started ---" << std::endl;

    // 1. GPU 메모리 할당
    Gaussian* d_gaussians;
    KdTreeNode* d_kdtree_nodes;
    unsigned int* d_kdtree_offsets;
    float* d_framebuffer;

    CUDA_CHECK(cudaMalloc(&d_gaussians, g_gaussians.size() * sizeof(Gaussian)));
    CUDA_CHECK(cudaMalloc(&d_kdtree_nodes, g_iKdTree_Node_Count * sizeof(KdTreeNode)));
    CUDA_CHECK(cudaMalloc(&d_kdtree_offsets, g_iKdTree_TriOffset_Count * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_framebuffer, width * height * 3 * sizeof(float)));

    // 2. 데이터 복사: Host -> Device
    CUDA_CHECK(cudaMemcpy(d_gaussians, g_gaussians.data(), g_gaussians.size() * sizeof(Gaussian), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kdtree_nodes, g_pKdTree_Node_Array, g_iKdTree_Node_Count * sizeof(KdTreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kdtree_offsets, g_pKdTree_TriOffset_Array, g_iKdTree_TriOffset_Count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // 3. 상수 메모리 설정 (카메라, 씬 정보 등)
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

    // CPU에서 계산한 씬 전체의 AABB를 GPU 상수 메모리로 복사합니다.
    float3 h_bbox_min = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    float3 h_bbox_max = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMin, &h_bbox_min, sizeof(float3)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMax, &h_bbox_max, sizeof(float3)));

    // 4. 새로운 커널 실행
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    gaussianRayTraceKernel << < blocks, threads >> > (
        d_framebuffer,
        d_kdtree_nodes,
        d_kdtree_offsets,
        d_gaussians
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. 결과 복사: Device -> Host
    if (out_framebuffer) delete[] out_framebuffer;
    out_framebuffer = new float[width * height * 3];
    CUDA_CHECK(cudaMemcpy(out_framebuffer, d_framebuffer, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    is_done = true;

    // 6. GPU 메모리 해제
    cudaFree(d_gaussians);
    cudaFree(d_kdtree_nodes);
    cudaFree(d_kdtree_offsets);
    cudaFree(d_framebuffer);

    std::cout << "--- Gaussian Ray Tracing Finished ---" << std::endl;
}
/*
size_t find_binary_start_offset(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return 0;

    char c;
    int state = 0;
    size_t offset = 0;

    while (fread(&c, 1, 1, fp) == 1) {
        offset++;
        if (c == '\n') {
            long prev = ftell(fp);
            char line[64] = { 0 };
            fgets(line, sizeof(line), fp);
            if (strncmp(line, "end_header", 10) == 0) {
                offset = ftell(fp);
                break;
            }
            fseek(fp, prev, SEEK_SET);
        }
    }

    fclose(fp);
    return offset;
}

int read_ply_and_upload_gaussians(const char* ply_filename, GPUParticle*& d_particles, int& n_particles) {
    static_assert(sizeof(FullPLYVertex) == 248, "FullPLYVertex size must be 248 bytes");
    printf("[DEBUG] sizeof(FullPLYVertex) = %zu\n", sizeof(FullPLYVertex));

    FILE* fp_txt = fopen(ply_filename, "r");
    if (!fp_txt) {
        fprintf(stderr, "[PLY] Cannot open file: %s\n", ply_filename);
        return 0;
    }

    int num_vertices = 0;
    char line[512];
    while (fgets(line, sizeof(line), fp_txt)) {
        if (strncmp(line, "element vertex", 14) == 0)
            sscanf(line, "element vertex %d", &num_vertices);
        else if (strncmp(line, "end_header", 10) == 0)
            break;
        }
    fclose(fp_txt);

    size_t header_end_offset = find_binary_start_offset(ply_filename);
    if (num_vertices == 0 || header_end_offset == 0) {
        fprintf(stderr, "[PLY] Invalid header or end_header not found.\n");
        return 0;
    }

    FILE* fp = fopen(ply_filename, "rb");
    if (!fp) return 0;
    fseek(fp, (long)header_end_offset, SEEK_SET);

    std::vector<FullPLYVertex> full_vertices(num_vertices);
    size_t read_count = fread(full_vertices.data(), sizeof(FullPLYVertex), num_vertices, fp);
    fclose(fp);

    if (read_count != (size_t)num_vertices) {
        fprintf(stderr, "[PLY] Failed to read all vertex data (%zu/%d)\n", read_count, num_vertices);
        return 0;
    }

    // Transform to GPUParticle
    std::vector<GPUParticle> host_particles(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        const auto& v = full_vertices[i];
        GPUParticle& p = host_particles[i];

        p.position = make_float3(v.x, v.y, v.z);
        p.scale = make_float3(v.scale[0], v.scale[1], v.scale[2]);
        p.rotation = make_float4(v.rot[0], v.rot[1], v.rot[2], v.rot[3]);
        p.color = make_float3(v.f_dc[0], v.f_dc[1], v.f_dc[2]);
        p.opacity = v.opacity;
    }

    // Upload to CUDA
    cudaMalloc(&d_particles, sizeof(GPUParticle) * num_vertices);
    cudaMemcpy(d_particles, host_particles.data(), sizeof(GPUParticle) * num_vertices, cudaMemcpyHostToDevice);

    n_particles = num_vertices;
    printf("[PLY] Successfully loaded %d particles to GPU.\n", n_particles);
    return 1;
}
*/