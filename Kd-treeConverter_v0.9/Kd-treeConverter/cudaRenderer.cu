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
    __device__ void init() { tHit = FLT_MAX; triIndex = -1; objectIndex = 0; }
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
#define SHORT_STACK_DEPTH 64
typedef struct { unsigned nodeID; float tMax; } cu_traceState;
extern __shared__ cu_traceState smemBuffer[];

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

__device__ bool BoundsRayIntersect(const float3 minB, const float3 maxB, const cuRay& ray, float& tmin, float& tmax) {
    float3 invD = 1.0f / ray.dir;
    float3 t0s = (minB - ray.pos) * invD;
    float3 t1s = (maxB - ray.pos) * invD;
    //float3 tsmaller = fminf(t0s, t1s);
    float3 tsmaller = make_float3(fminf(t0s.x, t1s.x), fminf(t0s.y, t1s.y), fminf(t0s.z, t1s.z));
    //float3 tbigger = fmaxf(t0s, t1s);
    float3 tbigger = make_float3(fmaxf(t0s.x, t1s.x), fmaxf(t0s.y, t1s.y), fmaxf(t0s.z, t1s.z));
    tmin = fmaxf(tmin, fmaxf(tsmaller.x, fmaxf(tsmaller.y, tsmaller.z)));
    tmax = fminf(tmax, fminf(tbigger.x, fminf(tbigger.y, tbigger.z)));
    return (tmin < tmax);
}

// =================================================================================
// 2. 텍스춰 및 상수 메모리 선언
// =================================================================================
texture<uint2, 1, cudaReadModeElementType> inKdTreeNodeTex;
texture<uint, 1, cudaReadModeElementType> inObjectOffsetListTex;
texture<float4, 1, cudaReadModeElementType> inTriAccelTex;

struct SceneInfo { int resX, resY; };
typedef struct _camera_info {
    float3 eye;
    float3 u, v, n;
    float fnear;
    float3 startPoint;		// 왼쪽상단 ray 의 position.
    float stepX, stepY;		// ray 하나당 이동거리.
} CameraInfo;
__constant__ SceneInfo g_SceneInfo;
__constant__ CameraInfo g_CameraInfo;
__constant__ float3 g_SceneBBoxMin;
__constant__ float3 g_SceneBBoxMax;
__constant__ cuObjectMaterial g_materials[1]; // 단일 객체이므로 재질 1개만 사용

// =================================================================================
// 3. CUDA 커널 코드 (사용자 제공 커널)
// =================================================================================

__device__ void getObjectMaterial(int objectIndex, cuObjectMaterial& material) {
    // 상수 메모리에서 유일한 재질을 가져옴
    material = g_materials[0];
}

__device__ float3 calDirectIllumination(cuIntersectionPoint& point, cuObjectMaterial& material) {
    // 간단한 램버트 음영. 광원은 (10,10,-10)에 흰색으로 가정
    float3 lightPos = make_float3(10.0f, 10.0f, -10.0f);
    float3 lightColor = make_float3(1.0f, 1.0f, 1.0f);

    float3 L = normalize(lightPos - point.pos);
    float NdotL = fmaxf(0.0f, dot(point.normal, L));

    return material.diffuse * lightColor * NdotL + material.ambient_emission;
}

__device__ void makeIntersectionPoint(const cuRay* pRay, const cuIntersectionCheck* pHit, cuIntersectionPoint* pPoint) {
    pPoint->pos = pRay->pos + pRay->dir * pHit->tHit;

    // TriAccel로부터 Normal 벡터 가져오기
    float4 N_packed = tex1Dfetch(inTriAccelTex, pHit->triIndex * 4 + 3);
    pPoint->normal = normalize(make_float3(N_packed.x, N_packed.y, N_packed.z));

    // 광선 방향의 반대 방향
    // pPoint->dir = -(pRay->dir);
    pPoint->dir = make_float3(-(pRay->dir.x), -(pRay->dir.y), -(pRay->dir.z));
}

__device__ void singlePassIntersectRoutine(const cuRay& ray, const int id, cuIntersectionCheck& hit, const float t_near, const float t_far) {
    float4 d0 = tex1Dfetch(inTriAccelTex, id * 4 + 0); // n_u, n_v, n_d, k
    float4 d1 = tex1Dfetch(inTriAccelTex, id * 4 + 1); // b_nu, b_nv, b_d, indexInObject
    float4 d2 = tex1Dfetch(inTriAccelTex, id * 4 + 2); // c_nu, c_nv, c_d, material_ID

    unsigned int k = float_as_uint(d0.w);
    float n_u = d0.x, n_v = d0.y, n_d = d0.z;
    float b_nu = d1.x, b_nv = d1.y;
    float c_nu = d2.x, c_nv = d2.y;

    float3 p_pos = ray.pos, p_dir = ray.dir;
    if (k == 1) { // Y-major
        p_pos = make_float3(ray.pos.y, ray.pos.z, ray.pos.x);
        p_dir = make_float3(ray.dir.y, ray.dir.z, ray.dir.x);
    }
    else if (k == 2) { // Z-major
        p_pos = make_float3(ray.pos.z, ray.pos.x, ray.pos.y);
        p_dir = make_float3(ray.dir.z, ray.dir.x, ray.dir.y);
    }

    float denum = p_dir.z + n_u * p_dir.x + n_v * p_dir.y;
    float t = (n_d - (p_pos.z + n_u * p_pos.x + n_v * p_pos.y)) / denum;

    if (t <= hit.tHit && t > t_near && t < t_far) {
        float hu = p_pos.x + t * p_dir.x;
        float hv = p_pos.y + t * p_dir.y;
        float beta = hu * b_nu + hv * b_nv;
        float gamma = hu * c_nu + hv * c_nv;

        if (beta >= -BARYCENTRY_EPSILON && gamma >= -BARYCENTRY_EPSILON && (beta + gamma) <= 1.0f + BARYCENTRY_EPSILON) {
            hit.tHit = t;
            hit.beta = beta;
            hit.gamma = gamma;
            hit.triIndex = id;
        }
    }
}

__device__ void singlePassIntersect(cuRay& currRay, cuIntersectionCheck& intersectionCheck) {
    float t_near = 0.0f, t_far = FLT_MAX;
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, currRay, t_near, t_far)) {
        const unsigned smem_baseOffset = umul24(threadIdx.y, blockDim.x) + threadIdx.x;
        shortStack stack;
        stack.init(smem_baseOffset);

        kdtreeNode node = tex1Dfetch(inKdTreeNodeTex, 0);
        while (true) {
            while (!IS_LEAF(node)) {
                const float t_split = (SPLIT_POS(node) - currRay.get_dir_pos(SPLIT_AXIS(node)).x) / currRay.get_dir_pos(SPLIT_AXIS(node)).y;
                const unsigned sign = signbit(currRay.get_dir_pos(SPLIT_AXIS(node)).y);
                const unsigned childOffset = FIRST_CHILD_OFFSET(node);
                unsigned idx = childOffset + (sign ^ (t_split <= t_near));

                if (t_split > t_near && t_split < t_far) {
                    stack.push(childOffset + (sign ^ 1), t_far);
                    t_far = t_split;
                }
                node = tex1Dfetch(inKdTreeNodeTex, idx);
            }

            unsigned baseOffset = OBJECTLIST_OFFSET(node);
            int objectSize = OBJECT_SIZE(node) + baseOffset;
            for (; baseOffset < objectSize; baseOffset++) {
                const unsigned objListOffset = tex1Dfetch(inObjectOffsetListTex, baseOffset);
                singlePassIntersectRoutine(currRay, objListOffset, intersectionCheck, t_near, t_far);
            }

            if (intersectionCheck.isHit() || stack.empty()) break;

            const cu_traceState& trace = stack.top();
            stack.pop();
            node = tex1Dfetch(inKdTreeNodeTex, trace.nodeID);
            t_near = t_far;
            t_far = trace.tMax;
        }
    }
}

__global__ void singlePassRayTracingKernel_ShadowOff(float* pFrameBuffer, int maxReflectionDepth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    // 1. Primary Ray Generation
    float sx = (float)x + 0.5f;
    float sy = (float)y + 0.5f;
    float3 dir = g_CameraInfo.startPoint + g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
    dir = normalize(dir - g_CameraInfo.eye);

    cuRay currRay;
    currRay.dir = dir;
    currRay.pos = g_CameraInfo.eye;

    cuIntersectionPoint point;
    point.init();

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);

    // 2. Recursive Ray Tracing Loop
    for (int depth = 0; depth < maxReflectionDepth; ++depth) {
        cuIntersectionCheck currIsectCheck;
        currIsectCheck.init();
        singlePassIntersect(currRay, currIsectCheck);

        if (currIsectCheck.isHit()) {
            makeIntersectionPoint(&currRay, &currIsectCheck, &point);
            cuObjectMaterial material;
            getObjectMaterial(currIsectCheck.objectIndex, material);

            finalColor += point.colorWeight * calDirectIllumination(point, material);

            // Reflection
            if (material.reflection > 0.0f) {
                point.colorWeight *= material.reflection;
                currRay.dir = reflection(currRay.dir, point.normal);
                currRay.pos = point.pos + currRay.dir * RAY_START_EPSILON;
            }
            else {
                break; // No more bounces
            }
        }
        else {
            // Background color (e.g., black)
            break;
        }
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

void renderWithCuda(const CompositeObject& object, const Camera& camera, int width, int height, float*& out_framebuffer, bool& is_done) {
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

    printf("0. exist KD-Tree\n");

    printf("[DEBUG] object.n_triangles = %d\n", object.n_triangles);
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
    //std::vector<float4> h_triangles(object.n_triangles * 4);
    float4* h_triangles = (float4*)malloc(sizeof(float4) * (object.n_triangles * 4));
    //printf("Data packing start\n");
    for (int i = 0; i < object.n_triangles; ++i) {
        const TriAccel& src = kdTree->tri_accel_list[i];
        //printf("Data packing %d - 0, src.k: % u\n", i, src.k);
        h_triangles[i * 4 + 0] = make_float4(src.n_u, src.n_v, src.n_d, uint_as_float_H(src.k));
        //printf("Data packing %d - 1, src.indexInObject: %d\n",i);
        h_triangles[i * 4 + 1] = make_float4(src.b_nu, src.b_nv, src.b_d, int_as_float_H(src.indexInObject));
        //printf("Data packing %d - 2, src.material_ID: %d\n", i);
        h_triangles[i * 4 + 2] = make_float4(src.c_nu, src.c_nv, src.c_d, int_as_float_H(src.material_ID));
        //printf("Data packing %d - 3\n", i);
        h_triangles[i * 4 + 3] = make_float4(src.N[0], src.N[1], src.N[2], 0.0f);
        //printf("Data packing %d - 4\n", i);
    }
    printf("1. Data packing done\n");

    // 2. GPU 메모리 할당 및 데이터 전송
    cudaError_t err;
    //cudaArray* d_kdtree_nodes, * d_tri_offsets, * d_tri_accel;
    uint2* d_kdtree_nodes;
    unsigned int* d_tri_offsets;
    float4* d_tri_accel;

    printf("kdtree node count: %d\n", kdTree->tree_node_count);
    cudaChannelFormatDesc node_desc = cudaCreateChannelDesc<uint2>();
    //CUDA_CHECK(cudaMallocArray(&d_kdtree_nodes, &node_desc, kdTree->tree_node_count, 0));
    //CUDA_CHECK(cudaMemcpyToArray(d_kdtree_nodes, 0, 0, kdTree->tree, kdTree->tree_node_count * sizeof(kdtreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_kdtree_nodes, kdTree->tree_node_count * sizeof(uint2)));
    CUDA_CHECK(cudaMemcpy(d_kdtree_nodes, kdTree->tree, kdTree->tree_node_count * sizeof(kdtreeNode), cudaMemcpyHostToDevice));

    cudaChannelFormatDesc offset_desc = cudaCreateChannelDesc<uint>();
    //CUDA_CHECK(cudaMallocArray(&d_tri_offsets, &offset_desc, kdTree->tri_offset_count, 0));
    //CUDA_CHECK(cudaMemcpyToArray(d_tri_offsets, 0, 0, kdTree->tri_offset_list, kdTree->tri_offset_count * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_tri_offsets, kdTree->tri_offset_count * sizeof(uint)));
    CUDA_CHECK(cudaMemcpy(d_tri_offsets, kdTree->tri_offset_list, kdTree->tri_offset_count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    cudaChannelFormatDesc tri_desc = cudaCreateChannelDesc<float4>();
    //CUDA_CHECK(cudaMallocArray(&d_tri_accel, &tri_desc, object.n_triangles * 4, 0));
    //CUDA_CHECK(cudaMemcpyToArray(d_tri_accel, 0, 0, h_triangles.data(), h_triangles.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_tri_accel, object.n_triangles * sizeof(float4)));
    //CUDA_CHECK(cudaMemcpy(d_tri_accel, h_triangles.data(), h_triangles.size() * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri_accel, h_triangles, object.n_triangles * sizeof(float4), cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    printf("2. gpu memcpy done\n");
    
    // 3. 텍스처 바인딩
    //cudaBindTextureToArray(inKdTreeNodeTex, d_kdtree_nodes);
    //cudaBindTextureToArray(inObjectOffsetListTex, d_tri_offsets);
    //cudaBindTextureToArray(inTriAccelTex, d_tri_accel);
    //CUDA_CHECK(cudaBindTextureToArray(&inKdTreeNodeTex, d_kdtree_nodes, &node_desc));
    //CUDA_CHECK(cudaBindTextureToArray(&inObjectOffsetListTex, d_tri_offsets, &offset_desc));
    //CUDA_CHECK(cudaBindTextureToArray(&inTriAccelTex, d_tri_accel, &tri_desc));
    CUDA_CHECK(cudaBindTexture(0, &inKdTreeNodeTex, d_kdtree_nodes, &node_desc, kdTree->tree_node_count * sizeof(kdtreeNode)));
    CUDA_CHECK(cudaBindTexture(0, &inObjectOffsetListTex, d_tri_offsets, &offset_desc, kdTree->tri_offset_count * sizeof(unsigned int)));
    CUDA_CHECK(cudaBindTexture(0, &inTriAccelTex, d_tri_accel, &tri_desc, object.n_triangles * sizeof(float4)));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    printf("3. texture Bind done\n");
    
    // 4. 상수 메모리 설정
    SceneInfo h_scene_info = { width, height };
    CUDA_CHECK(cudaMemcpyToSymbol(&g_SceneInfo, &h_scene_info, sizeof(SceneInfo)));

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
    CUDA_CHECK(cudaMemcpyToSymbol(&g_CameraInfo, &h_camera_info, sizeof(CameraInfo)));

    float3 h_bbox_min = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    float3 h_bbox_max = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    CUDA_CHECK(cudaMemcpyToSymbol(&g_SceneBBoxMin, &h_bbox_min, sizeof(float3)));
    CUDA_CHECK(cudaMemcpyToSymbol(&g_SceneBBoxMax, &h_bbox_max, sizeof(float3)));

    cuObjectMaterial h_material;
    h_material.ambient_emission = make_float3(0.1f, 0.1f, 0.1f);
    h_material.diffuse = make_float3(0.8f, 0.7f, 0.6f);
    h_material.specular = make_float3(0.2f, 0.2f, 0.2f);
    h_material.reflection = 0.05f;
    h_material.transparency = 0.0f;
    h_material.roughness = 32.0f;
    h_material.refractionIndex = 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(g_materials, &h_material, sizeof(cuObjectMaterial)));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    printf("4. const memory set done\n");

    // 5. 커널 실행
    float* d_framebuffer;
    size_t framebuffer_size = width * height * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_framebuffer, framebuffer_size));
    CUDA_CHECK(cudaMemset(d_framebuffer, 0, framebuffer_size));

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    size_t shared_mem_size = threads.x * threads.y * SHORT_STACK_DEPTH * sizeof(cu_traceState);

    singlePassRayTracingKernel_ShadowOff <<< blocks, threads, shared_mem_size >>> (d_framebuffer, 3);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    printf("5. kernel launch done\n");

    // 6. 결과 복사 및 메모리 해제
    if (out_framebuffer) delete[] out_framebuffer;
    out_framebuffer = new float[width * height * 3];
    CUDA_CHECK(cudaMemcpy(out_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost));
    is_done = true;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    free(h_triangles);

    cudaFree(d_framebuffer);
    //cudaFreeArray(d_kdtree_nodes);
    //cudaFreeArray(d_tri_offsets);
    //cudaFreeArray(d_tri_accel);
    cudaFree(d_kdtree_nodes);
    cudaFree(d_tri_offsets);
    cudaFree(d_tri_accel);
    cudaUnbindTexture(inKdTreeNodeTex);
    cudaUnbindTexture(inObjectOffsetListTex);
    cudaUnbindTexture(inTriAccelTex);
    printf("6. free done\n");
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    std::cout << "--- Minimal CUDA Renderer Finished ---" << std::endl;
}