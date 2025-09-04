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
//#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
//#include <texture_fetch_functions.h>
//#include <texture_indirect_functions.h>
#include <vector_types.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// =================================================================================
// CUDA 커널 및 디바이스 헬퍼 함수/구조체 (SGRT 파일들에서 필요한 부분만 추출)
// =================================================================================
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
//#define IS_LEAF(node)               (((node).x & 3) == 3)
//#define SPLIT_AXIS(node)            ( (node).x & 3)
//#define FIRST_CHILD_OFFSET(node)    ( (node).x >> 3)
//#define SPLIT_POS(node)             (*(float *)&((node).y))
//#define OBJECT_SIZE(node)           ( (node).x >> 3)
//#define OBJECTLIST_OFFSET(node)     ( (node).y)

// 스택 (cudaRenderPipelineCommonKernel.cu에서 추출)
extern __shared__ cu_traceState smemBuffer[SHORT_STACK_DEPTH * DIM_X * DIM_Y];

struct shortStack {
    unsigned _top, quant, baseOffset;
    __device__ shortStack() : _top(SHORT_STACK_DEPTH - 1), quant(0) {}
    __device__ inline void init(const unsigned smem_baseOffset) {
        baseOffset = smem_baseOffset * SHORT_STACK_DEPTH;
        _top = SHORT_STACK_DEPTH - 1;
        quant = 0;
    }
    __device__ inline cu_traceState top() { return smemBuffer[baseOffset + _top]; }
    __device__ inline void push(unsigned id, float t_max) {
        if (++_top == SHORT_STACK_DEPTH) _top = 0;
        //quant = min(quant + 1, SHORT_STACK_DEPTH);
        quant = (quant + 1) % SHORT_STACK_DEPTH;
        smemBuffer[baseOffset + _top].nodeID = id;
        smemBuffer[baseOffset + _top].tMax = t_max;
    }
    __device__ inline bool empty() { return quant == 0; }
    __device__ inline int full() { return quant == SHORT_STACK_DEPTH; }
    __device__ inline void pop() {
        if (_top == 0) _top = SHORT_STACK_DEPTH;
        --_top; --quant;
    }
};
// LIFO 캐시 구조체
struct ShortStackCache {
    unsigned int head;       // 가장 오래된 데이터의 위치 (다음에 밀려날 대상)
    unsigned int tail;       // 다음에 데이터를 쓸 위치
    unsigned int count;      // 현재 캐시 안의 데이터 개수
    unsigned int baseOffset;

    __device__ void init(const unsigned int smem_baseOffset) {
        baseOffset = smem_baseOffset * SHORT_STACK_DEPTH;
        head = 0;
        tail = 0;
        count = 0;
    }

    __device__ bool is_empty() const { return count == 0; }
    __device__ bool is_full() const { return count >= SHORT_STACK_DEPTH; }

    // 데이터를 캐시에 PUSH하는 함수. 캐시가 꽉 찼으면 밀려나는 데이터를 반환.
    __device__ cu_traceState push(const cu_traceState& item) {
        cu_traceState evicted_item = {}; // 기본값으로 초기화

        if (is_full()) {
            // 가장 오래된 데이터를 evicted_item에 저장
            evicted_item = smemBuffer[baseOffset + head];
            // head 포인터를 다음으로 이동 (순환)
            head = (head + 1) % SHORT_STACK_DEPTH;
        }
        else {
            count++;
        }

        // tail 위치에 새로운 아이템을 쓰고 tail 포인터를 다음으로 이동 (순환)
        smemBuffer[baseOffset + tail] = item;
        tail = (tail + 1) % SHORT_STACK_DEPTH;

        return evicted_item;
    }

    // 캐시에서 데이터를 POP하는 함수 (LIFO: 가장 나중에 들어온 것부터)
    __device__ cu_traceState pop() {
        // tail 포인터를 뒤로 돌려 가장 마지막에 쓴 데이터 위치로 이동 (순환)
        tail = (tail == 0) ? SHORT_STACK_DEPTH - 1 : tail - 1;
        count--;
        return smemBuffer[baseOffset + tail];
    }
};

//// --- Device-side Helper Functions ---
//
//__device__ float3 reflection(float3 I, float3 N) {
//    return I - 2.0f * N * dot(I, N);
//}
//
//__device__ float3 refraction(float3 I, float3 N, float eta) {
//    float dotNI = dot(N, I);
//    float k = 1.0f - eta * eta * (1.0f - dotNI * dotNI);
//    if (k < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
//    return eta * I - (eta * dotNI + sqrtf(k)) * N;
//}

// =================================================================================
// 텍스춰 및 상수 메모리 선언
// =================================================================================
//texture<uint2, 1, cudaReadModeElementType> inKdTreeNodeTex;
//texture<uint, 1, cudaReadModeElementType> inObjectOffsetListTex;
//texture<float4, 1, cudaReadModeElementType> inTriAccelTex;
//texture<float4, 1, cudaReadModeElementType> inGaussianTex;

__device__ cudaTextureObject_t g_d_kdtree_tex = 0;
__device__ cudaTextureObject_t g_d_offsets_tex = 0;
__device__ cudaTextureObject_t g_d_triaccel_tex = 0;
__device__ cudaTextureObject_t g_d_gaussian_tex = 0;

struct SceneInfo { int resX, resY; };
//struct SceneInfo {
//    int resX, resY;
//    int num_gaussians; // 전체 가우시안 개수를 저장할 변수 추가
//};
struct CameraInfo { float3 eye, u, v, startPoint; float stepX, stepY; };

__constant__ SceneInfo g_SceneInfo;
__constant__ CameraInfo g_CameraInfo;
__constant__ float3 g_SceneBBoxMin;
__constant__ float3 g_SceneBBoxMax;
//__device__ Gaussian* g_d_gaussians;
//__device__ ExtendedVertex* g_d_all_vertices;

//__device__ kdtreeNode* g_kdtree_nodes = nullptr;
//__device__ unsigned int* g_tri_offsets = nullptr;
//__device__ float4* g_tri_accel = nullptr;
//__device__ Gaussian* g_gaussians = nullptr;

kdtreeNode* g_d_kdtree_nodes = nullptr;
unsigned int* g_d_tri_offsets = nullptr;
TriAccel* g_d_tri_accel = nullptr;
Gaussian* g_d_gaussians_persistent = nullptr;

cudaEvent_t start_ev, stop_ev;
float k_fps = 0.0f; // FPS를 저장할 전역 변수
int frame_count = 0;
float total_frame = 0.0f;

// =================================================================================
// CUDA 커널 코드 (사용자 제공 커널)
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

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("(%d, %d) BoundsRayIntersect Done\n",x,y);
    return ((tmin < tmax) & (tmax >= 0.f));
}

// =================================================================================
// Host-Side Public Render Function
// =================================================================================

bool initCuda() {
    CUDA_CHECK(cudaGetLastError());

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    printf("CUDA devices: %d\n", deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "[CUDA Init] No CUDA devices found." << std::endl;
        return false;
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Init] Failed to set device 0." << cudaGetErrorString(err) << std::endl;
        return false;
    }
    std::cout << "[CUDA Init] CUDA device initialized successfully." << std::endl;
    return true;
}

//-----------------------------------------------------------------------
// Gaussian Render
//-----------------------------------------------------------------------
struct HitRecord {
    float t;
    int triIndex;
};

__device__ void sortHits(HitRecord* hits, int count) {
    //if (count > 25) printf("local sort count: %d\n", count);
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

__device__ void shellSort(HitRecord* hits, int count) {
    if(count>25) printf("local sort count: %d\n", count);
    for (int gap = count / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < count; i += 1) {
            HitRecord temp = hits[i];
            int j;
            for (j = i; j >= gap && hits[j - gap].t > temp.t; j -= gap) {
                hits[j] = hits[j - gap];
            }
            hits[j] = temp;
        }
    }
}

__device__ __forceinline__ float3 sigmoid(const float3& v) {
    return make_float3(
        1.0f / (1.0f + expf(-v.x)),
        1.0f / (1.0f + expf(-v.y)),
        1.0f / (1.0f + expf(-v.z))
    );
}
/**
 * @brief 3dgrt 원본 코드(radianceFromSpH)의 로직을 그대로 구현한 구면 조화 함수(SH) 평가 함수
 * @param degree 계산할 SH 차수 (최대 3)
 * @param view_dir 뷰 방향 벡터 (정규화 필요)
 * @param g 가우시안 데이터
 * @return 최종 계산된 색상 (0~1 범위로 클램핑됨)
 */
__device__ __forceinline__ float3 eval_sh_final(
    const int degree,
    const float3& view_dir,
    const Gaussian& g
) {
    // 1. 계산을 용이하게 하기 위해 g.f_dc와 g.f_rest를 하나의 배열로 합칩니다.
    float3 sphCoefficients[16];
    sphCoefficients[0] = make_float3(g.f_dc[0], g.f_dc[1], g.f_dc[2]);
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        sphCoefficients[i + 1] = make_float3(g.f_rest[i * 3 + 0], g.f_rest[i * 3 + 1], g.f_rest[i * 3 + 2]);
    }
    //sphCoefficients[1] = make_float3(g.f_rest[3], g.f_rest[4], g.f_rest[5]);
    //sphCoefficients[2] = make_float3(g.f_rest[6], g.f_rest[7], g.f_rest[8]);
    //sphCoefficients[3] = make_float3(g.f_rest[9], g.f_rest[10], g.f_rest[11]);
    //sphCoefficients[4] = make_float3(g.f_rest[12], g.f_rest[13], g.f_rest[14]);
    //sphCoefficients[5] = make_float3(g.f_rest[15], g.f_rest[16], g.f_rest[17]);
    //sphCoefficients[6] = make_float3(g.f_rest[18], g.f_rest[19], g.f_rest[20]);
    //sphCoefficients[7] = make_float3(g.f_rest[21], g.f_rest[22], g.f_rest[23]);
    //sphCoefficients[8] = make_float3(g.f_rest[24], g.f_rest[25], g.f_rest[26]);
    //sphCoefficients[9] = make_float3(g.f_rest[27], g.f_rest[28], g.f_rest[29]);
    //sphCoefficients[10] = make_float3(g.f_rest[30], g.f_rest[31], g.f_rest[32]);
    //sphCoefficients[11] = make_float3(g.f_rest[33], g.f_rest[34], g.f_rest[35]);
    //sphCoefficients[12] = make_float3(g.f_rest[36], g.f_rest[37], g.f_rest[38]);
    //sphCoefficients[13] = make_float3(g.f_rest[39], g.f_rest[40], g.f_rest[41]);
    //sphCoefficients[14] = make_float3(g.f_rest[42], g.f_rest[43], g.f_rest[44]);

    // --- 2. 3dgrt의 radianceFromSpH 로직을 그대로 적용 ---
    float3 rad = SH_C0 * sphCoefficients[0]; // 0차 SH

    if (degree > 0) {
        const float x = view_dir.x;
        const float y = view_dir.y;
        const float z = view_dir.z;

        // 1차 SH
        rad = rad - SH_C1 * y * sphCoefficients[1]
            + SH_C1 * z * sphCoefficients[2]
            - SH_C1 * x * sphCoefficients[3];

        if (degree > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;

            // 2차 SH
            rad = rad + SH_C2_0 * xy * sphCoefficients[4]
                + SH_C2_1 * yz * sphCoefficients[5]
                + SH_C2_2 * (2.0f * zz - xx - yy) * sphCoefficients[6]
                + SH_C2_3 * xz * sphCoefficients[7]
                + SH_C2_4 * (xx - yy) * sphCoefficients[8];

            if (degree > 2) {
                // 3차 SH
                rad = rad + SH_C3_0 * y * (3.0f * xx - yy) * sphCoefficients[9]
                    + SH_C3_1 * xy * z * sphCoefficients[10]
                    + SH_C3_2 * y * (4.0f * zz - xx - yy) * sphCoefficients[11]
                    + SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sphCoefficients[12]
                    + SH_C3_4 * x * (4.0f * zz - xx - yy) * sphCoefficients[13]
                    + SH_C3_5 * z * (xx - yy) * sphCoefficients[14]
                    + SH_C3_6 * x * (xx - 3.0f * yy) * sphCoefficients[15];
            }
        }
    }

    // 3. 최종 활성화: 원본과 동일하게 0.5를 더하고, 0 미만 값은 0으로 클램핑합니다.
    rad += make_float3(0.5f);
    return max(rad, make_float3(0.0f));
    //return sigmoid(rad);
}

// 쿼터니언의 역(conjugate)을 계산합니다.
__device__ __forceinline__ float4 quat_inverse(const float4& q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

// 쿼터니언을 사용하여 벡터를 회전시킵니다.
__device__ __forceinline__ float3 quat_rotate(const float3& v, const float4& q) {
    float3 t = 2.0f * cross(make_float3(q.x, q.y, q.z), v);
    return v + q.w * t + cross(make_float3(q.x, q.y, q.z), t);
}

/**
 * @brief 광선과 3D 가우시안의 상호작용을 평가하여 샘플의 투명도를 계산합니다.
 * @param ray 현재 추적 중인 광선.
 * @param g 평가할 가우시안 데이터.
 * @return 계산된 최종 샘플 투명도 (alpha).
 */
__device__ __forceinline__ float evaluateGaussianResponse(const cuRay& ray, const Gaussian& g)
{
    // 파라미터 정리
    const float3 g_pos = make_float3(g.pos[0], g.pos[1], g.pos[2]);   // μ
    const float3 g_scale = make_float3(g.scale[0], g.scale[1], g.scale[2]); // 대각 S (표준편차)

    const float3 p = ray.pos - g_pos; // (o - μ)

    // 위치 벡터 회전: o_os = R^T * p
    float3 o_os;
    o_os.x = g.rot_matrix.m[0][0] * p.x + g.rot_matrix.m[0][1] * p.y + g.rot_matrix.m[0][2] * p.z;
    o_os.y = g.rot_matrix.m[1][0] * p.x + g.rot_matrix.m[1][1] * p.y + g.rot_matrix.m[1][2] * p.z;
    o_os.z = g.rot_matrix.m[2][0] * p.x + g.rot_matrix.m[2][1] * p.y + g.rot_matrix.m[2][2] * p.z;

    // 방향 벡터 회전: d_os = R^T * d
    float3 d_os;
    d_os.x = g.rot_matrix.m[0][0] * ray.dir.x + g.rot_matrix.m[0][1] * ray.dir.y + g.rot_matrix.m[0][2] * ray.dir.z;
    d_os.y = g.rot_matrix.m[1][0] * ray.dir.x + g.rot_matrix.m[1][1] * ray.dir.y + g.rot_matrix.m[1][2] * ray.dir.z;
    d_os.z = g.rot_matrix.m[2][0] * ray.dir.x + g.rot_matrix.m[2][1] * ray.dir.y + g.rot_matrix.m[2][2] * ray.dir.z;

    //const float4 g_rot = make_float4(g.rot[1], g.rot[2], g.rot[3], g.rot[0]); // (x,y,z,w)
    //const float4 inv_rot = quat_inverse(g_rot); // R^T

    // 월드 → 가우시안 정렬좌표로 회전(R^T)한 뒤, 스케일의 역수(S^{-1})를 적용
    // o_g = S^{-1} R^T (o - μ),  d_g = S^{-1} R^T d
    //float3 o_os = quat_rotate(ray.pos - g_pos, inv_rot);                                  //R^T (o-μ)
    //float3 d_os = quat_rotate(ray.dir, inv_rot);                                          //R^T d
    float3 o_g = make_float3(o_os.x / g_scale.x, o_os.y / g_scale.y, o_os.z / g_scale.z); //S^-1 R^T (o-μ)
    float3 d_g = make_float3(d_os.x / g_scale.x, d_os.y / g_scale.y, d_os.z / g_scale.z); //S^-1 R^T d

    // τ_max = - (o_g·d_g) / (d_g·d_g)  (식 8)
    float denom = dot(d_g, d_g);
    // 안전장치: 방향이 너무 작으면 밀도는 원점에서 평가
    float tau = (denom > 1e-8f) ? (-dot(o_g, d_g) / denom) : 0.0f;

    // τ_max 위치의 가우시안 밀도: ρ = exp(-0.5 * ||o_g + τ_max d_g||^2)
    //float3 p_g = ray.pos + tau * ray.dir;
    float3 p_g = o_g + tau * d_g;
    float  expo = -0.5f * dot(p_g, p_g);
    //float  expo = -1.f * dot(p_g, p_g);
    float  rho = expf(expo);

    return g.opacity * rho;
}

/**
 * @brief grayDist 값을 기반으로 밀도 감쇠(density falloff)를 계산합니다.
 * 표준 2차(Quadratic) 가우시안 분포를 사용합니다.
 * @param grayDist 제곱된 마할라노비스 거리
 * @return 밀도 응답 값 (0.0 ~ 1.0)
 */
template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponse(float grayDist) {
    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        constexpr float s = -0.000685871056241f;
        const float grayDistSq = grayDist * grayDist;
        return expf(s * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        constexpr float s = -0.0185185185185f;
        return expf(s * grayDist * grayDist * sqrtf(grayDist));
    }
    case 4: // Tesseractic
    {
        constexpr float s = -0.0555555555556f;
        return expf(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        constexpr float s = -0.166666666667f;
        return expf(s * grayDist * sqrtf(grayDist));
    }
    case 1: // Laplacian
    {
        constexpr float s = -1.5f;
        return expf(s * sqrtf(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487f;
        return fmaxf(1.f + s * sqrtf(grayDist), 0.f);
    }
    default: // Quadratic
    {
        constexpr float s = -0.5f;
        return expf(s * grayDist);
    }
    }
}

/**
 * @brief 3dgrt 원본 코드의 cross product 방식을 사용하여
 * 광선과 가우시안의 최대 응답을 계산하고, 최종 샘플 불투명도를 반환합니다.
 * @param ray 렌더링에 사용되는 원본 광선
 * @param g 평가 대상 가우시안
 * @return 계산된 최종 샘플 불투명도
 */
__device__ __forceinline__ float evaluateGaussianResponse_3dgrt(const cuRay& ray, const Gaussian& g)
{
    // 가우시안 파라미터 준비
    const float3 g_pos = make_float3(g.pos[0], g.pos[1], g.pos[2]);
    const float3 g_scale = make_float3(g.scale[0], g.scale[1], g.scale[2]);
    //const float4 g_rot = make_float4(g.rot[1], g.rot[2], g.rot[3], g.rot[0]);

    // 광선을 가우시안의 로컬 좌표계로 변환 (회전 및 스케일링)
    //const float4 inv_rot = quat_inverse(g_rot);
    const float3 gposc = ray.pos - g_pos; // (o - μ)
    //const float3 gposcr = quat_rotate(gposc, inv_rot);

    float3 gposcr; // R^T * (o - μ)
    gposcr.x = g.rot_matrix.m[0][0] * gposc.x + g.rot_matrix.m[0][1] * gposc.y + g.rot_matrix.m[0][2] * gposc.z;
    gposcr.y = g.rot_matrix.m[1][0] * gposc.x + g.rot_matrix.m[1][1] * gposc.y + g.rot_matrix.m[1][2] * gposc.z;
    gposcr.z = g.rot_matrix.m[2][0] * gposc.x + g.rot_matrix.m[2][1] * gposc.y + g.rot_matrix.m[2][2] * gposc.z;

    float3 rayDirR; // R^T * d
    rayDirR.x = g.rot_matrix.m[0][0] * ray.dir.x + g.rot_matrix.m[0][1] * ray.dir.y + g.rot_matrix.m[0][2] * ray.dir.z;
    rayDirR.y = g.rot_matrix.m[1][0] * ray.dir.x + g.rot_matrix.m[1][1] * ray.dir.y + g.rot_matrix.m[1][2] * ray.dir.z;
    rayDirR.z = g.rot_matrix.m[2][0] * ray.dir.x + g.rot_matrix.m[2][1] * ray.dir.y + g.rot_matrix.m[2][2] * ray.dir.z;

    const float3 gro = gposcr / g_scale;

    //const float3 rayDirR = quat_rotate(ray.dir, inv_rot);
    const float3 grdu = rayDirR / g_scale;
    const float3 grd = normalize(grdu);

    // cross product를 이용해 grayDist(제곱된 마할라노비스 거리) 계산
    const float3 gcrod = cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    // particleResponse 함수를 통해 밀도 계산
    const float density = particleResponse<4>(grayDist);

    // 기본 불투명도와 밀도를 곱하여 최종 결과 반환
    return g.opacity * density;
}

// Gaussian 데이터를 텍스처에서 읽어오는 헬퍼 함수
//__device__ Gaussian fetch_gaussian(cudaTextureObject_t gaussianTex, int gaussianID) {
__device__ Gaussian fetch_gaussian(int gaussianID) {
    Gaussian g;
    // 패딩이 추가된 Gaussian 크기는 240바이트 = float4(16바이트) * 15개
    const int num_float4s = sizeof(Gaussian) / sizeof(float4);
    int base_idx = gaussianID * num_float4s;

    // 텍스처에서 float4 단위로 15번 데이터를 가져옵니다.
    float4 data[num_float4s];
#pragma unroll
    for (int i = 0; i < num_float4s; ++i) {
        //data[i] = tex1Dfetch(inGaussianTex, base_idx + i);
        data[i] = tex1D<float4>(g_d_gaussian_tex, base_idx + i);
    }

    // 가져온 데이터를 Gaussian 구조체로 복사
    memcpy(&g, data, sizeof(Gaussian));
    return g;
}

__device__ void singlePassIntersectRoutineGaussian_sortNode(const cuRay& ray, int id, float t_near, float t_far, HitRecord* local_hits, int& local_hit_count) {
    if (local_hit_count >= MAX_HITS) return;

    float4 d0 = tex1D<float4>(g_d_triaccel_tex, id * 4 + 0);
    //float4 d0 = g_tri_accel[id * 4 + 0];
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

    float4 d1 = tex1D<float4>(g_d_triaccel_tex, id * 4 + 1);
    float4 d2 = tex1D<float4>(g_d_triaccel_tex, id * 4 + 2);
    //float4 d1 = g_tri_accel[id * 4 + 1];
    //float4 d2 = g_tri_accel[id * 4 + 2];

    float u_coord = p_pos.x + t * p_dir.x;
    float v_coord = p_pos.y + t * p_dir.y;

    float beta = u_coord * d1.x + v_coord * d1.y + d1.z;
    float gamma = u_coord * d2.x + v_coord * d2.y + d2.z;

    if (beta >= -BARYCENTRY_EPSILON && gamma >= -BARYCENTRY_EPSILON && (beta + gamma) <= 1.0f + BARYCENTRY_EPSILON) {
        float4 N_packed = tex1D<float4>(g_d_triaccel_tex, id * 4 + 3);
        //float4 N_packed = g_tri_accel[id * 4 + 3];
        float3 N = make_float3(N_packed.x, N_packed.y, N_packed.z);

        // 법선 벡터와 광선 방향의 내적(dot product)을 계산
        // 내적 값이 0보다 크면 광선이 삼각형의 뒷면
        if (dot(N, ray.dir) > 0.0f) {
            return; // 뒷면이므로 이 충돌을 무시
        }

        local_hits[local_hit_count].t = t;
        local_hits[local_hit_count].triIndex = id;
        local_hit_count++;
    }
}

#if HIT_AND_NODE_COUNT_DEBUG
__device__ int singlePassIntersectGaussian_sortNode_globalStack(
    cuRay& currRay,
    float3& accumulated_color,      // 누적 색상을 직접 업데이트
    float& accumulated_opacity,      // 누적 알파를 직접 업데이트
    int& hitCount
    , cu_traceState* global_stack, int& global_stack_ptr
) {
    hitCount = 0;
    int node_visit_count = 0;
#else
__device__ void singlePassIntersectGaussian_sortNode_globalStack(
    cuRay & currRay,
    float3 & accumulated_color,      // 누적 색상을 직접 업데이트
    float& accumulated_opacity,      // 누적 알파를 직접 업데이트
    cu_traceState * global_stack, int& global_stack_ptr
) {
#endif
    // 광선의 유효 범위 설정
    //printf("test\n");
    float t_near = RAY_START_EPSILON, t_far = FLT_MAX;
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, currRay, t_near, t_far)) {
        //printf("test ");
        // Kd-tree 순회를 위한 스택 초기화
        ShortStackCache cache;
        cache.init(threadIdx.x + threadIdx.y * blockDim.x);
        unsigned int node_idx = 0;
        kdtreeNode node = tex1D<uint2>(g_d_kdtree_tex, node_idx);
        //kdtreeNode node = g_kdtree_nodes[node_idx];
        // 메인 순회 루프
        //while (true) {
        while (accumulated_opacity < OPACITY_THRESHOLD) {
            while (!IS_LEAF(node)) {
                node = tex1D<uint2>(g_d_kdtree_tex, node_idx);
                //node = g_kdtree_nodes[node_idx];
#if HIT_AND_NODE_COUNT_DEBUG
                node_visit_count++;
#endif
                unsigned axis = SPLIT_AXIS(node);
                float split_pos = SPLIT_POS(node);
                float dir_axis = (&(currRay.dir.x))[axis];
                if (fabsf(dir_axis) < 1e-8f) { // 광선이 축과 평행한 경우
                    // 임의로 한쪽으로 보냄
                    node_idx = FIRST_CHILD_OFFSET(node) + 1;
                    //node = tex1D<uint2>(kdtreeTex, FIRST_CHILD_OFFSET(node) + 1);
                    continue;
                }
                float pos_axis = (&(currRay.pos.x))[axis];
                float t_split = (split_pos - pos_axis) / dir_axis;
                unsigned near_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 0 : 1);
                unsigned far_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 1 : 0);
                if (t_split > t_far || t_split < 0) {
                    node_idx = near_child;
                    //node = tex1D<uint2>(kdtreeTex, near_child);
                }
                else if (t_split < t_near) {
                    node_idx = far_child;
                    //node = tex1D<uint2>(kdtreeTex, far_child);
                }
                else {
                    //stack.push(far_child, t_far);
                    cu_traceState item_to_push = { far_child, t_far };
                    bool was_full = cache.is_full();
                    cu_traceState evicted_item = cache.push(item_to_push);
                    if (was_full) { // 캐시가 꽉 차서 아이템이 밀려났다면 global_stack으로 보냄
                        if (global_stack_ptr < MAX_GLOBAL_STACK_DEPTH) {
                            global_stack[global_stack_ptr++] = evicted_item;
                            //if(global_stack_ptr>=2)
                                //printf("global Stack access(push): %d\n", global_stack_ptr);
                        }
                    }
                    node_idx = near_child;
                    //node = tex1D<uint2>(kdtreeTex, near_child);
                    t_far = t_split;
                }
                node = tex1D<uint2>(g_d_kdtree_tex, node_idx);
            }

            // --- 리프 노드 처리 로직 ---
#if HIT_AND_NODE_COUNT_DEBUG
            node_visit_count++;
#endif
            unsigned int offset = OBJECTLIST_OFFSET(node);
            unsigned int count = OBJECT_SIZE(node);
            //if (count <= 0) continue;
            if (count > 0) {
                // 수집: 이 리프 노드 내의 모든 충돌을 임시 로컬 배열에 저장
                HitRecord local_hits[MAX_HITS];
                int local_hit_count = 0;

                for (unsigned i = 0; i < count; ++i) {
                    unsigned tri_idx = tex1D<unsigned int>(g_d_offsets_tex, offset + i);
                    //unsigned tri_idx = g_tri_offsets[offset + i];
                    singlePassIntersectRoutineGaussian_sortNode(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count);
                }

                if (local_hit_count > 0) {
                    // 정렬: 이 리프 노드 내의 충돌만 정렬
                    sortHits(local_hits, local_hit_count);
                    // 블렌딩: 정렬된 순서대로 알파 블렌딩 수행
                    for (int i = 0; i < local_hit_count; ++i) {
                        float4 d2 = tex1D<float4>(g_d_triaccel_tex, local_hits[i].triIndex * 4 + 2);
                        //float4 d2 = g_tri_accel[local_hits[i].triIndex * 4 + 2];
                        int gaussianID = __float_as_int(d2.w);
                        //Gaussian g = g_gaussians[gaussianID];
                        Gaussian g = fetch_gaussian(gaussianID);

                        //sample_opacity = g.opacity;
#if USE_KERNEL_SCALE
                        float sample_opacity = evaluateGaussianResponse_3dgrt(currRay, g);
#else
                        float sample_opacity = evaluateGaussianResponse(currRay, g);
#endif

                        float3 view_dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - currRay.pos);
                        float3 sample_color = eval_sh_final(3, view_dir, g);

                        accumulated_color += sample_color * sample_opacity * (1.0f - accumulated_opacity);
                        accumulated_opacity += sample_opacity * (1.0f - accumulated_opacity);
#if HIT_AND_NODE_COUNT_DEBUG
                        hitCount++;
#endif
                        // 블렌딩 중에도 조기 종료 조건을 계속 확인
                        if (accumulated_opacity > OPACITY_THRESHOLD) {
                            //printf("hitCount:%d\n", hitCount);
                            break;
                        }
                    }
                } // if (local_hit_count > 0)
            } //if (count > 0)

            //if (stack.empty()) {
            if (cache.is_empty() && global_stack_ptr == 0) {
                //printf("(empty) hitCount:%d\n", hitCount);
                break;
            }
            //cu_traceState next = stack.top(); stack.pop();
            cu_traceState next;
            if (!cache.is_empty()) {
                next = cache.pop();
            }
            else {
                next = global_stack[--global_stack_ptr];
                //printf("global Stack access(pop): %d\n", global_stack_ptr);
            }
            t_near = t_far; t_far = next.tMax;
            node_idx = next.nodeID;
            node = tex1D<uint2>(g_d_kdtree_tex, node_idx);
            //node = g_kdtree_nodes[node_idx];
        } // while(true)
    } // if (BoundsRayIntersect)
#if HIT_AND_NODE_COUNT_DEBUG
    return node_visit_count;
#endif
}


#if HIT_AND_NODE_COUNT_DEBUG
__device__ int singlePassIntersectGaussian_sortNode_shortStack(
    cuRay& currRay,
    float3& accumulated_color,       // 누적 색상을 직접 업데이트
    float& accumulated_opacity,      // 누적 알파를 직접 업데이트
    cudaTextureObject_t kdtreeTex,
    cudaTextureObject_t offsetTex,
    cudaTextureObject_t triAccelTex,
    cudaTextureObject_t gaussianTex
    int& hitCount
) {
    hitCount = 0;
    int node_visit_count = 0;
#else
__device__ void singlePassIntersectGaussian_sortNode_shortStack(
    cuRay & currRay,
    float3 & accumulated_color,      // 누적 색상을 직접 업데이트
    float& accumulated_opacity,      // 누적 알파를 직접 업데이트
    cudaTextureObject_t kdtreeTex,
    cudaTextureObject_t offsetTex,
    cudaTextureObject_t triAccelTex,
    cudaTextureObject_t gaussianTex
) {
#endif
    // 광선의 유효 범위 설정
    float t_near_global = RAY_START_EPSILON, t_far_global = FLT_MAX;
    if (!BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, currRay, t_near_global, t_far_global)) {
#if HIT_AND_NODE_COUNT_DEBUG
        return 0;
#else
        return;
#endif
    }
   
    while (t_near_global < t_far_global) {

        unsigned int node_idx = 0;
        // Kd-tree 순회를 위한 스택 초기화
        ShortStackCache stack; // 그림의 로직을 따르는 새로운 캐시 구조체 사용
        stack.init(threadIdx.x + threadIdx.y * blockDim.x);

        float t_near = t_near_global;
        float t_far = t_far_global;

        // 메인 순회 루프
        //while (true) {
        while (accumulated_opacity < OPACITY_THRESHOLD) {
            kdtreeNode node = tex1D<uint2>(kdtreeTex, node_idx);

            while (!IS_LEAF(node)) {
#if HIT_AND_NODE_COUNT_DEBUG
                node_visit_count++;
#endif
                unsigned axis = SPLIT_AXIS(node);
                float split_pos = SPLIT_POS(node);
                float dir_axis = (&currRay.dir.x)[axis];
                if (fabsf(dir_axis) < 1e-8f) { // 광선이 축과 평행한 경우
                    node_idx = FIRST_CHILD_OFFSET(node) + 1; // 임의로 한쪽으로 보냄
                    continue;
                }
                float pos_axis = (&(currRay.pos.x))[axis];
                float t_split = (split_pos - pos_axis) / dir_axis;
                unsigned near_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 0 : 1);
                unsigned far_child = FIRST_CHILD_OFFSET(node) + (pos_axis < split_pos || (pos_axis == split_pos && dir_axis < 0) ? 1 : 0);


                if (t_split > t_far || t_split < 0) {
                    node_idx = near_child;
                }
                else if (t_split < t_near) {
                    node_idx = far_child;
                }
                else {
                    //stack.push(far_child, t_far);
                    cu_traceState item_to_push = { far_child, t_far };
                    cu_traceState evicted_item = stack.push(item_to_push);
                    node_idx = near_child;
                    t_far = t_split;
                }
                node = tex1D<uint2>(kdtreeTex, node_idx);
            }

#if HIT_AND_NODE_COUNT_DEBUG
            node_visit_count++;
#endif
            // --- 리프 노드 처리 로직 ---
            unsigned int offset = OBJECTLIST_OFFSET(node);
            unsigned int count = OBJECT_SIZE(node);
            //if (count <= 0) continue;
            if (count > 0) {
                // 수집: 이 리프 노드 내의 모든 충돌을 임시 로컬 배열에 저장
                HitRecord local_hits[MAX_HITS];
                int local_hit_count = 0;

                for (unsigned i = 0; i < count; ++i) {
                    unsigned tri_idx = tex1D<unsigned int>(offsetTex, offset + i);
                    //singlePassIntersectRoutineGaussian_sortNode(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count);
                }

                if (local_hit_count > 0) {
                    // 정렬: 이 리프 노드 내의 충돌만 정렬
                    sortHits(local_hits, local_hit_count);
                    // 블렌딩: 정렬된 순서대로 알파 블렌딩 수행
                    for (int i = 0; i < local_hit_count; ++i) {
                        float4 d2 = tex1D<float4>(triAccelTex, local_hits[i].triIndex * 4 + 2);
                        int gaussianID = __float_as_int(d2.w);
                        //Gaussian g = g_d_gaussians[gaussianID];
                        //Gaussian g = fetch_gaussian(gaussianTex, gaussianID);
                        Gaussian g = fetch_gaussian(gaussianID);
#if USE_KERNEL_SCALE
                        float sample_opacity = evaluateGaussianResponse_3dgrt(currRay, g);
#else
                        float sample_opacity = evaluateGaussianResponse(currRay, g);
#endif
                        //sample_opacity = g.opacity;

                        float3 view_dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - currRay.pos);
                        float3 sample_color = eval_sh_final(3, view_dir, g);
                        accumulated_color += sample_color * sample_opacity * (1.0f - accumulated_opacity);
                        accumulated_opacity += sample_opacity * (1.0f - accumulated_opacity);
#if HIT_AND_NODE_COUNT_DEBUG
                        //hitCount++;
                        atomicAdd(&hitCount, 1);
#endif
                        // 블렌딩 중에도 조기 종료 조건을 계속 확인
                        if (accumulated_opacity > OPACITY_THRESHOLD) {
                            //printf("hitCount:%d\n", hitCount);
                            break;
                        }
                    }
                } // if (local_hit_count > 0)
            } //if (count > 0)

            if (stack.is_empty()) {
                //printf("(empty) hitCount:%d\n", hitCount);
                break;
            }
            //cu_traceState next = stack.top(); stack.pop();
            cu_traceState next = stack.pop();
            t_near = t_far; t_far = next.tMax;
            node_idx = next.nodeID;
        } // while(true): while(accumulated_opacity < OPACITY_THRESHOLD)

        //TODO: Ray를 다시 쏘면서 진행
        if (accumulated_opacity < OPACITY_THRESHOLD) break;
        t_near_global = t_far;
    } // while(t_near_global < t_far_global)
#if HIT_AND_NODE_COUNT_DEBUG
    return node_visit_count;
#endif
}

__global__ void renderKernelGaussian_sortNode(float* pFrameBuffer
#if HIT_AND_NODE_COUNT_DEBUG
    , int* hitsum, int* maxhit, int* hcount, int* g_d_total_nodes, int* g_d_max_nodes
#endif
#if USE_GLOBAL_STACK
    , cu_traceState* d_global_stack, int* d_global_stack_pointers
#endif
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x == 0 && y == 0) { // for debug
        printf("[Step 1] Kernel has started.\n");
    }
    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    // 광선 생성
    float sx = (float)x + 0.5f, sy = (float)y + 0.5f;

    float3 dir = g_CameraInfo.startPoint + g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
    cuRay ray = { g_CameraInfo.eye, normalize(dir - g_CameraInfo.eye) };

    // 누적 변수 초기화
    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    float accumulated_opacity = 0.0f;
#if USE_GLOBAL_STACK
    // 스레드에 해당하는 전역 스택 포인터 가져오기
    int thread_idx = y * g_SceneInfo.resX + x;
    cu_traceState* my_global_stack = d_global_stack + thread_idx * MAX_GLOBAL_STACK_DEPTH;
    int my_global_stack_ptr = d_global_stack_pointers[thread_idx];
#endif

#if HIT_AND_NODE_COUNT_DEBUG
    int hitCount = 0;
    #if USE_GLOBAL_STACK
    int node_visits = singlePassIntersectGaussian_sortNode_globalStack(ray, accumulated_color, accumulated_opacity, hitCount,
        kdtreeTex, offsetTex, triAccelTex, gaussianTex
        , my_global_stack, my_global_stack_ptr
        );
    #else
    int node_visits = singlePassIntersectGaussian_sortNode_shortStack(ray, accumulated_color, accumulated_opacity, hitCount);
    #endif
#else
    #if USE_GLOBAL_STACK
    singlePassIntersectGaussian_sortNode_globalStack(ray, accumulated_color, accumulated_opacity
        , my_global_stack, my_global_stack_ptr
    );
    #else
    singlePassIntersectGaussian_sortNode_shortStack(ray, accumulated_color, accumulated_opacity);
    #endif
#endif

#if USE_GLOBAL_STACK
    // 최종 스택 포인터 저장
    d_global_stack_pointers[thread_idx] = my_global_stack_ptr;
#endif

    // 최종 색상 계산
    float3 background_color = make_float3(0.0f, 0.0f, 0.0f);
    //float3 final_color = accumulated_color + background_color * (1.0f - accumulated_opacity);
    float3 final_color = accumulated_color / accumulated_opacity;

    int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
    pFrameBuffer[idx + 0] = final_color.x;
    pFrameBuffer[idx + 1] = final_color.y;
    pFrameBuffer[idx + 2] = final_color.z;

#if HIT_AND_NODE_COUNT_DEBUG
    if (hitCount > 0) {
        //if(accumulated_opacity<OPACITY_THRESHOLD) printf("hitCount %d | ao: %f\n", hit.hitCount, accumulated_opacity);
        atomicAdd(hitsum, hitCount);
        atomicMax(maxhit, hitCount);
        atomicAdd(hcount, 1);
        atomicAdd(g_d_total_nodes, node_visits);
        atomicMax(g_d_max_nodes, node_visits);
    }
#endif
}

void renderGaussianWithCudaSetup(const CompositeObject& object, const std::vector<Gaussian>& gaussians) { 
    printf("Setting up static data for CUDA rendering...\n");

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Using GPU %d: %s\n", dev, prop.name);
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        printf("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
    }

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

    // 데이터 패킹 (Host)
    // TriAccel -> float4[4] (n_u, n_v, n_d, k | b_nu, b_nv, b_d, idx | c_nu, c_nv, c_d, matID | N.x, N.y, N.z, pad)
    std::vector<float4> h_triangles(object.n_triangles * 4);
    for (int i = 0; i < object.n_triangles; ++i) {
        const TriAccel& src = kdTree->tri_accel_list[i];
        h_triangles[i * 4 + 0] = make_float4(src.n_u, src.n_v, src.n_d, uint_as_float_H(src.k));
        h_triangles[i * 4 + 1] = make_float4(src.b_nu, src.b_nv, src.b_d, int_as_float_H(src.indexInObject));
        h_triangles[i * 4 + 2] = make_float4(src.c_nu, src.c_nv, src.c_d, int_as_float_H(src.material_ID));
        h_triangles[i * 4 + 3] = make_float4(src.N[0], src.N[1], src.N[2], 0.0f);
    }
    printf("1. Data packing done\n");

    // GPU 메모리 할당 및 데이터 전송
    if (g_d_kdtree_nodes) CUDA_CHECK(cudaFree(g_d_kdtree_nodes));
    if (g_d_tri_offsets) CUDA_CHECK(cudaFree(g_d_tri_offsets));
    if (g_d_tri_accel) CUDA_CHECK(cudaFree(g_d_tri_accel));
    if (g_d_gaussians_persistent) CUDA_CHECK(cudaFree(g_d_gaussians_persistent));

    size_t node_size = kdTree->tree_node_count * sizeof(kdtreeNode);
    CUDA_CHECK(cudaMalloc(&g_d_kdtree_nodes, node_size));
    CUDA_CHECK(cudaMemcpy(g_d_kdtree_nodes, kdTree->tree, node_size, cudaMemcpyHostToDevice));

    size_t offset_size = kdTree->tri_offset_count * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&g_d_tri_offsets, offset_size));
    CUDA_CHECK(cudaMemcpy(g_d_tri_offsets, kdTree->tri_offset_list, offset_size, cudaMemcpyHostToDevice));

    size_t accel_size = h_triangles.size() * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&g_d_tri_accel, accel_size));
    CUDA_CHECK(cudaMemcpy(g_d_tri_accel, h_triangles.data(), accel_size, cudaMemcpyHostToDevice));

    //new
    size_t gaussians_bytes = gaussians.size() * sizeof(Gaussian);
    CUDA_CHECK(cudaMalloc(&g_d_gaussians_persistent, gaussians_bytes));
    CUDA_CHECK(cudaMemcpy(g_d_gaussians_persistent, gaussians.data(), gaussians_bytes, cudaMemcpyHostToDevice));

    //CUDA_CHECK(cudaMemcpyToSymbol(g_kdtree_nodes, &g_d_kdtree_nodes, sizeof(kdtreeNode*)));
    //CUDA_CHECK(cudaMemcpyToSymbol(g_tri_offsets, &g_d_tri_offsets, sizeof(unsigned int*)));
    //CUDA_CHECK(cudaMemcpyToSymbol(g_tri_accel, &g_d_tri_accel, sizeof(float4*)));
    //CUDA_CHECK(cudaMemcpyToSymbol(g_gaussians, &g_d_gaussians_persistent, sizeof(Gaussian*)));
    //new end

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] GPU memcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    printf("2. gpu memcpy done\n");

    //// 텍스처 바인딩
    //cudaChannelFormatDesc node_desc = cudaCreateChannelDesc<uint2>();
    //cudaChannelFormatDesc offset_desc = cudaCreateChannelDesc<unsigned int>();
    //cudaChannelFormatDesc tri_desc = cudaCreateChannelDesc<float4>();
    //CUDA_CHECK(cudaBindTexture(0, &inKdTreeNodeTex, g_d_kdtree_nodes, &node_desc, node_size));
    //CUDA_CHECK(cudaBindTexture(0, &inObjectOffsetListTex, g_d_tri_offsets, &offset_desc, offset_size));
    //CUDA_CHECK(cudaBindTexture(0, &inTriAccelTex, g_d_tri_accel, &tri_desc, accel_size));

    //cudaChannelFormatDesc gaussian_desc = cudaCreateChannelDesc<float4>();
    //CUDA_CHECK(cudaBindTexture(0, &inGaussianTex, g_d_gaussians_persistent, &gaussian_desc, gaussians_bytes));

    //---텍스처 객체 생성--------------------------------------------------------------

    //  Kd-Tree 노드 텍스처 객체 생성
    cudaResourceDesc resDescNode;
    memset(&resDescNode, 0, sizeof(resDescNode));
    resDescNode.resType = cudaResourceTypeLinear;
    resDescNode.res.linear.devPtr = g_d_kdtree_nodes;
    resDescNode.res.linear.desc = cudaCreateChannelDesc<uint2>();
    resDescNode.res.linear.sizeInBytes = kdTree->tree_node_count * sizeof(kdtreeNode);

    cudaTextureDesc texDesc; // 모든 텍스처에 재사용 가능
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    if (g_d_kdtree_tex) CUDA_CHECK(cudaDestroyTextureObject(g_d_kdtree_tex)); // 기존 객체 파괴
    CUDA_CHECK(cudaCreateTextureObject(&g_d_kdtree_tex, &resDescNode, &texDesc, NULL));

    // 오프셋 리스트 텍스처 객체 생성 (resDesc만 변경)
    cudaResourceDesc resDescOffset;
    memset(&resDescOffset, 0, sizeof(resDescOffset));
    resDescOffset.resType = cudaResourceTypeLinear;
    resDescOffset.res.linear.devPtr = g_d_tri_offsets;
    resDescOffset.res.linear.desc = cudaCreateChannelDesc<unsigned int>();
    resDescOffset.res.linear.sizeInBytes = kdTree->tri_offset_count * sizeof(unsigned int);

    if (g_d_offsets_tex) CUDA_CHECK(cudaDestroyTextureObject(g_d_offsets_tex));
    CUDA_CHECK(cudaCreateTextureObject(&g_d_offsets_tex, &resDescOffset, &texDesc, NULL));

    // Triangle Acceleration 텍스처 객체 생성 (resDesc만 변경)
    cudaResourceDesc resDescAccel;
    memset(&resDescAccel, 0, sizeof(resDescAccel));
    resDescAccel.resType = cudaResourceTypeLinear;
    resDescAccel.res.linear.devPtr = g_d_tri_accel;
    resDescAccel.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDescAccel.res.linear.sizeInBytes = h_triangles.size() * sizeof(float4);

    if (g_d_triaccel_tex) CUDA_CHECK(cudaDestroyTextureObject(g_d_triaccel_tex));
    CUDA_CHECK(cudaCreateTextureObject(&g_d_triaccel_tex, &resDescAccel, &texDesc, NULL));

    // Gaussian 데이터 텍스처 객체 생성 (resDesc만 변경)
    cudaResourceDesc resDescGaussian;
    memset(&resDescGaussian, 0, sizeof(resDescGaussian));
    resDescGaussian.resType = cudaResourceTypeLinear;
    resDescGaussian.res.linear.devPtr = g_d_gaussians_persistent;
    resDescGaussian.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDescGaussian.res.linear.sizeInBytes = gaussians.size() * sizeof(Gaussian);

    if (g_d_gaussian_tex) CUDA_CHECK(cudaDestroyTextureObject(g_d_gaussian_tex));
    CUDA_CHECK(cudaCreateTextureObject(&g_d_gaussian_tex, &resDescGaussian, &texDesc, NULL));

    //---------------------------------------------------------------------------------*/



    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] texture bind failed: " << cudaGetErrorString(err) << std::endl;
    }
    printf("3. texture Bind done\n");

    // 상수 메모리 설정
    float3 h_bbox_min = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    float3 h_bbox_max = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMin, &h_bbox_min, sizeof(float3)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMax, &h_bbox_max, sizeof(float3)));


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] const memory set failed: " << cudaGetErrorString(err) << std::endl;
    }
    printf("4. const memory set done\n");


    //int deviceID;
    //cudaGetDevice(&deviceID);

    //int maxSharedMemPerBlock;
    //// 현재 GPU의 "블록 당 최대 공유 메모리" 속성 값을 가져옵니다.
    //cudaDeviceGetAttribute(
    //    &maxSharedMemPerBlock,
    //    cudaDevAttrMaxSharedMemoryPerBlock,
    //    deviceID
    //);
    //printf("This GPU's max shared memory per block: %d bytes\n", maxSharedMemPerBlock);
    //// 49152 bytes
    //// 49152 / (256 * 8) = 24

    //cudaFuncSetAttribute(
    //    renderKernelGaussian_sortNode,
    //    cudaFuncAttributeMaxDynamicSharedMemorySize,
    //    maxSharedMemPerBlock
    //);

    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
}

int renderGaussianWithCudaFrame(const Camera& camera, int width, int height, float* d_framebuffer
#if USE_GLOBAL_STACK
    , cu_traceState* d_global_stack, int* d_global_stack_pointers
#endif
) {
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
#if USE_GLOBAL_STACK
    CUDA_CHECK(cudaMemset(d_global_stack_pointers, 0, (size_t)width * height * sizeof(int)));
#endif
    printf("3\n");

    // 커널 실행
    dim3 threads(DIM_X, DIM_Y);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    size_t shared_mem_size = threads.x * threads.y * SHORT_STACK_DEPTH * sizeof(cu_traceState);

#if HIT_AND_NODE_COUNT_DEBUG
    int* d_hitsum, * d_maxhit, * d_hcount, * d_total_nodes, * d_max_nodes;
    cudaMalloc(&d_hitsum, sizeof(int));
    cudaMalloc(&d_maxhit, sizeof(int));
    cudaMalloc(&d_hcount, sizeof(int));
    cudaMalloc(&d_total_nodes, sizeof(int));
    cudaMalloc(&d_max_nodes, sizeof(int));

    cudaMemset(d_hitsum, 0, sizeof(int));
    cudaMemset(d_maxhit, 0, sizeof(int));
    cudaMemset(d_hcount, 0, sizeof(int));
    cudaMemset(d_total_nodes, 0, sizeof(int));
    cudaMemset(d_max_nodes, 0, sizeof(int));

    cudaEventRecord(start_ev); // 시작 기록
    //renderKernelGaussian << < blocks, threads, shared_mem_size >> > (d_framebuffer, d_hitsum, d_maxhit, hitcount);
    renderKernelGaussian_sortNode << < blocks, threads, shared_mem_size >> > (d_framebuffer, d_hitsum, d_maxhit, d_hcount, d_total_nodes, d_max_nodes
#if USE_GLOBAL_STACK
        , d_global_stack, d_global_stack_pointers
#endif
        , d_debug_log_buffer, d_debug_log_counter // 커널에 인자 전달
        );
    cudaEventRecord(stop_ev); // 종료 기록
    cudaEventSynchronize(stop_ev); // GPU 작업 완료까지 대기
    //CUDA_CHECK(cudaGetLastError());        // launch 실패 확인
    //CUDA_CHECK(cudaDeviceSynchronize()); // 실행 중 오류 확인

    int h_total_hits = 0, h_max_hit = 0, h_pixel_count = 0, h_total_nodes = 0, h_max_nodes = 0;
    cudaMemcpy(&h_total_hits, d_hitsum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_hit, d_maxhit, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pixel_count, d_hcount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_nodes, d_total_nodes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_nodes, d_max_nodes, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_pixel_count > 0) {
        float avg_hits = (float)h_total_hits / h_pixel_count;
        float avg_nodes = (float)h_total_nodes / h_pixel_count; // ★ 평균 계산
        printf("Avg Hits/Pixel: %.2f (Max: %d) | Avg Nodes/Pixel: %.2f (Max: %d) | Rendered Pixels: %d\n",
            avg_hits, h_max_hit, avg_nodes, h_max_nodes, h_pixel_count);
    }

    // --- 4. 메모리 해제 ---
    cudaFree(d_hitsum);
    cudaFree(d_maxhit);
    cudaFree(d_hcount);
    cudaFree(d_total_nodes);
    cudaFree(d_max_nodes);
#else
    cudaEventRecord(start_ev); // 시작 기록
    renderKernelGaussian_sortNode << < blocks, threads, shared_mem_size >> > (d_framebuffer
#if USE_GLOBAL_STACK
        , d_global_stack, d_global_stack_pointers
#endif
        );
    CUDA_CHECK(cudaGetLastError());        // launch 실패 확인
    CUDA_CHECK(cudaDeviceSynchronize()); // 실행 중 오류 확인
    cudaEventRecord(stop_ev); // 종료 기록
    cudaEventSynchronize(stop_ev); // GPU 작업 완료까지 대기
#endif
    printf("4\n");
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_ev, stop_ev);
    k_fps = 1000.0f / milliseconds; // 전역 변수에 FPS 저장
    //total_frame += k_fps;
    printf("FPS : %f\n", k_fps);
    //if (++frame_count >= 100) {
    //    printf("avg FPS for 100 frame : %f\n", (float)(total_frame / frame_count));
    //    frame_count = 0;
    //    total_frame = 0.0f;
    //}
    
    // 결과 복사 및 메모리 해제
    //if (out_framebuffer) delete[] out_framebuffer;
    //out_framebuffer = new float[width * height * 3];
    //CUDA_CHECK(cudaMemcpy(out_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost));
    //is_done = true;

    //CUDA_CHECK(cudaFree(d_framebuffer));
    return k_fps;
}

void cleanupCudaResources() {
    printf("Cleaning up CUDA resources...\n");
    if (g_d_kdtree_nodes) cudaFree(g_d_kdtree_nodes);
    if (g_d_tri_offsets) cudaFree(g_d_tri_offsets);
    if (g_d_tri_accel) cudaFree(g_d_tri_accel);
    if (g_d_gaussians_persistent) cudaFree(g_d_gaussians_persistent);

    //if (g_d_gaussians) cudaFree(g_d_gaussians);
    //if (g_kdtree_nodes) cudaFree(g_kdtree_nodes);
    //if (g_tri_offsets) cudaFree(g_tri_offsets);
    //if (g_tri_accel) cudaFree(g_tri_accel);
    //if (g_gaussians) cudaFree(g_gaussians);

    //cudaUnbindTexture(inKdTreeNodeTex);
    //cudaUnbindTexture(inObjectOffsetListTex);
    //cudaUnbindTexture(inTriAccelTex);
    //cudaUnbindTexture(inGaussianTex);
    if (g_d_kdtree_tex) cudaDestroyTextureObject(g_d_kdtree_tex);
    if (g_d_offsets_tex) cudaDestroyTextureObject(g_d_offsets_tex);
    if (g_d_triaccel_tex) cudaDestroyTextureObject(g_d_triaccel_tex);
    if (g_d_gaussian_tex) cudaDestroyTextureObject(g_d_gaussian_tex);

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
}