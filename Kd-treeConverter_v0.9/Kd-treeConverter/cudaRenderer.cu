#include <GL/glew.h>
#include <GL/freeglut.h>
#include <curand_kernel.h>
#include "OpenGLStuffs.h"
#include "CudaRenderer.h"
#include "MyMathUtility.h"
//#include "SGRTx2Lib/cudaRenderPipeline.h"
#include "SGRTx2Lib/cuda_math.h"

//#include <vector>
#include <iostream>
#include <cuda_runtime.h>
//#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
//#include <texture_fetch_functions.h>
#include <texture_indirect_functions.h>
#include <vector_types.h>

#undef Y
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"
#define Y 1

cudaDeviceProp deviceProp;
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
#define RAY_START_EPSILON		EPSILON2
#define BARYCENTRY_EPSILON		EPSILON7

#define EPSILON1 1e-1f
#define EPSILON2 1e-2f
#define EPSILON3 1e-3f
#define EPSILON4 1e-4f
#define EPSILON5 1e-5f
#define EPSILON7 1e-7f
#define EPSILON8 1e-8f
#define EPSILON9 1e-9f

using float33 = float3[3]; // row major matrix
static __device__ inline float3 operator*(const float3& p, const float33& m) {
    return make_float3(dot(m[0], p), dot(m[1], p), dot(m[2], p));
}

static __device__ inline float3 safe_normalize(float3 v) {
    const float l = v.x * v.x + v.y * v.y + v.z * v.z;
    return l > 0.0f ? (v * rsqrtf(l)) : v;
}

#if !QUATERNION
// vec * mat
static __device__ inline float3 multiplyMatrixVector(const float3& p, const float3x3& m) {
    return make_float3(
        m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z,
        m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z,
        m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z
    );
}

// mat * vec
static __device__ inline float3 multMatrixTransposeVector(const float3& p, const float3x3& m) {
    return make_float3(
        m.m[0][0] * p.x + m.m[1][0] * p.y + m.m[2][0] * p.z,
        m.m[0][1] * p.x + m.m[1][1] * p.y + m.m[2][1] * p.z,
        m.m[0][2] * p.x + m.m[1][2] * p.y + m.m[2][2] * p.z
    );
}
#endif

// --- Device-side Data Structures ---
struct cuRay {
    float3 pos;
    float3 dir;
    __device__ float2 get_dir_pos(const unsigned axis) const {
        float2 ret;
        switch (axis) {
            case 0:     ret.x = pos.x; ret.y = dir.x; return ret;
            case 1:     ret.x = pos.y; ret.y = dir.y; return ret;
            default:    ret.x = pos.z; ret.y = dir.z; return ret;
        }
    }
    __host__ __device__ inline float3 dir_perm_x(void) const {
        return make_float3(dir.x, dir.y, dir.z);
    }
    __host__ __device__ inline float3 dir_perm_y(void) const {
        return make_float3(dir.y, dir.z, dir.x);
    }
    __host__ __device__ inline float3 dir_perm_z(void) const {
        return make_float3(dir.z, dir.x, dir.y);
    }
    __host__ __device__ inline float3 pos_perm_x(void) const {
        return make_float3(pos.x, pos.y, pos.z);
    }
    __host__ __device__ inline float3 pos_perm_y(void) const {
        return make_float3(pos.y, pos.z, pos.x);
    }
    __host__ __device__ inline float3 pos_perm_z(void) const {
        return make_float3(pos.z, pos.x, pos.y);
    }
};

struct cuIntersectionCheck {
    float tHit;
    float beta, gamma;
    int primIndex;
    int objectIndex;
    __device__ void init() { tHit = FLT_MAX; primIndex = -1; objectIndex = 0; }
    __device__ bool isHit() const { return primIndex != -1; }
};

struct cuIntersectionPoint {
    float3 pos, dir, normal;
    float3 colorWeight;
    __device__ void init() { colorWeight = make_float3(1.0f, 1.0f, 1.0f); }
};

// Kd-tree 노드 (GKDTreeNode.h에서 추출)
typedef uint2 kdtreeNode;

/**
*	intersection 을 계산하기 위한 triangle 정보. wald 방법
*	internal2.w 를 object index 를 위한 값으로 사용한다.
*/
struct __align__(16) cuWaldTriangleInfo {
    float4 internal0, internal1, internal2;

    __device__ unsigned k() const { return __float_as_int(internal0.x); }
    __device__ float n_u() const { return internal0.y; }
    __device__ float n_v() const { return internal0.z; }
    __device__ float n_d() const { return internal0.w; }
    __device__ float vert_ku() const { return internal1.x; }
    __device__ float vert_kv() const { return internal1.y; }
    __device__ float b_nu() const { return internal1.z; }
    __device__ float b_nv() const { return internal1.w; }
    __device__ float c_nu() const { return internal2.x; }
    __device__ float c_nv() const { return internal2.y; }

    __device__ bool isTransparent() const {
        return (__float_as_int(internal2.z) == 1 || __float_as_int(internal2.z) == -1);
    }

    /** wald 방법에서 n' 를 구할때 나눈값이 양수인지 여부. */
    __device__ bool isPositiveDir() const { return (__float_as_int(internal2.z) > 0); }

    struct perm_t { float3 dir, pos; };
    __device__ inline perm_t get_perm(const cuRay & ray) const {
        perm_t perm;
        unsigned const axis = k();
        switch (axis) {
        case 0:
            perm.dir = ray.dir_perm_x();
            perm.pos = ray.pos_perm_x();
            return perm;
        case 1:
            perm.dir = ray.dir_perm_y();
            perm.pos = ray.pos_perm_y();
            return perm;
        default:
            perm.dir = ray.dir_perm_z();
            perm.pos = ray.pos_perm_z();
            return perm;
        }
    }

    /** selective supersampling 을 위해서.. 하위3바이트는 object id, 상위 1byte 는 물체선택여부 */
    __device__ inline int getObjectIndex(void) const {
        return (__float_as_int(internal2.w) & 0x00ffffff);
    }

    /** selective supersampling 을 위해서.. 하위3바이트는 object id, 상위 1byte 는 물체선택여부 */
    __device__ inline int isSelection(void) const {
        return ((__float_as_int(internal2.w) & 0xff000000) > 0);
    }
};

#if SHORT_STACK_DEPTH > 0
// 스택 (cudaRenderPipelineCommonKernel.cu에서 추출)
//extern __shared__ cu_traceState smemBuffer[SHORT_STACK_DEPTH * DIM_X * DIM_Y];
extern __shared__ cu_traceState smemBuffer[];

struct shortStack {
    unsigned _top, quant, baseOffset;
    __device__ shortStack() : _top(SHORT_STACK_DEPTH - 1), quant(0) {}
    __device__ inline void init(const unsigned smem_baseOffset) {
        baseOffset = smem_baseOffset * (SHORT_STACK_DEPTH);
    }
    __device__ inline cu_traceState top() { return smemBuffer[baseOffset + _top]; }
    __device__ inline void push(unsigned id, float t_max) {
        if (++_top == SHORT_STACK_DEPTH)
            _top = 0;
        quant = min(quant + 1, SHORT_STACK_DEPTH);
        smemBuffer[baseOffset + _top].nodeID = id;
        smemBuffer[baseOffset + _top].tMax = t_max;
    }
    __device__ inline int empty() { return quant == 0; }
    __device__ inline int full() { return quant == SHORT_STACK_DEPTH; }
    __device__ inline void pop() {
        if (_top == 0)
            _top = SHORT_STACK_DEPTH;
        --_top; --quant;
    }
};
//// LIFO 캐시 구조체
//struct ShortStackCache {
//    unsigned int head;       // 가장 오래된 데이터의 위치 (다음에 밀려날 대상)
//    unsigned int tail;       // 다음에 데이터를 쓸 위치
//    unsigned int count;      // 현재 캐시 안의 데이터 개수
//    unsigned int baseOffset;
//
//    __device__ void init(const unsigned int smem_baseOffset) {
//        baseOffset = smem_baseOffset * SHORT_STACK_DEPTH;
//        head = 0;
//        tail = 0;
//        count = 0;
//    }
//
//    __device__ bool is_empty() const { return count == 0; }
//    __device__ bool is_full() const { return count >= SHORT_STACK_DEPTH; }
//
//    // 데이터를 캐시에 PUSH하는 함수. 캐시가 꽉 찼으면 밀려나는 데이터를 반환.
//    __device__ cu_traceState push(const cu_traceState& item) {
//        cu_traceState evicted_item = {}; // 기본값으로 초기화
//
//        if (is_full()) {
//            // 가장 오래된 데이터를 evicted_item에 저장
//            evicted_item = smemBuffer[baseOffset + head];
//            // head 포인터를 다음으로 이동 (순환)
//            head = (head + 1) % SHORT_STACK_DEPTH;
//        }
//        else {
//            count++;
//        }
//
//        // tail 위치에 새로운 아이템을 쓰고 tail 포인터를 다음으로 이동 (순환)
//        smemBuffer[baseOffset + tail] = item;
//        tail = (tail + 1) % SHORT_STACK_DEPTH;
//
//        return evicted_item;
//    }
//
//    // 캐시에서 데이터를 POP하는 함수 (LIFO: 가장 나중에 들어온 것부터)
//    __device__ cu_traceState pop() {
//        // tail 포인터를 뒤로 돌려 가장 마지막에 쓴 데이터 위치로 이동 (순환)
//        tail = (tail == 0) ? SHORT_STACK_DEPTH - 1 : tail - 1;
//        count--;
//        return smemBuffer[baseOffset + tail];
//    }
//
//    __device__ int cnt() { return count; }
//};
#endif

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

__device__ cudaTextureObject_t inKdTreeNodeTex;
#if OFFSET_TEXTURE
__device__ cudaTextureObject_t inObjectOffsetListTex;
#else
__device__ unsigned int* g_d_tri_offsets_dev = nullptr;
#endif
#if TRIACC_TEXTURE
__device__ cudaTextureObject_t inTriAccelTex;
#else
__device__ float4* g_d_tri_acc_dev = nullptr;
#endif
#if GAUSSIAN_TEXTURE
__device__ cudaTextureObject_t inGaussianTex;

__device__ Gaussian fetch_gaussian(int gaussianID) {
    Gaussian g;
    // Gaussian g의 메모리 주소를 float4 포인터로 재해석
    float4* g_as_float4 = reinterpret_cast<float4*>(&g);

    const int num_float4s = sizeof(Gaussian) / sizeof(float4);
    int base_idx = gaussianID * num_float4s;

#pragma unroll
    for (int i = 0; i < num_float4s; ++i) {
        // 중간 배열 없이 g의 메모리에 직접 tex1Dfetch 결과를 씀
        g_as_float4[i] = tex1Dfetch<float4>(inGaussianTex, base_idx + i);
    }

    return g;
}
#else
__device__ Gaussian* g_d_gaussians;
#endif
#if KSCALE_TEXTURE
__device__ cudaTextureObject_t inKScaleTex;
#else
__device__ float* g_d_kScale = nullptr;
#endif

struct SceneInfo { int resX, resY; };
struct CameraInfo { float3 eye, u, v, startPoint; float stepX, stepY; };

__constant__ SceneInfo g_SceneInfo;
__constant__ CameraInfo g_CameraInfo;
__constant__ float3 g_SceneBBoxMin;
__constant__ float3 g_SceneBBoxMax;

#if GLOBAL_DEVICE_VAR
kdtreeNode* g_d_kdtree_nodes = nullptr;
unsigned int* g_d_prim_offsets = nullptr;
    #if WALD_METHOD
cuWaldTriangleInfo* g_d_waldInfo = nullptr;
    #else
TriAccel* g_d_tri_accel = nullptr;
    #endif
Gaussian* g_d_gaussians_persistent = nullptr;
    #if !QUATERNION
float* g_d_kScale_persistent = nullptr;
    #endif
#endif


cudaEvent_t start_ev, stop_ev;

// =================================================================================
// CUDA 커널 코드 (사용자 제공 커널)
// =================================================================================

/*__device__ bool BoundsRayIntersect(const float3& bmin, const float3& bmax, const cuRay& ray, float& tmin, float& tmax) {
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
}*/

__device__ inline bool BoundsRayIntersect(const float3& bmin, const float3& bmax, const cuRay* ray, float* tmin, float* tmax) {
    float l1 = 0.0f;
    float l2 = 0.0f;

    l1 = __fdividef(bmin.x - ray->pos.x, ray->dir.x);
    l2 = __fdividef(bmax.x - ray->pos.x, ray->dir.x);
    *tmin = fmaxf(fminf(l1, l2), *tmin);
    *tmax = fminf(fmaxf(l1, l2), *tmax);

    l1 = __fdividef(bmin.y - ray->pos.y, ray->dir.y);
    l2 = __fdividef(bmax.y - ray->pos.y, ray->dir.y);
    *tmin = fmaxf(fminf(l1, l2), *tmin);
    *tmax = fminf(fmaxf(l1, l2), *tmax);

    l1 = __fdividef(bmin.z - ray->pos.z, ray->dir.z);
    l2 = __fdividef(bmax.z - ray->pos.z, ray->dir.z);
    *tmin = fmaxf(fminf(l1, l2), *tmin);
    *tmax = fminf(fmaxf(l1, l2), *tmax);

    return ((*tmax >= *tmin) & (*tmax >= 0.f));
}

//-----------------------------------------------------------------------
// Gaussian Render
//-----------------------------------------------------------------------
struct HitRecord {
    float t;
    int primIndex;
#if STORE_GRAYDIST
    float grayDist;
#endif
#if BLEND_SELECT
    cuWaldTriangleInfo::perm_t perm;
#endif
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

__device__ void sortHits_Hybrid(HitRecord* hits, int count) {
    // Case 1: 데이터가 매우 적을 때 (대다수 케이스) -> Insertion Sort가 가장 빠름
    if (count <= 32) {
        for (int i = 1; i < count; i++) {
            HitRecord key = hits[i];
            int j = i - 1;
            while (j >= 0 && hits[j].t > key.t) {
                hits[j + 1] = hits[j];
                j--;
            }
            hits[j + 1] = key;
        }
    }
    // Case 2: 데이터가 많을 때 (Long Tail 케이스) -> Shell Sort 사용
    // 140개 정도면 Shell Sort가 Insertion Sort보다 훨씬 효율적입니다.
    else {
        // Ciura gap sequence (경험적으로 가장 효율적인 간격)
        const int gaps[] = { 57, 23, 10, 4, 1 };

        for (int g = 0; g < 5; ++g) {
            int gap = gaps[g];
            if (gap >= count) continue;

            for (int i = gap; i < count; ++i) {
                HitRecord temp = hits[i];
                int j;
                for (j = i; j >= gap && hits[j - gap].t > temp.t; j -= gap) {
                    hits[j] = hits[j - gap];
                }
                hits[j] = temp;
            }
        }
    }
}

__device__ void selectionSortStep(HitRecord* hits, int count, int i)
{
    float k = hits[i].t;
    int idx = i;

    for (int j = i + 1; j < count; j++) {
        if (k > hits[j].t) {
            k = hits[j].t;
            idx = j;
        }
    }

    HitRecord t = hits[idx];
    hits[idx] = hits[i];
    hits[i] = t;
}

/**
 * @brief 구면 조화 함수(SH) 평가 함수
 * @param degree 계산할 SH 차수 (최대 3)
 * @param view_dir 뷰 방향 벡터 (정규화 필요)
 * @param g 가우시안 데이터
 * @return 최종 계산된 색상 (0~1 범위로 클램핑됨)
 */
__device__ __forceinline__ float3 eval_sh_final(
    const int degree,
    const float3& view_dir,
    const Gaussian& g,
    bool clamped = true
) {
    // 계산을 용이하게 하기 위해 g.f_dc와 g.f_rest를 하나의 배열로
    float3 sphCoefficients[16];
    sphCoefficients[0] = make_float3(g.f_dc[0], g.f_dc[1], g.f_dc[2]);
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        sphCoefficients[i + 1] = make_float3(g.f_rest[i], g.f_rest[15 + i], g.f_rest[30 + i]);
    }
    //sphCoefficients[0] = make_float3(g.f_dc[0], g.f_dc[1], g.f_dc[2]);
    //sphCoefficients[1] = make_float3(g.f_rest[0], g.f_rest[15], g.f_rest[30]);
    //sphCoefficients[2] = make_float3(g.f_rest[1], g.f_rest[16], g.f_rest[31]);
    //sphCoefficients[3] = make_float3(g.f_rest[2], g.f_rest[17], g.f_rest[32]);
    //sphCoefficients[4] = make_float3(g.f_rest[3], g.f_rest[18], g.f_rest[33]);
    //sphCoefficients[5] = make_float3(g.f_rest[4], g.f_rest[19], g.f_rest[34]);
    //sphCoefficients[6] = make_float3(g.f_rest[5], g.f_rest[20], g.f_rest[35]);
    //sphCoefficients[7] = make_float3(g.f_rest[6], g.f_rest[21], g.f_rest[36]);
    //sphCoefficients[8] = make_float3(g.f_rest[7], g.f_rest[22], g.f_rest[37]);
    //sphCoefficients[9] = make_float3(g.f_rest[8], g.f_rest[23], g.f_rest[38]);
    //sphCoefficients[10] = make_float3(g.f_rest[9], g.f_rest[24], g.f_rest[39]);
    //sphCoefficients[11] = make_float3(g.f_rest[10], g.f_rest[25], g.f_rest[40]);
    //sphCoefficients[12] = make_float3(g.f_rest[11], g.f_rest[26], g.f_rest[41]);
    //sphCoefficients[13] = make_float3(g.f_rest[12], g.f_rest[27], g.f_rest[42]);
    //sphCoefficients[14] = make_float3(g.f_rest[13], g.f_rest[28], g.f_rest[43]);
    //sphCoefficients[15] = make_float3(g.f_rest[14], g.f_rest[29], g.f_rest[44]);

    // --- 3dgrt의 radianceFromSpH 로직을 그대로 적용 ---
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

    // 최종 활성화: 원본과 동일하게 0.5를 더하고, 0 미만 값은 0으로 클램핑
    rad += make_float3(0.5f);
    //return min(max(rad, make_float3(0.f)), make_float3(1.f));
    return clamped ? max(rad, make_float3(0.0f)) : rad;
}

__device__ __forceinline__ float3 eval_sh_final2(
    const int degree,
    const float3& view_dir,
    const Gaussian g
) {

    // 계산을 용이하게 하기 위해 g.f_dc와 g.f_rest를 하나의 배열로
    float3 sphCoefficients[16];
    sphCoefficients[0] = make_float3(g.f_dc[0], g.f_dc[1], g.f_dc[2]);
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        sphCoefficients[i + 1] = make_float3(g.f_rest[i], g.f_rest[15 + i], g.f_rest[30 + i]);
    }

    const float x = view_dir.x;
    const float y = view_dir.y;
    const float z = view_dir.z;

    // --- 1. SH 기저 함수 계산 (SHEval4 로직을 스칼라 코드로 변환) ---
    // 이 부분은 논문의 코드 생성기가 만드는 출력과 동일한 계산을 수행합니다[cite: 76].
    float basis[16]; // 16개의 SH 기저 함수 값을 저장할 배열

    float z2 = z * z;

    // m=0 항들 계산
    basis[0] = 0.2820947917738781f;
    basis[2] = 0.4886025119029199f * z;
    basis[6] = 0.9461746957575601f * z2 - 0.31539156525252f;
    basis[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);

    // m=1, m=-1 항들 계산
    float c0 = x;
    float s0 = y;

    float tmpA = -0.48860251190292f;
    basis[3] = tmpA * c0; // m=1
    basis[1] = tmpA * s0; // m=-1

    float tmpB = -1.092548430592079f * z;
    basis[7] = tmpB * c0; // m=1
    basis[5] = tmpB * s0; // m=-1

    float tmpC = 0.4570457994644658f - 2.285228997322329f * z2;
    basis[13] = tmpC * c0; // m=1
    basis[11] = tmpC * s0; // m=-1

    // m=2, m=-2 항들 계산
    // 삼각함수 덧셈 정리를 이용한 효율적인 계산 [cite: 71]
    float c1 = x * c0 - y * s0; // cos(2*phi) 관련 항
    float s1 = x * s0 + y * c0; // sin(2*phi) 관련 항

    tmpA = 0.5462742152960395f;
    basis[8] = tmpA * c1; // m=2
    basis[4] = tmpA * s1; // m=-2

    tmpB = 1.445305721320277f * z;
    basis[14] = tmpB * c1; // m=2
    basis[10] = tmpB * s1; // m=-2

    // m=3, m=-3 항들 계산
    float c2 = x * c1 - y * s1; // cos(3*phi) 관련 항
    float s2 = x * s1 + y * c1; // sin(3*phi) 관련 항

    tmpC = -0.5900435899266435f;
    basis[15] = tmpC * c2; // m=3
    basis[9] = tmpC * s2; // m=-3

    // --- 2. 기저 함수와 계수를 곱하여 최종 색상 계산 ---
    float3 rad = make_float3(0.0f, 0.0f, 0.0f);
    int numBands = (degree + 1) * (degree + 1);

#pragma unroll
    for (int i = 0; i < numBands; ++i) {
        rad += basis[i] * sphCoefficients[i];
    }


    rad += make_float3(0.5f);
    return min(max(rad, make_float3(0.f)), make_float3(1.f));
}

// 쿼터니언의 역(conjugate)을 계산
__device__ __forceinline__ float4 quat_inverse(const float4& q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

// 쿼터니언을 사용하여 벡터를 회전
__device__ __forceinline__ float3 quat_rotate(const float3& v, const float4& q) {
    float3 t = 2.0f * cross(make_float3(q.x, q.y, q.z), v);
    return v + q.w * t + cross(make_float3(q.x, q.y, q.z), t);
    //float3 q_vec = make_float3(q.x, q.y, q.z);
    //return (q.w * q.w - dot(q_vec, q_vec)) * v
    //    + 2.0f * q_vec * dot(q_vec, v)
    //    + 2.0f * q.w * cross(q_vec, v);
}

/**
 * @brief grayDist 값을 기반으로 밀도 감쇠(density falloff)를 계산
 * 표준 2차(Quadratic) 가우시안 분포를 사용
 * @param grayDist 제곱된 마할라노비스 거리
 * @return 밀도 응답 값 (0.0 ~ 1.0)
 */
template <int GeneralizedGaussianDegree = 4>
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
        constexpr/* static const */ float s = -0.329630334487f;
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
 * @brief 광선과 3D 가우시안의 상호작용을 평가하여 샘플의 투명도를 계산
 * @param ray 현재 추적 중인 광선.
 * @param g 평가할 가우시안 데이터.
 * @return 계산된 최종 샘플 투명도 (alpha).
 */
__device__ __forceinline__ float evaluateGaussianResponse(const cuRay& ray, const Gaussian& g)
{
    // 파라미터 정리
    const float3 g_pos = make_float3(g.pos[0], g.pos[1], g.pos[2]);   // μ
    const float3 g_scale = make_float3(g.scale[0], g.scale[1], g.scale[2]); // 대각 S (표준편차)
#if QUATERNION
    const float4 g_rot = make_float4(g.rot[1], g.rot[2], g.rot[3], g.rot[0]); // (x,y,z,w)
    const float4 inv_rot = quat_inverse(g_rot);

    // 월드 → 가우시안 정렬좌표로 회전(R^T)한 뒤, 스케일의 역수(S^{-1})를 적용
    float3 o_os = quat_rotate(ray.pos - g_pos, inv_rot);                                  //R^T (o-μ)
    float3 d_os = quat_rotate(ray.dir, inv_rot);                                          //R^T d
#else
    const float3 p = ray.pos - g_pos; // (o - μ)

    // 위치 벡터 회전: o_os = R^T * p
    float3 o_os;
    o_os.x = g.rotMat.m[0][0] * p.x + g.rotMat.m[0][1] * p.y + g.rotMat.m[0][2] * p.z;
    o_os.y = g.rotMat.m[1][0] * p.x + g.rotMat.m[1][1] * p.y + g.rotMat.m[1][2] * p.z;
    o_os.z = g.rotMat.m[2][0] * p.x + g.rotMat.m[2][1] * p.y + g.rotMat.m[2][2] * p.z;

    // 방향 벡터 회전: d_os = R^T * d
    float3 d_os;
    d_os.x = g.rotMat.m[0][0] * ray.dir.x + g.rotMat.m[0][1] * ray.dir.y + g.rotMat.m[0][2] * ray.dir.z;
    d_os.y = g.rotMat.m[1][0] * ray.dir.x + g.rotMat.m[1][1] * ray.dir.y + g.rotMat.m[1][2] * ray.dir.z;
    d_os.z = g.rotMat.m[2][0] * ray.dir.x + g.rotMat.m[2][1] * ray.dir.y + g.rotMat.m[2][2] * ray.dir.z;
#endif
    // o_g = S^{-1} R^T (o - μ),  d_g = S^{-1} R^T d
    float3 o_g = make_float3(o_os.x / g_scale.x, o_os.y / g_scale.y, o_os.z / g_scale.z); //S^-1 R^T (o-μ)
    float3 d_g = make_float3(d_os.x / g_scale.x, d_os.y / g_scale.y, d_os.z / g_scale.z); //S^-1 R^T d


    // τ_max = - (o_g·d_g) / (d_g·d_g)  (식 8)
    float denom = dot(d_g, d_g);
    // 안전장치: 방향이 너무 작으면 밀도는 원점에서 평가
    float tau = (denom > 1e-8f) ? (-dot(o_g, d_g) / denom) : 0.0f;

    // τ_max 위치의 가우시안 밀도: ρ = exp(-0.5 * ||o_g + τ_max d_g||^2)
    //float3 p_g = ray.pos + tau * ray.dir;
    float3 p_g = o_g + tau * d_g;

    //float  expo = -0.5f * dot(p_g, p_g);
    //float  rho = expf(expo);

    float  expo = dot(p_g, p_g);
    const float rho = particleResponse<GAUSSIAN_DEGREE>(expo);

    return g.opacity * rho;
}

/**
 * @brief 3dgrt 원본 코드의 cross product 방식을 사용하여
 * 광선과 가우시안의 최대 응답을 계산하고, 최종 샘플 불투명도를 반환
 * @param ray 렌더링에 사용되는 원본 광선
 * @param g 평가 대상 가우시안
 * @return 계산된 최종 샘플 불투명도
 */
__device__ __forceinline__ float evaluateGaussianResponse_3dgrt(const cuRay& ray, const Gaussian& g)
{
    // 가우시안 파라미터 준비
    const float3 g_pos = make_float3(g.pos[0], g.pos[1], g.pos[2]);
    const float3 g_scale = make_float3(g.scale[0], g.scale[1], g.scale[2]);
    const float3 gposc = ray.pos - g_pos; // (o - μ)
    // 광선을 가우시안의 로컬 좌표계로 변환 (회전 및 스케일링)
#if QUATERNION
    const float4 g_rot = make_float4(g.rot[1], g.rot[2], g.rot[3], g.rot[0]);
    const float4 inv_rot = quat_inverse(g_rot);
    const float3 gposcr = quat_rotate(gposc, inv_rot); // R^T * (o - μ)
    const float3 rayDirR = quat_rotate(ray.dir, inv_rot); // R^T * d
#else
    float3 gposcr; // R^T * (o - μ)
    gposcr.x = g.rotMat.m[0][0] * gposc.x + g.rotMat.m[0][1] * gposc.y + g.rotMat.m[0][2] * gposc.z;
    gposcr.y = g.rotMat.m[1][0] * gposc.x + g.rotMat.m[1][1] * gposc.y + g.rotMat.m[1][2] * gposc.z;
    gposcr.z = g.rotMat.m[2][0] * gposc.x + g.rotMat.m[2][1] * gposc.y + g.rotMat.m[2][2] * gposc.z;

    float3 rayDirR; // R^T * d
    rayDirR.x = g.rotMat.m[0][0] * ray.dir.x + g.rotMat.m[0][1] * ray.dir.y + g.rotMat.m[0][2] * ray.dir.z;
    rayDirR.y = g.rotMat.m[1][0] * ray.dir.x + g.rotMat.m[1][1] * ray.dir.y + g.rotMat.m[1][2] * ray.dir.z;
    rayDirR.z = g.rotMat.m[2][0] * ray.dir.x + g.rotMat.m[2][1] * ray.dir.y + g.rotMat.m[2][2] * ray.dir.z;
#endif

    const float3 gro = gposcr / g_scale; //o_g

    const float3 grdu = rayDirR / g_scale; //d_g
    const float3 grd = normalize(grdu);

    // cross product를 이용해 grayDist(제곱된 마할라노비스 거리) 계산
    const float3 gcrod = cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    // particleResponse 함수를 통해 밀도 계산
    const float density = particleResponse<GAUSSIAN_DEGREE>(grayDist);

    // 기본 불투명도와 밀도를 곱하여 최종 결과 반환
    return fminf(0.99f, g.opacity * density);
}

//world to local space
__device__ __forceinline__ void quaternionWXYZToMatrixTranspose(const float4& q, float33& ret) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float rx = r * x;
    const float ry = r * y;
    const float rz = r * z;

    // Compute rotation matrix from quaternion
    ret[0] = make_float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry));
    ret[1] = make_float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx));
    ret[2] = make_float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy)));
}

//__device__ __forceinline__ float evaluateGaussianResponse_origin(const cuRay& ray, const Gaussian& g)
//{
//    // 가우시안 파라미터 준비
//    const float3 particlePosition = make_float3(g.pos[0], g.pos[1], g.pos[2]);
//    const float3 particleScale = make_float3(g.scale[0], g.scale[1], g.scale[2]);
//    //float4 particleQquaternion = make_float4(g.rot[1], g.rot[2], g.rot[3], g.rot[0]);
//    float4 particleQquaternion = make_float4(g.rot[0], g.rot[1], g.rot[2], g.rot[3]);
//    float33 particleRotation;
//    quaternionWXYZToMatrixTranspose(particleQquaternion, particleRotation);
//    // 광선을 가우시안의 로컬 좌표계로 변환 (회전 및 스케일링)
//    const float3 giscl = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
//    const float3 gposc = (ray.pos - particlePosition);
//    const float3 gposcr = (gposc * particleRotation);
//    const float3 gro = giscl * gposcr;
//    const float3 rayDirR = ray.dir * particleRotation;
//    const float3 grdu = giscl * rayDirR;
//    const float3 grd = normalize(grdu);
//
//    // cross product를 이용해 grayDist(제곱된 마할라노비스 거리) 계산
//    const float3 gcrod = cross(grd, gro);
//    const float grayDist = dot(gcrod, gcrod);
//
//    // particleResponse 함수를 통해 밀도 계산
//    const float gres = particleResponse<GAUSSIAN_DEGREE>(grayDist);
//
//    // 기본 불투명도와 밀도를 곱하여 최종 결과 반환
//    return fminf(0.99f, g.opacity * gres);
//}

__device__ __forceinline__ float3 eval_sh_final_ptr(
    const int degree,
    const float3& view_dir,
    const Gaussian* g
) {
    // 계산을 용이하게 하기 위해 g.f_dc와 g.f_rest를 하나의 배열로 합침
    float3 sphCoefficients[16];
    sphCoefficients[0] = make_float3(g->f_dc[0], g->f_dc[1], g->f_dc[2]);
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        sphCoefficients[i + 1] = make_float3(g->f_rest[i], g->f_rest[15 + i], g->f_rest[30 + i]);
    }

    // --- 3dgrt의 radianceFromSpH 로직을 그대로 적용 ---
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

    // 최종 활성화: 원본과 동일하게 0.5를 더하고, 0 미만 값은 0으로 클램핑
    rad += make_float3(0.5f);
    return min(max(rad, make_float3(0.f)), make_float3(1.f));
}

//WALD_METHOD == false
__device__ void singlePassIntersectRoutineGaussian_sortNode(const cuRay& ray, int id, float t_near, float t_far, HitRecord* local_hits, int& local_hit_count
    //, cudaTextureObject_t inTriAccelTex
) {
    if (local_hit_count >= MAX_HITS) return;
#if TRIACC_TEXTURE
    float4 d0 = tex1Dfetch<float4>(inTriAccelTex, id * 4 + 0);
    float4 d1 = tex1Dfetch<float4>(inTriAccelTex, id * 4 + 1);
    float4 d2 = tex1Dfetch<float4>(inTriAccelTex, id * 4 + 2);
#else
    float4 d0 = g_d_tri_acc_dev[id * 4 + 0];
    float4 d1 = g_d_tri_acc_dev[id * 4 + 1];
    float4 d2 = g_d_tri_acc_dev[id * 4 + 2];
#endif

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
    //if (isnan(t)) return;
    //if ((t < t_near - EPSILON4) | (t > t_far + EPSILON4)) return;
     if (t <= t_near || t >= t_far) return;

    float u_coord = p_pos.x + t * p_dir.x;
    float v_coord = p_pos.y + t * p_dir.y;

    float beta = u_coord * d1.x + v_coord * d1.y + d1.z;
    float gamma = u_coord * d2.x + v_coord * d2.y + d2.z;
    if ((beta < 0.f - BARYCENTRY_EPSILON) || (gamma < 0.f - BARYCENTRY_EPSILON) ||
        ((1.0f - beta - gamma) < 0.0f - BARYCENTRY_EPSILON)) return;

#if TRIACC_TEXTURE
    float4 N_packed = tex1Dfetch<float4>(inTriAccelTex, id * 4 + 3);
#else
    float4 N_packed = g_d_tri_acc_dev[id * 4 + 3];
#endif
    float3 N = make_float3(N_packed.x, N_packed.y, N_packed.z);

    // 법선 벡터와 광선 방향의 내적(dot product)을 계산
    // 내적 값이 0보다 크면 광선이 삼각형의 뒷면
    if (dot(N, ray.dir) > 0.0f) {
        return; // 뒷면이므로 이 충돌을 무시
    }

    local_hits[local_hit_count].t = t;
    local_hits[local_hit_count].primIndex = id;
    local_hit_count++;
}

#if PRIMITIVE_TYPE == ELLIPSOID || PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
//only translate, rotation done
//no scaling
__device__ inline float ellipsoidIntersect(const float3& ocn, const float3& rdn) {
    //float3 safeScale = scale;
    //safeScale.x = max(scale.x, EPSILON);
    //safeScale.y = max(scale.y, EPSILON);
    //safeScale.z = max(scale.z, EPSILON);
    //float3 ocn = pos / safeScale;
    //float3 rdn = dir / safeScale;
    float a = dot(rdn, rdn);
    float b = dot(ocn, rdn);
    float c = dot(ocn, ocn);
    float h = b * b - a * (c - 1.0);
    //if (h < 0.0) return -1.0;

    //if (h < -EPSILON4) return -1.0;
    //h = sqrt(h) + EPSILON3;
    //return (-b-h) / a;

    float t1, t2;
    if (h < -EPSILON4) return -1.0;
    //else if (h == 0) return -0.5 * b / a;
    else {
        //float q = (b > 0) ? -(b + sqrt(h)) : -(b - sqrt(h));
        //t1 = q / a;
        //t2 = (c-1.0) / q;

        if (b > 0) {
            float q = -(b + sqrt(h));
            //t1 = c / q;
            t2 = q / a;
        }
        else {
            float q = -(b - sqrt(h));
            //t1 = q / a;
            t2 = (c-1) / q;
        }
    }
    //return (t1 <  t2) ? t1 : t2;
    return t2;
}

__device__ inline float calculateKernelScale(float density, float kernelMinResponse = KERNEL_MIN_RESPONSE, uint32_t opts = 1, float kernelDegree = KERNEL_DEGREE) {
    const float responseModulation = (opts & 1 /* MOGRenderAdaptiveKernelClamping */) ? density : 1.0f;
    const float minResponse = min(kernelMinResponse / responseModulation, 0.97f);

    const float b = kernelDegree;
    const float a = -4.5f / pow(3.0f, b);

    // 3. e^{a * r^b} = minResponse 를 만족하는 r(반지름) 계산
    // r = (ln(minResponse) / a)^(1/b)
    return pow(log(minResponse) / a, 1.0f / b);
}
#if VOLUME_ISECT
__device__ inline void rayPrimIntersect(const cuRay& currRay, const unsigned id
    , const float t_near, const float t_far
    , HitRecord* local_hits, int& local_hit_count) {
    //fetch gaussian
    Gaussian g;
    float4* g_as_float4 = reinterpret_cast<float4*>(&g);
    const int num_float4s = sizeof(Gaussian) / sizeof(float4);
    int base_idx = id * num_float4s;
#pragma unroll
#if QUATERNION
    for (int i = 0; i < 3; ++i) {
#else
    for (int i = 0; i < 4; i++) {
#endif
        g_as_float4[i] = tex1Dfetch<float4>(inGaussianTex, base_idx + i);
    }

    const float3 particlePosition = make_float3(g.pos[0], g.pos[1], g.pos[2]);
#if UPLOAD_INV_SCALE
    float3 giscl = make_float3(g.scale[0], g.scale[1], g.scale[2]);
    float3 particleScale = 1 / giscl;
#else
    float3 particleScale = make_float3(g.scale[0], g.scale[1], g.scale[2]);
    float3 giscl = 1 / particleScale;
#endif
    float33 particleRotation;
#if QUATERNION
    float4 particleQquaternion = make_float4(g.rot[0], g.rot[1], g.rot[2], g.rot[3]);
    quaternionWXYZToMatrixTranspose(particleQquaternion, particleRotation);
#else
    #if !DIRECT_ROT_CALC
    particleRotation[0] = make_float3(g.rotMat.m[0][0], g.rotMat.m[1][0], g.rotMat.m[2][0]);
    particleRotation[1] = make_float3(g.rotMat.m[0][1], g.rotMat.m[1][1], g.rotMat.m[2][1]);
    particleRotation[2] = make_float3(g.rotMat.m[0][2], g.rotMat.m[1][2], g.rotMat.m[2][2]);
    #endif
#endif

    const float3 gposc = (currRay.pos - particlePosition);
#if QUATERNION || !DIRECT_ROT_CALC
    const float3 gposcr = (gposc * particleRotation);
    const float3 gro = giscl * gposcr;
    const float3 rayDirR = currRay.dir * particleRotation;
    const float3 grdu = rayDirR * giscl;
#else
    #if UPLOAD_INVSR_MAT
    const float3 gro = multMatrixTransposeVector(gposc, g.rotMat);
    const float3 grdu = multMatrixTransposeVector(currRay.dir, g.rotMat);
    #else
    const float3 gposcr = multMatrixTransposeVector(gposc, g.rotMat);
    const float3 gro = giscl * gposcr;
    const float3 rayDirR = multMatrixTransposeVector(currRay.dir, g.rotMat);
    const float3 grdu = rayDirR * giscl;
    #endif
#endif
    const float3 grd = safe_normalize(grdu);
    const float grp = -dot(grd, gro);
    const float3 grds = particleScale * grd * grp;
    float t = (grp < 0.f ? -1.f : 1.f) * sqrtf(dot(grds, grds));
    if (t < t_near || t > t_far) return;
    const float3 gcrod = cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);
    if (grayDist < 8.f) {
        local_hits[local_hit_count].t = t;
        local_hits[local_hit_count].primIndex = id;
#if STORE_GRAYDIST
        local_hits[local_hit_count].grayDist = grayDist;
#endif
        local_hit_count++;
    }
    return;
}
#else   //VOLUME_ISECT
__device__ inline void rayPrimIntersect(const cuRay& currRay, const unsigned id
    , const float t_near, const float t_far
    , HitRecord* local_hits, int& local_hit_count) {
    //fetch gaussian
    Gaussian g;
    float4* g_as_float4 = reinterpret_cast<float4*>(&g);
    const int num_float4s = sizeof(Gaussian) / sizeof(float4);
    int base_idx = id * num_float4s;
#pragma unroll
#if QUATERNION
    for (int i = 0; i < 3; ++i) {
#else
    for (int i = 0; i < 4; i++){
#endif
        g_as_float4[i] = tex1Dfetch<float4>(inGaussianTex, base_idx + i);
    }
    
    const float3 particlePosition = make_float3(g.pos[0], g.pos[1], g.pos[2]);
    float3 giscl = make_float3(g.scale[0], g.scale[1], g.scale[2]);
    float33 particleRotation;
    float k_scale;
#if QUATERNION
    float4 particleQquaternion = make_float4(g.rot[0], g.rot[1], g.rot[2], g.rot[3]);
    quaternionWXYZToMatrixTranspose(particleQquaternion, particleRotation);
    k_scale = g.k_scale;
#else
    #if !DIRECT_ROT_CALC
    particleRotation[0] = make_float3(g.rotMat.m[0][0], g.rotMat.m[1][0], g.rotMat.m[2][0]);
    particleRotation[1] = make_float3(g.rotMat.m[0][1], g.rotMat.m[1][1], g.rotMat.m[2][1]);
    particleRotation[2] = make_float3(g.rotMat.m[0][2], g.rotMat.m[1][2], g.rotMat.m[2][2]);
    #endif
    #if PRE_CALC_KSCALE
        k_scale = tex1Dfetch<float>(inKScaleTex, id);
    #else
        k_scale = calculateKernelScale(g.opacity);
    #endif
#endif

#if UPLOAD_INV_SCALE
    giscl = giscl;
#else
    giscl = 1 / giscl;
#endif
#if UPLOAD_INV_KSCALE
    giscl = giscl * k_scale;
#else
    giscl = giscl / k_scale;
#endif
    const float3 gposc = (currRay.pos - particlePosition);
#if QUATERNION || !DIRECT_ROT_CALC
    const float3 gposcr = (gposc * particleRotation);
    const float3 gro = giscl * gposcr;
    const float3 rayDirR = currRay.dir * particleRotation;
    const float3 grd = rayDirR * giscl;
#else
    #if UPLOAD_INVSR_MAT
    k_scale = 1 / k_scale;
    const float3 gro = multMatrixTransposeVector(gposc, g.rotMat) * (k_scale);
    const float3 grd = multMatrixTransposeVector(currRay.dir, g.rotMat) * (k_scale);
    #else
    const float3 gposcr = multMatrixTransposeVector(gposc, g.rotMat);
    const float3 gro = giscl * gposcr;
    const float3 rayDirR = multMatrixTransposeVector(currRay.dir, g.rotMat);
    const float3 grd = rayDirR * giscl;
    #endif
#endif

    float t = ellipsoidIntersect(gro, grd);
    if (t < t_near || t > t_far) return;
#if STORE_GRAYDIST
    const float3 grdn = normalize(grd);
    const float3 gron = gro * k_scale;
    const float3 gcrod = cross(grdn, gron);
    const float grayDist = dot(gcrod, gcrod);
    local_hits[local_hit_count].grayDist = grayDist;
#endif

    local_hits[local_hit_count].t = t;
    local_hits[local_hit_count].primIndex = id;
    local_hit_count++;
}
#endif //VOLUME_ISECT
#elif PRIMITIVE_TYPE == TRI
__device__ inline void rayPrimIntersect(const cuRay& ray, const int id,
    const float t_near, const float t_far
    , HitRecord* local_hits, int& local_hit_count) {
    cuWaldTriangleInfo tri;
#if TRIACC_TEXTURE
    tri.internal0 = tex1Dfetch<float4>(inTriAccelTex, 3 * id);
    tri.internal1 = tex1Dfetch<float4>(inTriAccelTex, 3 * id + 1);
    tri.internal2 = tex1Dfetch<float4>(inTriAccelTex, 3 * id + 2);
#else
    tri.internal0 = g_d_tri_acc_dev[3 * id];
    tri.internal1 = g_d_tri_acc_dev[3 * id + 1];
    tri.internal2 = g_d_tri_acc_dev[3 * id + 2];
#endif
    cuWaldTriangleInfo::perm_t p = tri.get_perm(ray);
    p.pos.x = (tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z);
    const float denum = (p.dir.x + tri.n_u() * p.dir.y + tri.n_v() * p.dir.z);
    int flag = __float_as_int(tri.internal2.z);
    if (denum * (float)flag > 0.0f) return; //뒷면 확인
    
    const float t = __fdividef(p.pos.x, denum);
    if (isnan(t)) return;
    if ((t < t_near - EPSILON4) | (t > t_far + EPSILON4)) return;
    /**
    * culling 옵션이 있고, object 가 transparent 하지 않다면
    * 앞면인지 뒷면인지 체크. 뒷면에 맞은거면 hit 처리 안함.
    */
    const float hu = p.pos.y + t * p.dir.y - tri.vert_ku();
    const float hv = p.pos.z + t * p.dir.z - tri.vert_kv();
    const float beta = hv * tri.b_nu() + hu * tri.b_nv();
    const float gamma = hu * tri.c_nu() + hv * tri.c_nv();
    /** 삼각형의 edge 와 부딪힐때, 수치오차가 있으므로 epsilon 을 좀 준다. */
    if ((beta < 0.f - BARYCENTRY_EPSILON) | (gamma < 0.f - BARYCENTRY_EPSILON) |
        ((1.0f - beta - gamma) < 0.0f - BARYCENTRY_EPSILON)) return;
    

    local_hits[local_hit_count].t = t;
    local_hits[local_hit_count].primIndex = id;
    local_hit_count++;
}
#endif //PRIMITIVE_TYPE


// --- Mailbox Optimization Structure ---
#define MAILBOX_SIZE 6
#define MAILBOX_MASK (MAILBOX_SIZE - 1)
#define PROBE_LENGTH 4  // 충돌 시 몇 칸까지 더 찾아볼지 (4칸 정도면 충분)

//struct Mailbox {
//    int cached_ids[MAILBOX_SIZE]; // 방문한 Gaussian ID 저장
//
//    __device__ void init() {
//        // 루프 언롤링: 컴파일러가 레지스터에 즉시 할당하도록 유도
//#pragma unroll
//        for (int i = 0; i < MAILBOX_SIZE; ++i) {
//            cached_ids[i] = -1; // -1: 비어있음
//        }
//    }
//
//    // 캐시 조회: ID가 있으면 true (이미 처리했음)
//    __device__ bool contains(int gaussianID) {
//        // [비트 연산 최적화] '%' 대신 '&' 사용
//        int idx = gaussianID & MAILBOX_MASK;
//        // Tag Check: 해시 충돌 방지
//        return (cached_ids[idx] == gaussianID);
//    }
//
//    // 캐시 등록: "이 가우시안은 처리 완료"
//    __device__ void mark(int gaussianID) {
//        int idx = gaussianID & MAILBOX_MASK;
//        cached_ids[idx] = gaussianID;
//    }
//};
struct Mailbox {
    int cached_ids[MAILBOX_SIZE];

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < MAILBOX_SIZE; ++i) {
            cached_ids[i] = -1;
        }
    }

    // 캐시 조회: 내 ID가 저장되어 있는지 주변 4칸을 뒤져봄
    __device__ __forceinline__ bool contains(int gaussianID) {
        int base_idx = gaussianID & MAILBOX_MASK;

        // Loop Unrolling으로 성능 최적화
#pragma unroll
        for (int i = 0; i < PROBE_LENGTH; ++i) {
            // 순환 구조: 15번 다음은 0번
            int curr_idx = (base_idx + i) & MAILBOX_MASK;

            if (cached_ids[curr_idx] == gaussianID) {
                return true; // 찾았다!
            }
        }
        return false; // 없다.
    }

    // 캐시 등록: 빈칸을 찾아 넣거나, 없으면 덮어씀
    __device__ __forceinline__ void mark(int gaussianID) {
        int base_idx = gaussianID & MAILBOX_MASK;

        // 1. 빈칸(-1)이 있는지 먼저 확인
#pragma unroll
        for (int i = 0; i < PROBE_LENGTH; ++i) {
            int curr_idx = (base_idx + i) & MAILBOX_MASK;

            // 이미 내가 등록되어 있거나, 빈칸이면 거기에 저장하고 종료
            if (cached_ids[curr_idx] == gaussianID || cached_ids[curr_idx] == -1) {
                cached_ids[curr_idx] = gaussianID;
                return;
            }
        }

        // 2. 빈칸이 없으면(꽉 찼으면), 그냥 원래 자리(base_idx)를 덮어씀 (Eviction)
        cached_ids[base_idx] = gaussianID;
    }
};
__device__ inline void singlePassIntersectRoutineMailBox(
    const cuRay& ray,
    const int id, // primIndex
    const float t_near,
    const float t_far,
    HitRecord* local_hits,
    int& local_hit_count,
    Mailbox* mailbox
) {
    // 1. 최대 히트 수 체크
    if (local_hit_count >= MAX_HITS) return;

    // 2. Gaussian ID 추출 (가장 저렴한 메모리 접근)
    // internal2.w에 Gaussian ID가 있음
#if TRIACC_TEXTURE
    float4 internal2 = tex1Dfetch<float4>(inTriAccelTex, 3 * id + 2);
#else
    float4 internal2 = g_d_tri_acc_dev[3 * id + 2];
#endif
    int gaussianID = __float_as_int(internal2.w);

    // 3. Mailbox 확인 (핵심 최적화)
    if (mailbox->contains(gaussianID)) {
        return; // 이미 처리했으므로 중복 연산 및 렌더링 방지
    }

    // 4. 직접 교차 검사 수행 (singlePassIntersectRoutine 로직 복사)
    cuWaldTriangleInfo tri;
    // tri.internal2는 위에서 이미 읽었으므로 재사용
    tri.internal2 = internal2;
    // 나머지 데이터 로드
#if TRIACC_TEXTURE
    tri.internal0 = tex1Dfetch<float4>(inTriAccelTex, 3 * id + 0);
    tri.internal1 = tex1Dfetch<float4>(inTriAccelTex, 3 * id + 1);
#else
    tri.internal0 = g_d_tri_acc_dev[3 * id + 0];
    tri.internal1 = g_d_tri_acc_dev[3 * id + 1];
#endif

    // Permutation 및 교차점 계산
    cuWaldTriangleInfo::perm_t p = tri.get_perm(ray);
    p.pos.x = (tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z);
    const float denum = (p.dir.x + tri.n_u() * p.dir.y + tri.n_v() * p.dir.z);

    // Backface Culling 및 평행 검사
    int flag = __float_as_int(tri.internal2.z);
    if (denum * (float)flag > 0.0f) {
        return; // 뒷면이므로 실패 -> Mailbox에 등록하지 않음 (나중에 앞면 만날 기회 줌)
    }

    const float t = __fdividef(p.pos.x, denum);
    if (isnan(t)) return;

    // 거리(Spatial) 검사
    // t가 유효하더라도 현재 노드 범위 밖이면, 
    // "이 노드에서는 안 그림" (다른 노드에서 그릴 것임) -> Mailbox 등록 안 함
    if ((t < t_near - EPSILON4) | (t > t_far + EPSILON4)) return;

    // Barycentric 좌표 검사 (삼각형 내부인지 확인)
    const float hu = p.pos.y + t * p.dir.y - tri.vert_ku();
    const float hv = p.pos.z + t * p.dir.z - tri.vert_kv();
    const float beta = hv * tri.b_nu() + hu * tri.b_nv();
    const float gamma = hu * tri.c_nu() + hv * tri.c_nv();

    if ((beta < 0.f - BARYCENTRY_EPSILON) | (gamma < 0.f - BARYCENTRY_EPSILON) |
        ((1.0f - beta - gamma) < 0.0f - BARYCENTRY_EPSILON)) {
        return; // ★ 삼각형 밖이므로 실패 -> Mailbox에 등록하지 않음
    }

    // -----------------------------------------------------------
    // [성공] 유효한 앞면(Front-face) Hit!
    // -----------------------------------------------------------

    // 1. 결과 저장 (perm 생략)
    local_hits[local_hit_count].t = t;
    local_hits[local_hit_count].primIndex = id;

    local_hit_count++;

    // 2. Mailbox에 "방문 완료" 마킹
    // 이제부터 이 Ray가 끝날 때까지, 같은 ID를 가진 다른 삼각형(뒷면 등)이나
    // 다른 노드에서 마주치는 중복 가우시안은 모두 무시됨.
    mailbox->mark(gaussianID);
}

#if BLEND_SELECT
__device__ inline void singlePassIntersectPlane(const cuRay& ray, const int id,
    const float t_near, const float t_far
    , HitRecord* local_hits, int& local_hit_count) {
    cuWaldTriangleInfo tri;
    tri.internal0 = tex1Dfetch<float4>(inTriAccelTex, 3 * id);
    //tri.internal1 = tex1Dfetch<float4>(inTriAccelTex, 3 * id + 1);
    //tri.internal2 = tex1Dfetch<float4>(inTriAccelTex, 3 * id + 2);
    cuWaldTriangleInfo::perm_t p = tri.get_perm(ray);
    p.pos.x = (tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z);
    const float denum = (p.dir.x + tri.n_u() * p.dir.y + tri.n_v() * p.dir.z);
    //if (denum > 0.0f) return; //뒷면 확인
    int flag = __float_as_int(tri.internal2.z);
    if (denum * (float)flag > 0.0f) return;
    const float t = __fdividef(p.pos.x, denum);
    if (isnan(t)) return;
    if ((t < t_near - EPSILON4) | (t > t_far + EPSILON4)) return;
    /**
    * culling 옵션이 있고, object 가 transparent 하지 않다면
    * 앞면인지 뒷면인지 체크. 뒷면에 맞은거면 hit 처리 안함.
    */

    local_hits[local_hit_count].t = t;
    local_hits[local_hit_count].primIndex = id;
    local_hits[local_hit_count].perm = p;
    local_hit_count++;
}

__device__ inline int singlePassIntersectCheck(const cuRay& ray, HitRecord local_hits) {
    cuWaldTriangleInfo tri;
    tri.internal1 = tex1Dfetch<float4>(inTriAccelTex, 3 * local_hits.primIndex + 1);
    tri.internal2 = tex1Dfetch<float4>(inTriAccelTex, 3 * local_hits.primIndex + 2);
    cuWaldTriangleInfo::perm_t p = local_hits.perm;
    const float t = local_hits.t;
    //이미 isnan, t_near, tfar 검사함
    const float hu = p.pos.y + t * p.dir.y - tri.vert_ku();
    const float hv = p.pos.z + t * p.dir.z - tri.vert_kv();
    const float beta = hv * tri.b_nu() + hu * tri.b_nv();
    const float gamma = hu * tri.c_nu() + hv * tri.c_nv();
    /** 삼각형의 edge 와 부딪힐때, 수치오차가 있으므로 epsilon 을 좀 준다. */
    if ((beta < 0.f - BARYCENTRY_EPSILON) | (gamma < 0.f - BARYCENTRY_EPSILON) |
        ((1.0f - beta - gamma) < 0.0f - BARYCENTRY_EPSILON)) return -1;

    return __float_as_int(tri.internal2.w);
}
#endif
#if DEBUG_LEAF_CUDA
__device__ float3 turboColormap(float x)
{
    const float4 kRedVec4 = make_float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const float4 kGreenVec4 = make_float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const float4 kBlueVec4 = make_float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const float2 kRedVec2 = make_float2(-152.94239396, 59.28637943);
    const float2 kGreenVec2 = make_float2(4.27729857, 2.82956604);
    const float2 kBlueVec2 = make_float2(-89.90310912, 27.34824973);

    
    x = __saturatef(x);
    float4 v4 = make_float4(1.0, x, x * x, x * x * x);
    float2 v2 = make_float2(v4.z * v4.z, v4.w * v4.z);
    return make_float3(
        dot(v4, kRedVec4) + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4) + dot(v2, kBlueVec2));
}

__device__ float3 computeVisualizationColor(float value, float2 minMax) {
    float normalized_val = (value - minMax.x) / (minMax.y - minMax.x);
    return turboColormap(__saturatef((value - minMax.x) / (minMax.y - minMax.x)));
}
#endif

#if PRIMITIVE_TYPE == ELLIPSOID || PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
__device__ void singlePassIntersectGaussian_sortNode_onlyShortStack(
    cuRay& currRay,
    float3& accumulated_color,
    float& accumulated_opacity
#if HIT_AND_NODE_COUNT_DEBUG
    , int& node_visits
    , int& leaf_visits
    , int& intersection_tests
    , int& hits_found
    , int& blend_ops
    , int& max_sort_size
#endif
) {
#if HIT_AND_NODE_COUNT_DEBUG
    node_visits = 0;
    leaf_visits = 0;
    intersection_tests = 0;
    hits_found = 0;
    blend_ops = 0;
    max_sort_size = 0;
#endif

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
#if DEBUG_LEAF_CUDA
    int maxLeaf = 0;
#endif

    float t_scene_near = RAY_START_EPSILON, t_scene_far = FLT_MAX;
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, &currRay, &t_scene_near, &t_scene_far)) {
        kdtreeNode node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, 0);
        float t_near = t_scene_near, t_far = t_scene_far;
        // Kd-tree 순회를 위한 스택 초기화
        shortStack cache;
        //ShortStackCache cache;
        cache.init(threadIdx.y * blockDim.x + threadIdx.x);

        while (accumulated_opacity < OPACITY_THRESHOLD) { //while (true) {
            //traverse internal nodes
            while (!IS_LEAF(node)) {
                const unsigned childOffset = FIRST_CHILD_OFFSET(node);

                const float2 pos_dir = currRay.get_dir_pos(SPLIT_AXIS(node));

                const float t_split = __fdividef(SPLIT_POS(node) - pos_dir.x, pos_dir.y); //__fdividef() : faster than "/" but less precise
                const unsigned sign = signbit(pos_dir.y);

                unsigned idx = childOffset + (sign ^ (t_split <= t_near));

                if (t_near < t_split && t_split < t_far) {
                    cache.push(childOffset + (sign ^ 1), t_far);
                    t_far = t_split;
                }
#if HIT_AND_NODE_COUNT_DEBUG
                node_visits++;
#endif
                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, idx);
            }

            // --- 리프 노드 처리 로직 ---
            unsigned int baseOffset = OBJECTLIST_OFFSET(node);
            int objectSize = OBJECT_SIZE(node) + baseOffset;
#if DEBUG_LEAF_CUDA
            if (OBJECT_SIZE(node) > maxLeaf) {
                maxLeaf = OBJECT_SIZE(node);
            }
#endif

            //if (count <= 0) continue;
                // 수집: 이 리프 노드 내의 모든 충돌을 임시 로컬 배열에 저장
            HitRecord local_hits[MAX_HITS];
            int local_hit_count = 0;

#if HIT_AND_NODE_COUNT_DEBUG
            leaf_visits++;
#endif

            for (; baseOffset < objectSize; baseOffset++) {
#if OFFSET_TEXTURE
                const unsigned primIdx = tex1Dfetch<unsigned int>(inObjectOffsetListTex, baseOffset);
#else
                const unsigned primIdx = g_d_tri_offsets_dev[baseOffset];
#endif
                rayPrimIntersect(currRay, primIdx, t_near, t_far, local_hits, local_hit_count);
            }

#if HIT_AND_NODE_COUNT_DEBUG
            intersection_tests += OBJECT_SIZE(node);
            hits_found += local_hit_count;
            max_sort_size = MyMAX(max_sort_size, local_hit_count);
#endif

            if (local_hit_count > 0) {
                sortHits(local_hits, local_hit_count);

                for (int i = 0; i < local_hit_count; ++i) {
                    int gaussianID = 0;
                    gaussianID = local_hits[i].primIndex;
#if GAUSSIAN_TEXTURE
                    Gaussian g = fetch_gaussian(gaussianID);
#else
                    Gaussian g = g_d_gaussians[gaussianID];
#endif
#if !STORE_GRAYDIST
                    const float3 particlePosition = make_float3(g.pos[0], g.pos[1], g.pos[2]);
    #if UPLOAD_INV_SCALE
                    const float3 giscl = make_float3(g.scale[0], g.scale[1], g.scale[2]);
    #else
                    const float3 giscl = make_float3(1/g.scale[0], 1/g.scale[1], 1/g.scale[2]);
    #endif
                    float33 particleRotation;
    #if QUATERNION
                    float4 particleQquaternion = make_float4(g.rot[0], g.rot[1], g.rot[2], g.rot[3]);
                    quaternionWXYZToMatrixTranspose(particleQquaternion, particleRotation);
    #else
        #if !DIRECT_ROT_CALC
                    particleRotation[0] = make_float3(g.rotMat.m[0][0], g.rotMat.m[1][0], g.rotMat.m[2][0]);
                    particleRotation[1] = make_float3(g.rotMat.m[0][1], g.rotMat.m[1][1], g.rotMat.m[2][1]);
                    particleRotation[2] = make_float3(g.rotMat.m[0][2], g.rotMat.m[1][2], g.rotMat.m[2][2]);
        #endif
    #endif

                    const float3 gposc = (currRay.pos - particlePosition);
    #if QUATERNION || !DIRECT_ROT_CALC
                    const float3 gposcr = (gposc * particleRotation);
                    const float3 gro = giscl * gposcr;
                    const float3 rayDirR = currRay.dir * particleRotation;
                    const float3 grdu = giscl * rayDirR;
                    const float3 grd = normalize(grdu);
    #else
        #if UPLOAD_INVSR_MAT
                    const float3 gro = multMatrixTransposeVector(gposc, g.rotMat);
                    const float3 grd = normalize(multMatrixTransposeVector(currRay.dir, g.rotMat));
        #else
                    const float3 gposcr = multMatrixTransposeVector(gposc, g.rotMat);
                    const float3 gro = giscl * gposcr;
                    const float3 rayDirR = multMatrixTransposeVector(currRay.dir, g.rotMat);
                    const float3 grd = normalize(rayDirR * giscl);
        #endif
    #endif
                    const float3 gcrod = cross(grd, gro);
                    const float grayDist = dot(gcrod, gcrod);
                    const float gres = particleResponse<GAUSSIAN_DEGREE>(grayDist);
#else
                    const float gres = particleResponse<GAUSSIAN_DEGREE>(local_hits[i].grayDist);
#endif
                    float sample_opacity = fminf(0.99f, g.opacity * gres);
                    
                    //if (x == g_SceneInfo.resX / 2 && y == g_SceneInfo.resY / 2)
                    //    printf("opacity, gres, mul: %f %f, %f\n", g.opacity, gres, g.opacity * gres);
                    //float sample_opacity = evaluateGaussianResponse_origin(currRay, g);
                    //float sample_opacity = evaluateGaussianResponse_3dgrt(currRay, g);
                    //float sample_opacity = evaluateGaussianResponse(currRay, g);
                    if (gres < KERNEL_MIN_RESPONSE || sample_opacity < 1.0f / 255.0f) // 0.004
                        continue;
                    float3 sample_color = eval_sh_final(SPH_EVAL_DEGREE, currRay.dir, g);
                    //float3 view_dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - currRay.pos);
                    //float3 sample_color = eval_sh_final2(SPH_EVAL_DEGREE, view_dir, g);

                    accumulated_color += sample_color * sample_opacity * (1.0f - accumulated_opacity);
                    accumulated_opacity += sample_opacity * (1.0f - accumulated_opacity);

#if HIT_AND_NODE_COUNT_DEBUG
                    blend_ops++;
#endif
#if !NO_EARLY_TERMINATION
                    if (accumulated_opacity > OPACITY_THRESHOLD) {
                        break;
                    }
#endif
                } // for (i < local_hit_count)
            } // if (local_hit_count > 0)

#if NO_EARLY_TERMINATION
            if (t_far >= t_scene_far)
                break;
#else
            if (accumulated_opacity > OPACITY_THRESHOLD | t_far >= t_scene_far)
                break;
#endif
            
            if (cache.empty()) {
                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, 0); // 루트에서 재시작
                t_near = t_far; t_far = t_scene_far; // 탐색 구간을 뒤로 미룸
            }
            else {
                const cu_traceState& trace = cache.top(); cache.pop();

                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, trace.nodeID);
                t_near = t_far;
                t_far = trace.tMax;
            }
        } // while(true) == while(accumulated_opacity < OPACITY_THRESHOLD)
        if (x == g_SceneInfo.resX / 2 && y == g_SceneInfo.resY) printf("\n");
    } // if (BoundsRayIntersect)
#if DEBUG_LEAF_CUDA
    /* debug */
    float2 minMax = make_float2(0.0f, COLORMAP_MAX);
    accumulated_color = computeVisualizationColor(maxLeaf, minMax);
#endif
}
#elif PRIMITIVE_TYPE == TRI
#if USE_STACK == SHORT_STACK

__device__ void singlePassIntersectGaussian_sortNode_onlyShortStack(
    cuRay& currRay,
    float3& accumulated_color,      // 수정: 누적 색상을 직접 업데이트
    float& accumulated_opacity    // 수정: 누적 알파를 직접 업데이트
#if HIT_AND_NODE_COUNT_DEBUG
    , int& node_visits
    , int& leaf_visits
    , int& intersection_tests
    , int& hits_found
    , int& blend_ops
    , int& max_sort_size
#endif
) {
#if HIT_AND_NODE_COUNT_DEBUG
    node_visits = 0;
    leaf_visits = 0;
    intersection_tests = 0;
    hits_found = 0;
    blend_ops = 0;
    max_sort_size = 0;
#elif BLEND_SELECT
    int blend_ops = 0;
#endif
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

#if DEBUG_LEAF_CUDA
    int maxLeaf = 0 ;
#endif
    //int prevGaussianID = -1; // 이전 ID 기억
    // 광선의 유효 범위 설정
    float t_scene_near = RAY_START_EPSILON, t_scene_far = FLT_MAX;
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, &currRay, &t_scene_near, &t_scene_far)) {
        kdtreeNode node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, 0);
        float t_near = t_scene_near, t_far = t_scene_far;
        // Kd-tree 순회를 위한 스택 초기화
        shortStack cache;
        //ShortStackCache cache;
        cache.init(threadIdx.y * blockDim.x + threadIdx.x);
#if MAIL_BOX
        Mailbox mailbox;
        mailbox.init();
#endif

        while (accumulated_opacity < OPACITY_THRESHOLD) { //while (true) {
            //traverse internal nodes
            while (!IS_LEAF(node)) {
                const unsigned childOffset = FIRST_CHILD_OFFSET(node);

                const float2 pos_dir = currRay.get_dir_pos(SPLIT_AXIS(node));

                const float t_split = __fdividef(SPLIT_POS(node) - pos_dir.x, pos_dir.y); //__fdividef() : faster than "/" but less precise
                const unsigned sign = signbit(pos_dir.y);

                unsigned idx = childOffset + (sign ^ (t_split <= t_near));

                if (t_near < t_split && t_split < t_far) {
                    cache.push(childOffset + (sign ^ 1), t_far);
                    t_far = t_split;
                }
#if HIT_AND_NODE_COUNT_DEBUG
                node_visits++;
#endif
                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, idx);
            }

            // --- 리프 노드 처리 로직 ---

            unsigned int baseOffset = OBJECTLIST_OFFSET(node);
            int objectSize = OBJECT_SIZE(node) + baseOffset;
            //if (count <= 0) continue;
                // 수집: 이 리프 노드 내의 모든 충돌을 임시 로컬 배열에 저장
            HitRecord local_hits[MAX_HITS];
            int local_hit_count = 0;

#if DEBUG_LEAF_CUDA
            if (OBJECT_SIZE(node) > maxLeaf) {
                maxLeaf = OBJECT_SIZE(node);
            }
#endif

#if HIT_AND_NODE_COUNT_DEBUG
            leaf_visits++;
#endif
            for (; baseOffset < objectSize; baseOffset++) {
#if HIT_AND_NODE_COUNT_DEBUG
                intersection_tests++;
                int tmp = local_hit_count;
#endif

#if OFFSET_TEXTURE
                const unsigned tri_idx = tex1Dfetch<unsigned int>(inObjectOffsetListTex, baseOffset);
#else
                const unsigned tri_idx = g_d_tri_offsets_dev[baseOffset];
#endif
#if WALD_METHOD
    #if BLEND_SELECT
                if (blend_ops <= 25)
                    rayPrimIntersect(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count);
                else
                    singlePassIntersectPlane(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count);
    #else
        #if MAIL_BOX
                singlePassIntersectRoutineMailBox(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count, &mailbox);
        #else
                rayPrimIntersect(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count);
        #endif

    #endif

#else
                singlePassIntersectRoutineGaussian_sortNode(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count);
#endif
            }
#if HIT_AND_NODE_COUNT_DEBUG
            max_sort_size = MyMAX(max_sort_size, local_hit_count);
#endif

            if (local_hit_count > 0) {
#if HIT_AND_NODE_COUNT_DEBUG
                hits_found+= local_hit_count;
#endif
                // 정렬: 이 리프 노드 내의 충돌만 정렬
#if BLEND_SELECT
                if(blend_ops <= 25)
#endif
                sortHits(local_hits, local_hit_count);
                //sortHits_Hybrid(local_hits, local_hit_count);
                // 블렌딩: 정렬된 순서대로 알파 블렌딩 수행
                for (int i = 0; i < local_hit_count; ++i) {
                    int gaussianID = 0;
#if WALD_METHOD

    #if BLEND_SELECT
                    if (blend_ops > 20) {
                        selectionSortStep(local_hits, local_hit_count, i);
                        gaussianID = singlePassIntersectCheck(currRay, local_hits[i]);
                        if (gaussianID < 0) {
                            continue;
                        }
                    }
                    else {
                        float4 internal2 = tex1Dfetch<float4>(inTriAccelTex, 3 * local_hits[i].primIndex + 2);
                        gaussianID = __float_as_int(internal2.w);
                    }
    #else
        #if TRIACC_TEXTURE
                    float4 internal2 = tex1Dfetch<float4>(inTriAccelTex, 3 * local_hits[i].primIndex + 2);
        #else
                    float4 internal2 = g_d_tri_acc_dev[3 * local_hits[i].primIndex + 2];
        #endif
                    gaussianID = __float_as_int(internal2.w);

                    //if (gaussianID == prevGaussianID) continue;
                    //prevGaussianID = gaussianID;
        #if HIT_AND_NODE_COUNT_DEBUG
                    if (x == g_SceneInfo.resX / 2 && y == g_SceneInfo.resY / 2) printf("%d ", gaussianID);
        #endif
    #endif

#else
                    float4 d2 = tex1Dfetch<float4>(inTriAccelTex, local_hits[i].primIndex * 4 + 2);
                    gaussianID = __float_as_int(d2.w);
#endif  //WALD_METHOD
#if GAUSSIAN_TEXTURE
                    Gaussian g = fetch_gaussian(gaussianID);
#else
                    Gaussian g = g_d_gaussians[gaussianID];
#endif
                    // 가우시안 파라미터 준비
                    const float3 particlePosition = make_float3(g.pos[0], g.pos[1], g.pos[2]);
                    const float3 particleScale = make_float3(g.scale[0], g.scale[1], g.scale[2]);
                    //float4 particleQquaternion = make_float4(g.rot[1], g.rot[2], g.rot[3], g.rot[0]);
                    float4 particleQquaternion = make_float4(g.rot[0], g.rot[1], g.rot[2], g.rot[3]);
                    float33 particleRotation;
                    quaternionWXYZToMatrixTranspose(particleQquaternion, particleRotation);
                    // 광선을 가우시안의 로컬 좌표계로 변환 (회전 및 스케일링)
#if UPLOAD_INV_SCALE
                    const float3 giscl = particleScale;
#else
                    const float3 giscl = make_float3(1 / particleScale.x, 1 / 
                        particleScale.y, 1 / particleScale.z);
#endif
                    const float3 gposc = (currRay.pos - particlePosition);
                    const float3 gposcr = (gposc * particleRotation);
                    const float3 gro = giscl * gposcr;
                    const float3 rayDirR = currRay.dir * particleRotation;
                    const float3 grdu = giscl * rayDirR;
                    const float3 grd = normalize(grdu);

                    // cross product를 이용해 grayDist(제곱된 마할라노비스 거리) 계산
                    const float3 gcrod = cross(grd, gro);
                    const float grayDist = dot(gcrod, gcrod);

                    // particleResponse 함수를 통해 밀도 계산
                    const float gres = particleResponse<GAUSSIAN_DEGREE>(grayDist);

                    // 기본 불투명도와 밀도를 곱하여 최종 결과 반환
                    float sample_opacity = fminf(0.99f, g.opacity * gres);
                    //if (x == g_SceneInfo.resX / 2 && y == g_SceneInfo.resY / 2)
                    //    printf("opacity, gres, mul: %f %f, %f\n", g.opacity, gres, g.opacity * gres);
                    //float sample_opacity = evaluateGaussianResponse_origin(currRay, g);
                    //float sample_opacity = evaluateGaussianResponse_3dgrt(currRay, g);
                    //float sample_opacity = evaluateGaussianResponse(currRay, g);
                    if (gres < KERNEL_MIN_RESPONSE || sample_opacity < 1.0f / 255.0f) // 0.004
                        continue;
                    float3 sample_color = eval_sh_final(SPH_EVAL_DEGREE, currRay.dir, g);
                    //float3 view_dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - currRay.pos);
                    //float3 sample_color = eval_sh_final2(SPH_EVAL_DEGREE, view_dir, g);

                    accumulated_color += sample_color * sample_opacity * (1.0f - accumulated_opacity);
                    accumulated_opacity += sample_opacity * (1.0f - accumulated_opacity);

#if HIT_AND_NODE_COUNT_DEBUG | BLEND_SELECT
                    blend_ops++;
#endif

                    /*/
                    float sample_opacity = evaluateGaussianResponse_3dgrt(currRay, g);
                    sample_opacity = fminf(0.99f, sample_opacity);

                    //float3 view_dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - currRay.pos);
                    float3 sample_color = eval_sh_final(SPH_EVAL_DEGREE, currRay.dir, g);
                    //float3 sample_color = eval_sh_final2(SPH_EVAL_DEGREE, view_dir, g);

                    float transmittance = 1.0f - accumulated_opacity;
                    accumulated_color += sample_color * sample_opacity * accumulated_opacity;
                    accumulated_opacity += sample_opacity * (1.0f - accumulated_opacity);
                    //*/
                    // 블렌딩 중에도 조기 종료 조건을 계속 확인
                    if (accumulated_opacity > OPACITY_THRESHOLD) {
                        break;
                    }
                } // for (i < local_hit_count)
            } // if (local_hit_count > 0)

            if (accumulated_opacity > OPACITY_THRESHOLD | t_far >= t_scene_far)
                break;
            if (cache.empty()) {
                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, 0); // 루트에서 재시작
                t_near = t_far; t_far = t_scene_far; // 탐색 구간을 뒤로 미룸
            }
            else {
                const cu_traceState& trace = cache.top(); cache.pop();

                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, trace.nodeID);
                t_near = t_far;
                t_far = trace.tMax;
            }
        } // while(true) == while(accumulated_opacity < OPACITY_THRESHOLD)
        if (x == g_SceneInfo.resX / 2 && y == g_SceneInfo.resY) printf("\n");
    } // if (BoundsRayIntersect)

#if DEBUG_LEAF_CUDA
/* debug */
    float k;
    float3 blue = make_float3(0.f, 0.0f, 1.f);
    float3 green = make_float3(0.f, 1.0f, 0.f);
    float3 yellow = make_float3(1.f, 1.0f, 0.f);
    float3 red = make_float3(1.f, 0.0f, 0.f);
    float3 p = make_float3(1.f, 0.0f, 1.f);
    float3 white = make_float3(1.0f, 1.0f, 1.0f);
    float alpha;
    if (maxLeaf < 8 && maxLeaf > 0) {
        k = 8;
        alpha = (maxLeaf - 0) / k;
        accumulated_color = blue * (1 - alpha) + green * alpha;
    }
    else if (maxLeaf < 16 && maxLeaf > 0) {
        k = 8;
        alpha = (maxLeaf - 8) / k;
        accumulated_color = green * (1 - alpha) + yellow * alpha;
    }
    else if (maxLeaf < 32 && maxLeaf > 0) {
        k = 16;
        alpha = (maxLeaf - 16) / k;
        accumulated_color = yellow * (1 - alpha) + red * alpha;
    }
    else if (maxLeaf < 48 && maxLeaf > 0) {
        k = 16;
        alpha = (maxLeaf - 32) / k;
        accumulated_color = red * (1 - alpha) + p * alpha;
    }
    else if (maxLeaf > 0) {
        k = 16;
        alpha = (maxLeaf - 48) / k;
        accumulated_color = p * (1 - alpha) + white * alpha;
    }
    //if (maxLeaf < 32) {
    //    accumulated_color = make_float3(0.f, 0.0f, maxLeaf / 32.f);
    //}
    //else if(maxLeaf < 64)
    //    accumulated_color = make_float3((maxLeaf - 32) / 32.f, (maxLeaf - 32) / 32.f, 0.f);
    //else if (maxLeaf < 96)
    //    accumulated_color = make_float3(0.0f, (maxLeaf - 64) / 32.f, 0.f);
    //else
    //    accumulated_color = make_float3((maxLeaf - 96) / 32.0f, 0.f, 0.f);
#endif
}

#elif USE_STACK == HYBRID_STACK

#if HIT_AND_NODE_COUNT_DEBUG
__device__ int singlePassIntersectGaussian_sortNode_hybridStack(
    int& hitCount,
#else
__device__ void singlePassIntersectGaussian_sortNode_hybridStack(
#endif
    cuRay& currRay,
    float3& accumulated_color,      // 수정: 누적 색상을 직접 업데이트
    float& accumulated_opacity    // 수정: 누적 알파를 직접 업데이트
    , cu_traceState* global_stack
) {
    int global_stack_ptr = 0;
#if HIT_AND_NODE_COUNT_DEBUG
    hitCount = 0;
    int node_visit_count = 0;
#endif
    // 광선의 유효 범위 설정
    float t_scene_near = RAY_START_EPSILON, t_scene_far = FLT_MAX;
    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("(%d,%d) ", x, y);
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, &currRay, &t_scene_near, &t_scene_far)) {
    //if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, currRay, t_scene_near, t_scene_far)) {
        float t_near = t_scene_near, t_far = t_scene_far;
        // Kd-tree 순회를 위한 스택 초기화
        ShortStackCache cache;
        //shortStack cache;
        cache.init(threadIdx.y * blockDim.x + threadIdx.x);
        kdtreeNode node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, 0);
        // 메인 순회 루프
        //while (true) {
        while (accumulated_opacity < OPACITY_THRESHOLD) {
            while (!IS_LEAF(node)) {
                const unsigned childOffset = FIRST_CHILD_OFFSET(node);

                const float2 pos_dir = currRay.get_dir_pos(SPLIT_AXIS(node));

                const float t_split = __fdividef(SPLIT_POS(node) - pos_dir.x, pos_dir.y);
                const unsigned sign = signbit(pos_dir.y);

                unsigned idx = childOffset + (sign ^ (t_split <= t_near));

                if (t_near < t_split && t_split < t_far) {
                    cu_traceState item_to_push = { childOffset + (sign ^ 1), t_far };
                    bool was_full = cache.is_full();
                    cu_traceState evicted_item = cache.push(item_to_push);
                    if (was_full) { // 캐시가 꽉 차서 아이템이 밀려났다면 global_stack으로 보냄
                        if (global_stack_ptr < MAX_GLOBAL_STACK_DEPTH) {
                            global_stack[global_stack_ptr++] = evicted_item;
                            //if(global_stack_ptr>=2)
                                //printf("global Stack access(push): %d\n", global_stack_ptr);
                        }
                    }
                    t_far = t_split;
                }
                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, idx);
#if HIT_AND_NODE_COUNT_DEBUG
                node_visit_count++;
#endif
            }
            // --- 리프 노드 처리 로직 ---
            unsigned baseOffset = OBJECTLIST_OFFSET(node);
            int objectSize = OBJECT_SIZE(node) + baseOffset;
            //if (count <= 0) continue;
            if (objectSize > 0) {
                // 수집: 이 리프 노드 내의 모든 충돌을 임시 로컬 배열에 저장
                HitRecord local_hits[MAX_HITS];
                int local_hit_count = 0;

                for (; baseOffset < objectSize; baseOffset++) {
#if OFFSET_TEXTURE
                    const unsigned tri_idx = tex1Dfetch<unsigned int>(inObjectOffsetListTex, baseOffset);
#else
                    const unsigned tri_idx = g_d_tri_offsets_dev[baseOffset];
#endif
                    singlePassIntersectRoutineGaussian_sortNode(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count
                        //, inTriAccelTex
                    );
                }

                if (local_hit_count > 0) {
                    // 정렬: 이 리프 노드 내의 충돌만 정렬
                    sortHits(local_hits, local_hit_count);
                    // 블렌딩: 정렬된 순서대로 알파 블렌딩 수행
                    for (int i = 0; i < local_hit_count; ++i) {
                        //float4 d2 = tex1Dfetch(inTriAccelTex, local_hits[i].primIndex * 4 + 2);
                        float4 d2 = tex1Dfetch<float4>(inTriAccelTex, local_hits[i].primIndex * 4 + 2);
                        //float4 d2 = g_d_accel[local_hits[i].primIndex * 4 + 2];
                        int gaussianID = __float_as_int(d2.w);
                        //Gaussian g = fetch_gaussian(gaussianID);
                        Gaussian g = g_d_gaussians[gaussianID];

                        //sample_opacity = g.opacity;
                        //float sample_opacity = evaluateGaussianResponse_3dgrt(currRay, g);
                        float sample_opacity = evaluateGaussianResponse(currRay, g);

                        float3 view_dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - currRay.pos);
                        float3 sample_color = eval_sh_final(SPH_EVAL_DEGREE, view_dir, g);

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
            if (accumulated_opacity > OPACITY_THRESHOLD | t_far >= t_scene_far) {
                break;
            }
            //if (cache.empty()) {
            if (cache.is_empty() && global_stack_ptr == 0) {
                //printf("(empty) hitCount:%d\n", hitCount);
                //node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, 0); // 루트에서 재시작
                //t_near = t_far; t_far = t_scene_far; // 탐색 구간을 뒤로 미룸
                break;
            }
            cu_traceState next;
            if (!cache.is_empty()) {
                next = cache.pop();
            }
            else {
                //cu_traceState next;
                next = global_stack[--global_stack_ptr];
                //printf("global Stack access(pop): %d\n", global_stack_ptr);
            }
            //const cu_traceState& trace = stack.top(); stack.pop();
            node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, next.nodeID);
            t_near = t_far;
            t_far = next.tMax;
            //}
        } // while(true)
    } // if (BoundsRayIntersect)
#if HIT_AND_NODE_COUNT_DEBUG
    return node_visit_count;
#endif
}

#elif USE_STACK == GLOBAL_STACK

#if HIT_AND_NODE_COUNT_DEBUG
__device__ int singlePassIntersectGaussian_sortNode_onlyGlobalStack(
    int& hitCount,
#else
__device__ void singlePassIntersectGaussian_sortNode_onlyGlobalStack(
#endif
    cuRay& currRay,
    float3& accumulated_color,      // 수정: 누적 색상을 직접 업데이트
    float& accumulated_opacity    // 수정: 누적 알파를 직접 업데이트
    , cu_traceState* global_stack
) {
    int global_stack_ptr = 0;
#if HIT_AND_NODE_COUNT_DEBUG
    hitCount = 0;
    int node_visit_count = 0;
#endif
    // 광선의 유효 범위 설정
    float t_scene_near = RAY_START_EPSILON, t_scene_far = FLT_MAX;
    if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, &currRay, &t_scene_near, &t_scene_far)) {
        //if (BoundsRayIntersect(g_SceneBBoxMin, g_SceneBBoxMax, currRay, t_scene_near, t_scene_far)) {
        float t_near = t_scene_near, t_far = t_scene_far;
        // Kd-tree 순회를 위한 스택 초기화
        kdtreeNode node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, 0);
        // 메인 순회 루프
        //while (true) {
        while (accumulated_opacity < OPACITY_THRESHOLD) {
            while (!IS_LEAF(node)) {
                const unsigned childOffset = FIRST_CHILD_OFFSET(node);

                const float2 pos_dir = currRay.get_dir_pos(SPLIT_AXIS(node));

                const float t_split = __fdividef(SPLIT_POS(node) - pos_dir.x, pos_dir.y);
                const unsigned sign = signbit(pos_dir.y);

                unsigned idx = childOffset + (sign ^ (t_split <= t_near));

                if (t_near < t_split && t_split < t_far) {
                    //cache.push(childOffset + (sign ^ 1), t_far);
                    cu_traceState item_to_push = { childOffset + (sign ^ 1), t_far };
                    if (global_stack_ptr < MAX_GLOBAL_STACK_DEPTH) {
                        global_stack[global_stack_ptr++] = item_to_push;
                    }
                    t_far = t_split;
                }
                node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, idx);
#if HIT_AND_NODE_COUNT_DEBUG
                node_visit_count++;
#endif
            }
            // --- 리프 노드 처리 로직 ---
            unsigned baseOffset = OBJECTLIST_OFFSET(node);
            int objectSize = OBJECT_SIZE(node) + baseOffset;
            //if (count <= 0) continue;
                // 수집: 이 리프 노드 내의 모든 충돌을 임시 로컬 배열에 저장
            HitRecord local_hits[MAX_HITS];
            int local_hit_count = 0;

            for (; baseOffset < objectSize; baseOffset++) {
#if OFFSET_TEXTURE
                const unsigned tri_idx = tex1Dfetch<unsigned int>(inObjectOffsetListTex, baseOffset);
#else
                const unsigned tri_idx = g_d_tri_offsets_dev[baseOffset];
#endif
                singlePassIntersectRoutineGaussian_sortNode(currRay, tri_idx, t_near, t_far, local_hits, local_hit_count
                    //, inTriAccelTex
                );
            }

            if (local_hit_count > 0) {
                // 정렬: 이 리프 노드 내의 충돌만 정렬
                sortHits(local_hits, local_hit_count);
                // 블렌딩: 정렬된 순서대로 알파 블렌딩 수행
                for (int i = 0; i < local_hit_count; ++i) {
                    //float4 d2 = tex1Dfetch(inTriAccelTex, local_hits[i].primIndex * 4 + 2);
                    float4 d2 = tex1Dfetch<float4>(inTriAccelTex, local_hits[i].primIndex * 4 + 2);
                    //float4 d2 = g_d_accel[local_hits[i].primIndex * 4 + 2];
                    int gaussianID = __float_as_int(d2.w);
                    //Gaussian g = fetch_gaussian(gaussianID);
                    const Gaussian g = g_d_gaussians[gaussianID];
                    //const Gaussian* g = &g_d_gaussians[gaussianID]; // 포인터로 접근

                    //sample_opacity = g.opacity;
                    
                    //float sample_opacity = evaluateGaussianResponse_3dgrt(currRay, g);
                    float sample_opacity = evaluateGaussianResponse(currRay, g);
                    //float sample_opacity = evaluateGaussianResponse_ptr(currRay, g);

                    float3 view_dir = normalize(make_float3(g.pos[0], g.pos[1], g.pos[2]) - currRay.pos);
                    //float3 view_dir = normalize(make_float3(g->pos[0], g->pos[1], g->pos[2]) - currRay.pos);
                    float3 sample_color = eval_sh_final(SPH_EVAL_DEGREE, view_dir, g);
                    //float3 sample_color = eval_sh_final_ptr(SPH_EVAL_DEGREE, view_dir, g);

                    accumulated_color += sample_color * sample_opacity * (1.0f - accumulated_opacity);
                    accumulated_opacity += sample_opacity * (1.0f - accumulated_opacity);

                    // 블렌딩 중에도 조기 종료 조건을 계속 확인
                    if (accumulated_opacity > OPACITY_THRESHOLD) {
                        break;
                    }
                }
            } // if (local_hit_count > 0)
            if (accumulated_opacity > OPACITY_THRESHOLD | t_far >= t_scene_far) {
                break;
            }
            if (global_stack_ptr == 0) {
                break;
            }
            cu_traceState next = global_stack[--global_stack_ptr];
            node = tex1Dfetch<kdtreeNode>(inKdTreeNodeTex, next.nodeID);
            t_near = t_far;
            t_far = next.tMax;
        } // while(true)
    } // if (BoundsRayIntersect)
}

#endif
#endif //PRIMITIVE_TYPE
__global__ void renderKernelGaussian_sortNode(float* pFrameBuffer
#if HIT_AND_NODE_COUNT_DEBUG
    , float3* d_debug_buffer1, float3* d_debug_buffer2
#endif
#if USE_STACK > SHORT_STACK
    , cu_traceState* d_global_stack
#endif
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //if (x == 0 && y == 0) {
    //    printf("SceneInfo: %d %d\n", g_SceneInfo.resX, g_SceneInfo.resY);
    //    //float3 eye, u, v, startPoint; float stepX, stepY;
    //    printf("Cam Info:\n");
    //    printf("eye: %f %f %f\n", g_CameraInfo.eye.x, g_CameraInfo.eye.y, g_CameraInfo.eye.z);
    //    printf("u: %f %f %f\n", g_CameraInfo.u.x, g_CameraInfo.u.y, g_CameraInfo.u.z);
    //    printf("v: %f %f %f\n", g_CameraInfo.v.x, g_CameraInfo.v.y, g_CameraInfo.v.z);
    //    printf("startPoint: %f %f %f\n", g_CameraInfo.startPoint.x, g_CameraInfo.startPoint.y, g_CameraInfo.startPoint.z);
    //    printf("step: %f %f\n", g_CameraInfo.stepX, g_CameraInfo.stepY);
    //}
    if (x >= g_SceneInfo.resX || y >= g_SceneInfo.resY) return;

    // 광선 생성
    float sx = (float)x + 0.5f, sy = (float)y + 0.5f;

    float3 dir = g_CameraInfo.startPoint + g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
    cuRay ray = { g_CameraInfo.eye, normalize(dir - g_CameraInfo.eye) };

    // 누적 변수 초기화
    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    float accumulated_opacity = 0.0f;

#if USE_STACK > SHORT_STACK
    // 스레드에 해당하는 전역 스택 포인터 가져오기
    int thread_idx = y * g_SceneInfo.resX + x;
    cu_traceState* my_global_stack = d_global_stack + thread_idx * MAX_GLOBAL_STACK_DEPTH;
#endif

#if HIT_AND_NODE_COUNT_DEBUG
    int node_visits = 0;        // 총 순회 노드 수
    int leaf_visits = 0;        // 방문한 리프 노드 수
    int intersection_tests = 0; // 총 삼각형 교차 판정 횟수
    int hits_found = 0;         // 교차에 성공한 삼각형 수
    int blend_ops = 0;          // 최종 블렌딩에 사용된 가우시안 수
    int max_sort_size = 0;      // 단일 리프에서 정렬한 최대 아이템 수
#endif

#if USE_STACK == SHORT_STACK
    singlePassIntersectGaussian_sortNode_onlyShortStack(
#elif USE_STACK == HYBRID_STACK
    singlePassIntersectGaussian_sortNode_hybridStack(
#elif USE_STACK == GLOBAL_STACK
    singlePassIntersectGaussian_sortNode_onlyGlobalStack(
#endif
        ray, accumulated_color, accumulated_opacity
#if HIT_AND_NODE_COUNT_DEBUG
        , node_visits
        , leaf_visits
        , intersection_tests
        , hits_found
        , blend_ops
        , max_sort_size
#endif
#if USE_STACK > SHORT_STACK
        , my_global_stack
#endif
        );

    // 최종 색상 계산
    float3 background_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 final_color = accumulated_color + background_color * (1.0f - accumulated_opacity);
    //float3 final_color = accumulated_color / accumulated_opacity;

    int idx = 3 * ((g_SceneInfo.resY - y - 1) * g_SceneInfo.resX + x);
    pFrameBuffer[idx + 0] = final_color.x;
    pFrameBuffer[idx + 1] = final_color.y;
    pFrameBuffer[idx + 2] = final_color.z;

#if HIT_AND_NODE_COUNT_DEBUG
    if (d_debug_buffer1 == nullptr || d_debug_buffer2 == nullptr) return; // for dummy run
    int id = y * g_SceneInfo.resX + x;
    //printf("id(%d, %d) %d\n",x,y,id);
    d_debug_buffer1[id] = make_float3(
        (float)node_visits,
        (float)leaf_visits,
        (float)intersection_tests
    );
    d_debug_buffer2[id] = make_float3(
        (float)hits_found,
        (float)blend_ops,
        (float)max_sort_size
    );
#endif
}

// =================================================================================
// Host-Side Public Render Function
// =================================================================================
//TODO: print device prop on start of app
void printDeviceLimits() {
    int deviceId;
    cudaError_t err = cudaGetDevice(&deviceId);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));

    std::cout << "\nDevice Name: " << deviceProp.name << std::endl;

    // 1. maxTexture1D (CUDA Array 기반, 작은 값)
    std::cout << "props.maxTexture1D: " << deviceProp.maxTexture1D << " elements" << std::endl;

    // 2. maxTexture1DLinear (선형 메모리 기반, 우리가 찾는 값)
    int linearWidth = 0;
    err = cudaDeviceGetAttribute(&linearWidth, cudaDevAttrMaxTexture1DLinearWidth, deviceId);
    if (err == cudaSuccess) {
        std::cout << "cudaDevAttrMaxTexture1DLinearWidth: " << linearWidth << " elements" << std::endl;
    }
    else {
        std::cerr << "Failed to get MaxTexture1DLinearWidth attribute: " << cudaGetErrorString(err) << std::endl;
    }
    std::cout << "\n";

    //get device prop
    int deviceID;
    cudaGetDevice(&deviceID);

    int maxSharedMemPerBlock;
    // 현재 GPU의 "블록 당 최대 공유 메모리" 속성
    cudaDeviceGetAttribute(
        &maxSharedMemPerBlock,
        cudaDevAttrMaxSharedMemoryPerBlock,
        deviceId
    );
    printf("This GPU's max shared memory per block: %d bytes\n", maxSharedMemPerBlock);
    // 49152 bytes
    // 49152 / (256 * 8) = 24
}

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
    printDeviceLimits();
    return true;
}

void build_waldInfoList_from_model(const CompositeObject* poly_model, cuWaldTriangleInfo*& pWaldInfo) {
    int iTriangleSize = poly_model->n_triangles;
    printf("Building final cuWaldTriangleInfo list for %d triangles...\n", iTriangleSize);

    // 메모리 할당
    if (!pWaldInfo) {
        pWaldInfo = (cuWaldTriangleInfo*)_aligned_malloc(sizeof(cuWaldTriangleInfo) * iTriangleSize, 16);
        if (!pWaldInfo) {
            fprintf(stderr, "ERROR: Failed to allocate memory for cuWaldTriangleInfo list!\n");
            throw std::bad_alloc();
        }
    }

    float A[3], B[3], C[3];
    float b[3], c[3];
    float N[3];

    for (unsigned int i = 0; i < iTriangleSize; i++) {
        const ExtendedVertex* pVerts = &poly_model->extended_vertices[3 * i];
        A[0] = pVerts[0].vertex[0]; A[1] = pVerts[0].vertex[1]; A[2] = pVerts[0].vertex[2];
        B[0] = pVerts[1].vertex[0]; B[1] = pVerts[1].vertex[1]; B[2] = pVerts[1].vertex[2];
        C[0] = pVerts[2].vertex[0]; C[1] = pVerts[2].vertex[1]; C[2] = pVerts[2].vertex[2];

        // 에지 및 법선 계산 (원본 함수와 동일한 순서)
        b[0] = C[0] - A[0]; b[1] = C[1] - A[1]; b[2] = C[2] - A[2]; // Edge AC
        c[0] = B[0] - A[0]; c[1] = B[1] - A[1]; c[2] = B[2] - A[2]; // Edge AB
        fMyVecCrossProduct(b, c, N); // N = (C-A) x (B-A)
        fMyVecNormalize(N);

        // 주축 및 기타 축 계산 (원본 함수와 동일)
        unsigned k = 0;
        k = fabsf(N[1]) > fabsf(N[k]) ? 1 : k;
        k = fabsf(N[2]) > fabsf(N[k]) ? 2 : k;

        const unsigned u = (k + 1) % 3;
        const unsigned v = (k + 2) % 3;

        cuWaldTriangleInfo& wald = pWaldInfo[i];

        // 평면 방정식 계수 계산 (원본 함수와 동일)
        const float krec = N[k];
        const float nu = N[u] / krec;
        const float nv = N[v] / krec;
        const float nd = fMyVecDotProduct(A, N) / krec;

        // 무게중심 좌표 계수 계산
        const float denom = b[u] * c[v] - b[v] * c[u];
        const float bnu = b[u] / denom;
        const float bnv = -b[v] / denom;
        const float cnu = c[v] / denom;
        const float cnv = -c[u] / denom;

        // 5. cuWaldTriangleInfo 구조체에 모든 값 패킹
        wald.internal0.x = uint_as_float_H(k);
        wald.internal0.y = nu;
        wald.internal0.z = nv;
        wald.internal0.w = nd;

        wald.internal1.x = A[u];
        wald.internal1.y = A[v];
        wald.internal1.z = bnu;
        wald.internal1.w = bnv;

        wald.internal2.x = cnu;
        wald.internal2.y = cnv;

        // 투명도 및 기타 플래그 (원본과 동일하게 단순화)
        int flag = 2;
        if (krec < 0.0f) flag = -flag;
        wald.internal2.z = int_as_float_H(flag); // 투명도 정보 등

        // material_ID 패킹 (getObjectIndex()가 하위 24비트만 사용)
        unsigned int object_id = pVerts[0].material_ID;
        wald.internal2.w = uint_as_float_H(object_id);
    }
}

void warmUp(float* d_framebuffer, cudaStream_t stream) {
    dim3 threads(DIM_X, DIM_Y);
    dim3 blocks((MAIN_WINDOW_WIDTH + threads.x - 1) / threads.x, (MAIN_WINDOW_HEIGHT + threads.y - 1) / threads.y);
    size_t shared_mem_size = threads.x * threads.y * SHORT_STACK_DEPTH * sizeof(cu_traceState);
    //printf("Warming up GPU...\n");
    CUDA_CHECK(cudaEventRecord(start_ev, stream)); // 시작 기록
    renderKernelGaussian_sortNode << < blocks, threads, shared_mem_size, stream >> > (d_framebuffer
#if HIT_AND_NODE_COUNT_DEBUG
        , nullptr, nullptr
#endif
#if USE_STACK > SHORT_STACK
        , d_global_stack
#endif
        );
    //CUDA_CHECK(cudaGetLastError());        // DEBUG: launch 실패 확인
    //CUDA_CHECK(cudaDeviceSynchronize());   // DEBUG: 실행 중 오류 확인
    CUDA_CHECK(cudaEventRecord(stop_ev, stream)); // 종료 기록
    CUDA_CHECK(cudaEventSynchronize(stop_ev)); // GPU 작업 완료까지 대기
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_ev, stop_ev));
    printf("WarmUp: %f\n", 1000.0f / milliseconds); // 전역 변수에 FPS 저장
    //printf("Warm-up complete.\n");
}

#if PRIMITIVE_TYPE == ELLIPSOID || PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
void renderGaussianWithCudaSetup(const CompositeObject& object, const std::vector<Gaussian>& gaussians
#if !QUATERNION
    , std::vector<float>& kScales
#endif
) {
    printf("Setting up static data for CUDA rendering...\n");

    KdTree* kdTree = object.kd_tree;
    //if (!kdTree || object.n_triangles == 0) {
    //    std::cerr << "[CUDA Error] Object or Kd-tree is empty." << std::endl;
    //    return;
    //}
    if (!kdTree || gaussians.empty()) {
        std::cerr << "[CUDA Error] Kd-tree or Gaussian data is empty." << std::endl;
        return;
    }
    if (kdTree == nullptr) {
        printf("[FATAL] kdTree == nullptr\n");
        return;
    }

    cudaError_t err;

#if !GLOBAL_DEVICE_VAR
    kdtreeNode* g_d_kdtree_nodes = nullptr;
#endif
    /* kdtree node */
    size_t node_size = kdTree->tree_node_count * sizeof(kdtreeNode);
    CUDA_CHECK(cudaMalloc(&g_d_kdtree_nodes, node_size));
    CUDA_CHECK(cudaMemcpy(g_d_kdtree_nodes, kdTree->tree, node_size, cudaMemcpyHostToDevice));
    /* prim offset */
    size_t offset_size = kdTree->prim_offset_count * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&g_d_prim_offsets, offset_size));
    CUDA_CHECK(cudaMemcpy(g_d_prim_offsets, kdTree->prim_offset_list, offset_size, cudaMemcpyHostToDevice));
    /* gaussians */
    size_t gaussians_bytes = gaussians.size() * sizeof(Gaussian);
    CUDA_CHECK(cudaMalloc(&g_d_gaussians_persistent, gaussians_bytes));
    CUDA_CHECK(cudaMemcpy(g_d_gaussians_persistent, gaussians.data(), gaussians_bytes, cudaMemcpyHostToDevice));

#if !QUATERNION
    size_t kScale_bytes = kScales.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&g_d_kScale_persistent, kScale_bytes));
    CUDA_CHECK(cudaMemcpy(g_d_kScale_persistent, kScales.data(), kScale_bytes, cudaMemcpyHostToDevice));
#endif

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] GPU memcpy failed: " << cudaGetErrorString(err) << std::endl;
    }

    // 리소스 디스크립터(Resource Descriptor) 설정
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear; // 1D 배열이므로 Linear 타입

    // 텍스처 디스크립터(Texture Descriptor) 설정
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; // 주소 지정 모드
    texDesc.filterMode = cudaFilterModePoint;      // 필터링 없음 (tex1Dfetch와 동일)
    texDesc.readMode = cudaReadModeElementType;    // 원본 타입 그대로 읽기
    texDesc.normalizedCoords = 0;                  // 정규화되지 않은 좌표 사용

    /* kdtree node */
    resDesc.res.linear.devPtr = g_d_kdtree_nodes;
    resDesc.res.linear.desc = cudaCreateChannelDesc<uint2>();
    //resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
    resDesc.res.linear.sizeInBytes = node_size;
    cudaTextureObject_t h_inKdTreeNodeTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inKdTreeNodeTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inKdTreeNodeTex, &h_inKdTreeNodeTex, sizeof(cudaTextureObject_t)));

    /* prim offsets */
    resDesc.res.linear.devPtr = g_d_prim_offsets;
    resDesc.res.linear.desc = cudaCreateChannelDesc<unsigned int>();
    resDesc.res.linear.sizeInBytes = offset_size;
    printf("g_d_prim_offsets: %u\n", offset_size);
    cudaTextureObject_t h_inObjectOffsetListTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inObjectOffsetListTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inObjectOffsetListTex, &h_inObjectOffsetListTex, sizeof(cudaTextureObject_t)));

    /* gaussian infos */
#if GAUSSIAN_TEXTURE
    resDesc.res.linear.devPtr = g_d_gaussians_persistent;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDesc.res.linear.sizeInBytes = gaussians_bytes;
    cudaTextureObject_t h_inGaussianTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inGaussianTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inGaussianTex, &h_inGaussianTex, sizeof(cudaTextureObject_t)));
#else
    if (g_d_gaussians) cudaFree(g_d_gaussians);
    CUDA_CHECK(cudaMemcpyToSymbol(g_d_gaussians, &g_d_gaussians_persistent, sizeof(Gaussian*)));
#endif
#if !QUATERNION
    #if KSCALE_TEXTURE
    resDesc.res.linear.devPtr = g_d_kScale_persistent;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = gaussians_bytes;
    cudaTextureObject_t h_inKScaleTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inKScaleTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inKScaleTex, &h_inKScaleTex, sizeof(cudaTextureObject_t)));
    #else
    if (g_d_kScale) cudaFree(g_d_kScale);
    CUDA_CHECK(cudaMemcpyToSymbol(g_d_kScale, &g_d_kScale_persistent, sizeof(Gaussian*)));
    #endif
#endif

    // 상수 메모리 설정
    float3 h_bbox_min = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    float3 h_bbox_max = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMin, &h_bbox_min, sizeof(float3)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMax, &h_bbox_max, sizeof(float3)));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] const memory set failed: " << cudaGetErrorString(err) << std::endl;
    }

#if SHORT_STACK_DEPTH > 24
    cudaFuncSetAttribute(
        renderKernelGaussian_sortNode,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        maxSharedMemPerBlock
    );
#endif
#if USE_STACK == GLOBAL_STACK
    cudaFuncSetCacheConfig(renderKernelGaussian_sortNode, cudaFuncCachePreferL1);
#endif

    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));

    if (h_inKdTreeNodeTex)          cudaDestroyTextureObject(h_inKdTreeNodeTex);
#if OFFSET_TEXTURE
    if (h_inObjectOffsetListTex)    cudaDestroyTextureObject(h_inObjectOffsetListTex);
#endif
#if GAUSSIAN_TEXTURE
    if (h_inGaussianTex)            cudaDestroyTextureObject(h_inGaussianTex);
#endif
#if !QUATERNION && KSCALE_TEXTURE
    if (h_inKScaleTex)            cudaDestroyTextureObject(h_inKScaleTex);
#endif
}
#elif PRIMITIVE_TYPE == TRI
void renderGaussianWithCudaSetup(const CompositeObject& object, const std::vector<Gaussian>& gaussians) {
    printf("Setting up static data for CUDA rendering...\n");

    KdTree* kdTree = object.kd_tree;
    if (!kdTree || object.n_triangles == 0) {
        std::cerr << "[CUDA Error] Object or Kd-tree is empty." << std::endl;
        return;
    }
    if (!kdTree || gaussians.empty()) {
        std::cerr << "[CUDA Error] Kd-tree or Gaussian data is empty." << std::endl;
        return;
    }

    if (kdTree == nullptr) {
        printf("[FATAL] kdTree == nullptr\n");
        return;
    }

    //get device prop
    int deviceID;
    cudaGetDevice(&deviceID);

    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));

    printf("[DEBUG] kdTree->prim_offset_count = %zu\n", kdTree->prim_offset_count);
    printf("[DEBUG] maxTexture1D: %d\n", deviceProp.maxTexture1D);

    /* build tri acc */
#if WALD_METHOD
    cuWaldTriangleInfo* h_waldInfo = nullptr;
    build_waldInfoList_from_model(&object, h_waldInfo);
#else
    if (kdTree->tri_accel_list == nullptr) {
        printf("[FATAL] tri_accel_list == nullptr\n");
        return;
    }
    if (kdTree->tri_accel_list + object.n_triangles <= kdTree->tri_accel_list) {
        printf("[FATAL] tri_accel_list too small or corrupt pointer\n");
        return;
    }
    // TriAccel -> float4[4] (n_u, n_v, n_d, k | b_nu, b_nv, b_d, idx | c_nu, c_nv, c_d, matID | N.x, N.y, N.z, pad)
    std::vector<float4> h_triangles(object.n_triangles * 4);
    for (int i = 0; i < object.n_triangles; ++i) {
        const TriAccel& src = kdTree->tri_accel_list[i];
        h_triangles[i * 4 + 0] = make_float4(src.n_u, src.n_v, src.n_d, uint_as_float_H(src.k));
        h_triangles[i * 4 + 1] = make_float4(src.b_nu, src.b_nv, src.b_d, int_as_float_H(src.indexInObject));
        h_triangles[i * 4 + 2] = make_float4(src.c_nu, src.c_nv, src.c_d, int_as_float_H(src.material_ID));
        h_triangles[i * 4 + 3] = make_float4(src.N[0], src.N[1], src.N[2], 0.0f);
    }
#endif
    //printf("1. Data packing done\n");

    // GPU 메모리 할당 및 데이터 전송
    cudaError_t err;

#if !GLOBAL_DEVICE_VAR
    kdtreeNode* g_d_kdtree_nodes = nullptr;
#endif
    size_t node_size = kdTree->tree_node_count * sizeof(kdtreeNode);
    CUDA_CHECK(cudaMalloc(&g_d_kdtree_nodes, node_size));
    CUDA_CHECK(cudaMemcpy(g_d_kdtree_nodes, kdTree->tree, node_size, cudaMemcpyHostToDevice));

#if !GLOBAL_DEVICE_VAR
    unsigned int* g_d_prim_offsets = nullptr;
#endif
    size_t offset_size = kdTree->prim_offset_count * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&g_d_prim_offsets, offset_size));
    CUDA_CHECK(cudaMemcpy(g_d_prim_offsets, kdTree->prim_offset_list, offset_size, cudaMemcpyHostToDevice));

#if WALD_METHOD
    #if !GLOBAL_DEVICE_VAR
    cuWaldTriangleInfo* g_d_waldInfo = nullptr; // Device-side pointer
    #endif
    size_t wald_info_size = sizeof(cuWaldTriangleInfo) * object.n_triangles;
    CUDA_CHECK(cudaMalloc(&g_d_waldInfo, wald_info_size));
    CUDA_CHECK(cudaMemcpy(g_d_waldInfo, h_waldInfo, wald_info_size, cudaMemcpyHostToDevice));
#else
    #if !GLOBAL_DEVICE_VAR
    TriAccel* g_d_tri_accel = nullptr;
    #endif
    size_t accel_size = h_triangles.size() * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&g_d_tri_accel, accel_size));
    CUDA_CHECK(cudaMemcpy(g_d_tri_accel, h_triangles.data(), accel_size, cudaMemcpyHostToDevice));
#endif

#if !GLOBAL_DEVICE_VAR
    Gaussian* g_d_gaussians_persistent = nullptr;
#endif
    size_t gaussians_bytes = gaussians.size() * sizeof(Gaussian);
    CUDA_CHECK(cudaMalloc(&g_d_gaussians_persistent, gaussians_bytes));
    CUDA_CHECK(cudaMemcpy(g_d_gaussians_persistent, gaussians.data(), gaussians_bytes, cudaMemcpyHostToDevice));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] GPU memcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("2. gpu memcpy done\n");

    // 텍스처 바인딩

    // 리소스 디스크립터(Resource Descriptor) 설정
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear; // 1D 배열이므로 Linear 타입

    // 텍스처 디스크립터(Texture Descriptor) 설정
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; // 주소 지정 모드
    texDesc.filterMode = cudaFilterModePoint;      // 필터링 없음 (tex1Dfetch와 동일)
    texDesc.readMode = cudaReadModeElementType;    // 원본 타입 그대로 읽기
    texDesc.normalizedCoords = 0;                  // 정규화되지 않은 좌표 사용

    // 각 버퍼에 대해 텍스처 객체 생성 및 전역 변수에 복사
    // k-d 트리 노드 텍스처 객체 생성
    resDesc.res.linear.devPtr = g_d_kdtree_nodes;
    resDesc.res.linear.desc = cudaCreateChannelDesc<uint2>();
    //resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
    resDesc.res.linear.sizeInBytes = node_size;

    // 지역 변수(texNode) 대신 호스트 전역 변수(h_inKdTreeNodeTex)에 핸들을 저장
    cudaTextureObject_t h_inKdTreeNodeTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inKdTreeNodeTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inKdTreeNodeTex, &h_inKdTreeNodeTex, sizeof(cudaTextureObject_t)));

#if OFFSET_TEXTURE
    // 삼각형 오프셋 텍스처 객체 생성
    resDesc.res.linear.devPtr = g_d_prim_offsets;
    resDesc.res.linear.desc = cudaCreateChannelDesc<unsigned int>();
    resDesc.res.linear.sizeInBytes = offset_size;
    printf("g_d_prim_offsets: %u\n", offset_size);

    // 호스트 전역 변수에 핸들을 저장
    cudaTextureObject_t h_inObjectOffsetListTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inObjectOffsetListTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inObjectOffsetListTex, &h_inObjectOffsetListTex, sizeof(cudaTextureObject_t)));
#else
    CUDA_CHECK(cudaMemcpyToSymbol(g_d_tri_offsets_dev, &g_d_prim_offsets, sizeof(unsigned int*)));
#endif

#if TRIACC_TEXTURE
    // 삼각형 가속 구조체 텍스처 객체 생성
    #if WALD_METHOD
    resDesc.res.linear.devPtr = g_d_waldInfo;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>(); // WaldInfo는 float4 3개로 구성
    resDesc.res.linear.sizeInBytes = wald_info_size;
    #else
    resDesc.res.linear.devPtr = g_d_tri_accel;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDesc.res.linear.sizeInBytes = accel_size;
    #endif

    // 호스트 전역 변수에 핸들을 저장
    cudaTextureObject_t h_inTriAccelTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inTriAccelTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inTriAccelTex, &h_inTriAccelTex, sizeof(cudaTextureObject_t)));
#else
    #if WALD_METHOD
    CUDA_CHECK(cudaMemcpyToSymbol(g_d_tri_acc_dev, &g_d_waldInfo, sizeof(float4*)));
    #else
    CUDA_CHECK(cudaMemcpyToSymbol(g_d_tri_acc_dev, &g_d_tri_accel, sizeof(unsigned int*));
    #endif
#endif

#if GAUSSIAN_TEXTURE
    // 가우시안 데이터 텍스춰 객체 생성
    resDesc.res.linear.devPtr = g_d_gaussians_persistent;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDesc.res.linear.sizeInBytes = gaussians_bytes;

    // 호스트 전역 변수에 핸들을 저장
    cudaTextureObject_t h_inGaussianTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&h_inGaussianTex, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaMemcpyToSymbol(inGaussianTex, &h_inGaussianTex, sizeof(cudaTextureObject_t)));
#else
    if (g_d_gaussians) cudaFree(g_d_gaussians);
    CUDA_CHECK(cudaMemcpyToSymbol(g_d_gaussians, &g_d_gaussians_persistent, sizeof(Gaussian*)));
#endif

    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] texture bind failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("3. texture Bind done\n");

    // 상수 메모리 설정
    float3 h_bbox_min = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    float3 h_bbox_max = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMin, &h_bbox_min, sizeof(float3)));
    CUDA_CHECK(cudaMemcpyToSymbol(g_SceneBBoxMax, &h_bbox_max, sizeof(float3)));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] const memory set failed: " << cudaGetErrorString(err) << std::endl;
    }
    //printf("4. const memory set done\n");

    int maxSharedMemPerBlock;
    // 현재 GPU의 "블록 당 최대 공유 메모리" 속성
    cudaDeviceGetAttribute(
        &maxSharedMemPerBlock,
        cudaDevAttrMaxSharedMemoryPerBlock,
        deviceID
    );
    printf("This GPU's max shared memory per block: %d bytes\n", maxSharedMemPerBlock);
    // 49152 bytes
    // 49152 / (256 * 8) = 24
#if SHORT_STACK_DEPTH > 24
    cudaFuncSetAttribute(
        renderKernelGaussian_sortNode,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        maxSharedMemPerBlock
    );
#endif
    //cudaFuncSetCacheConfig(renderKernelGaussian_sortNode, cudaFuncCachePreferShared);
#if USE_STACK == GLOBAL_STACK
    cudaFuncSetCacheConfig(renderKernelGaussian_sortNode, cudaFuncCachePreferL1);
#endif
    //cudaFuncSetCacheConfig(renderKernelGaussian_sortNode, cudaFuncCachePreferEqual);

    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));


#if WALD_METHOD
    _aligned_free(h_waldInfo);
#endif

    if (h_inKdTreeNodeTex)          cudaDestroyTextureObject(h_inKdTreeNodeTex);
#if OFFSET_TEXTURE
    if (h_inObjectOffsetListTex)    cudaDestroyTextureObject(h_inObjectOffsetListTex);
#endif
#if TRIACC_TEXTURE
    if (h_inTriAccelTex)            cudaDestroyTextureObject(h_inTriAccelTex);
#endif
#if GAUSSIAN_TEXTURE
    if (h_inGaussianTex)            cudaDestroyTextureObject(h_inGaussianTex);
#endif
}
#endif //PRIMITIVE_TYPE

#include <fstream>

void save_matrix_csv(const std::string& filename, float* data, int width, int height, int stride = 1) {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 구조체 배열에서 특정 채널만 뽑아내기 위한 인덱싱
            // float4 형태이므로 stride(보통 4)를 곱하고 오프셋을 더함
            float val = data[y * width + x];

            file << (int)val;
            if (x < width - 1) file << ","; // 마지막 열이 아니면 쉼표 추가
        }
        file << "\n"; // 행 바꿈
    }
    file.close();
}

// 0.0~1.0 사이의 값을 Jet 컬러맵 RGB로 변환하는 함수
void get_jet_color(float v, unsigned char& r, unsigned char& g, unsigned char& b) {
    // 값을 0~1 사이로 클램핑
    v = std::max(0.0f, std::min(1.0f, v));

    float r_val = std::max(0.0f, std::min(1.0f, 4.0f * v - 3.0f));
    float g_val = std::max(0.0f, std::min(1.0f, 4.0f * v - 2.0f)) - std::max(0.0f, std::min(1.0f, 4.0f * v - 3.5f)); // 근사치 조정
    // 단순화된 Jet 로직
    if (v < 0.125f) { r = 0; g = 0; b = 127 + (unsigned char)(v * 1024); }
    else if (v < 0.375f) { r = 0; g = (unsigned char)((v - 0.125f) * 1020); b = 255; }
    else if (v < 0.625f) { r = (unsigned char)((v - 0.375f) * 1020); g = 255; b = 255 - (unsigned char)((v - 0.375f) * 1020); }
    else if (v < 0.875f) { r = 255; g = 255 - (unsigned char)((v - 0.625f) * 1020); b = 0; }
    else { r = 255 - (unsigned char)((v - 0.875f) * 1020); g = 0; b = 0; }
}

inline float saturate(float x) {
    // C++17 미만 버전을 사용하신다면 std::max(0.0f, std::min(1.0f, x)) 로 대체하세요.
    return std::max(0.0f, std::min(1.0f, x));
}

inline unsigned char floatToUchar(float val) {
    return static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, val * 255.0f + 0.5f)));
}

uchar3 turboColormapHost(float x)
{
    const float4 kRedVec4 = { 0.13572138f,  4.61539260f, -42.66032258f,  132.13108234f };
    const float4 kGreenVec4 = { 0.09140261f,  2.19418839f,   4.84296658f,  -14.18503333f };
    const float4 kBlueVec4 = { 0.10667330f, 12.64194608f, -60.58204836f,  110.36276771f };
    const float2 kRedVec2 = { -152.94239396f,  59.28637943f };
    const float2 kGreenVec2 = { 4.27729857f,   2.82956604f };
    const float2 kBlueVec2 = { -89.90310912f,  27.34824973f };

    x = saturate(x);

    float4 v4 = { 1.0f, x, x * x, x * x * x };
    float2 v2 = { v4.z * v4.z, v4.w * v4.z };

    // 다항식 계산 결과를 float으로 먼저 구합니다.
    float r = dot(v4, kRedVec4) + dot(v2, kRedVec2);
    float g = dot(v4, kGreenVec4) + dot(v2, kGreenVec2);
    float b = dot(v4, kBlueVec4) + dot(v2, kBlueVec2);

    // 0~255 범위의 unsigned char로 변환하여 반환합니다.
    return { floatToUchar(r), floatToUchar(g), floatToUchar(b) };
}

uchar3 computeVisualizationColorHost(float value, float2 minMax)
{
    float normalized_val = (value - minMax.x) / (minMax.y - minMax.x);
    return turboColormapHost(saturate(normalized_val));
}

// 히트맵 저장 메인 함수
void save_heatmap_stb(const char* filename, float* values, int width, int height, float max_val) {
    std::vector<unsigned char> image_data(width * height * 3); // RGB 3채널
    max_val = 120;
    for (int i = 0; i < width * height; i++) {
        if (values[i] <= 0.0f) {
            image_data[i * 3 + 0] = 0;
            image_data[i * 3 + 1] = 0;
            image_data[i * 3 + 2] = 0;
            continue;
        }

        //float normalized = (max_val > 0) ? (values[i] / max_val) : 0.0f;

        //unsigned char r, g, b;
        //get_jet_color(normalized, r, g, b);
        //image_data[i * 3 + 0] = r;
        //image_data[i * 3 + 1] = g;
        //image_data[i * 3 + 2] = b;

        float2 minMax = make_float2(0.0f, max_val);
        uchar3 rgb = computeVisualizationColorHost(values[i], minMax);
        image_data[i * 3 + 0] = rgb.x;
        image_data[i * 3 + 1] = rgb.y;
        image_data[i * 3 + 2] = rgb.z;
    }

    stbi_write_png(filename, width, height, 3, image_data.data(), width * 3);
}

void save_histograms_to_csv(const std::string& filename,
    const std::vector<int>& node_hist,
    const std::vector<int>& leaf_hist) {
    std::ofstream file(filename);
    file << "Value,Node_Visits,Leaf_Visits\n";

    int max_range = std::max(node_hist.size(), leaf_hist.size());
    for (int i = 0; i < max_range; ++i) {
        int v1 = (i < node_hist.size()) ? node_hist[i] : 0;
        int v2 = (i < leaf_hist.size()) ? leaf_hist[i] : 0;

        if (v1 > 0 || v2 > 0) { // 둘 중 하나라도 데이터가 있는 행만 기록
            file << i << "," << v1 << "," << v2 << "\n";
        }
    }
    file.close();
    std::cout << "Histogram saved to " << filename << std::endl;
}

#if DUMP_RENDER_IMAGE
void saveDeviceFramebufferToPNG(float* d_framebuffer, int width, int height, int channels = 3) {
    static int call = 0;
    char filename[256];
    snprintf(filename, sizeof(filename), "../../output/r_%d.png", call);
    call++;
    size_t num_float_bytes = width * height * channels * sizeof(float);

    float* h_float_pixels = (float*)malloc(num_float_bytes);

    cudaError_t err = cudaMemcpy(h_float_pixels, d_framebuffer, num_float_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Memcpy fail: %s\n", cudaGetErrorString(err));
        free(h_float_pixels);
        return;
    }
    unsigned char* h_byte_pixels = (unsigned char*)malloc(width * height * channels);

    for (int i = 0; i < width * height * channels; ++i) {
        float val = std::max(0.0f, std::min(1.0f, h_float_pixels[i]));
        h_byte_pixels[i] = (unsigned char)(val * 255.0f);
    }

    stbi_flip_vertically_on_write(1);

    stbi_write_png(filename, width, height, channels, h_byte_pixels, width * channels);

    free(h_float_pixels);
    free(h_byte_pixels);

    printf("image save success: %s\n", filename);
}
#endif

float renderGaussianWithCudaFrame(const Camera& camera, int width, int height, float* d_framebuffer, cudaStream_t stream
#if HIT_AND_NODE_COUNT_DEBUG
    , float3*& h_debug_buffer1, float3*& h_debug_buffer2
#endif
#if USE_STACK > SHORT_STACK
    , cu_traceState* d_global_stack
#endif
) {

    SceneInfo h_scene_info = { width, height };
    //CUDA_CHECK(cudaMemcpyToSymbol(g_SceneInfo, &h_scene_info, sizeof(SceneInfo), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(g_SceneInfo, &h_scene_info, sizeof(SceneInfo), 0, cudaMemcpyHostToDevice, stream));

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
    //CUDA_CHECK(cudaMemcpyToSymbol(g_CameraInfo, &h_camera_info, sizeof(CameraInfo), 0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbolAsync(g_CameraInfo, &h_camera_info, sizeof(CameraInfo), 0, cudaMemcpyHostToDevice, stream));

    // 커널 실행
    dim3 threads(DIM_X, DIM_Y);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    size_t shared_mem_size = threads.x * threads.y * SHORT_STACK_DEPTH * sizeof(cu_traceState);

#if HIT_AND_NODE_COUNT_DEBUG
    float3* d_debug_buffer1;
    float3* d_debug_buffer2;
    CUDA_CHECK(cudaMalloc(&d_debug_buffer1, width * height * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_debug_buffer2, width * height * sizeof(float3)));

    CUDA_CHECK(cudaEventRecord(start_ev));
    renderKernelGaussian_sortNode << < blocks, threads, shared_mem_size, stream >> > (d_framebuffer
        , d_debug_buffer1, d_debug_buffer2
    #if USE_STACK > SHORT_STACK
        , d_global_stack
    #endif
        );
    //CUDA_CHECK(cudaGetLastError());        // DEBUG: launch 실패 확인
    //CUDA_CHECK(cudaDeviceSynchronize());   // DEBUG: 실행 중 오류 확인
    CUDA_CHECK(cudaEventRecord(stop_ev));
    CUDA_CHECK(cudaEventSynchronize(stop_ev));

    //float3* h_debug_buffer1 = new float3[width * height];
    //float3* h_debug_buffer2 = new float3[width * height];
    if (h_debug_buffer1 == nullptr) { // 처음 한 번만 할당
        h_debug_buffer1 = new float3[width * height];
    }
    if (h_debug_buffer2 == nullptr) { // 처음 한 번만 할당
        h_debug_buffer2 = new float3[width * height];
    }
    CUDA_CHECK(cudaMemcpy(h_debug_buffer1, d_debug_buffer1, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_debug_buffer2, d_debug_buffer2, width * height * sizeof(float3), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_debug_buffer1));
    CUDA_CHECK(cudaFree(d_debug_buffer2));

    int max_node_visits = 0;
    int max_leaf_visits = 0;
    int max_intersection_tests = 0;
    int max_hits_found = 0;
    int max_blend_ops = 0;
    int max_max_sort_size = 0;

    int node_visits = 0;
    int leaf_visits = 0;
    int intersection_tests = 0;
    int hits_found = 0;
    int blend_ops = 0;
    int max_sort_size = 0;

    float avg_node_visits = 0;
    float avg_leaf_visits = 0;
    float avg_intersection_tests = 0;
    float avg_hits_found = 0;
    float avg_blend_ops = 0;
    float avg_max_sort_size = 0;

    std::vector<int> hist_node_visits(1024, 0);
    std::vector<int> hist_leaf_visits(1024, 0);
    std::vector<int> hist_intersection_tests(1024, 0);
    std::vector<int> hist_hits_found(256, 0);      // 히트 수는 보통 노드 방문보다 적음
    // max_sort_size는 보통 값이 작으므로 작게 설정
    std::vector<int> hist_max_sort_size(128, 0);

    // 벡터 크기를 안전하게 관리하기 위한 람다 함수 (범위 초과 방지)
    auto add_to_hist = [](std::vector<int>& hist, int value) {
        if (value >= 0) {
            if (value >= hist.size()) {
                hist.resize(value + 1, 0);
            }
            hist[value]++;
        }
        };

    // 각 버퍼에서 올바른 값을 가져와 최댓값을 계산
    for (int i = 0; i < width * height; i++) {
        int cur_node_visits = (int)h_debug_buffer1[i].x;
        int cur_leaf_visits = (int)h_debug_buffer1[i].y;
        int cur_intersection_tests = (int)h_debug_buffer1[i].z;
        int cur_hits_found = (int)h_debug_buffer2[i].x;
        int cur_max_sort_size = (int)h_debug_buffer2[i].z;

        max_node_visits = max(max_node_visits, (int)h_debug_buffer1[i].x);
        max_leaf_visits = max(max_leaf_visits, (int)h_debug_buffer1[i].y);
        max_intersection_tests = max(max_intersection_tests, (int)h_debug_buffer1[i].z);

        max_hits_found = max(max_hits_found, (int)h_debug_buffer2[i].x);
        max_blend_ops = max(max_blend_ops, (int)h_debug_buffer2[i].y);
        max_max_sort_size = max(max_max_sort_size, (int)h_debug_buffer2[i].z);

        if (h_debug_buffer2[i].y != 0.0f) { // blending 일어난 픽셀만 계산
            avg_node_visits += h_debug_buffer1[i].x;
            avg_leaf_visits += h_debug_buffer1[i].y;
            avg_intersection_tests += h_debug_buffer1[i].z;

            avg_hits_found += h_debug_buffer2[i].x;
            avg_blend_ops += h_debug_buffer2[i].y;
            avg_max_sort_size += h_debug_buffer2[i].z;
            
            node_visits++;
            leaf_visits++;
            intersection_tests++;
            hits_found++;
            blend_ops++;
            max_sort_size++;

            add_to_hist(hist_node_visits, cur_node_visits);
            add_to_hist(hist_leaf_visits, cur_leaf_visits);
            add_to_hist(hist_intersection_tests, cur_intersection_tests);
            add_to_hist(hist_hits_found, cur_hits_found);
            add_to_hist(hist_max_sort_size, cur_max_sort_size);
        }
    }



    // 데이터를 임시로 담을 float 배열 생성
    std::vector<float> temp_buffer(width * height);
    std::string dirPath = "../../output/";

    save_histograms_to_csv(dirPath + "render_stats.csv", hist_node_visits, hist_leaf_visits);

    // 1. Node Visits 히트맵
    for (int i = 0; i < width * height; ++i) temp_buffer[i] = (float)h_debug_buffer1[i].x;
    save_heatmap_stb((dirPath + "heatmap_node_visits.png").c_str(), temp_buffer.data(), width, height, (float)max_node_visits);
    save_matrix_csv((dirPath + "heatmap_node_visits.csv").c_str(), temp_buffer.data(), width, height, (float)max_node_visits);

    // 2. Leaf Visits 히트맵
    for (int i = 0; i < width * height; ++i) temp_buffer[i] = (float)h_debug_buffer1[i].y;
    save_heatmap_stb((dirPath + "heatmap_leaf_visits.png").c_str(), temp_buffer.data(), width, height, (float)max_leaf_visits);
    save_matrix_csv((dirPath + "heatmap_leaf_visits.csv").c_str(), temp_buffer.data(), width, height, (float)max_leaf_visits);

    // 3. Intersection Tests 히트맵
    for (int i = 0; i < width * height; ++i) temp_buffer[i] = (float)h_debug_buffer1[i].z;
    save_heatmap_stb((dirPath + "heatmap_intersection.png").c_str(), temp_buffer.data(), width, height, (float)max_intersection_tests);
    save_matrix_csv((dirPath + "heatmap_intersection.csv").c_str(), temp_buffer.data(), width, height, (float)max_intersection_tests);

    // 4. Hits Found 히트맵
    for (int i = 0; i < width * height; ++i) temp_buffer[i] = (float)h_debug_buffer2[i].x;
    save_heatmap_stb((dirPath + "heatmap_hits_found.png").c_str(), temp_buffer.data(), width, height, (float)max_hits_found);
    save_matrix_csv((dirPath + "heatmap_hits_found.csv").c_str(), temp_buffer.data(), width, height, (float)max_hits_found);

    // 5. Blend Ops 히트맵
    for (int i = 0; i < width * height; ++i) temp_buffer[i] = (float)h_debug_buffer2[i].y;
    save_heatmap_stb((dirPath + "heatmap_blend_ops.png").c_str(), temp_buffer.data(), width, height, (float)max_blend_ops);
    save_matrix_csv((dirPath + "heatmap_blend_ops.csv").c_str(), temp_buffer.data(), width, height, (float)max_blend_ops);

    // 6. Max Sort Size 히트맵
    for (int i = 0; i < width * height; ++i) temp_buffer[i] = (float)h_debug_buffer2[i].z;
    save_heatmap_stb((dirPath + "heatmap_max_sort_size.png").c_str(), temp_buffer.data(), width, height, (float)max_max_sort_size);
    save_matrix_csv((dirPath + "heatmap_max_sort_size.csv").c_str(), temp_buffer.data(), width, height, (float)max_max_sort_size);

    std::string path = dirPath + "renderStats.txt";
    FILE* fp = fopen(path.c_str(), "w");
    if (fp != NULL) {
        fprintf(fp, "\n===========================================\n");
        fprintf(fp, "avg_node_visits\t\t: %f\n", avg_node_visits / node_visits);
        fprintf(fp, "avg_leaf_visits\t\t: %f\n", avg_leaf_visits / leaf_visits);
        fprintf(fp, "avg_intersection_tests\t: %f\n", avg_intersection_tests / intersection_tests);
        fprintf(fp, "avg_hits_found\t\t: %f\n", avg_hits_found / hits_found);
        fprintf(fp, "avg_blend_ops\t\t: %f\n", avg_blend_ops / blend_ops);
        fprintf(fp, "avg_max_sort_size\t: %f\n", avg_max_sort_size / max_sort_size);
        fprintf(fp, "===========================================\n");
        fprintf(fp, "max_node_visits\t\t: %d\n", max_node_visits);
        fprintf(fp, "max_leaf_visits\t\t: %d\n", max_leaf_visits);
        fprintf(fp, "max_intersection_tests\t: %d\n", max_intersection_tests);
        fprintf(fp, "max_hits_found\t\t: %d\n", max_hits_found);
        fprintf(fp, "max_blend_ops\t\t: %d\n", max_blend_ops);
        fprintf(fp, "max_max_sort_size\t: %d\n", max_max_sort_size);
        fprintf(fp, "===========================================\n");
        fprintf(fp, "tot_node_visits\t\t: %f\n", avg_node_visits);
        fprintf(fp, "tot_leaf_visits\t\t: %f\n", avg_leaf_visits);
        fprintf(fp, "tot_intersection_tests\t: %f\n", avg_intersection_tests);
        fprintf(fp, "tot_hits_found\t\t: %f\n", avg_hits_found);
        fprintf(fp, "tot_blend_ops\t\t: %f\n", avg_blend_ops);
        fprintf(fp, "tot_max_sort_size\t: %f\n", avg_max_sort_size);
        fprintf(fp, "===========================================\n");

        // 작업이 끝나면 반드시 파일을 닫아주어야 안전하게 저장됩니다.
        fclose(fp);
    }
    else {
        printf("cannot open file: %s.\n", path);
    }
    printf("\n===========================================\n");
    printf("avg_node_visits\t\t: %f\n", avg_node_visits / node_visits);
    printf("avg_leaf_visits\t\t: %f\n", avg_leaf_visits / leaf_visits);
    printf("avg_intersection_tests\t: %f\n", avg_intersection_tests / intersection_tests);
    printf("avg_hits_found\t\t: %f\n", avg_hits_found / hits_found);
    printf("avg_blend_ops\t\t: %f\n", avg_blend_ops / blend_ops);
    printf("avg_max_sort_size\t: %f\n", avg_max_sort_size / max_sort_size);
    printf("===========================================\n");
    printf("max_node_visits\t\t: %d\n", max_node_visits);
    printf("max_leaf_visits\t\t: %d\n", max_leaf_visits);
    printf("max_intersection_tests\t: %d\n", max_intersection_tests);
    printf("max_hits_found\t\t: %d\n", max_hits_found);
    printf("max_blend_ops\t\t: %d\n", max_blend_ops);
    printf("max_max_sort_size\t: %d\n", max_max_sort_size);
    printf("===========================================\n");
    printf("tot_node_visits\t\t: %f\n", avg_node_visits);
    printf("tot_leaf_visits\t\t: %f\n", avg_leaf_visits);
    printf("tot_intersection_tests\t: %f\n", avg_intersection_tests);
    printf("tot_hits_found\t\t: %f\n", avg_hits_found);
    printf("tot_blend_ops\t\t: %f\n", avg_blend_ops);
    printf("tot_max_sort_size\t: %f\n", avg_max_sort_size);
    printf("===========================================\n");

    //delete[] h_debug_buffer1;
    //delete[] h_debug_buffer2;
#else
    CUDA_CHECK(cudaEventRecord(start_ev, stream)); // 시작 기록
    renderKernelGaussian_sortNode << < blocks, threads, shared_mem_size, stream >> > (d_framebuffer
    #if USE_STACK > SHORT_STACK
        , d_global_stack
    #endif
        );
    CUDA_CHECK(cudaEventRecord(stop_ev, stream)); // 종료 기록
    CUDA_CHECK(cudaEventSynchronize(stop_ev)); // GPU 작업 완료까지 대기
#endif

#if DUMP_RENDER_IMAGE
    saveDeviceFramebufferToPNG(d_framebuffer, width, height);
#endif

#if WARP_OCCUPANCY
    std::cout << "\n========= GPU Device Properties =========" << std::endl;
    std::cout << "Device: " << deviceProp.name << std::endl;
    std::cout << "Max Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Max Shared Memory per SM: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "Max Registers per Block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Max Registers per SM: " << deviceProp.regsPerMultiprocessor << std::endl;

    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, renderKernelGaussian_sortNode));

    std::cout << "========= Kernel Properties ('renderKernelGaussian_sortNode') =========" << std::endl;
    std::cout << "Registers used per Thread          : " << attr.numRegs << std::endl;
    std::cout << "Static Shared Memory used per Block: " << attr.sharedSizeBytes << " bytes" << std::endl;
    std::cout << "Constant      Memory used per Block: " << attr.constSizeBytes << " bytes" << std::endl;
    std::cout << "Local         Memory used per Block: " << attr.localSizeBytes << " bytes" << std::endl;

    int blockSize = DIM_X * DIM_Y;
    // SM 당 최대 활성 블록 개수 계산
    int maxActiveBlocks;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        renderKernelGaussian_sortNode,
        blockSize,
        shared_mem_size
    ));

    // 점유율(Occupancy) 계산
    // SM이 최대로 가질 수 있는 스레드(Warp) 수
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int warpSize = deviceProp.warpSize;
    int maxWarpsPerSM = maxThreadsPerSM / warpSize;

    // 현재 커널 설정에서 활성화되는 Warp 수
    int activeWarpsPerSM = maxActiveBlocks * (blockSize / warpSize);

    // 최종 점유율 (%)
    double occupancy = (double)activeWarpsPerSM / maxWarpsPerSM;

    std::cout << "========= Occupancy Calculation =========" << std::endl;
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Max Active Blocks per SM: " << maxActiveBlocks << std::endl;
    std::cout << "Dynamic Shared Memory per Block: " << shared_mem_size << " bytes" << std::endl;
    std::cout << "Max Warps per SM on this device: " << maxWarpsPerSM << std::endl;
    std::cout << "Active Warps per SM for this kernel: " << activeWarpsPerSM << std::endl;
    std::cout << "Theoretical Occupancy: " << occupancy * 100.0 << "%" << std::endl;
#endif

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_ev, stop_ev));
    float k_fps = 1000.0f / milliseconds;

#if HIT_AND_NODE_COUNT_DEBUG
    //if(k_fps < 100.0f)
    printf("FPS : %f-------------------------------------------------------------\n", k_fps);
    //if (++frame_count >= 100) {
    //    printf("avg FPS for 100 frame : %f\n", (float)(total_frame / frame_count));
    //    frame_count = 0;
    //    total_frame = 0.0f;
    //}
#endif

    return k_fps;
}

void cleanupCudaResources() {
    printf("Cleaning up CUDA resources...\n");
#if GLOBAL_DEVICE_VAR
    if (g_d_kdtree_nodes) cudaFree(g_d_kdtree_nodes);
    if (g_d_prim_offsets) cudaFree(g_d_prim_offsets);
#if WALD_METHOD
    if (g_d_waldInfo) cudaFree(g_d_waldInfo);
#else
    if (g_d_tri_accel) cudaFree(g_d_tri_accel);
#endif
    if (g_d_gaussians_persistent) cudaFree(g_d_gaussians_persistent);
#endif

    //cudaUnbindTexture(inKdTreeNodeTex);
    //cudaUnbindTexture(inObjectOffsetListTex);
    //cudaUnbindTexture(inTriAccelTex);
    //cudaUnbindTexture(inGaussianTex);

    if (inKdTreeNodeTex)       cudaDestroyTextureObject(inKdTreeNodeTex);
#if OFFSET_TEXTURE
    if (inObjectOffsetListTex) cudaDestroyTextureObject(inObjectOffsetListTex);
#else
    if (g_d_tri_offsets_dev)   cudaFree(g_d_tri_offsets_dev);
#endif
#if TRIACC_TEXTURE
    if (inTriAccelTex)         cudaDestroyTextureObject(inTriAccelTex);
#endif
#if GAUSSIAN_TEXTURE
    if (inGaussianTex)         cudaDestroyTextureObject(inGaussianTex);
#else
    if (g_d_gaussians) cudaFree(g_d_gaussians);
#endif

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
}