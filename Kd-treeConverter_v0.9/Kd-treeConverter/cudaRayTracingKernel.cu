//#include "sgrt_interface.h"
//#include "SGRTx2Lib/cudaRenderCommon.cuh"
//#include "SGRTx2Lib/cuda_math.h"
//#include "Kd-treeConverter.h"
//
//#include <cuda.h>
//#include <cuda_runtime.h>
//
//struct IntersectionResult {
//    bool hit;              // 교차 여부
//    float t;               // ray.tHit (ray origin으로부터의 거리)
//    float3 hitPoint;       // 교차 지점 위치
//    float3 normal;         // 교차 지점에서의 법선 벡터
//    float3 color;          // 재질 기반 색상
//    int triangleIndex;     // 교차된 삼각형 인덱스
//    int materialID;        // 재질 ID (예: 반사, 투과 여부 등)
//
//    __device__ __host__ void init() {
//        hit = false;
//        t = 1e30f;
//        hitPoint = make_float3(0, 0, 0);
//        normal = make_float3(0, 0, 0);
//        color = make_float3(0.1f, 0.1f, 0.1f); // 기본 색상
//        triangleIndex = -1;
//        materialID = -1;
//    }
//};
//
//__device__ bool intersectRayWithTriAccel(const Ray& ray, const TriAccel& tri, float* t, float* u, float* v) {
//    int k = tri.k;
//
//    float nu = tri.n_u;
//    float nv = tri.n_v;
//    float nd = tri.n_d;
//
//    float dir_k = ray.df[k];
//    float denom = dir_k + nu * ray.df[(k + 1) % 3] + nv * ray.df[(k + 2) % 3];
//    if (fabsf(denom) < 1e-6f) return false;
//
//    float tval = -(
//        ray.of[k] + nu * ray.of[(k + 1) % 3] + nv * ray.of[(k + 2) % 3] + nd
//        ) / denom;
//
//    if (tval < 0.0f || tval > 1e30f) return false;
//
//    float hu = ray.of[(k + 1) % 3] + ray.df[(k + 1) % 3] * tval;
//    float hv = ray.of[(k + 2) % 3] + ray.df[(k + 2) % 3] * tval;
//
//    float alpha = hu * tri.b_nu + hv * tri.b_nv + tri.b_d;
//    float beta = hu * tri.c_nu + hv * tri.c_nv + tri.c_d;
//
//    if (alpha < 0.0f || beta < 0.0f || (alpha + beta) > 1.0f) return false;
//
//    *t = tval;
//    *u = alpha;
//    *v = beta;
//    return true;
//}
//
//__device__ bool BoundsRayIntersect(const float* aabb, const Ray& ray, float& tmin, float& tmax) {
//    tmin = 0.0f;
//    tmax = 1e30f;
//
//    for (int i = 0; i < 3; ++i) {
//        float invD = 1.0f / ray.df[i];
//        float t0 = (aabb[2 * i] - ray.of[i]) * invD;
//        float t1 = (aabb[2 * i + 1] - ray.of[i]) * invD;
//
//        if (invD < 0.0f) {
//            float temp = t0;
//            t0 = t1;
//            t1 = temp;
//        }
//
//        tmin = t0 > tmin ? t0 : tmin;
//        tmax = t1 < tmax ? t1 : tmax;
//
//        if (tmax <= tmin) return false;
//    }
//    return true;
//}
//
//__device__ bool kdTreeIntersect(CompositeObject* obj, const Ray& ray, IntersectionResult* result) {
//    const KdTree* tree = obj->kd_tree;
//    const KdTreeNode* nodes = tree->tree;
//    const unsigned int* triOffsetList = tree->tri_offset_list;
//    const TriAccel* triAccels = tree->tri_accel_list;
//
//    float t_near = 0.0f, t_far = 1e30f;
//
//    if (!BoundsRayIntersect(tree->AABB, ray, t_near, t_far)) return false;
//
//    const int maxStack = 64;
//    int stack[maxStack];
//    float tstack[maxStack];
//    int stackPtr = 0;
//
//    int nodeIdx = 0;
//    KdTreeNode node = nodes[nodeIdx];
//
//    while (true) {
//        while (!IS_LEAF(node)) {
//            int axis = SPLIT_AXIS(node);
//            float splitPos = SPLIT_POS(node);
//
//            float origin = ray.of[axis];
//            float dir = ray.df[axis];
//
//            float t_split = (splitPos - origin) / dir;
//
//            int left = FIRST_CHILD_OFFSET(node);
//            int right = SECOND_CHILD_OFFSET(node);
//
//            bool below = (origin < splitPos) || (origin == splitPos && dir <= 0);
//
//            if (t_split > t_far || t_split <= 0.0f) {
//                nodeIdx = below ? left : right;
//            }
//            else if (t_split < t_near) {
//                nodeIdx = below ? right : left;
//            }
//            else {
//                if (stackPtr < maxStack) {
//                    stack[stackPtr] = below ? right : left;
//                    tstack[stackPtr] = t_far;
//                    ++stackPtr;
//                }
//                nodeIdx = below ? left : right;
//                t_far = t_split;
//            }
//
//            node = nodes[nodeIdx];
//        }
//
//        // 리프 노드: 삼각형 리스트 검사
//        int base = OBJECTLIST_OFFSET(node);
//        int count = OBJECT_SIZE(node);
//
//        for (int i = 0; i < count; ++i) {
//            int triIndex = triOffsetList[base + i];
//            const TriAccel& tri = triAccels[triIndex];
//
//            float t, u, v;
//            if (intersectRayWithTriAccel(ray, tri, &t, &u, &v)) {
//                if (t > 0.0f && t < result->t) {
//                    result->hit = true;
//                    result->t = t;
//                    result->triangleIndex = tri.indexInObject;
//                    //result->materialID = tri.material_ID;
//                    result->hitPoint = make_float3(
//                        ray.of[0] + ray.df[0] * t,
//                        ray.of[1] + ray.df[1] * t,
//                        ray.of[2] + ray.df[2] * t
//                    );
//                    result->normal = make_float3(tri.N[0], tri.N[1], tri.N[2]);
//                    result->color = make_float3(1, 1, 1); // 기본 diffuse 예시
//                }
//            }
//        }
//
//        if (stackPtr == 0) break;
//        --stackPtr;
//        nodeIdx = stack[stackPtr];
//        t_far = tstack[stackPtr];
//        node = nodes[nodeIdx];
//    }
//
//    return result->hit;
//}
//
//__global__ void singlePassRayTracingKernel(
//    float* framebuffer,
//    int maxReflectionDepth,
//    int sampleX, int sampleY,
//    bool jittering,
//    cuCamera camera,
//    SceneInfo scene,
//    CompositeObject* obj
//) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if (x >= scene.iResolutionX || y >= scene.iResolutionY)
//        return;
//
//    // Supersampling position 계산
//    float invSX = 1.0f / scene.iSuperSamplingX;
//    float invSY = 1.0f / scene.iSuperSamplingY;
//    float sx = (float)x + ((float)sampleX + 0.5f) * invSX;
//    float sy = (float)y + ((float)sampleY + 0.5f) * invSY;
//
//    // Ray 생성
//    float3 dir = camera.startPoint +
//        camera.u * sx * camera.stepX -
//        camera.v * sy * camera.stepY;
//    dir = normalize(dir - camera.eye);
//
//    Ray ray;
//    ray.of[0] = camera.eye.x;
//    ray.of[1] = camera.eye.y;
//    ray.of[2] = camera.eye.z;
//    //ray.of = camera.eye;
//    ray.df[0] = dir.x;
//    ray.df[1] = dir.y;
//    ray.df[2] = dir.z;
//    //ray.df = dir;
//    ray.Depth = 0;
//
//    float3 color = make_float3(0.0f, 0.0f, 0.0f);
//    IntersectionResult result;
//    result.init();
//
//    if (kdTreeIntersect(obj, ray, &result)) {
//        color = result.color;  // Diffuse shading
//    }
//
//    int idx = 3 * ((scene.iResolutionY - y - 1) * scene.iResolutionX + x);
//    framebuffer[idx + 0] += color.x * invSX * invSY;
//    framebuffer[idx + 1] += color.y * invSX * invSY;
//    framebuffer[idx + 2] += color.z * invSX * invSY;
//}
