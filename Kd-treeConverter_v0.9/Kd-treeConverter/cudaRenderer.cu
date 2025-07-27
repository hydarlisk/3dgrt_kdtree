#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaRenderer.h"
#define M_PI 3.14159f

// =================================================================================
// 1. 제공된 SGRTx2Lib 커널 및 관련 구조체/헬퍼 함수
// (이전 질문에서 제공된 코드를 그대로 사용합니다)
// =================================================================================

// Scene, Camera 정보 (상수 메모리 사용)
struct SceneInfo {
    int iResolutionX, iResolutionY;
    int iSuperSamplingX, iSuperSamplingY;
    int iBlockSizeX, iBlockSizeY;
};

struct CameraInfo {
    float3 eye;
    float3 u, v;
    float3 startPoint;
    float stepX, stepY;
};

__constant__ SceneInfo g_SceneInfo;
__constant__ CameraInfo g_CameraInfo;
__constant__ float3 g_SceneBBox[2]; // Scene의 AABB

// 광선(Ray) 구조체
struct cuRay {
    float3 pos;
    float3 dir;

    __device__ float2 get_dir_pos(const unsigned axis) const {
        if (axis == 0) return make_float2(pos.x, dir.x);
        if (axis == 1) return make_float2(pos.y, dir.y);
        return make_float2(pos.z, dir.z);
    }
};

// 교차점(Intersection) 정보 구조체
struct cuIntersectionCheck {
    float tHit;
    float beta, gamma;
    int triIndex;
    int objectIndex;
    bool bSelected;

    __device__ void init() { tHit = FLT_MAX; triIndex = -1; objectIndex = -1; bSelected = false; }
    __device__ bool isHit() const { return triIndex != -1; }
};

// Kd-tree 텍스처 참조 선언
texture<uint2, 1, cudaReadModeElementType> inKdTreeNodeTex;
texture<uint, 1, cudaReadModeElementType> inObjectOffsetListTex;
texture<float4, 1, cudaReadModeElementType> inWaldTriangleTex;

// --- 여기에 singlePassIntersectRoutine, singlePassIntersect, singlePassRayTracingKernel_ShadowOff 등
// --- 제공된 모든 __device__ 및 __global__ 함수를 붙여넣으세요.
// --- (내용이 길어 생략합니다. 반드시 이전 질문의 커널 코드를 모두 복사해와야 합니다.)

/**
 *	WALD Intersection method.
 */
__device__ inline void singlePassIntersectRoutine(const cuRay& ray, const int id, cuIntersectionCheck& hit,
    const float t_near, const float t_far)
{
    cuWaldTriangleInfo tri;
    tri.internal0 = tex1Dfetch(inWaldTriangleTex, 3 * id);
    tri.internal1 = tex1Dfetch(inWaldTriangleTex, 3 * id + 1);
    tri.internal2 = tex1Dfetch(inWaldTriangleTex, 3 * id + 2);

    cuWaldTriangleInfo::perm_t p = tri.get_perm(ray);
    //const float dot = ( tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z );
    p.pos.x = (tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z);
    const float denum = (p.dir.x + tri.n_u() * p.dir.y + tri.n_v() * p.dir.z);
    const float t = __fdividef(p.pos.x, denum);

    if (isnan(t)) return;
    if ((hit.tHit <= t) | (t < t_near - EPSILON4) | (t > t_far + EPSILON4)) return;

    /**
     *	culling 옵션이 있고, object 가 transparent 하지 않다면
     *	앞면인지 뒷면인지 체크. 뒷면에 맞은거면 hit 처리 안함.
     */
    const float hu = p.pos.y + t * p.dir.y - tri.vert_ku();
    const float hv = p.pos.z + t * p.dir.z - tri.vert_kv();
    const float beta = hv * tri.b_nu() + hu * tri.b_nv();
    const float gamma = hu * tri.c_nu() + hv * tri.c_nv();

    /** 삼각형의 edge 와 부딪힐때, 수치오차가 있으므로 epsilon 을 좀 준다. */
    //if ( isnan( beta * gamma ) ) return;
    if ((beta < 0.f - BARYCENTRY_EPSILON) | (gamma < 0.f - BARYCENTRY_EPSILON) | ((1.0f - beta - gamma) < 0.0f - BARYCENTRY_EPSILON)) return;

    hit.tHit = t;
    hit.beta = beta;
    hit.gamma = gamma;
    hit.triIndex = id;
    hit.objectIndex = tri.getObjectIndex();
    hit.bSelected = tri.isSelection();
}

__device__ inline void singlePassIntersect(cuRay& currRay, cuIntersectionCheck& intersectionCheck)
{
    float t_scene_near = 0.0f, t_scene_far = FLT_MAX;
    //float t_scene_near = currRay.mint, t_scene_far = currRay.maxt;
    if (BoundsRayIntersect(g_SceneBBox, currRay, t_scene_near, t_scene_far))
    {
        float t_near = t_scene_near, t_far = t_scene_far;
        const unsigned smem_baseOffset = umul24(threadIdx.y, blockDim.x) + threadIdx.x;
        shortStack stack;
        stack.init(smem_baseOffset);

        kdtreeNode node = tex1Dfetch(inKdTreeNodeTex, 0);
        while (true)
        {
            while (!IS_LEAF(node))
            {
                //const int axis = SPLIT_AXIS(node);
                //const float splitPos = SPLIT_POS(node);
                const float2 pos_dir = currRay.get_dir_pos(SPLIT_AXIS(node));
                //const float dir = pos_dir.y;			
                const float t_split = __fdividef(SPLIT_POS(node) - pos_dir.x, pos_dir.y);
                const unsigned sign = signbit(pos_dir.y);
                const unsigned childOffset = FIRST_CHILD_OFFSET(node);
                unsigned idx = childOffset + (sign ^ (t_split <= t_near));
                //if(t_split <= t_near) 
                //	idx = childOffset + (sign^1);
                if (t_near < t_split && t_split < t_far) {
                    stack.push(childOffset + (sign ^ 1), t_far);
                    t_far = t_split;
                }
                node = tex1Dfetch(inKdTreeNodeTex, idx);
            }


            unsigned baseOffset = OBJECTLIST_OFFSET(node);
            int objectSize = OBJECT_SIZE(node) + baseOffset;

            for (; baseOffset < objectSize; baseOffset++) {
                const unsigned objListOffset = tex1Dfetch(inObjectOffsetListTex, baseOffset);
#if INTERSECTION_METHOD == 0
                singlePassIntersectRoutine(currRay, objListOffset, intersectionCheck, t_near, t_far);
#elif INTERSECTION_METHOD == 1
                PlueckerIntersection(currRay, objListOffset, intersectionCheck, t_near, t_far, faceCCW, bCulling);
#endif
            }
            if (intersectionCheck.tHit <= t_far | t_far >= t_scene_far)
                break;
            if (stack.empty())
            {
                node = tex1Dfetch(inKdTreeNodeTex, 0);
                t_near = t_far;		t_far = t_scene_far;
            }
            else
            {
                const cu_traceState& trace = stack.top(); stack.pop();
                node = tex1Dfetch(inKdTreeNodeTex, trace.nodeID);
                t_near = t_far;
                t_far = trace.tMax;
            }
        }
    }
}

/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_ShadowOff(float* pFrameBuffer,
    int maxReflectionDepth,
    int sampleX, int sampleY, bool bJittering)
{
    float3 dir;
    float sx, sy;
    float invSamplingX = __fdividef(1.0f, g_SceneInfo.iSuperSamplingX);
    float invSamplingY = __fdividef(1.0f, g_SceneInfo.iSuperSamplingY);
    int depth = 0;

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    int x = umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int y = umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY)
        return;

    /**
     *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
     */

    if (g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering) {
        sx = (float)x + ((float)sampleX + radicalInverse(x * 256 + sampleX, 3)) * invSamplingX;
        sy = (float)y + ((float)sampleY + radicalInverse(y * 256 + sampleY, 5)) * invSamplingY;
    }
    else {
        sx = (float)x + ((float)sampleX + 0.5f) * invSamplingX;
        sy = (float)y + ((float)sampleY + 0.5f) * invSamplingY;
    }

    dir = g_CameraInfo.startPoint +
        g_CameraInfo.u * sx * g_CameraInfo.stepX -
        g_CameraInfo.v * sy * g_CameraInfo.stepY;
    dir = normalize(dir - g_CameraInfo.eye);

    cuRay currRay;
    currRay.dir = dir;
    currRay.pos = g_CameraInfo.eye;

    cuIntersectionPoint point;
    point.init();
    point.colorWeight.x = 1.0f;
    point.colorWeight.y = 1.0f;
    point.colorWeight.z = 1.0f;

    bool secondary;

    do {
        secondary = false;
        cuIntersectionCheck currIsectCheck;
        currIsectCheck.init();
        singlePassIntersect(currRay, currIsectCheck);

        /**
         *	intersection check
         */
        if (currIsectCheck.isHit()) {

            makeIntersectionPoint(&currRay, &currIsectCheck, &point);

            cuObjectMaterial material;
            getObjectMaterial(currIsectCheck.objectIndex, material);

            /** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
            float3 diffuse = material.diffuse;
            calTextureColor(point, material, diffuse);

            color += point.colorWeight *
                (1.0f - (material.transparency + material.reflection) * (maxReflectionDepth > depth))
                * calSinglePassDirectIllumination_ShadowOff(point, material, diffuse);

            if (maxReflectionDepth > depth && material.transparency > 0.0f) {

                dir = refraction(point.dir, point.normal, material.refractionIndex);
                point.colorWeight = point.colorWeight * material.transparency * diffuse;
                currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
                currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
                currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
                currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

                secondary = true;

            }
            else if (maxReflectionDepth > depth && material.reflection > 0.0f) {

                dir = reflection(point.dir, point.normal);
                point.colorWeight = point.colorWeight * material.reflection * diffuse;
                currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
                currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
                currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
                currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


                secondary = true;

            }


        }

        depth++;

    } while (secondary);

    int idx = 3 * ((g_SceneInfo.iResolutionY - y - 1) * g_SceneInfo.iResolutionX + x);
    pFrameBuffer[idx + 0] += color.x * invSamplingX * invSamplingY;
    pFrameBuffer[idx + 1] += color.y * invSamplingX * invSamplingY;
    pFrameBuffer[idx + 2] += color.z * invSamplingX * invSamplingY;
}

// =================================================================================
// 2. Host -> Device 데이터 변환을 위한 코드
// =================================================================================

// GPU 텍스처 포맷에 맞는 삼각형 데이터 구조
struct PackedTriangle {
    float4 data[3];
};

// GPU 텍스처 포맷에 맞는 Kd-tree 노드 구조
typedef uint2 PackedKdNode;

// TriAccel 구조체를 PackedTriangle로 변환
void packTriangle(PackedTriangle& dest, const TriAccel& src) {
    // data[0]: Plane equation & projection info
    dest.data[0].x = src.n_u;
    dest.data[0].y = src.n_v;
    dest.data[0].z = src.n_d;
    memcpy(&dest.data[0].w, (float*)src.k, sizeof(unsigned int)); // Bit-fields

    // data[1]: Line equation for edge 'ac' & index info
    dest.data[1].x = src.b_nu;
    dest.data[1].y = src.b_nv;
    dest.data[1].z = src.b_d;
    memcpy(&dest.data[1].w, &src.indexInObject, sizeof(int));

    // data[2]: Line equation for edge 'ab' & material ID
    dest.data[2].x = src.c_nu;
    dest.data[2].y = src.c_nv;
    dest.data[2].z = src.c_d;
    memcpy(&dest.data[2].w, &src.material_ID, sizeof(int));
}

// KdTreeNode 구조체를 PackedKdNode로 변환 (SGRT 형식에 맞춤)
void packKdNode(PackedKdNode& dest, const KdTreeNode& src) {
    // Kd-treeConverter의 KdTreeNode가 SGRT와 동일한 포맷을 사용한다고 가정
    // src.x, src.y 를 dest.x, dest.y에 그대로 복사
    dest.x = src.x;
    dest.y = src.y;
}


// =================================================================================
// 3. CudaRenderer.h 에 선언된 메인 렌더링 함수 구현
// =================================================================================

void launchCudaRender(const CompositeObject& object, const Camera& camera, int width, int height, float*& out_framebuffer, bool& is_done) {
    printf("--- Launching CUDA Rendering ---\n");
    is_done = false;

    // 0. Kd-tree 데이터 유효성 검사
    KdTree* kdTree = object.kd_tree;
    if (!kdTree || kdTree->tree_node_count == 0 || object.n_triangles == 0) {
        fprintf(stderr, "[CUDA] Error: Kd-tree data is not valid or empty.\n");
        return;
    }

    // 1. Host에서 변환된 데이터용 메모리 할당 및 변환 수행
    printf("[CUDA] Packing data for GPU...\n");
    std::vector<PackedTriangle> host_packed_triangles(object.n_triangles);
    for (int i = 0; i < object.n_triangles; ++i) {
        packTriangle(host_packed_triangles[i], kdTree->tri_accel_list[i]);
    }

    std::vector<PackedKdNode> host_packed_nodes(kdTree->tree_node_count);
    for (int i = 0; i < kdTree->tree_node_count; ++i) {
        packKdNode(host_packed_nodes[i], kdTree->tree[i]);
    }

    // 2. GPU 메모리 할당 (CUDA Array) 및 데이터 복사
    printf("[CUDA] Allocating GPU memory and copying data...\n");
    cudaError_t err;

    // 삼각형 데이터
    cudaChannelFormatDesc channelDescFloat4 = cudaCreateChannelDesc<float4>();
    cudaArray* cuArrayTriangles;
    size_t total_float4_for_tris = object.n_triangles * 3;
    err = cudaMallocArray(&cuArrayTriangles, &channelDescFloat4, total_float4_for_tris, 1);
    err = cudaMemcpyToArray(cuArrayTriangles, 0, 0, host_packed_triangles.data(), host_packed_triangles.size() * sizeof(PackedTriangle), cudaMemcpyHostToDevice);

    // Kd-tree 노드 데이터
    cudaChannelFormatDesc channelDescUint2 = cudaCreateChannelDesc<uint2>();
    cudaArray* cuArrayNodes;
    err = cudaMallocArray(&cuArrayNodes, &channelDescUint2, kdTree->tree_node_count, 1);
    err = cudaMemcpyToArray(cuArrayNodes, 0, 0, host_packed_nodes.data(), host_packed_nodes.size() * sizeof(PackedKdNode), cudaMemcpyHostToDevice);

    // 삼각형 오프셋 리스트
    cudaChannelFormatDesc channelDescUint = cudaCreateChannelDesc<uint>();
    cudaArray* cuArrayOffsets;
    err = cudaMallocArray(&cuArrayOffsets, &channelDescUint, kdTree->tri_offset_count, 1);
    err = cudaMemcpyToArray(cuArrayOffsets, 0, 0, kdTree->tri_offset_list, kdTree->tri_offset_count * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // 3. CUDA Array를 텍스처에 바인딩
    printf("[CUDA] Binding textures...\n");
    err = cudaBindTextureToArray(inWaldTriangleTex, cuArrayTriangles);
    err = cudaBindTextureToArray(inKdTreeNodeTex, cuArrayNodes);
    err = cudaBindTextureToArray(inObjectOffsetListTex, cuArrayOffsets);

    // 4. Scene/Camera 정보 등 상수 메모리 설정
    printf("[CUDA] Setting up scene and camera...\n");
    SceneInfo h_scene_info;
    h_scene_info.iResolutionX = width;
    h_scene_info.iResolutionY = height;
    h_scene_info.iSuperSamplingX = 1;
    h_scene_info.iSuperSamplingY = 1;
    h_scene_info.iBlockSizeX = 16;
    h_scene_info.iBlockSizeY = 16;
    cudaMemcpyToSymbol(g_SceneInfo, &h_scene_info, sizeof(SceneInfo));

    // Camera 정보 변환
    CameraInfo h_camera_info;
    h_camera_info.eye = make_float3(camera.pos[0], camera.pos[1], camera.pos[2]);
    float3 look_at = make_float3(0, 0, 0); // AABB 중심으로 변경 가능
    float3 up = make_float3(camera.vaxis[0], camera.vaxis[1], camera.vaxis[2]);
    float aspect = (float)width / (float)height;
    float fov_rad = camera.fovy * (M_PI / 180.0f);

    float3 view = normalize(look_at - h_camera_info.eye);
    h_camera_info.u = normalize(cross(view, up));
    h_camera_info.v = normalize(cross(h_camera_info.u, view));

    float plane_height = 2.0f * tanf(fov_rad * 0.5f);
    float plane_width = plane_height * aspect;
    h_camera_info.stepX = plane_width / width;
    h_camera_info.stepY = plane_height / height;

    h_camera_info.startPoint = view - (h_camera_info.u * plane_width * 0.5f) + (h_camera_info.v * plane_height * 0.5f);
    cudaMemcpyToSymbol(g_CameraInfo, &h_camera_info, sizeof(CameraInfo));

    // Scene BBox 설정
    float3 scene_bbox[2];
    scene_bbox[0] = make_float3(object.AABB[XMIN], object.AABB[YMIN], object.AABB[ZMIN]);
    scene_bbox[1] = make_float3(object.AABB[XMAX], object.AABB[YMAX], object.AABB[ZMAX]);
    cudaMemcpyToSymbol(g_SceneBBox, scene_bbox, sizeof(float3) * 2);

    // 5. CUDA 커널 실행
    printf("[CUDA] Launching kernel...\n");
    size_t framebuffer_size = width * height * 3 * sizeof(float);
    float* d_framebuffer;
    cudaMalloc(&d_framebuffer, framebuffer_size);
    cudaMemset(d_framebuffer, 0, framebuffer_size);

    dim3 threadsPerBlock(h_scene_info.iBlockSizeX, h_scene_info.iBlockSizeY);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    singlePassRayTracingKernel_ShadowOff << <numBlocks, threadsPerBlock >> > (d_framebuffer, 1, 0, 0, false);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return; // 실패 시 여기서 종료
    }

    // 6. 결과 프레임버퍼를 Host로 복사
    printf("[CUDA] Copying result back to host...\n");
    if (out_framebuffer) delete[] out_framebuffer; // 이전 버퍼가 있으면 삭제
    out_framebuffer = new float[width * height * 3];
    cudaMemcpy(out_framebuffer, d_framebuffer, framebuffer_size, cudaMemcpyDeviceToHost);

    // 7. 할당 해제
    printf("[CUDA] Cleaning up...\n");
    cudaFree(d_framebuffer);
    cudaFreeArray(cuArrayTriangles);
    cudaFreeArray(cuArrayNodes);
    cudaFreeArray(cuArrayOffsets);
    cudaUnbindTexture(inWaldTriangleTex);
    cudaUnbindTexture(inKdTreeNodeTex);
    cudaUnbindTexture(inObjectOffsetListTex);

    is_done = true;
    printf("--- CUDA Rendering Finished ---\n");
}