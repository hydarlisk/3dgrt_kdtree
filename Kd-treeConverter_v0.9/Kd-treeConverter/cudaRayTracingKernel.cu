//#include "sgrt_interface.h"
//#include "sgrtx2lib/cudarendercommon.cuh"
//#include "sgrtx2lib/cuda_math.h"
//#include "kd-treeconverter.h"
//
//#include <cuda.h>
//#include <cuda_runtime.h>
//
//#include "SGRTx2Lib/cudaRenderCommon.cuh"
//#include <cuda_runtime.h>
//#include <cstdio>
//#include <math.h>
//#include "SGRTx2Lib/GScene.h"
//#include "SGRTx2Lib/GGPUExperimentalRayTracer.h"

/**
 *	CUDA 로 Rendering 을 수행하기 위한 기능들.
 *
 *	by graphicsian.
 */
//#include "SGRTx2Lib/GKDTreeNode.h"
//#include "SGRTx2Lib/cuda_math.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cmath>
#include "sgrt_interface.h"
//#include "cuCommonDefs.cuh"
#include "SGRTx2Lib/cuda_math.h"
//#include <cutil.h>
//#include "SGRTx2Lib/cudaRenderPipelineCommonKernel.cu"
//#include "SGRTx2Lib/cudaRenderPipelineKernel.cu"

int g_render_width = 800;
int g_render_height = 600;

// CUDA 에러 체크 함수
void checkCudaErrors(cudaError err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s", cudaGetErrorString(err));
		exit(1);
	}
}

//// 임시: CompositeObject를 SGRTx2Lib의 GScene으로 변환
//GScene* convertCompositeObjectToGScene2(const CompositeObject* compObj) {
//    if (!compObj || !compObj->kd_tree || !compObj->extended_vertices)
//        return nullptr;
//
//    GScene* scene = new GScene();
//    scene->setResolution(800, 600);
//    scene->setSuperSampling(1, 1);
//    scene->setMaxReflectionDepth(1);
//    scene->setFrontFace(faceCCW);
//    scene->setEnableShadow(false);
//    scene->setEnableLocalShading(false);
//    scene->setUseTexture(false);
//
//    // [1] GMeshObject 생성
//    GMeshObject* mesh = new GMeshObject();
//    for (int i = 0; i < compObj->n_triangles; ++i) {
//        const ExtendedVertex* v0 = &compObj->extended_vertices[i * 3 + 0];
//        const ExtendedVertex* v1 = &compObj->extended_vertices[i * 3 + 1];
//        const ExtendedVertex* v2 = &compObj->extended_vertices[i * 3 + 2];
//
//        mesh->addTriangle(
//            make_float3(v0->vertex[0], v0->vertex[1], v0->vertex[2]),
//            make_float3(v1->vertex[0], v1->vertex[1], v1->vertex[2]),
//            make_float3(v2->vertex[0], v2->vertex[1], v2->vertex[2])
//        );
//    }
//
//    scene->addObject(mesh);
//
//    // [2] GKdTreeForCuda에 KdTree 정보 세팅
//    GKdTreeForCuda* gKd = new GKdTreeForCuda();
//    gKd->setFromRawKDTree(compObj->kd_tree, compObj->n_triangles);
//    scene->setKDTreeStructure(gKd);
//
//    // [3] 추가 정보
//    float3 amb = make_float3(0.1f, 0.1f, 0.1f);
//    scene->setGlobalAmbient(amb);
//    scene->createImageBuffer();  // 내부에서 resolution 기준으로 할당됨
//    scene->setSceneNumber(1);
//    scene->setGeometryChangeTimestamp(1);
//
//    return scene;
//}
//
//
//// 이후 SGRT 렌더링 예시
//extern "C" void launchSGRTRenderFromCompositeObject(const CompositeObject * compObj) {
//    GScene* scene = convertCompositeObjectToGScene(compObj);
//    if (!scene) {
//        printf("[SGRT] Failed to convert CompositeObject to GScene\n");
//        return;
//    }
//
//    GGPUExperimentalRayTracer raytracer;
//    GError err = raytracer.rendering(scene, false);
//
//    if (err != errorNo) {
//        printf("[SGRT] Rendering failed.\n");
//    }
//    else {
//        printf("[SGRT] Rendering succeeded.\n");
//    }
//
//    delete scene;
//}


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

//void LaunchRenderKernel(float* h_framebuffer, float width, float height) {
//	int frameBufferSize = width * height;
//
//	// Allocate device framebuffer
//	float* d_framebuffer = nullptr;
//	cudaMalloc(&d_framebuffer, frameBufferSize * sizeof(float));
//
//	// 커널 파라미터
//	int maxReflectionDepth = 1;
//	int sampleX = 1;
//	int sampleY = 1;
//	bool bJittering = false;
//
//	dim3 blockSize(16, 16);
//	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
//		(height + blockSize.y - 1) / blockSize.y);
//
//	singlePassRayTracingKernel_ShadowOff <<<gridSize, blockSize >>> (
//		d_framebuffer,
//		maxReflectionDepth,
//		sampleX,
//		sampleY,
//		bJittering
//		);
//
//	h_framebuffer = new float[frameBufferSize];
//	cudaMemcpy(h_framebuffer, d_framebuffer, frameBufferSize * sizeof(float), cudaMemcpyDeviceToHost);
//
//	cudaFree(d_framebuffer);
//}

// CUDA 커널 함수

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

// rayTraceKernel 함수
__global__ void rayTraceKernel(CUDACompositeObject object, float* output, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		// 1. 광선 생성 (카메라 위치, 방향 등 설정)
		float3 rayOrigin = make_float3(0.0f, 0.0f, 5.0f); // 예시
		float3 rayDirection;
		rayDirection.x = (x - width / 2.0f) / (width / 2.0f); // 예시
		rayDirection.y = (y - height / 2.0f) / (height / 2.0f); // 예시
		rayDirection.z = -1.0f; // 예시
		rayDirection = normalize(rayDirection);

		// 2. cuRay 구조체 생성
		cuRay ray;
		ray.pos = make_float3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
		ray.dir = make_float3(rayDirection.x, rayDirection.y, rayDirection.z);

		// 3. intersectKdTree 함수 호출
		cuIntersectionCheck currIsectCheck;
		intersectKdTree(object.kd_tree, ray, currIsectCheck);

		// 4. 결과 색상 결정
		float3 color;
		if (t > 0.0f) {
			// 교차 발생: 교점에서의 색상 계산 (normal, material 등 고려)
			color = make_float3(1.0f, 0.0f, 0.0f); // 예시: 빨간색
		}
		else {
			// 교차 없음: 배경색
			color = make_float3(0.0f, 0.0f, 0.0f); // 예시: 검은색
		}

		// 5. 결과 저장
		int index = (y * width + x) * 4; // RGBA
		output[index] = color.x;
		output[index + 1] = color.y;
		output[index + 2] = color.z;
		output[index + 3] = 1.0f; // Alpha
	}
}

//// KdTree Traversal 함수
//__device__ float intersectKdTree(CUDAKdTree* tree, float3 origin, float3 direction) {
//	// KdTree Traversal 로직 구현
//	// ...
//}

__device__ void intersectKdTree(CUDAKdTree* tree, cuRay ray, cuIntersectionCheck& intersectionCheck) {
	// 1. Kd-Tree Traversal
	float t_scene_near = 0.0f, t_scene_far = FLT_MAX;
	if (BoundsRayIntersect(g_SceneBBox, ray, t_scene_near, t_scene_far)) {
		float t_near = t_scene_near, t_far = t_scene_far;
		const unsigned smem_baseOffset = umul24(threadIdx.y, blockDim.x) + threadIdx.x;
		shortStack stack;
		stack.init(smem_baseOffset);

		// Kd-Tree Traversal 로직 구현
		kdtreeNode node = tex1Dfetch(inKdTreeNodeTex, 0); // 루트 노드

		while (true) {
			while (!IS_LEAF(node)) {
				//const int axis = SPLIT_AXIS(node);
				//const float splitPos = SPLIT_POS(node);
				const float2 pos_dir = ray.get_dir_pos(SPLIT_AXIS(node));
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
				// CUDA 텍스처에서 Kd-Tree 노드 정보 읽기
				node = tex1Dfetch(inKdTreeNodeTex, idx);
			}

			// Leaf 노드에 있는 Triangle 리스트 순회
			unsigned baseOffset = OBJECTLIST_OFFSET(node);
			int objectSize = OBJECT_SIZE(node) + baseOffset;

			for (; baseOffset < objectSize; baseOffset++) {
				const unsigned objListOffset = tex1Dfetch(inObjectOffsetListTex, baseOffset);
				singlePassIntersectRoutine(ray, objListOffset, intersectionCheck, t_near, t_far);
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

// CompositeObject 데이터를 CUDA 메모리로 복사
void copyCompositeObjectToCUDA(const CompositeObject* hostObject, CUDACompositeObject*& cudaObject) {
	// 1. CUDACompositeObject 할당
	checkCudaErrors(cudaMalloc((void**)&cudaObject, sizeof(CUDACompositeObject)));

	// 2. 필요한 데이터 할당 및 복사
	// ExtendedVertex 복사
	CUDAExtendedVertex* cudaVertices;
	checkCudaErrors(cudaMalloc((void**)&cudaVertices, sizeof(CUDAExtendedVertex) * hostObject->n_triangles * 3)); // 삼각형 당 3개의 정점
	checkCudaErrors(cudaMemcpy(cudaVertices, hostObject->extended_vertices, sizeof(CUDAExtendedVertex) * hostObject->n_triangles * 3, cudaMemcpyHostToDevice));

	// TriAccel 복사 (KdTree 내부)
	CUDATriAccel* cudaTriAccelList;
	checkCudaErrors(cudaMalloc((void**)&cudaTriAccelList, sizeof(CUDATriAccel) * hostObject->kd_tree->tri_offset_count));
	checkCudaErrors(cudaMemcpy(cudaTriAccelList, hostObject->kd_tree->tri_accel_list, sizeof(CUDATriAccel) * hostObject->kd_tree->tri_offset_count, cudaMemcpyHostToDevice));

	// KdTreeNode 복사 (KdTree 내부)
	CUDAKdTreeNode* cudaKdTreeNodes;
	checkCudaErrors(cudaMalloc((void**)&cudaKdTreeNodes, sizeof(CUDAKdTreeNode) * hostObject->kd_tree->tree_node_count));
	checkCudaErrors(cudaMemcpy(cudaKdTreeNodes, hostObject->kd_tree->tree, sizeof(CUDAKdTreeNode) * hostObject->kd_tree->tree_node_count, cudaMemcpyHostToDevice));

	// tri_offset_list 복사 (KdTree 내부)
	unsigned int* cudaTriOffsetList;
	checkCudaErrors(cudaMalloc((void**)&cudaTriOffsetList, sizeof(unsigned int) * hostObject->kd_tree->tri_offset_count));
	checkCudaErrors(cudaMemcpy(cudaTriOffsetList, hostObject->kd_tree->tri_offset_list, sizeof(unsigned int) * hostObject->kd_tree->tri_offset_count, cudaMemcpyHostToDevice));


	// CUDAKdTree 할당 및 데이터 복사
	CUDAKdTree cudaKdTree;
	checkCudaErrors(cudaMalloc((void**)&cudaKdTree.tree, sizeof(CUDAKdTreeNode) * hostObject->kd_tree->tree_node_count));
	checkCudaErrors(cudaMemcpy(cudaKdTree.tree, hostObject->kd_tree->tree, sizeof(CUDAKdTreeNode) * hostObject->kd_tree->tree_node_count, cudaMemcpyHostToDevice));

	cudaKdTree.tree_node_count = hostObject->kd_tree->tree_node_count;

	checkCudaErrors(cudaMalloc((void**)&cudaKdTree.tri_offset_list, sizeof(unsigned int) * hostObject->kd_tree->tri_offset_count));
	checkCudaErrors(cudaMemcpy(cudaKdTree.tri_offset_list, hostObject->kd_tree->tri_offset_list, sizeof(unsigned int) * hostObject->kd_tree->tri_offset_count, cudaMemcpyHostToDevice));

	cudaKdTree.tri_offset_count = hostObject->kd_tree->tri_offset_count;

	checkCudaErrors(cudaMalloc((void**)&cudaKdTree.tri_accel_list, sizeof(CUDATriAccel) * hostObject->kd_tree->tri_offset_count));
	checkCudaErrors(cudaMemcpy(cudaKdTree.tri_accel_list, hostObject->kd_tree->tri_accel_list, sizeof(CUDATriAccel) * hostObject->kd_tree->tri_offset_count, cudaMemcpyHostToDevice));

	// CUDACompositeObject에 데이터 연결
	cudaObject->n_triangles = hostObject->n_triangles;
	cudaMemcpy(cudaObject->AABB, hostObject->AABB, sizeof(float) * 6, cudaMemcpyHostToDevice);
	cudaObject->extended_vertices = cudaVertices;
	cudaObject->kd_tree = &cudaKdTree; // 포인터 연결

}

// CUDA 초기화 및 렌더링 함수
void initCudaRendering_Orig(CompositeObject& compositeObject, float* frameBuffer, bool* renderFlag) {
	// 1. CUDA 디바이스 초기화 (필요한 경우)

	// 2. CUDACompositeObject 생성 및 데이터 복사
	CUDACompositeObject* cudaObject;
	copyCompositeObjectToCUDA(&compositeObject, cudaObject);

	// 3. 렌더링 결과를 저장할 프레임 버퍼 할당
	if (frameBuffer == nullptr) {
		g_render_width = 512; // 예시 해상도
		g_render_height = 512;
		frameBuffer = new float[g_render_width * g_render_height * 4]; // RGBA
	}

	// 4. CUDA 커널 실행
	dim3 blockDim(16, 16);
	dim3 gridDim((g_render_width + blockDim.x - 1) / blockDim.x, (g_render_height + blockDim.y - 1) / blockDim.y);
	rayTraceKernel <<<gridDim, blockDim >>> (*cudaObject, frameBuffer, g_render_width, g_render_height);
	checkCudaErrors(cudaDeviceSynchronize());

	// 5. CUDA 렌더링 완료 플래그 설정
	*renderFlag = true;

	// 6. CUDA 메모리 해제 (나중에 필요할 수 있음)
	cudaFree(cudaObject->extended_vertices);
	cudaFree(cudaObject->kd_tree->tree);
	cudaFree(cudaObject->kd_tree->tri_offset_list);
	cudaFree(cudaObject->kd_tree->tri_accel_list);
	cudaFree(cudaObject->kd_tree);
	cudaFree(cudaObject);
}
#endif