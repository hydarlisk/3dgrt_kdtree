#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cmath>
#include "cudaKDTreeTracer.h"
//#include "sgrt_interface.h"
//#include "RayTraversal.h"
//#include "cuCommonDefs.cuh"
//#include "SGRTx2Lib/cuda_math.h"

int g_render_width = 800;
int g_render_height = 600;

// CUDA 에러 체크 함수
void checkCudaErrors(cudaError err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s", cudaGetErrorString(err));
		exit(1);
	}
}

// rayTraceKernel 함수
__global__ void rayTraceKernel(float* framebuffer, CompositeObject scene) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * g_render_width + x;

	cuRay ray = generateRay(x, y);
	cuIntersectionCheck isect;
	isect.init();

	singlePassIntersect(ray, scene, isect, 0.0f, 1e30f);

	if (isect.triIndex != -1) {
		framebuffer[3 * idx + 0] = 1.0f;
		framebuffer[3 * idx + 1] = 0.0f;
		framebuffer[3 * idx + 2] = 0.0f;
	}
}

__device__ void singlePassIntersect(
	const cuRay& ray,
	const CompositeObject& scene,
	cuIntersectionCheck& isectCheck,
	float t_near,
	float t_far
) {
	isectCheck.tHit = t_far;
	isectCheck.triIndex = -1;

	int todoStack[64];
	int todoIndex = 0;
	todoStack[todoIndex++] = scene.kd_tree_root_index;

	const KdNode* nodes = scene.kd_tree_nodes;
	const TriAccel* tris = scene.triangle_accel;
	const ExtendedVertex* verts = scene.extended_vertices;

	float3 invDir = make_float3(1.0f) / ray.dir;
	int dirIsNeg[3] = { ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0 };

	while (todoIndex > 0) {
		int nodeIdx = todoStack[--todoIndex];
		const KdNode& node = nodes[nodeIdx];

		if (node.isLeaf()) {
			for (int i = 0; i < node.triangleCount; ++i) {
				int triIdx = node.triangleIndices[i];
				float t, u, v;
				if (intersectRayTriAccel(ray, tris[triIdx], verts, t, u, v)) {
					if (t > t_near && t < isectCheck.tHit) {
						isectCheck.tHit = t;
						isectCheck.beta = u;
						isectCheck.gamma = v;
						isectCheck.triIndex = triIdx;
					}
				}
			}
		}
		else {
			int axis = node.axis;
			float split = node.split;
			float tSplit = (split - ray.origin[axis]) * invDir[axis];

			int first = node.left;
			int second = node.right;
			if (dirIsNeg[axis]) {
				int temp = first;
				first = second;
				second = temp;
			}

			if (tSplit <= t_near) {
				todoStack[todoIndex++] = second;
			}
			else if (tSplit >= t_far) {
				todoStack[todoIndex++] = first;
			}
			else {
				todoStack[todoIndex++] = second;
				todoStack[todoIndex++] = first;
			}
		}
	}
}


// CompositeObject 데이터를 CUDA 메모리로 복사
void copyCompositeObjectToCUDA(const CompositeObject* hostObject, CUDACompositeObject*& cudaObject) {
	// CUDACompositeObject 할당
	checkCudaErrors(cudaMalloc((void**)&cudaObject, sizeof(CUDACompositeObject)));

	// 필요한 데이터 할당 및 복사
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
void initCudaRendering(CompositeObject& compositeObject, float* frameBuffer, bool* renderFlag) {
	// CUDA 디바이스 초기화 (필요한 경우)
	g_render_width = 512; // 예시 해상도
	g_render_height = 512;

	// CUDACompositeObject 생성 및 데이터 복사
	CUDACompositeObject* cudaObject;
	copyCompositeObjectToCUDA(&compositeObject, cudaObject);

	// 렌더링 결과를 저장할 프레임 버퍼 할당
	float* d_render_framebuffer;
	size_t framebuffer_size = sizeof(float) * g_render_width * g_render_height * 3;

	cudaMalloc(&d_render_framebuffer, framebuffer_size);
	cudaMemset(d_render_framebuffer, 0, framebuffer_size);

	// CUDA 커널 실행
	dim3 blockDim(16, 16);
	dim3 gridDim((g_render_width + blockDim.x - 1) / blockDim.x, (g_render_height + blockDim.y - 1) / blockDim.y);
	rayTraceKernel <<< gridDim, blockDim >>> (*cudaObject, frameBuffer, g_render_width, g_render_height);
	checkCudaErrors(cudaDeviceSynchronize());

	// CUDA 렌더링 완료 플래그 설정
	*renderFlag = true;

	// CUDA 메모리 해제 (나중에 필요할 수 있음)
	cudaFree(cudaObject->extended_vertices);
	cudaFree(cudaObject->kd_tree->tree);
	cudaFree(cudaObject->kd_tree->tri_offset_list);
	cudaFree(cudaObject->kd_tree->tri_accel_list);
	cudaFree(cudaObject->kd_tree);
	cudaFree(cudaObject);
}