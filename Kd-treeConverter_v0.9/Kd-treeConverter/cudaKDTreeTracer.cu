#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cmath>
#include "sgrt_interface.h"
#include "RayTraversal.h"
#include "cuCommonDefs.cuh"
#include "SGRTx2Lib/cuda_math.h"

// CUDA 에러 체크 함수
void checkCudaErrors(cudaError err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s", cudaGetErrorString(err));
		exit(1);
	}
}

__device__ void InitRay(int nIdx, Ray* a_Ray, Ray* g_Ray) {
	Ray* rp = &g_Ray[nIdx];

	rp->o = a_Ray->o;
	rp->d = a_Ray->d;

	// Normalize ray's direction vector
	float v1 = sqrtf((rp->d.x * rp->d.x) + (rp->d.y * rp->d.y) + (rp->d.z * rp->d.z));
	rp->d.x *= v1;
	rp->d.y *= v1;
	rp->d.z *= v1;

	// Clear initial color
	//memset( &is->color, 0, sizeof(is->color) );
}

// __device__ 함수로 변환된 IsectRay 함수
__device__ void IsectRay(const KdTreeNode node, int nIdx, Ray* g_Ray, Hit* g_Hit,
	const KdTree* kd_tree, cudaTextureObject_t triAccelTexObj,
	cudaTextureObject_t triOffsetListTexObj) {
	int i;

	Ray* rp = &g_Ray[nIdx];
	Hit* is = &g_Hit[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(node);
	const int nObjs = OBJECT_SIZE(node);

	for (i = baseOffset; i < baseOffset + nObjs; i++) {
		unsigned int triID = tex1Dfetch(triOffsetListTexObj, i); // 텍스처에서 triID 읽기
		TriAccel acc; // 텍스처에서 TriAccel 읽기

		// 텍스처 메모리에서 TriAccel 정보를 읽어옵니다.
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinearBuffer;
		resDesc.res.linearBuf.devPtr = kd_tree->tri_accel_list;
		resDesc.res.linearBuf.format = cudaChannelFormatKindFloat;
		resDesc.res.linearBuf.numChannels = 4; // Assuming 4 floats in TriAccel
		resDesc.res.linearBuf.sizeInBytes = sizeof(TriAccel) * kd_tree->tri_offset_count;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = false;

		cudaTextureObject_t texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc);

		acc = tex1Dfetch(texObj, triID);

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayID) continue;
		acc.mbox = rp->RayID;

		// ---------------------------------------------------------------
		// Backface Culling :  ʴ ü ش
		// ---------------------------------------------------------------
		if (!acc.isTransparent && BACKFACE_CULLING) {
			//if (fMyVecDotProduct(rp->df, acc.N) < 0) { //fMyVecDotProduct가 없으므로 내적을 직접 계산
			if (rp->df.x * acc.N[0] + rp->df.y * acc.N[1] + rp->df.z * acc.N[2] < 0) {
				continue;
			}
		}

		const unsigned int k = acc.k;

		float nd, f;
		nd = 1.0f / (rp->df.x/*[k]*/ + acc.n_u * rp->df.y/*[ku]*/ + acc.n_v * rp->df.z/*[kv]*/);
		f = acc.n_d - (rp->of.x/*[k]*/ + acc.n_u * rp->of.y/*[ku]*/ + acc.n_v * rp->of.z/*[kv]*/);
		f = f * nd;

		if (!(is->dist >= f && f > RAY_DIST_EPSILON)) continue;    // eps < f <= Hit4.dist

		float hu, hv;
		float lambda, mue;
		hu = rp->of.y/*[ku]*/ + f * rp->df.y/*[ku]*/;
		hv = rp->of.z/*[kv]*/ + f * rp->df.z/*[kv]*/;

		lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
		mue = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

		if (lambda < 0.0f) continue;
		if (mue < 0.0f) continue;
		if (lambda + mue > 1.0f) continue;

		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID + 1;
		is->material_ID = acc.material_ID;
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
void initCudaRendering(CompositeObject& compositeObject, float* frameBuffer, bool* renderFlag) {
	// 1. CUDA 디바이스 초기화 (필요한 경우)
	int g_render_width = 512; // 예시 해상도
	int g_render_height = 512;

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
	rayTraceKernel << <gridDim, blockDim >> > (*cudaObject, frameBuffer, g_render_width, g_render_height);
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