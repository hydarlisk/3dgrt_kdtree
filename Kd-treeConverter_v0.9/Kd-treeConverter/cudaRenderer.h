#pragma once

// C++ 코드에서 참조할 구조체들
#include "Kd-treeConverter.h"
#include "OpenGLStuffs.h"
//#include <vector_types.h>
#include <cstring>

// 스택 연산을 기록할 로그 구조체
struct DebugLog {
	int operation;  // 1: push_short, 2: push_global, -1: pop_short, -2: pop_global
	int short_top;  // 연산 후 shortStack의 _top 값
	int global_ptr; // 연산 후 global_stack_ptr 값
	int node_id;    // 처리 대상이 된 노드 ID
};

// 추적할 픽셀 좌표와 로그 버퍼 크기 정의
#define TRACE_PIXEL_X 519
#define TRACE_PIXEL_Y 413
#define MAX_LOG_ENTRIES 512 // 기록할 최대 로그 수

typedef struct { unsigned nodeID; float tMax; } cu_traceState;

#define M_PI 3.14159265358979323846f
#define icosaHedronNumVrt 12
#define icosaHedronNumTri 20

const float goldenRatio = 1.618033988749895f;
const float unitspherefactor = 0.5257311121191335703f;

const float icosaEdge = 1.323169076499215f;

const float ICO_VERTICES[icosaHedronNumVrt][3] = {
	{-1, goldenRatio, 0}, {1, goldenRatio, 0}, {0, 1, -goldenRatio},
	{-goldenRatio, 0, -1}, {-goldenRatio, 0, 1}, {0, 1, goldenRatio},
	{goldenRatio, 0, 1}, {0, -1, goldenRatio}, {-1, -goldenRatio, 0},
	{0, -1, -goldenRatio}, {goldenRatio, 0, -1}, {1, -goldenRatio, 0} };
const int ICO_FACES[icosaHedronNumTri][3] = {
	{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}, {0, 5, 1},
	{6, 1, 5}, {6, 5, 7}, {6, 7, 11}, {6, 11, 10}, {6, 10, 1},
	{8, 4, 3}, {8, 3, 9}, {8, 9, 11}, {8, 11, 7}, {8, 7, 4},
	{9, 3, 2}, {9, 2, 10}, {9, 10, 11},
	{5, 4, 7}, {1, 10, 2} };


#define SH_C0 0.28209479177387814f	// sqrt(1 / (4 * pi))
#define SH_C1 0.4886025119029199f	// sqrt(3 / (4 * pi))

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

inline float fminf(const float a, const float b) { return (b > a) ? a : b; }
inline float fmaxf(const float a, const float b) { return (b < a) ? a : b; }
inline float int_as_float_H(const int a) { return *(float*)&(a); }
inline float uint_as_float_H(const unsigned int a) { return *(float*)&(a); }
//inline float uint_as_float_H(unsigned int a) {
//	float f;
//	std::memcpy(&f, &a, sizeof(float));
//	return f;
//}

bool initCuda();

/**
 * @brief CompositeObject를 CUDA로 렌더링하는 Public 함수.
 * @param object 렌더링할 CompositeObject (Kd-tree 포함).
 * @param camera 현재 카메라 정보.
 * @param width 결과 이미지 너비.
 * @param height 결과 이미지 높이.
 * @param out_framebuffer [out] 렌더링 결과가 저장될 Host 메모리 프레임버퍼 포인터.
 * @param is_done [out] 렌더링 완료 여부 플래그.
 */
void renderObjWithCuda(
	const CompositeObject& object,
	const Camera& camera,
	int width,
	int height,
	float*& out_framebuffer,
	bool& is_done
);

//void renderGaussianWithCuda(
//	const CompositeObject& object,
//	const std::vector<Gaussian>& gaussians,
//	const Camera& camera,
//	int width,
//	int height,
//	float*& out_framebuffer,
//	bool& is_done
//);
void warmUp(float* d_framebuffer, cudaStream_t stream);

void renderGaussianWithCudaSetup(const CompositeObject& object, const std::vector<Gaussian>& gaussians);

float renderGaussianWithCudaFrame(const Camera& camera, int width, int height, float* d_framebuffer, cudaStream_t stream
#if HIT_AND_NODE_COUNT_DEBUG
	, float3*& h_debug_buffer1, float3*& h_debug_buffer2
#endif
#if USE_STACK > SHORT_STACK
	, cu_traceState* d_global_stack
#endif
);

void cleanupCudaResources();