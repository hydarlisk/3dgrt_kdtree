#pragma once

// C++ 코드에서 참조할 구조체들
#include "Kd-treeConverter.h"
#include "OpenGLStuffs.h"
//#include <vector_types.h>
#include <cstring>

struct GPUParticle {
	float3 position;
	float3 scale;
	float4 rotation;
	float3 color;
	float opacity;
};

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
 * @brief CompositeObject를 CUDA로 렌더링하는 유일한 Public 함수.
 * @param object 렌더링할 CompositeObject (Kd-tree 포함).
 * @param camera 현재 카메라 정보.
 * @param width 결과 이미지 너비.
 * @param height 결과 이미지 높이.
 * @param out_framebuffer [out] 렌더링 결과가 저장될 Host 메모리 프레임버퍼 포인터.
 * @param is_done [out] 렌더링 완료 여부 플래그.
 */
void renderWithCuda(
	const CompositeObject& object,
	const Camera& camera,
	int width,
	int height,
	float*& out_framebuffer,
	bool& is_done
);

int read_ply_and_upload_gaussians(const char* ply_filename, GPUParticle*& d_particles, int& n_particles);


#if 1
// -----------------------------------------------------------
// kdtree node
// -----------------------------------------------------------
	// macros for extracting node information
	#define IS_LEAF(node)				(((node).x & 3) == 3)
	#define IS_ROPE(node)				(((node).x & 4) == 4)

	#define SPLIT_AXIS(node)			( (node).x & 3)
	#define FIRST_CHILD_OFFSET(node)	( (node).x >> 3)
	#define SECOND_CHILD_OFFSET(node)	(((node).x >> 3) + 1)

	#define SPLIT_POS(node)				(*(float *)&((node).y))
	#define OBJECT_SIZE(node)			( (node).x >> 3)

	#define OBJECTLIST_OFFSET(node)		( (node).y)
	#define ROPE_NODE_OFFSET(node)		( (node).y)
#endif