#pragma once

// C++ 코드에서 참조할 구조체들
#include "Kd-treeConverter.h"
#include "OpenGLStuffs.h"
//#include <vector_types.h>
#include <cstring>

#define MAX_HITS 32
#define SCENE_NUM 1

#if SCENE_NUM == 0
#define MODEL_PATH "../../Data/Obj/hotdog_3dgrt.obj"
#define KDTREE_PATH "../../Data/Obj/hotdog_tree.kdt"
#define IGEOM_PATH "../../Data/Obj/hotdog_igeom.bin"
#elif SCENE_NUM == 1
#define MODEL_PATH "../../Data/ply/hotdog/hotdog_3dgrt.ply"
#define KDTREE_PATH "../../Data/ply/hotdog/hotdog_tree.kdt"
#define IGEOM_PATH "../../Data/ply/hotdog/hotdog_igeom.bin"
#elif SCENE_NUM == 2
#define MODEL_PATH "../../Data/ply/lego/lego_3dgrt.ply"
#define KDTREE_PATH "../../Data/ply/lego/lego_tree.kdt"
#define IGEOM_PATH "../../Data/ply/lego/lego_igeom.bin"
#elif SCENE_NUM == 3
#define MODEL_PATH "../../Data/ply/bonsai/bonsai_3dgrt.ply"
#define KDTREE_PATH "../../Data/ply/bonsai/bonsai_tree.kdt"
#define IGEOM_PATH "../../Data/ply/bonsai/bonsai_igeom.bin"
#elif SCENE_NUM == 4
#define MODEL_PATH "../../Data/ply/chair/chair_3dgrt.ply"
#define KDTREE_PATH "../../Data/ply/chair/chair_tree.kdt"
#define IGEOM_PATH "../../Data/ply/chair/chair_igeom.bin"
#elif SCENE_NUM == 5
#define MODEL_PATH "../../Data/ply/flowers/flowers_3dgrt.ply"
#define KDTREE_PATH "../../Data/ply/flowers/flowers_tree.kdt"
#define IGEOM_PATH "../../Data/ply/flowers/flowers_igeom.bin"
#endif

const float iX = 0.525731112119133606f;
const float iZ = 0.850650808352039932f;

const float ICO_VERTICES[12][3] = {
	{-iX, 0.0, iZ}, {iX, 0.0, iZ}, {-iX, 0.0, -iZ}, {iX, 0.0, -iZ},
	{0.0, iZ, iX}, {0.0, iZ, -iX}, {0.0, -iZ, iX}, {0.0, -iZ, -iX},
	{iZ, iX, 0.0}, {-iZ, iX, 0.0}, {iZ, -iZ, 0.0}, {-iZ, -iZ, 0.0}
};

const int ICO_FACES[20][3] = {
	{0, 4, 1}, {0, 9, 4}, {9, 5, 4}, {4, 5, 8}, {4, 8, 1},
	{8, 10, 1}, {8, 3, 10}, {5, 3, 8}, {5, 2, 3}, {2, 7, 3},
	{7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
	{6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5}, {7, 2, 11}
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
void renderObjWithCuda(
	const CompositeObject& object,
	const Camera& camera,
	int width,
	int height,
	float*& out_framebuffer,
	bool& is_done
);

void renderGaussianWithCuda(
	const CompositeObject& object,
	const std::vector<Gaussian>& gaussians,
	const Camera& camera,
	int width,
	int height,
	float*& out_framebuffer,
	bool& is_done
);


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