#pragma once

// C++ 코드에서 참조할 구조체들
#include "Kd-treeConverter.h"
#include "OpenGLStuffs.h"
//#include <vector_types.h>
#include <cstring>

#define ALPHA_MIN 0.01f
#define KERNEL_DEGREE 2.0f

#define OPACITY_THRESHOLD 0.95f

#define SUPER_SAMPLING false

#define MAX_HITS 256
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
#define MODEL_PATH "../../Data/ply/bonsai/bonsai.ply"
#define KDTREE_PATH "../../Data/ply/bonsai/bonsai_tree.kdt"
#define IGEOM_PATH "../../Data/ply/bonsai/bonsai_igeom.bin"
#elif SCENE_NUM == 4
#define MODEL_PATH "../../Data/ply/chair/chair_3dgrt.ply"
#define KDTREE_PATH "../../Data/ply/chair/chair_tree.kdt"
#define IGEOM_PATH "../../Data/ply/chair/chair_igeom.bin"
#elif SCENE_NUM == 5
#define MODEL_PATH "../../Data/ply/flowers/flowers.ply"
#define KDTREE_PATH "../../Data/ply/flowers/flowers_tree.kdt"
#define IGEOM_PATH "../../Data/ply/flowers/flowers_igeom.bin"
#endif

struct float3x3 {
	float m[3][3];
};

#define M_PI 3.14159265358979323846f
#define icosaHedronNumVrt 12
#define icosaHedronNumTri 20

const float goldenRatio = 1.618033988749895f;
//const float icosaEdge = 1.323169076499215f;
const float unitspherefactor = 0.5257311121191335703f;

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

#define SH_C2_0 1.0925484306f
#define SH_C2_1 -1.0925484306f
#define SH_C2_2 0.3153915652f
#define SH_C2_3 -1.0925484306f
#define SH_C2_4 0.5462742153f

#define SH_C3_0 -0.5900435899f
#define SH_C3_1 2.8906114426f
#define SH_C3_2 -0.4570457996f
#define SH_C3_3 0.3731763326f
#define SH_C3_4 -0.4570457996f
#define SH_C3_5 1.4453057213f
#define SH_C3_6 -0.5900435899f

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

//void renderGaussianWithCuda(
//	const CompositeObject& object,
//	const std::vector<Gaussian>& gaussians,
//	const Camera& camera,
//	int width,
//	int height,
//	float*& out_framebuffer,
//	bool& is_done
//);
void renderGaussianWithCudaSetup(const CompositeObject& object, const std::vector<Gaussian>& gaussians);
void renderGaussianWithCudaFrame(const Camera& camera, int width, int height, float* d_framebuffer);
void cleanupCudaResources();

#if 0
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
#else
	#define IS_LEAF(node)				(((node).x & 7) == 3)
	#define SPLIT_AXIS(node)			( (node).x & 3)
	#define FIRST_CHILD_OFFSET(node)	( (node).x >> 3)
	#define SECOND_CHILD_OFFSET(node)	(((node).x >> 3) + 1)
	#define SPLIT_POS(node)				(*(float *)&((node).y))
	#define OBJECT_SIZE(node)			( (node).x >> 3)
	#define OBJECTLIST_OFFSET(node)		( (node).y)
#endif