/**************************************************************
  File name: Kd-treeConverter.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#pragma once
#include <vector>

//shyun added begin
#define TRAVL_COST 1.0
#define ISCET_COST 5.0
#define MAX_LEVEL 128
#define MIN_TRI 32
#define EMTPY_BONUS 0.9

#define FORCE_SPLIT_THRESHOLD 128				// kd-tree 강제분할
//#define SOFT_SPLIT_THRESHOLD 128				// kd-tree 강제분할(완화)
#define SOFT_SPLIT_THRESHOLD2 256				// kd-tree 강제분할(완화)

#define BLEND_SELECT false
#define ADAPTIVE_MESH true
#define EXPORTED false

#define USE_KERNEL_SCALE true					// 기존의 OptiX 방식 kernelScale 사용

//#define LESS_TRI false

#define SAH_OPACITY 0
#define CLIP_AREA false							// 부모 노드의 AABB로 삼각형 면적 clip
#define SAH_MAXIMIZE false
//0 - P_s * N_s																									//235
//1 - P_s * SUM(sigma)																							//227
#define TRANSPARENCY false
//2 - P_s * SUM(sigma(i) * area(i))																				//187
//2 - P_s * SUM(sigma(i) * area_clip_parent(i))																	//154
//21 - P_s * SUM(sigma(i) * area(i) * OPACITY_PENALTY)															//182
//21 - P_s * SUM(sigma(i) * area_clip_parent(i) * OPACITY_PENALTY)												//71
#define OPACITY_PENALTY 10.0f					//for SAH 21
//22 - P_s * ( SUM(sigma(i) * area(i)) + HYBRID_BETA * N_s)														//196
//22 - P_s * ( SUM(sigma(i) * area_clip_parent(i)) + HYBRID_BETA * N_s)											//
#define HYBRID_BETA 0.3f						//for SAH 22, 23
//23 - P_s * ( (1-HYBRID_BETA) * SUM(sigma(i) * area(i))_normalize + HYBRID_BETA * N_s_normalize)				//194
//23 - P_s * ( (1-HYBRID_BETA) * SUM(sigma(i) * area_clip_parent(i))_normalize + HYBRID_BETA * N_s_normalize)	//
//3 - P_s * SUM(sigma(i) * area(i) / MAX(area(V_s))																//4
//3 - P_s * SUM(sigma(i) * area_clip_parent(i) / MAX(area(V_s))													//
//4 - P_s * SUM(sigma(i) * area(i) / MAX(area(V))																//208
//4 - P_s * SUM(sigma(i) * area_clip_parent(i) / MAX(area(V))													//
//5 - SUM(sigma(i) * area(i) / MAX(area(V))																		//
//5 - SUM(sigma(i) * area_clip_parent(i) / MAX(area_clip_parent_in_V))											//
//6 - P_s * SUM(sigma(i) * area(i_real)): (실패)
//7 - P_s * SUM(sigma(i) * (A_tri_leaf_AABB/A_leaf_AABB)): (실패)
//8 - P_s * SUM(sigma(i) * (A_tri_clip_parent_AABB/A_parent_AABB))												//
//81 - P_s * SUM(sigma(i) * area(i) * (A_tri_clip_parent_AABB/A_parent_AABB)): TODO								//
//81 - P_s * SUM(sigma(i) * area_clip_parent(i) * (A_tri_clip_parent_AABB/A_parent_AABB)): TODO					//
//9 - P_s * SUM(sigma(i) * (A_tri_clip_parent_AABB/A_tri_origin_AABB)): TODO
//91 - P_s * SUM(sigma(i) * area(i) * (A_tri_clip_parent_AABB/A_tri_origin_AABB)): TODO
//91 - P_s * SUM(sigma(i) * area_clip_parent(i) * (A_tri_clip_parent_AABB/A_tri_origin_AABB)): TODO
//10 - P_s * ( SUM(sigma(i)) / Volume(leaf_AABB) ): (실패)
//101 - ( SUM(sigma(i)) / Volume(leaf_AABB) ): (실패)
//1000 - 일정 레벨까지 1, 이후 0
#define HYBRID_SAH_DEPTH_THRESHOLD 5			//for SAH 1000, 2000
//1001 - 일정 갯수까지 1, 이후 0
#define HYBRID_SAH_TRIANGLE_THRESHOLD 10000		//for SAH 1001, 2001
//20 - 4V_s/S_s * N_s
//201 - 4V_s/S_s * ( SUM(sigma(i)) )
//in 20, 201 P_s * ()
#define MUL_PROP false
//2000 - 일정 레벨까지 20, 이후 0
//2001 - 일정 갯수까지 20, 이후 0
//2010 - 일정 레벨까지 201, 이후 0
//2011 - 일정 갯수까지 201, 이후 0

#define ALPHA_MIN 0.0113f
#define KERNEL_DEGREE 4.0f

#define ROTATION false							// R(45degree,1,1,1)

#define OPACITY_THRESHOLD 0.95f					// 충족할때까지 kd-tree 탐색
//#define OPACITY_THRESHOLD 0.9961f					// 충족할때까지 kd-tree 탐색

#define MAX_HITS 256								// leaf node에서 blending을 위한 최대 sort 크기

#define DIM_X 32
#define DIM_Y 8

#define SHORT_STACK 0
#define HYBRID_STACK 1
#define GLOBAL_STACK 2
#define USE_STACK SHORT_STACK					// SHORT_STACK	(0): ShortStack만 사용
												// HYBRID_STACK	(1): GlobalStack 같이 사용
												// GLOBAL_STACK	(2): GlobalStack만 사용
#if USE_STACK > HYBRID_STACK
#define SHORT_STACK_DEPTH 0
#else
#define SHORT_STACK_DEPTH 12 					// for kernel sh.mem
#endif

#define MAX_GLOBAL_STACK_DEPTH 64

#define SPH_EVAL_DEGREE 3
#define GAUSSIAN_DEGREE 4
#define QUATERNION true
#define WALD_METHOD true
#define SIGMA_THRESHOLD 0//.03   					// sigma(density) 작은 가우시안 제거

#define GLOBAL_DEVICE_VAR true
#define GAUSSIAN_TEXTURE true

#define OFFSET_TEXTURE true

#define DUMMY_RUN false
//Debug Flags=====================================================================================
#define DEBUG_SIGMA_HISTOGRAM 100 				// gaussian의 sigma들의 histogram 출력
#define DEBUG_SCALE_HISTOGRAM 100 				// gaussian의 sigma들의 histogram 출력
#define DEBUG_TRILEN_HISTOGRAM 100 				// gaussian의 sigma들의 histogram 출력
#define WARP_OCCUPANCY false   					// warp occupancy 출력
#define HIT_AND_NODE_COUNT_DEBUG false			// Ray 마다 hitcount, node count, ... 확인
												// kd-tree hitmap 확인 가능
#define LEAF_NODE_DEBUG false					// kd-tree leaf node 렌더링

#define MEASURE_START_FRAME 500
#define MEASURE_END_FRAME 1000

#define TID_X (blockDim.x * blockIdx.x + threadIdx.x)
#define TID_Y (blockDim.y * blockIdx.y + threadIdx.y)
//================================================================================================

//#define MAIN_WINDOW_WIDTH 800
//#define MAIN_WINDOW_HEIGHT 800
#define RESOLUTION 4
#if RESOLUTION == 0		// FHD
#define MAIN_WINDOW_WIDTH 1920
#define MAIN_WINDOW_HEIGHT 1080
#define FOV_Y 19.6f
#elif RESOLUTION == 1	// QHD
#define MAIN_WINDOW_WIDTH 2560
#define MAIN_WINDOW_HEIGHT 1440
#define FOV_Y 19.6f
#elif RESOLUTION == 2	// 4K
#define MAIN_WINDOW_WIDTH 3840
#define MAIN_WINDOW_HEIGHT 2160
#define FOV_Y 19.6f
#elif RESOLUTION == 3	// 8K
#define MAIN_WINDOW_WIDTH 7680
#define MAIN_WINDOW_HEIGHT 4320
#define FOV_Y 19.6f
#elif RESOLUTION == 4	// VR
#define MAIN_WINDOW_WIDTH 1440
#define MAIN_WINDOW_HEIGHT 1540
#define FOV_Y 96.0f
#endif
 /*
 * List of resolutions
 *
 *		  width		*		height
 * 8K	: 7680		*		4320
 * 4K	: 3840		*		2160
 * QHD	: 2560		*		1440
 * FHD	: 1920		*		1080
 */

/* Camera */
//#define RENDERING_WIDTH 800
//#define RENDERING_HEIGHT 800
//#define RENDERING_WIDTH 960
//#define RENDERING_HEIGHT 640
//#define RENDERING_WIDTH 1920
//#define RENDERING_HEIGHT 1080

#define NEAR_PLANE 0.005f
#define FAR_PLANE 20.00f

#define SCENE_NUM 1								// obj 로드할때만 0으로 변경

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
//shyun added end

#define X 0
#define Y 1
#define Z 2

#define XMIN 0
#define XMAX 1
#define YMIN 2
#define YMAX 3
#define ZMIN 4
#define ZMAX 5

#define KD_TREE_DUMP_IN_ASCII 0
#define KD_TREE_DUMP_IN_BINARY 1

typedef struct _MeshGeom {
	int nvertices;
	int nfaces;
	float AABB[6]; // Axis-aligned Bounding Box: (xmin, xmax, ymin, ymax, zmin, zmax)
	float *vertices;
	unsigned int *faces;
} MeshGeom;

typedef __declspec(align(16)) struct _TriAccel {
	// plane
	float n_u;		// normal.u / normal.k
	float n_v;		// normal.v / normal.k
	float n_d;		// constant of plane equation
	union {
		struct {
			unsigned int k : 2;				// projection dimension
			unsigned int isTransparent : 1;	// transparent material 
			unsigned int mbox : 29;			// mail box
		};
		unsigned int paccked_flags;
	};

	// line equation for line ac
	float b_nu;
	float b_nv;
	float b_d;
	int indexInObject;

	// line equation for line ab
	float c_nu;
	float c_nv;
	float c_d;
	int material_ID;

	// normal vector
	float N[3];
	int pad;
} TriAccel;

typedef struct _KdTreeNode {
	// 8 bytes
	unsigned int x; unsigned int y;
} KdTreeNode;

typedef struct _KdTree {
	KdTreeNode *tree;
	int tree_node_count;
	unsigned int *tri_offset_list;
	int tri_offset_count;
	TriAccel *tri_accel_list;
	float AABB[6];
} KdTree;

typedef struct _ExtendedVertex {
	float vertex[3];
	float normal[3]; //
	int material_ID; // Need to be modified
	char pad[4]; // For 32 byte-alignement
} ExtendedVertex;

typedef struct _CompositeObject {
	int n_triangles;
	float AABB[6];
	ExtendedVertex *extended_vertices;
	KdTree *kd_tree;
} CompositeObject;

//shyun added begin
struct Gaussian {
	float pos[3];       // 3D 위치 (x, y, z)
	float scale[3];     // 3축 스케일 (sx, sy, sz)
#if QUATERNION
	float rot[4];       // 회전 쿼터니언 (qw, qx, qy, qz)
#else
	float3x3 rot_matrix; // 미리 계산된 회전 행렬 (전치된 상태, R^T)
#endif
	float opacity;      // 불투명도 (0.0 ~ 1.0)
	float f_dc[3];      // 기본 색상 (R, G, B)
	float f_rest[45];
#if QUATERNION
	float pad[5];
#endif
};
// pad    : 236 + 4 (240 = 16 * 15)
// pad[5] : 236 + 20 (256)
// rot_mat: 256
//shyun added end

typedef struct _Ray {
	//float origin[3];
	//float direction[3];
	union {
		float of[3];
		struct {
			float x, y, z;
		} o;
	};
	union {
		float df[3];
		struct {
			float x, y, z;
		} d;
	};
	int RayID;
	int Depth;
} Ray;

int build_kd_tree_for_composite_object(CompositeObject *);
//shyun added begin
int build_kd_tree_for_composite_object2(CompositeObject* c_object, const char* filename);
void collectTriangleCounts_recursive(
	const KdTree* kd_tree,
	unsigned long long nodeIndex,
	std::vector<unsigned long long>& counts,
	unsigned int current_level,
	unsigned int& max_level,
	unsigned int& total_level
);
//shyun added end
void dump_kd_tree_for_composite_object(CompositeObject *, const char *, int, const char *);
int read_kd_tree_from_file(CompositeObject *, const char *, int);


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