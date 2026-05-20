/**************************************************************
  File name: Kd-treeConverter.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#pragma once
#include <vector>

/* camera */
#define CAM_MOVE_SPEED 0.5
#define CAM_ROT_SPEED 0.1
#define CAM_MOVE_SHIFT 0.01

/* data format */
#define JS_BIN false
#define COMPACT_VERTEX true
#define KDT_VERSION 0

/* ply read */
#define OCCLUDE_MIN_OPACITY 1
#define OCCLUDE_MIN_OPACITY_TRI 1
#define PROBLEMATIC_THRESHOLD 1	//remove dominant prim from gaussian data

/* primitive type */
#define TRI 0
#define ELLIPSOID 1
#define ELLIPSOID_BY_TRI 2
#define PRIMITIVE_TYPE 2

#define UPLOAD_INV_SCALE 1

#define QUATERNION true
	#define PRE_CALC_KSCALE 1
	#define UPLOAD_INVSR_MAT 0
	#define DIRECT_ROT_CALC 1
	#define STORE_GRAYDIST 0

/////////////**** Debug ****///////////////
/* Ellipsoid */
#define ELLIPSOID_DEBUG 1
#define DEBUG_LEAF_CUDA 0
#if PRIMITIVE_TYPE == ELLIPSOID && ELLIPSOID_DEBUG
	#define DEBUG_ELLIPSOID_CLIP_AABB 1
	#define DEBUG_ELLIPSOID_INTERNAL 1
	#define DEBUG_LEAF_GL 0
	#define DEBUG_LEAF_CUDA 1
	#undef OCCLUDE_MIN_OPACITY_TRI
	#define OCCLUDE_MIN_OPACITY_TRI 0
#endif

/* Triangle */
#define LEAF_NODE_DEBUG true					// kd-tree leaf node 렌더링

#define DUMP_LEAF_CSV 0
////////////////////////////////////////////

#define EPSILON 0.00001f

/* kd tree build */
#define TRAVL_COST 1.0
#define ISCET_COST 20.0
#define MAX_LEVEL 128
#define EMTPY_BONUS 0.9
//#define IGNORE_THRESHOLD 0.0001
#define IGNORE_THRESHOLD 0
#define REMOVE_SMALL_PRIM 0

#define MIN_TRI 16
#define MIN_ELLIPSOID 16

#if QUATERNION
#undef PRE_CALC_KSCALE
#define PRE_CALC_KSCALE 1
#undef UPLOAD_INVSR_MAT 
#define UPLOAD_INVSR_MAT 0
#endif

#define ADAPTIVE_KERNEL_CLAMPING 1

/* BSPT */
#define BSPT false
#define BSPT_MAX_STACK_DEPTH 64
#define BSPT_MAX_HITS 95
#define BSPT_NO_SPLIT false
#define BSPT_DUMP_STATISTICS false
#define CLIP_BEFORE_BSPT false

#define FORCE_SPLIT_THRESHOLD 64				// kd-tree 강제분할
//#define FORCE_SPLIT_THRESHOLD 256				// kd-tree 강제분할
//#define FORCE_SPLIT_THRESHOLD 256				// kd-tree 강제분할
//#define SOFT_SPLIT_THRESHOLD 128				// kd-tree 강제분할(완화)
#define SOFT_SPLIT_THRESHOLD2 256				// kd-tree 강제분할(완화)

#define SIGMA_THRESHOLD_MODE 1
#define ADAPTIVE_MESH false
#define USE_KERNEL_SCALE true					// 기존의 OptiX 방식 kernelScale 사용

#if PRIMITIVE_TYPE == TRI	|| PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
	#undef QUATERNION
	#define QUATERNION true
	#undef ISCET_COST
	#define ISCET_COST 5.0
	#undef KDT_VERSION
	#define KDT_VERSION 0
#endif
#if PRIMITIVE_TYPE == ELLIPSOID
	#undef ISCET_COST
	#define ISCET_COST 20.0
	#undef BSPT
	#define BSPT 0
#endif

#if BSPT
#undef JS_BIN
#define JS_BIN false
#endif

#define BLEND_SELECT false
#define EXPORTED false

//#define LESS_TRI false

/********* for fast asset setting **********/
#define HOTDOG 0
#define BICYCLE 1
#define ROOM 2
#define LEGO 3

///////////// change this part /////////////
#define ASSET HOTDOG
#define ORIGINAL false
////////////////////////////////////////////

#if PRIMITIVE_TYPE == TRI || PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
#if ASSET == HOTDOG
	#if ORIGINAL
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE false
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 0
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 0
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH false
	#else
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE false
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 64
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 1
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH false
	#endif
#elif ASSET == BICYCLE
	#if ORIGINAL
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE false
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 0
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 0
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH false
	#else
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE true
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 128
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 1
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH true
	#endif
#elif ASSET == ROOM
	#if ORIGINAL
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE false
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 0
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 0
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH false
	#else
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE true
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 256
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 1
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH true
	#endif
#elif ASSET == LEGO
	#if ORIGINAL
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE false
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 0
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 0
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH false
	#else
		#undef USE_KERNEL_SCALE
		#define USE_KERNEL_SCALE false
		#undef FORCE_SPLIT_THRESHOLD
		#define FORCE_SPLIT_THRESHOLD 64
		#undef SIGMA_THRESHOLD_MODE
		#define SIGMA_THRESHOLD_MODE 1
		#undef ADAPTIVE_MESH
		#define ADAPTIVE_MESH false
	#endif
#endif
#endif // PRIMITIVE_TYPE == TRI


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

#define KERNEL_MIN_RESPONSE 0.0113f
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
#define WALD_METHOD true
#define MAIL_BOX false
#define SIGMA_THRESHOLD 0//.03   					// sigma(density) 작은 가우시안 제거

#define GLOBAL_DEVICE_VAR true
#define GAUSSIAN_TEXTURE true
#define TRIACC_TEXTURE true
#define OFFSET_TEXTURE true
#define KSCALE_TEXTURE true

#define DUMMY_RUN false

#define DUMP_DIR_PATH "../../output/"
//Debug Flags=====================================================================================
#define DEBUG_SIGMA_HISTOGRAM 0 				// gaussian의 sigma들의 histogram 출력
#define DEBUG_SCALE_HISTOGRAM 0 				// gaussian의 sigma들의 histogram 출력
#define DEBUG_TRILEN_HISTOGRAM 0 				// gaussian의 sigma들의 histogram 출력
#define WARP_OCCUPANCY false   					// warp occupancy 출력
#define HIT_AND_NODE_COUNT_DEBUG false			// Ray 마다 hitcount, node count, ... 확인
												// kd-tree hitmap 확인 가능

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
#elif RESOLUTION ==	5	// sgmrt
#define MAIN_WINDOW_WIDTH 1080
#define MAIN_WINDOW_HEIGHT 2102
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


typedef struct _ExtendedVertex {
	float vertex[3];
	float normal[3]; //
	int material_ID; // Need to be modified
	char pad[4]; // For 32 byte-alignement
} ExtendedVertex;


typedef struct _BoundingBox {
	union {
		struct {
			float min[3];
			float max[3];
		};
		struct {
			float pos[2][3];
		};
	};
} BoundingBox;

typedef struct _TriangleList {
	int offset;
	BoundingBox AABB;				//	split 되었을때의 가상의 bounding box
	//GTriangleWrapper *pTriangleWrapper;
	ExtendedVertex point[3];
	int side;
	int cutAxis;
	float cutCenter[3];
	float ejCut, ekCut;
} PrimList;

#if BSPT
enum Side { FRONT, BACK, STRADDLE, ON_PLANE };

//bspt node for build
struct BSPNode_Build {
	float n[3];
	float d;

	//TODO
	//float x;	//x >> 3 : front child, x >> 3 + 1 : back child
				//IS_LEAF(node) node.x & 7 == 3
	BSPNode_Build* front = nullptr;
	BSPNode_Build* back = nullptr;

	std::vector<PrimList> onPlaneTriangles;
};

//final bspt node
struct BSPNode {
	float n[3];
	float d;

	//TODO
	//float x;	//x >> 3 : front child, x >> 3 + 1 : back child
				//IS_LEAF(node) node.x & 7 == 3
	int frontChild;
	int backChild;

	unsigned int triStart;
	unsigned int triCnt;
};

#endif

typedef struct _KdTreeNode {
	// 8 bytes
	unsigned int x; unsigned int y;
} KdTreeNode;

typedef struct _KdTree {
	KdTreeNode *tree;
	int tree_node_count;
	unsigned int *prim_offset_list;
	int prim_offset_count;
#if PRIMITIVE_TYPE == TRI
	TriAccel *tri_accel_list;
#endif
	float AABB[6];
#if BSPT
	std::vector<BSPNode> bsptTree;
	std::vector<PrimList> bsptTris;
#endif
} KdTree;

typedef struct LeafForDump {
	int idx;
	std::vector<float> vertices;
	std::vector<unsigned int> indices;
};

typedef struct _CompositeObject {
	int n_triangles;
	float AABB[6];	//macro: XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX
	ExtendedVertex *extended_vertices;
	KdTree *kd_tree;
#if PRIMITIVE_TYPE == ELLIPSOID
	std::vector<PrimList>* ellipsoidAabbDebug;
	std::vector<std::vector<std::vector<PrimList>>>* ellipsoidClipAabbDebug;
	std::vector<std::vector<PrimList>>* ellipsoidLeafDebug;
	std::vector<std::vector<std::vector<PrimList>>>* ellipsoidInternalDebug;
#endif
} CompositeObject;

//shyun added begin
struct Gaussian {
	float pos[3];       // 3D 위치 (x, y, z)
	float opacity;      // 불투명도 (0.0 ~ 1.0)
#if QUATERNION
	float scale[3];     // 3축 스케일 (sx, sy, sz)
	float k_scale;
	float rot[4];       // 회전 쿼터니언 (qw, qx, qy, qz)
#else
	float3x3 rotMat; // 미리 계산된 회전 행렬 (전치된 상태, R^T)
	float scale[3];     // 3축 스케일 (sx, sy, sz)
#endif
	float f_dc[3];      // 기본 색상 (R, G, B)
	float f_rest[45];
#if QUATERNION
	float pad[4];
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
// [추가] 바이너리 지오메트리 파일(.bin)을 읽어오는 함수
bool read_igeom_from_file(CompositeObject* c_object, const char* filename);
//shyun added end
void dump_kd_tree_for_composite_object(CompositeObject* c_object,
	int dump_format,
	const char* filename,
	const char* filename_igeom
);
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
//internal
#define SPLIT_AXIS(node)			( (node).x & 3)
#define FIRST_CHILD_OFFSET(node)	( (node).x >> 3)
#define SECOND_CHILD_OFFSET(node)	(((node).x >> 3) + 1)
#define SPLIT_POS(node)				(*(float *)&((node).y))
//leaf
#define OBJECT_SIZE(node)			( (node).x >> 3)
#define OBJECTLIST_OFFSET(node)		( (node).y)
#endif

#if BSPT
#define BSPT_OFFSET(node) ((node).y)
#endif

//int HS;
//#define TABLE(i, j) table[(i)*(HS+1)+(j)]
//
//void build_DP_table(int* a, int* table, int n) {
//	for (int i = 0; i <= n; i++) TABLE(i, 0) = 1;
//	for (int j = 0; j <= n; j++) TABLE(0, j) = 1;
//
//	for (int i = 1; i <= n; i++) {
//		for (int j = 1; j <= HS; j++) {
//			if ( < Part_B > )
//				TABLE(i, j) = 1;
//			else {
//				if ((a[i] <= j) && ( < Part_C > ))
//					TABLE(i, j) = 1;
//				else
//					TABLE(i, j) = 0;
//			}
//		}
//	}
//}