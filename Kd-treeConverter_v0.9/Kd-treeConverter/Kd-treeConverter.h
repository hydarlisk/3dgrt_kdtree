/**************************************************************
  File name: Kd-treeConverter.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#pragma once

#include "DefineConst.h"
#include "UnusedMacros.h"

#include <vector>

#define DEMO_CAMERA 1

/* ply read */
#define ADD_PLY_FILE_NAME ""
#define CAMERA_FILE_NAME "transforms_val"

#define PROBLEMATIC_THRESHOLD 1	//remove dominant prim from gaussian data

#define PRIMITIVE_TYPE 0	//0: TRI, 1: ELLIPSOID, 2: ELLIPSOID_TRI


/*** demo ***/
#define SECONDARY_RAY 1

#define TRI 0
#define SPHERE 1
#define SECONDARY_PRIM SPHERE
////////

#if PRIMITIVE_TYPE == TRI
	#undef SECONDARY_RAY
	#define SECONDARY_RAY 0
#endif

/*** ellipsoid_by_tri ***/
#define COUNT_BY_GID 1		// 가우시안 단위 병합 고려
//////////////////////////

/* kd tree build */
#define TRAVL_COST 1.0
#define EMTPY_BONUS 0.9


extern int MAX_LEVEL;
extern float ISCET_COST;
extern int FORCE_SPLIT_THRESHOLD;				// kd-tree 강제분할
extern int SAH_MODE;
extern int OFFSET_MAX;
extern float SORT_COST;

#define USE_KERNEL_SCALE false					// 기존의 OptiX 방식 kernelScale 사용

#define IGNORE_THRESHOLD 0
#define REMOVE_SMALL_PRIM 0

#define QUATERNION true
	#define PRE_CALC_KSCALE 1
	#define UPLOAD_INVSR_MAT 0
	#define DIRECT_ROT_CALC 1
	#define STORE_GRAYDIST 0

#define MIN_TRI 16
#define MIN_ELLIPSOID 16

#if PRIMITIVE_TYPE == TRI
#undef COUNT_BY_GID
#define COUNT_BY_GID 0
#endif

/////////////**** Debug ****///////////////
#define DUMP_RENDER_IMAGE 0
#define COLORMAP_MAX 64
#define NO_EARLY_TERMINATION 0

#define HIT_AND_NODE_COUNT_DEBUG false			// Ray 마다 hitcount, node count, ... 확인
												// kd-tree hitmap 확인 가능
#define DEBUG_LEAF_CUDA 0
/* Ellipsoid */
#define ELLIPSOID_DEBUG 1

#define DEBUG_LEAF_GL 0
#if PRIMITIVE_TYPE == ELLIPSOID && ELLIPSOID_DEBUG
	#define DEBUG_ELLIPSOID_CLIP_AABB 1
	#define DEBUG_ELLIPSOID_INTERNAL 1
	#define DEBUG_LEAF_CUDA 0
	#undef OCCLUDE_MIN_OPACITY_TRI
	#define OCCLUDE_MIN_OPACITY_TRI 0
#endif

/* Triangle */
#define LEAF_NODE_DEBUG false					// kd-tree leaf node 렌더링
////////////////////////////////////////////

#if QUATERNION
#undef PRE_CALC_KSCALE
#define PRE_CALC_KSCALE 1
#undef UPLOAD_INVSR_MAT 
#define UPLOAD_INVSR_MAT 0
#endif

#define USE_STACK GLOBAL_STACK					// SHORT_STACK	(0): ShortStack만 사용
												// HYBRID_STACK	(1): GlobalStack 같이 사용
												// GLOBAL_STACK	(2): GlobalStack만 사용
#define SHORT_STACK_DEPTH 12 					// for kernel sh.mem

#define WALD_METHOD true
#define MAIL_BOX false

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


#define MEASURE_START_FRAME 500
#define MEASURE_END_FRAME 1000

#define DIM_X 32
#define DIM_Y 8

//================================================================================================

#define RESOLUTION 5
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
#elif RESOLUTION ==	5	// FHD+
#define MAIN_WINDOW_WIDTH 2340
#define MAIN_WINDOW_HEIGHT 1080
#define FOV_Y 96.0f
#elif RESOLUTION == 6	//eval
// nerf synthtic
#define MAIN_WINDOW_WIDTH 800
#define MAIN_WINDOW_HEIGHT 800
//room
//#define MAIN_WINDOW_WIDTH 1557
//#define MAIN_WINDOW_HEIGHT 1038
//counter
//#define MAIN_WINDOW_WIDTH 1558
//#define MAIN_WINDOW_HEIGHT 1038
//kitchen
//#define MAIN_WINDOW_WIDTH 1558
//#define MAIN_WINDOW_HEIGHT 1039
//bonsai
//#define MAIN_WINDOW_WIDTH 1559
//#define MAIN_WINDOW_HEIGHT 1039
//bicycle
//#define MAIN_WINDOW_WIDTH 1237
//#define MAIN_WINDOW_HEIGHT 822
//garden
//#define MAIN_WINDOW_WIDTH 1297
//#define MAIN_WINDOW_HEIGHT 840

#define FOV_Y 96.0f
#elif RESOLUTION == 7 //record
#define MAIN_WINDOW_WIDTH 960
#define MAIN_WINDOW_HEIGHT 1080
#define FOV_Y 96.0f
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

typedef struct _PrimList {
	int offset;
	BoundingBox AABB;				//	split 되었을때의 가상의 bounding box
	ExtendedVertex point[3];
	int side;
#if PRIMITIVE_TYPE == ELLIPSOID
	int cutAxis;
	float cutCenter[3];
	float ejCut, ekCut;
#endif
} PrimList;

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

	std::vector<std::vector<PrimList>>* leafDebug;
#if PRIMITIVE_TYPE == ELLIPSOID
	std::vector<PrimList>* ellipsoidAabbDebug;
	std::vector<std::vector<std::vector<PrimList>>>* ellipsoidClipAabbDebug;
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
	const char* filename_igeom,
	const char* filename_leafInfo = nullptr
);
int read_kd_tree_from_file(CompositeObject *, const char *, int);
void loadLeafDebug(const char* filename, std::vector<std::vector<PrimList>>& leafDebug);


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