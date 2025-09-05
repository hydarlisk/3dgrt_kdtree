/**************************************************************
  File name: Kd-treeConverter.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#pragma once
#include <vector>

//shyun
#define TRAVL_COST 1.0
#define ISCET_COST 20.0
#define MAX_LEVEL 100
#define MIN_TRI 32
#define EMTPY_BONUS 0.9

#define FORCE_SPLIT_THRESHOLD 64				// kd-tree 강제분할

#define ALPHA_MIN 0.01f
#define KERNEL_DEGREE 2.0f

#define OPACITY_THRESHOLD 0.90f

#define SUPER_SAMPLING false

#define HIT_AND_NODE_COUNT_DEBUG true			// Ray 마다 hitcount, node count 확인
#define ROTATION true							// R(45degree,1,1,1)

#define MAX_HITS 128

#define DIM_X 16
#define DIM_Y 16
#define USE_GLOBAL_STACK 0						// 0: ShortStack만 사용 1: GlobalStack 같이 사용 2: GlobalStack만 사용

#if USE_GLOBAL_STACK < 2
#define SHORT_STACK_DEPTH 7					// for kernel sh.mem
#else
#define SHORT_STACK_DEPTH 0
#endif

#define MAX_GLOBAL_STACK_DEPTH 256

#define SPH_EVAL_DEGREE 3
#define QUATERNION true
#define USE_KERNEL_SCALE false					// 기존의 OptiX 방식 kernelScale 사용

#define MAIN_WINDOW_WIDTH 960
#define MAIN_WINDOW_HEIGHT 640

#define RENDERING_WIDTH 960
#define RENDERING_HEIGHT 640
 /*
 * List of resolutions
 *
 *		  width		*		height
 * 4K	: 3840		*		2160
 * S24	: 3200		*		1440
 * QHD	: 2560		*		1440
 * FHD	: 1920		*		1080
 */

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
//shyun end

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
	//float normal[3]; //
	int material_ID; // Need to be modified
	//char pad[4]; // For 32 byte-alignement
} ExtendedVertex;

typedef struct _CompositeObject {
	int n_triangles;
	float AABB[6];
	ExtendedVertex *extended_vertices;
	KdTree *kd_tree;
} CompositeObject;

//shyun
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
//shyun end

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
int build_kd_tree_for_composite_object2(CompositeObject* c_object, const char* filename);
void dump_kd_tree_for_composite_object(CompositeObject *, const char *, int, const char *);
int read_kd_tree_from_file(CompositeObject *, const char *, int);
int find_ray_object_intersection(KdTree *, Ray *, float *, float *);

//bool initialize_kdtree_for_gaussians(const std::vector<Gaussian>& gaussians);
void collectLeafNodeStats_recursive(
	const KdTree* kd_tree,
	int nodeIndex,
	unsigned int& leaf_count,
	unsigned int& total_triangles,
	unsigned int& max_triangles,
	unsigned int& min_triangles
);