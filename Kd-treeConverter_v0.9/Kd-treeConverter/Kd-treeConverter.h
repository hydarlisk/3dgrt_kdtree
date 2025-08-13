/**************************************************************
  File name: Kd-treeConverter.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#pragma once
#include <vector>

#define TRAVL_COST 0.2
#define ISCET_COST 60000.0

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
//namespace KDTConverter{
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
	float rot[4];       // 회전 쿼터니언 (qw, qx, qy, qz)
	float opacity;      // 불투명도 (0.0 ~ 1.0)
	float f_dc[3];      // 기본 색상 (R, G, B)
	float f_rest[45];
};
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
void dump_kd_tree_for_composite_object(CompositeObject *, const char *, int, const char *);
int read_kd_tree_from_file(CompositeObject *, const char *, int);
int find_ray_object_intersection(KdTree *, Ray *, float *, float *);

bool initialize_kdtree_for_gaussians(const std::vector<Gaussian>& gaussians);
//}