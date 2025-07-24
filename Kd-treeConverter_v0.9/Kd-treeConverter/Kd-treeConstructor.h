/**************************************************************
  File name: Kd-treeConstructor.h
  Version: 1.0
  Date: November 19, 2014
 **************************************************************/

#include <stdio.h>
#include "Kd-treeConverter.h"
//using namespace KDTConverter;

#define KD_TREE_EPSILON        0.00001f

// Kd-tree build parameters --> KD-treeConstructor.cpp 변수로 전환
//#define KD_TREE_TRAVL_COST     1.0
//#define KD_TREE_ISECT_COST     1.5
//#define KD_TREE_MAX_LEVEL      100
//#define KD_TREE_MIN_TRIANGLE   4
//#define KD_TREE_EMTPY_BONUS    0.9

typedef enum { BOTH_SIDE, MINCOST_SIDE } PlanarTriangleAddMode;
#define KD_TREE_PLANAR_TRIANGLE_ADD_MODE  MINCOST_SIDE
//namespace KDTConstructor {
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
} TriangleList;

typedef struct _BoundEdge {
	float t;
	enum { START, END } type;
	const TriangleList *triangleInfo;
	bool isPlanar;
	bool isNormalPositive;
} BoundEdge;

typedef struct _SplitCost {
	//split될 때의 cost, 위치, axis
	double		cost;
	float		splitPos;
	int			axis;
	//각 경우에 대해 따로 따로 집계
	int			n_onlyLeft, n_onlyRight, n_cross, n_planar;
	//n_left = n_onlyLeft+n_cross+n_planar
	int			n_left, n_right;
	//planar triangle들이 속하게 되는 side
	int			planar_side;
	int			n_leftW, n_rightW;
	int			splitIndex;

	struct _SplitCost(void):
		cost(DBL_MAX), splitPos(FLT_MAX), axis(-1), n_onlyLeft(0), n_onlyRight(0), n_cross(0), n_planar(0), n_left(0), n_right(0), planar_side(-1){}
	bool is_valid() const { return axis >= 0; }
} SplitCost;

// added for these new variables
extern float v_KD_TREE_TRAVL_COST;
extern float v_KD_TREE_ISECT_COST;
extern unsigned int v_KD_TREE_MAX_LEVEL;
extern unsigned int v_KD_TREE_MIN_TRIANGLE;
extern float v_KD_TREE_EMTPY_BONUS;

extern BoundingBox   g_root_AABB;
extern BoundEdge    *g_bEdge;
extern TriangleList *g_pTriangleInfos;
extern unsigned int  g_iTriangleSize;

extern unsigned int  g_iKdTree_Level;
extern unsigned int  g_iKdTree_TriOffset_Count;
extern unsigned int  g_iKdTree_TriOffset_CountAlloc;
extern unsigned int *g_pKdTree_TriOffset_Array;
extern unsigned int  g_iKdTree_Node_Count;
extern unsigned int  g_iKdTree_Node_CountAlloc;
extern KdTreeNode   *g_pKdTree_Node_Array;
extern unsigned int  g_iKdTree_EmptyNode_Count;
extern unsigned int  g_iKdTree_LeafNode_Count;
extern unsigned int  g_iKdTree_MaxTriInLeafNode_Count;


inline void setInnerNode(KdTreeNode* pNode, int _splitAxis, unsigned int _firstChildOffset, float _splitPos);
inline void setLeafNode(KdTreeNode* pNode, unsigned int _objectSize, unsigned int _objectListOffset);

void set_bound_edge(const int axis, const TriangleList *pTriangleInfo, const unsigned int n_bEdge, BoundEdge *bEdge);

void try_to_split(const int axis, BoundingBox &inBBox, const TriangleList *pTriangles, const int triangleSize, 
                        BoundEdge *bEdge,  SplitCost &bestCost);

bool initialize_kd_tree(CompositeObject *poly_model);
void uninitialize_kd_tree(void);

void build_kd_tree_recursive(BoundEdge *bEdge, const TriangleList *pTriangleInfos, unsigned int triangleSize,
                             BoundingBox &bbox, unsigned int inNodeLevel, KdTreeNode *inNode);

void build_TriAccList(CompositeObject *poly_model, TriAccel *pTriAcc);
//}