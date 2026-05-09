/**************************************************************
  File name: Kd-treeConstructor.h
  Version: 1.0
  Date: November 19, 2014
 **************************************************************/
#pragma once
#include <stdio.h>
#include "Kd-treeConverter.h"

#define KD_TREE_EPSILON        0.00001f

// Kd-tree build parameters --> KD-treeConstructor.cpp 변수로 전환
//#define KD_TREE_TRAVL_COST     1.0
//#define KD_TREE_ISECT_COST     1.5
//#define KD_TREE_MAX_LEVEL      100
//#define KD_TREE_MIN_TRIANGLE   4
//#define KD_TREE_EMTPY_BONUS    0.9

#define BOTH_SIDE 0
#define MINCOST_SIDE 1
#define KD_TREE_PLANAR_TRIANGLE_ADD_MODE  MINCOST_SIDE
//#define KD_TREE_PLANAR_TRIANGLE_ADD_MODE  BOTH_SIDE

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
extern float v_KD_TREE_EMTPY_BONUS;

extern unsigned int v_KD_TREE_MIN_PRIMITIVE;

extern BoundingBox   g_root_AABB;
extern BoundEdge    *g_bEdge;

extern unsigned int  g_iKdTree_Level;
extern unsigned int  g_iKdTree_Node_Count;
extern unsigned int  g_iKdTree_Node_CountAlloc;
extern KdTreeNode* g_pKdTree_Node_Array;
extern unsigned int  g_iKdTree_EmptyNode_Count;
extern unsigned int  g_iKdTree_LeafNode_Count;

#if PRIMITIVE_TYPE == TRI
extern TriangleList* g_pTriangleInfos;
extern unsigned int  g_iTriangleSize;
extern unsigned long long  g_iKdTree_TriOffset_Count;
extern unsigned long long  g_iKdTree_TriOffset_CountAlloc;
extern unsigned int *g_pKdTree_TriOffset_Array;
extern unsigned int  g_iKdTree_MaxTriInLeafNode_Count;
#elif PRIMITIVE_TYPE == ELLIPSOID
extern TriangleList* g_pEllipsoidInfos;
extern unsigned int g_iEllipsoidSize;
extern unsigned long long g_iKdTreeEllipsoidOffsetCnt;
extern unsigned long long g_iKdTreeEllipsoidOffsetCnt_Alloc;
extern unsigned int* g_pKdTreeEllipsoidOffsetArray;
extern unsigned int g_iKdTreeMaxEllipsoidInLeafNodeCnt;
extern std::vector<TriangleList> g_ellipsoidAabbDebug;
#endif

inline void setInnerNode(KdTreeNode* pNode, int _splitAxis, unsigned int _firstChildOffset, float _splitPos);
void setLeafNode(KdTreeNode* pNode, unsigned int _objectSize, unsigned int _objectListOffset);

void set_bound_edge(const int axis, const TriangleList *pTriangleInfo, const unsigned int n_bEdge, BoundEdge *bEdge);

void try_to_split(const int axis, BoundingBox &inBBox, const TriangleList *pTriangles, const int triangleSize, 
                        BoundEdge *bEdge,  SplitCost &bestCost
//shyun added begin
#if SAH_OPACITY == 1 | SAH_OPACITY == 10 | SAH_OPACITY == 101 | SAH_OPACITY == 1000 | SAH_OPACITY == 1001 | SAH_OPACITY == 201 | SAH_OPACITY == 2010
	, const double total_opacity_in_node
#elif SAH_OPACITY >= 2 & SAH_OPACITY != 20 & SAH_OPACITY != 2000
	, const double total_contribution_in_node
#endif
#if SAH_OPACITY >= 4 & SAH_OPACITY < 6
	, const float max_area_in_node
#endif
#if SAH_OPACITY == 1000 | SAH_OPACITY == 2000 | SAH_OPACITY == 2010
	, unsigned int inNodeLevel
#endif
//shyun added end
);

bool initialize_kd_tree(CompositeObject *poly_model);
void uninitialize_kd_tree(void);

void build_kd_tree_recursive(BoundEdge *bEdge, const TriangleList *pTriangleInfos, unsigned int triangleSize,
                             BoundingBox &bbox, unsigned int inNodeLevel, KdTreeNode *inNode);

void build_TriAccList(CompositeObject *poly_model, TriAccel*& pTriAcc);

//shyun added begin
struct LeafNodeInfo {
	BoundingBox aabb;
	std::vector<unsigned int> triangle_indices;
	
	bool operator<(const LeafNodeInfo& other) const {
		return triangle_indices.size() < other.triangle_indices.size();
	}
};
std::vector<BoundingBox> extract_leaves_from_kd_tree();
std::vector<LeafNodeInfo> extract_all_leaf_data(CompositeObject* c_object, int& largest_leaf_index);

inline double get_surface_area(const BoundingBox& box);
inline double get_surface_volume(const BoundingBox& box);
//shyun added end

/* binary space partitioning tree */
#if BSPT
extern std::vector<BSPNode> g_BSPTNodes;
extern std::vector<TriangleList> g_BSPTTris;
#endif