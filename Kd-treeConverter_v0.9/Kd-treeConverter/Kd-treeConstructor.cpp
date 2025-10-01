/**************************************************************
  File name: Kd-treeConstructor.cpp
  Version: 1.0
  Date: November 19, 2014
 **************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include <stack>
//#include <vector>

#include "Kd-treeConverter.h"
#include "Kd-treeConstructor.h"
#include "MyMathUtility.h"
//using namespace KDTConverter;
//using namespace KDTConstructor;

static const unsigned int modulo[] = { 0,1,2,0,1 };

// From macros to variables to be able to modify through the configuration file
float v_KD_TREE_TRAVL_COST = TRAVL_COST;
float v_KD_TREE_ISECT_COST = ISCET_COST;
unsigned int v_KD_TREE_MAX_LEVEL = MAX_LEVEL;
unsigned int v_KD_TREE_MIN_TRIANGLE = MIN_TRI;
float v_KD_TREE_EMTPY_BONUS = EMTPY_BONUS;

BoundingBox   g_root_AABB;
BoundEdge    *g_bEdge = NULL;
TriangleList *g_pTriangleInfos = NULL;
unsigned int  g_iTriangleSize;

unsigned int  g_iKdTree_Level;
unsigned int  g_iKdTree_TriOffset_Count;
unsigned int  g_iKdTree_TriOffset_CountAlloc;
unsigned int *g_pKdTree_TriOffset_Array = NULL;
unsigned int  g_iKdTree_Node_Count;
unsigned int  g_iKdTree_Node_CountAlloc;
KdTreeNode   *g_pKdTree_Node_Array = NULL;
unsigned int  g_iKdTree_EmptyNode_Count;
unsigned int  g_iKdTree_LeafNode_Count;
unsigned int  g_iKdTree_MaxTriInLeafNode_Count;

extern std::vector<Gaussian> g_gaussians;

void _reAllocTriangleOffsetList(unsigned int _newAllocSize, unsigned int &_oldAllocSize, unsigned int** _ppTriOffsetArray)
{
	unsigned int *tmpList = new unsigned int [ _newAllocSize ];
	memcpy( tmpList, *_ppTriOffsetArray, sizeof( unsigned int ) * _oldAllocSize );
	delete[] *_ppTriOffsetArray; 
	*_ppTriOffsetArray = tmpList;
	_oldAllocSize = _newAllocSize;
}

void _reAllocKdtreeNodes(unsigned int _newAllocSize, unsigned int &_oldAllocSize, KdTreeNode** _ppNodeArray)
{
	KdTreeNode *tmpList = new KdTreeNode[_newAllocSize];
	memcpy( tmpList, *_ppNodeArray, sizeof( KdTreeNode ) * _oldAllocSize );
	delete[] *_ppNodeArray;
	*_ppNodeArray = tmpList;	
	_oldAllocSize = _newAllocSize;
}

inline unsigned float_as_unsigned(const float a) { return *(unsigned *)&(a); }

void setInnerNode(KdTreeNode* pNode, int _splitAxis, unsigned int _firstChildOffset, float _splitPos)
{
	pNode->x = _firstChildOffset << 3;
	pNode->x |= _splitAxis;
	pNode->y = float_as_unsigned( _splitPos );
}

void setLeafNode(KdTreeNode* pNode, unsigned int _objectSize, unsigned int _objectListOffset)
{
	pNode->x = _objectSize << 3;
	pNode->x |= 3;
	pNode->y = _objectListOffset;
}

int compare_bound_edge(const void *elem0, const void *elem1)
{
	const BoundEdge* obj0 = (const BoundEdge *)elem0;
	const BoundEdge* obj1 = (const BoundEdge *)elem1;

	return obj0->t == obj1->t ?
		(obj0->triangleInfo->offset > obj1->triangleInfo->offset ? 1 : -1)
		:
		(obj0->t > obj1->t ? 1 : -1);
}

void set_bound_edge(const int axis, const TriangleList *pTriangleInfo, const unsigned int n_bEdge, BoundEdge *bEdge)
{
	int index = 0;

	for( unsigned int i = 0; i < n_bEdge; ) {
		BoundingBox worldbound = pTriangleInfo[index].AABB;
		bEdge[i].type = BoundEdge::START;	bEdge[i].triangleInfo = &pTriangleInfo[index];
		bEdge[i].t = worldbound.min[axis];
		bEdge[i].isPlanar = ( worldbound.min[axis] == worldbound.max[axis] );
		i++;
		bEdge[i].type = BoundEdge::END;		bEdge[i].triangleInfo = &pTriangleInfo[index];
		bEdge[i].t = worldbound.max[axis];
		bEdge[i].isPlanar = ( worldbound.min[axis] == worldbound.max[axis] );
		i++; index++;
	}
	qsort( &( bEdge[0] ), n_bEdge, sizeof( BoundEdge ), compare_bound_edge );
}

bool intersect_edge_plane(float *p0, float *p1, float *planePoint, float *planeNorm, float *hitPoint)
{
	float u[3], w[3];
	u[0] = p1[0] - p0[0];
	u[1] = p1[1] - p0[1];
	u[2] = p1[2] - p0[2];
	w[0] = p0[0] - planePoint[0];
	w[1] = p0[1] - planePoint[1];
	w[2] = p0[2] - planePoint[2];

	float D =  fMyVecDotProduct(planeNorm, u);
	float N = -fMyVecDotProduct(planeNorm, w);

	// N==0이면 plane에 엣지가 붙어 있는거고, 아니면 intersection안 한거.
	if ( fabs( D ) < KD_TREE_EPSILON )
		return false;

	float sI = N / D;
	if( sI < 0.f || sI > 1.f )
		return false;

	hitPoint[0] = p0[0] + u[0] * sI;
	hitPoint[1] = p0[1] + u[1] * sI;
	hitPoint[2] = p0[2] + u[2] * sI;

	return true;
}

void clip_triangle(const int triangleSize, const SplitCost &bestCost, TriangleList *pTriangleInfos, int side)
{
	int axis = bestCost.axis;
	float norm[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
	
	float planeNorm[3];
		planeNorm[0] = norm[ axis ][ 0 ];
		planeNorm[1] = norm[ axis ][ 1 ];
		planeNorm[2] = norm[ axis ][ 2 ];

	float planePoint[3];
		planePoint[0] = planeNorm[0] * bestCost.splitPos;
		planePoint[1] = planeNorm[1] * bestCost.splitPos;
		planePoint[2] = planeNorm[2] * bestCost.splitPos;

	for( int i = 0; i < triangleSize; i++ ) {
		BoundingBox currBBox = pTriangleInfos[i].AABB;

		/**
		 *	만약 삼각형이 Split Plane과 교차 한다면,
		 */
		if( ( currBBox.min[ axis ] < bestCost.splitPos ) && 
			( currBBox.max[ axis ] > bestCost.splitPos ) ) {
			float p[3][3];
			p[0][0] = pTriangleInfos[i].point[0].vertex[0];
			p[0][1] = pTriangleInfos[i].point[0].vertex[1];
			p[0][2] = pTriangleInfos[i].point[0].vertex[2];
			p[1][0] = pTriangleInfos[i].point[1].vertex[0];
			p[1][1] = pTriangleInfos[i].point[1].vertex[1];
			p[1][2] = pTriangleInfos[i].point[1].vertex[2];
			p[2][0] = pTriangleInfos[i].point[2].vertex[0];
			p[2][1] = pTriangleInfos[i].point[2].vertex[1];
			p[2][2] = pTriangleInfos[i].point[2].vertex[2];
			//GPoint p[3];
			//pTriangleInfos[i].pTriangleWrapper->getPoint( p[0], p[1], p[2] );

			int nLeft = 0,	nRight = 0;
			float leftVec[4][3], rightVec[4][3];
			//GPoint leftVec[4], rightVec[4];

			//왼쪽 subBox에 있는건 왼쪽에, 오른쪽도 마찬가지로.
			for( int nPoint = 0; nPoint < 3; nPoint++ ) {
				if( p[nPoint][axis] < bestCost.splitPos ) {
					leftVec[nLeft][0] = p[nPoint][0];
					leftVec[nLeft][1] = p[nPoint][1];
					leftVec[nLeft][2] = p[nPoint][2];
					nLeft++;
				} else {
					rightVec[nRight][0] = p[nPoint][0];
					rightVec[nRight][1] = p[nPoint][1];
					rightVec[nRight][2] = p[nPoint][2];
					nRight++;
				}
			}

			{
				const int leftSize = nLeft,
					rightSize = nRight;
				for(int j=0; j<leftSize; j++)
				{
					for(int k=0; k<rightSize; k++)
					{
						float hitPoint[3];
						if( intersect_edge_plane( leftVec[j], rightVec[k], planePoint, planeNorm, hitPoint ) ) {
							leftVec[nLeft][0] = rightVec[nRight][0] = hitPoint[0];
							leftVec[nLeft][1] = rightVec[nRight][1] = hitPoint[1];
							leftVec[nLeft][2] = rightVec[nRight][2] = hitPoint[2];
							nLeft++; nRight++;
						}					
					}
				}
			}

			if( side == 0 )
			{
				BoundingBox reducedBBox;
				reducedBBox.min[0] = reducedBBox.max[0] = leftVec[0][0];
				reducedBBox.min[1] = reducedBBox.max[1] = leftVec[0][1];
				reducedBBox.min[2] = reducedBBox.max[2] = leftVec[0][2];

				for(int j=1; j<nLeft; j++) {
					reducedBBox.min[0] = MyMIN(reducedBBox.min[0], leftVec[j][0]);
					reducedBBox.min[1] = MyMIN(reducedBBox.min[1], leftVec[j][1]);
					reducedBBox.min[2] = MyMIN(reducedBBox.min[2], leftVec[j][2]);
					reducedBBox.max[0] = MyMAX(reducedBBox.max[0], leftVec[j][0]);
					reducedBBox.max[1] = MyMAX(reducedBBox.max[1], leftVec[j][1]);
					reducedBBox.max[2] = MyMAX(reducedBBox.max[2], leftVec[j][2]);
				}
				reducedBBox.min[0] = MyMAX(reducedBBox.min[0], currBBox.min[0]);
				reducedBBox.min[1] = MyMAX(reducedBBox.min[1], currBBox.min[1]);
				reducedBBox.min[2] = MyMAX(reducedBBox.min[2], currBBox.min[2]);
				reducedBBox.max[0] = MyMIN(reducedBBox.max[0], currBBox.max[0]);
				reducedBBox.max[1] = MyMIN(reducedBBox.max[1], currBBox.max[1]);
				reducedBBox.max[2] = MyMIN(reducedBBox.max[2], currBBox.max[2]);
				pTriangleInfos[i].AABB = reducedBBox;
			}
			else
			{
				BoundingBox reducedBBox;
				reducedBBox.min[0] = reducedBBox.max[0] = rightVec[0][0];
				reducedBBox.min[1] = reducedBBox.max[1] = rightVec[0][1];
				reducedBBox.min[2] = reducedBBox.max[2] = rightVec[0][2];

				for(int j=1; j<nRight; j++)	{
					reducedBBox.min[0] = MyMIN(reducedBBox.min[0], rightVec[j][0]);
					reducedBBox.min[1] = MyMIN(reducedBBox.min[1], rightVec[j][1]);
					reducedBBox.min[2] = MyMIN(reducedBBox.min[2], rightVec[j][2]);
					reducedBBox.max[0] = MyMAX(reducedBBox.max[0], rightVec[j][0]);
					reducedBBox.max[1] = MyMAX(reducedBBox.max[1], rightVec[j][1]);
					reducedBBox.max[2] = MyMAX(reducedBBox.max[2], rightVec[j][2]);
				}
				reducedBBox.min[0] = MyMAX(reducedBBox.min[0], currBBox.min[0]);
				reducedBBox.min[1] = MyMAX(reducedBBox.min[1], currBBox.min[1]);
				reducedBBox.min[2] = MyMAX(reducedBBox.min[2], currBBox.min[2]);
				reducedBBox.max[0] = MyMIN(reducedBBox.max[0], currBBox.max[0]);
				reducedBBox.max[1] = MyMIN(reducedBBox.max[1], currBBox.max[1]);
				reducedBBox.max[2] = MyMIN(reducedBBox.max[2], currBBox.max[2]);
				pTriangleInfos[i].AABB = reducedBBox;
			}
		}
	}
}

void push_triangles_to_child(const unsigned n_bEdge, const BoundEdge *bEdge, 
                             TriangleList *pLeftTriangles, TriangleList *pRightTriangles,
                             const SplitCost &bestCost )
{
	int currLeftIndex = 0, currRightIndex = 0;

	if (KD_TREE_PLANAR_TRIANGLE_ADD_MODE == BOTH_SIDE) {
		// NlogN (in RTGPU) 방식
		for ( unsigned int i = 0; i < n_bEdge; ++i ) {
			if( !bEdge[i].isPlanar ) {

				if( bEdge[i].t < bestCost.splitPos && bEdge[i].type == BoundEdge::START )
					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
				else if(bEdge[i].t > bestCost.splitPos && bEdge[i].type == BoundEdge::END)
					pRightTriangles[currRightIndex++] = *(bEdge[i].triangleInfo);

			} else if(bEdge[i].type == BoundEdge::START) {

				if(bEdge[i].t < bestCost.splitPos)
					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
				else if(bEdge[i].t > bestCost.splitPos)
					pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
				else { 
					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
					pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
				}	

			}
		}
	} else if (KD_TREE_PLANAR_TRIANGLE_ADD_MODE == MINCOST_SIDE) {
		// 기존 SGRTx2 방식 (상락&혁 방법)
		for ( unsigned int i = 0; i < n_bEdge; ++i ) {
			if( !bEdge[i].isPlanar ) {

				if( bEdge[i].t < bestCost.splitPos && bEdge[i].type == BoundEdge::START )
					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
				else if(bEdge[i].t > bestCost.splitPos && bEdge[i].type == BoundEdge::END)
					pRightTriangles[currRightIndex++] = *(bEdge[i].triangleInfo);

			} else if(bEdge[i].type == BoundEdge::START) {

				if(bEdge[i].t < bestCost.splitPos)
					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
				else if(bEdge[i].t > bestCost.splitPos)
					pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
				else { 
					if(bestCost.planar_side == BoundEdge::START)
						pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
					else 
						pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
				}	

			}
		}
	}
}

void try_to_split(const int axis, BoundingBox &inBBox, const TriangleList *pTriangles, const int triangleSize, BoundEdge *bEdge,  SplitCost &bestCost
//shyun added begin
#if SAH_OPACITY == 1
	, const double total_opacity_in_node
#endif
#if SAH_OPACITY >= 2
	, const double total_contribution_in_node
#endif
#if SAH_OPACITY >= 4
	, const float max_area_in_node
#endif
//shyun added end
)
{
	const int axis1 = modulo[axis + 1], axis2 = modulo[axis + 2];
	float fCell_extent[3];
		fCell_extent[0] = inBBox.max[0] - inBBox.min[0];
		fCell_extent[1] = inBBox.max[1] - inBBox.min[1];
		fCell_extent[2] = inBBox.max[2] - inBBox.min[2];

	// 전체 Cell 넓이의 1/2
	const double cell_area      = (fCell_extent[0] * fCell_extent[1]) + (fCell_extent[1] * fCell_extent[2]) + (fCell_extent[0] * fCell_extent[2]);
	const double cell_area_rcp  = 1 / cell_area;
	const double area_mul       = double( fCell_extent[axis1] ) + double( fCell_extent[axis2] );
	const double area_add       = double( fCell_extent[axis1] ) * double( fCell_extent[axis2] );
	const float  cell_min       = inBBox.min[axis];
	const float  cell_max       = inBBox.max[axis];
	const float  cell_length    = cell_max - cell_min;

	// ===========================================================
	// 모든 split candidate 에 대해 cost 계산
	// ===========================================================
//shyun added begin
#if SAH_OPACITY == 1
	double opacity_open = 0.0, opacity_close = 0.0, opacity_planar_local = 0.0;
	double opacity_local_open = 0.0, opacity_local_close = 0.0;
#elif SAH_OPACITY >= 2
	double contrib_open = 0.0, contrib_close = 0.0, contrib_planar_local = 0.0;
	double contrib_local_open = 0.0, contrib_local_close = 0.0;
#endif
//shyun added end
	/**
	 * open			: triangle box 기준으로 min 에 해당하는 개수
	 * close		: triangle box 기준으로 max 에 해당하는 개수
	 * num_planars	: triangle box 가 min = max 가 되는 것들의 개수(local 임)
	 * local_open / local_close : 현재 curr_bEdge.t 인 split position 에 해당하는 개수
	*/
	int open = 0, close = 0, num_planars = 0, local_open = 0, local_close = 0;
	int num_normalPositive = 0;

	// ===========================================================
	// edge 정보 정렬
	// ===========================================================
	// nlogn 방식처럼 정렬하지 않음, edge 의 type 은 고려하지 않고 순수히 위치로만 정렬함
	const unsigned n_bEdge = triangleSize * 2;
	set_bound_edge( axis, pTriangles, n_bEdge, bEdge );

#if SAH_OPACITY == 3
	std::vector<float> from_left_max_area(n_bEdge, 0.0f);
	std::vector<float> from_right_max_area(n_bEdge, 0.0f);
	// Pass 1: L->R Sweep
	// 왼쪽에서 오른쪽으로 진행하며, 각 위치까지 '시작된' 삼각형들의 최대 넓이를 누적
	float running_max_L = 0.0f;
	for (unsigned int i = 0; i < n_bEdge; ++i) {
		if (bEdge[i].type == BoundEdge::START) {
			running_max_L = MyMAX(running_max_L, bEdge[i].triangleInfo->area);
		}
		from_left_max_area[i] = running_max_L;
	}

	// Pass 2: R->L Sweep
	// 오른쪽에서 왼쪽으로 진행하며, 각 위치부터 '끝나는' 삼각형들의 최대 넓이를 누적
	float running_max_R = 0.0f;
	for (int i = n_bEdge - 1; i >= 0; --i) {
		if (bEdge[i].type == BoundEdge::END) {
			running_max_R = MyMAX(running_max_R, bEdge[i].triangleInfo->area);
		}
		from_right_max_area[i] = running_max_R;
	}
#endif

	for (unsigned int i = 0; i < n_bEdge; i++) {
		// 현재 bEdge[i] 가 자르고자 하는 plane candidate
		BoundEdge curr_bEdge = bEdge[i];

		//planar는 open과 close에 둘 다 포함됨
		// (원래는 2개(min/max)가 planar 한개로 계산 됐으므로 min->open, max->close 로 각각 들어감.)
		open += local_open + num_planars;
		close += local_close + num_planars;

		local_open = 0;  local_close = 0; num_planars = 0; // num_planars 도 local 계산임
		num_normalPositive = 0;
#if SAH_OPACITY == 1
		opacity_open += opacity_local_open + opacity_planar_local;
		opacity_close += opacity_local_close + opacity_planar_local;

		opacity_planar_local = 0.0; opacity_local_open = 0.0; opacity_local_close = 0.0;
#elif SAH_OPACITY >= 2
		contrib_open += contrib_local_open + contrib_planar_local;
		contrib_close += contrib_local_close + contrib_planar_local;

		contrib_planar_local = 0.0; contrib_local_open = 0.0; contrib_local_close = 0.0;
#endif

		//현재 axis와 side에 대해 포지션구함
		const float cur_position = curr_bEdge.t;
#if SAH_OPACITY == 3
		float planar_max_area = 0.0f; // 현재 위치의 Planar 삼각형 최대 넓이
#endif

		// ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ --
		// Split plane candidate 와 같은 위치의 edge 들에 대한 처리
		// ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ --
		{
			//똑같은게 여러개 있을 때는 제일 오른쪽에서만 SAH 계산을 한다.
			for (unsigned int j = i; j < n_bEdge; j++) {
				BoundEdge tmp_bEdge = bEdge[j];
				if (tmp_bEdge.t != cur_position) break;

				const bool
					is_left = (tmp_bEdge.type == BoundEdge::START),
					is_planar = tmp_bEdge.isPlanar,
					is_normalPositive = tmp_bEdge.isNormalPositive;
//shyun added begin
#if SAH_OPACITY == 1
				const float tri_opacity = tmp_bEdge.triangleInfo->opacity;
#elif SAH_OPACITY >= 2
				const float tri_contrib = tmp_bEdge.triangleInfo->opacity
										* tmp_bEdge.triangleInfo->area;
#endif
//shyun added end
				//카운팅
				if (!is_planar) {
					local_open += is_left ? 1 : 0; //!< box 의 왼쪽은 local_open 을 증가
					local_close += is_left ? 0 : 1; //!< box 의 오른쪽은 local_close 를 증가
				}
				else {
					//플라나하다면 따로 카운팅
					num_planars += is_left ? 1 : 0;	// only count it once
					num_normalPositive += is_normalPositive ? 1 : 0;
#if SAH_OPACITY == 3
					if (is_left) planar_max_area = MyMAX(planar_max_area, tmp_bEdge.triangleInfo->area);
#endif
				}
//shyun added begin
#if SAH_OPACITY == 1
				if (!is_planar) {
					opacity_local_open += is_left ? tri_opacity : 0.0;
					opacity_local_close += is_left ? 0.0 : tri_opacity;
				}
				else {
					opacity_planar_local += is_left ? tri_opacity : 0.0;
				}
#elif SAH_OPACITY >= 2
				if (!is_planar) {
					contrib_local_open += is_left ? tri_contrib : 0.0;
					contrib_local_close += is_left ? 0.0 : tri_contrib;
				}
				else {
					contrib_planar_local += is_left ? tri_contrib : 0.0;
				}
#endif
//shyun added end

				curr_bEdge = tmp_bEdge;
				i = j;
			}
		}

		// Numerical error 
		if (cur_position <= cell_min + KD_TREE_EPSILON) continue;
		if (cur_position >= cell_max - KD_TREE_EPSILON) break;


		// ==================================================
		// Scoring
		// ==================================================
		{
			const double
				extent_l = double(cur_position) - cell_min,
				extent_r = cell_max - double(cur_position);
			const double
				prob_l = (extent_l * area_mul + area_add) * cell_area_rcp,
				prob_r = (extent_r * area_mul + area_add) * cell_area_rcp;

			const int
				n_leftOnly = close + local_close,  // close 가 된다면 그 삼각형은 오른쪽에 있지도 않게 됨 (겹치지도 않음)
				n_cross = open - n_leftOnly,   // open = n_leftOnly + n_cross (현재 local_open 은 포함하지 않음)
				n_rightOnly = triangleSize - (n_leftOnly + n_cross + num_planars);

			// planar  는 local_open 에 해당하는 것만 카운팅함.
			// n_cross 는 local_open 은 고려하지 않음. 즉, planar 도 고려하지 않음.
			// 전체 triangle_size = n_leftOnly + n_rightOnly + n_cross + num_planars

			double ExpectedCost;
			int nTri_left, nTri_right;
			int planar_side;

			// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			// The best spliting position search algorithm as decribed in 
			//  "On building fast kd-Trees for Ray Tracing, and on doing that in O(N log N)"
			//  Based on RTGPU code
			// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			if (KD_TREE_PLANAR_TRIANGLE_ADD_MODE == BOTH_SIDE) {
				const int tri_num_left = n_leftOnly + n_cross + num_planars;
				const int tri_num_right = n_rightOnly + n_cross + num_planars;
				const float	emptyBonus
					= (tri_num_left == 0 || tri_num_right == 0) ? v_KD_TREE_EMTPY_BONUS : 1.0f;

				ExpectedCost = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
					double(tri_num_left) * prob_l +
					double(tri_num_right) * prob_r) * emptyBonus;

				planar_side = 2;
				nTri_left = tri_num_left;
				nTri_right = tri_num_right;

				// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
				// Cost function 의 최소값에 따라
				// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			}
			else { // MINCOST_SIDE
//shyun added begin
#if SAH_OPACITY == 1
				// 불투명도 합계를 기반으로 각 영역(왼쪽만, 오른쪽만, 교차)을 계산
				const double op_leftOnly = opacity_close + opacity_local_close;
				const double op_cross = opacity_open - op_leftOnly;
				const double op_rightOnly = total_opacity_in_node - (op_leftOnly + op_cross + opacity_planar_local);

				// 두 가지 시나리오에 대한 불투명도 합계를 계산
				// - 시나리오 [0]: 평면 삼각형(Planar)을 왼쪽에 포함
				// - 시나리오 [1]: 평면 삼각형(Planar)을 오른쪽에 포함
				const double total_op_left[2] = { op_leftOnly + op_cross + opacity_planar_local, op_leftOnly + op_cross };
				const double total_op_right[2] = { op_rightOnly + op_cross, op_rightOnly + op_cross + opacity_planar_local };
#elif SAH_OPACITY >= 2
				const double contr_leftOnly = contrib_close + contrib_local_close;
				const double contr_cross = contrib_open - contr_leftOnly;
				const double contr_rightOnly = total_contribution_in_node - (contr_leftOnly + contr_cross + contrib_planar_local);

				const double total_contr_left[2] = { contr_leftOnly + contr_cross + contrib_planar_local, contr_leftOnly + contr_cross };
				const double total_contr_right[2] = { contr_rightOnly + contr_cross, contr_rightOnly + contr_cross + contrib_planar_local };
#endif
//shyun added end
				
				// 배열[0] 은 planar 가 왼쪽에 들어갔을 경우
				// 배열[1] 은 planar 가 오른쪽에 들어갔을 경우

				//최종적인 양쪽 갯수.
				const int tri_num_left[2] = { n_leftOnly + n_cross + num_planars, n_leftOnly + n_cross };
				const int tri_num_right[2] = { n_rightOnly + n_cross, n_rightOnly + n_cross + num_planars };

				const float	emptyBonus[2] = {
					(tri_num_left[0] == 0 || tri_num_right[0] == 0) ? v_KD_TREE_EMTPY_BONUS : 1.0f,
					(tri_num_left[1] == 0 || tri_num_right[1] == 0) ? v_KD_TREE_EMTPY_BONUS : 1.0f };

				double SAH[2];
				for (int side_idx = 0; side_idx < 2; side_idx++) {
					SAH[side_idx] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
//shyun added begin
#if SAH_OPACITY == 1
						// 개수 대신 불투명도 합계 사용
						total_op_left[side_idx] * prob_l +
						total_op_right[side_idx] * prob_r
#elif SAH_OPACITY == 2
						total_contr_left[side_idx] * prob_l +
						total_contr_right[side_idx] * prob_r
#elif SAH_OPACITY == 3
						//TODO: 분할 기준으 최대 area로 나누어주기
						[&] {
							// 분할 평면은 bEdge[i]와 bEdge[i+1] 사이에 위치합니다.
							// L 그룹의 max_area는 bEdge[i]까지 시작된 모든 삼각형의 최대 넓이입니다.
							const float max_area_L_base = from_left_max_area[i];
							// R 그룹의 max_area는 bEdge[i+1]부터 끝나는 모든 삼각형의 최대 넓이입니다.
							const float max_area_R_base = (i + 1 < n_bEdge) ? from_right_max_area[i + 1] : 0.0f;

							float current_max_L = max_area_L_base;
							float current_max_R = max_area_R_base;

							// Planar 삼각형들을 어느 쪽에 포함시킬지에 따라 max_area를 최종 결정합니다.
							if (num_planars > 0) {
								if (i == 0) { // Planar가 왼쪽에 포함될 경우
									current_max_L = MyMAX(current_max_L, planar_max_area);
								}
								else { // Planar가 오른쪽에 포함될 경우
									current_max_R = MyMAX(current_max_R, planar_max_area);
								}
							}

							const double cost_l = (current_max_L > KD_TREE_EPSILON) ?
								(total_contr_left[side_idx] / current_max_L) : total_contr_left[side_idx];
							const double cost_r = (current_max_R > KD_TREE_EPSILON) ?
								(total_contr_right[side_idx] / current_max_R) : total_contr_right[side_idx];

							return (cost_l * prob_l + cost_r * prob_r);
						}()
#elif SAH_OPACITY == 4
						(max_area_in_node > KD_TREE_EPSILON) ?
						(total_contr_left[side_idx] / max_area_in_node) * prob_l +
						(total_contr_right[side_idx] / max_area_in_node) * prob_r
						: // max_area_in_node가 0일 경우의 예외 처리 (SAH 2와 동일하게)
						total_contr_left[side_idx] * prob_l +
						total_contr_right[side_idx] * prob_r
#elif SAH_OPACITY == 5
						(max_area_in_node > KD_TREE_EPSILON) ?
						(total_contr_left[side_idx] / max_area_in_node) +
						(total_contr_right[side_idx] / max_area_in_node)
						: // max_area_in_node가 0일 경우의 예외 처리
						total_contr_left[side_idx] +
						total_contr_right[side_idx]
//shyun added end
#else
						double(tri_num_left[side_idx])* prob_l +
						double(tri_num_right[side_idx]) * prob_r
#endif
						)* emptyBonus[side_idx];
				}
#if SAH_MAXIMIZE
				if (SAH[0] >= SAH[1]) { // planar 를 왼쪽에 넣는 것이 낫다면,
#else
				if (SAH[0] <= SAH[1]) { // planar 를 왼쪽에 넣는 것이 낫다면,
#endif
					ExpectedCost = SAH[0];
					planar_side = BoundEdge::START;
					nTri_left = tri_num_left[0];
					nTri_right = tri_num_right[0];
				}
				else {
					ExpectedCost = SAH[1];
					planar_side = BoundEdge::END;
					nTri_left = tri_num_left[1];
					nTri_right = tri_num_right[1];
				}
			}

			if (ExpectedCost < bestCost.cost) {
				bestCost.cost = ExpectedCost;
				bestCost.splitPos = cur_position;
				bestCost.axis = axis;

				bestCost.n_onlyLeft = n_leftOnly;
				bestCost.n_onlyRight = n_rightOnly;
				bestCost.n_cross = n_cross;
				bestCost.n_planar = num_planars;
				bestCost.n_left = nTri_left;
				bestCost.n_right = nTri_right;

				bestCost.planar_side = planar_side;
			}

		} // scoring
	}
}

bool initialize_kd_tree(CompositeObject *poly_model) {
	// Returns 1 if kd-tree data was initialized successfully, or 0 otherwise.

	bool bError = false;

	g_iKdTree_Node_Count      = 0;
	g_iKdTree_TriOffset_Count = 0;
	g_pKdTree_Node_Array      = NULL;
	g_pKdTree_TriOffset_Array = NULL;
	g_iKdTree_Node_CountAlloc      =  8 * 1024 * 1024; 
	g_iKdTree_TriOffset_CountAlloc = 16 * 1024 * 1024;

	g_iKdTree_Level = 0;
	g_iKdTree_LeafNode_Count  = 0;
	g_iKdTree_EmptyNode_Count = 0;
	g_iKdTree_MaxTriInLeafNode_Count  = 0;

	g_iTriangleSize = poly_model->n_triangles;

	g_root_AABB.min[0] = poly_model->AABB[0];
	g_root_AABB.min[1] = poly_model->AABB[2];
	g_root_AABB.min[2] = poly_model->AABB[4];
	g_root_AABB.max[0] = poly_model->AABB[1];
	g_root_AABB.max[1] = poly_model->AABB[3];
	g_root_AABB.max[2] = poly_model->AABB[5];

	ExtendedVertex *pVertexList = poly_model->extended_vertices;

	g_pTriangleInfos = new TriangleList[g_iTriangleSize];
		if( g_pTriangleInfos == NULL ) {
			bError |= true;
		} else {
			for( int i = 0; i < g_iTriangleSize; i++ ) {
				g_pTriangleInfos[i].offset = i;
				g_pTriangleInfos[i].point[0] = pVertexList[3*i];
				g_pTriangleInfos[i].point[1] = pVertexList[3*i+1];
				g_pTriangleInfos[i].point[2] = pVertexList[3*i+2];
//shyun added begin
#if SAH_OPACITY
				int gaussian_idx = g_pTriangleInfos[i].point[0].material_ID;

				// 인덱스가 유효한지 확인하고 Opacity 값 가져오기
				if (gaussian_idx >= 0 && gaussian_idx < g_gaussians.size()) {
					g_pTriangleInfos[i].opacity = g_gaussians[gaussian_idx].opacity;
				}
				else {
					// 가우시안이 아닌 일반 지오메트리를 위한 기본값
					g_pTriangleInfos[i].opacity = 1.0f;
				}
#endif
//shyun added end
				g_pTriangleInfos[i].AABB.min[0] = MyMIN (MyMIN (g_pTriangleInfos[i].point[0].vertex[0], g_pTriangleInfos[i].point[1].vertex[0]), g_pTriangleInfos[i].point[2].vertex[0]);
				g_pTriangleInfos[i].AABB.min[1] = MyMIN (MyMIN (g_pTriangleInfos[i].point[0].vertex[1], g_pTriangleInfos[i].point[1].vertex[1]), g_pTriangleInfos[i].point[2].vertex[1]);
				g_pTriangleInfos[i].AABB.min[2] = MyMIN (MyMIN (g_pTriangleInfos[i].point[0].vertex[2], g_pTriangleInfos[i].point[1].vertex[2]), g_pTriangleInfos[i].point[2].vertex[2]);
				g_pTriangleInfos[i].AABB.max[0] = MyMAX (MyMAX (g_pTriangleInfos[i].point[0].vertex[0], g_pTriangleInfos[i].point[1].vertex[0]), g_pTriangleInfos[i].point[2].vertex[0]);
				g_pTriangleInfos[i].AABB.max[1] = MyMAX (MyMAX (g_pTriangleInfos[i].point[0].vertex[1], g_pTriangleInfos[i].point[1].vertex[1]), g_pTriangleInfos[i].point[2].vertex[1]);
				g_pTriangleInfos[i].AABB.max[2] = MyMAX (MyMAX (g_pTriangleInfos[i].point[0].vertex[2], g_pTriangleInfos[i].point[1].vertex[2]), g_pTriangleInfos[i].point[2].vertex[2]);

#if SAH_OPACITY >= 2
				// 삼각형의 면적을 계산하여 contribution 값을 채웁니다.
				float v0[3], v1[3], v2[3];
				memcpy(v0, g_pTriangleInfos[i].point[0].vertex, sizeof(float) * 3);
				memcpy(v1, g_pTriangleInfos[i].point[1].vertex, sizeof(float) * 3);
				memcpy(v2, g_pTriangleInfos[i].point[2].vertex, sizeof(float) * 3);

				// 두 개의 엣지 벡터 계산
				double edge1[3] = { (double)v1[0] - v0[0], (double)v1[1] - v0[1], (double)v1[2] - v0[2] };
				double edge2[3] = { (double)v2[0] - v0[0], (double)v2[1] - v0[1], (double)v2[2] - v0[2] };

				// 외적 계산
				double cross_product[3];
				dMyVecCrossProduct(edge1, edge2, cross_product);

				// 외적 벡터의 크기(길이) 계산
				double magnitude = dMyVecLength(cross_product);

				// 최종 면적 및 contribution 계산
				float area = (float)(0.5 * magnitude);
				g_pTriangleInfos[i].area = area;
#endif
			}
		}

	g_bEdge = new BoundEdge[g_iTriangleSize * 2];
		if( g_bEdge == NULL ) {
			bError |= true;
		} else {
			memset( g_bEdge, 0x00, sizeof( BoundEdge ) * g_iTriangleSize * 2 );
		}
		
	g_pKdTree_Node_Array = new KdTreeNode[ g_iKdTree_Node_CountAlloc ];
		if( g_pKdTree_Node_Array == NULL ) {
			bError |= true;
		} else {
			memset( g_pKdTree_Node_Array, 0x00, sizeof( KdTreeNode ) * g_iKdTree_Node_CountAlloc );
			g_iKdTree_Node_Count = 1; 
		}
		
	g_pKdTree_TriOffset_Array = new unsigned int [ g_iKdTree_TriOffset_CountAlloc ];
		if( g_pKdTree_TriOffset_Array == NULL )	{
			bError |= true;
		} else {
			memset( g_pKdTree_TriOffset_Array, 0x00, sizeof( unsigned int ) * g_iKdTree_TriOffset_CountAlloc );
		}

	if (bError) {
		printf("init kd-tree bError");
		uninitialize_kd_tree();
		return 0;
	}
	return 1;
}

void uninitialize_kd_tree(void) {
	if( g_pTriangleInfos ) {
		delete[] g_pTriangleInfos;
		g_pTriangleInfos = NULL;
	}
	if( g_bEdge ) {
		delete[] g_bEdge;
		g_bEdge = NULL;
	}
	if( g_pKdTree_Node_Array ) {
		delete[] g_pKdTree_Node_Array;
		g_pKdTree_Node_Array = NULL;
	}
	if( g_pKdTree_TriOffset_Array ) {
		delete[] g_pKdTree_TriOffset_Array;
		g_pKdTree_TriOffset_Array = NULL;
	}
}

#define DEBUG_FLAG 0

void build_kd_tree_recursive(BoundEdge* bEdge, const TriangleList* pTriangleInfos, unsigned int triangleSize,
	BoundingBox& bbox, unsigned int inNodeLevel, KdTreeNode* inNode)
{
	SplitCost bestCost;
	g_iKdTree_Level = MyMAX(inNodeLevel, g_iKdTree_Level);
	//printf("innodeLevel %d\n", g_iKdTree_Level);


	if (DEBUG_FLAG) {
		fprintf(stdout, "b_k_t_r: (S) triangleSize = %d, inNodeLevel = %d\n", triangleSize, inNodeLevel);
	}

	// Calculate cost function (in case of no partition)
	bestCost.cost = double(triangleSize) * v_KD_TREE_ISECT_COST;
//shyun added begin
#if SAH_OPACITY == 1
	double total_opacity_in_node = 0.0;
	for (unsigned int i = 0; i < triangleSize; ++i) {
		total_opacity_in_node += pTriangleInfos[i].opacity;
}
	bestCost.cost = total_opacity_in_node * v_KD_TREE_ISECT_COST;
#endif
#if SAH_OPACITY >= 2
	double total_contribution_in_node = 0.0;
	for (unsigned int i = 0; i < triangleSize; ++i) {
		total_contribution_in_node += pTriangleInfos[i].area * pTriangleInfos[i].opacity;
	}
	bestCost.cost = total_contribution_in_node * v_KD_TREE_ISECT_COST;
#endif
#if SAH_OPACITY >= 4
	float max_area_in_node = 0.0f;
	for (unsigned int i = 0; i < triangleSize; ++i) {
		max_area_in_node = MyMAX(max_area_in_node, pTriangleInfos[i].area);
	}
	bestCost.cost = total_contribution_in_node * v_KD_TREE_ISECT_COST;

	// 최대 면적이 0보다 큰 경우에만 정규화된 비용을 사용
	if (max_area_in_node > KD_TREE_EPSILON) {
		bestCost.cost /= max_area_in_node;
	}
	else if(DEBUG_FLAG) {
		printf("lv %d: tiny max area %f\n", inNodeLevel, max_area_in_node);
	}
#endif

#if FORCE_SPLIT_THRESHOLD
	if (triangleSize > FORCE_SPLIT_THRESHOLD) bestCost.cost = DBL_MAX; //shyun added
#endif
//shyun added end

	// Calculate cost function (in case of trying to partition)
	if (inNodeLevel < v_KD_TREE_MAX_LEVEL && triangleSize > v_KD_TREE_MIN_TRIANGLE) {
		// (모든 축에 대해 수행)
		for (int axis = 0; axis < 3; axis++) {
			try_to_split(axis, bbox, pTriangleInfos, triangleSize, bEdge, bestCost
//shyun added begin
#if SAH_OPACITY == 1
			,total_opacity_in_node
#endif
#if SAH_OPACITY >= 2
			, total_contribution_in_node
#endif
#if SAH_OPACITY >= 4
			, max_area_in_node
#endif
//shyun added end
				);
		}
	}
	
	// ----------------------------------------------------------------------------
	// Leaf node 생성
	// ----------------------------------------------------------------------------
	if (!bestCost.is_valid()) {
		//fprintf(stdout, "-> Leaf Node generated with %u triangles.\n", triangleSize);

		unsigned int iTriOffset;

		{
			iTriOffset = g_iKdTree_TriOffset_Count;
			setLeafNode(inNode, triangleSize, g_iKdTree_TriOffset_Count);
			g_iKdTree_TriOffset_Count += triangleSize;

			// 메모리 체크 : Triangle Offset Size
			if (g_iKdTree_TriOffset_Count >= g_iKdTree_TriOffset_CountAlloc) {
				_reAllocTriangleOffsetList(MyMAX(2 * g_iKdTree_TriOffset_CountAlloc, 512), g_iKdTree_TriOffset_CountAlloc, &g_pKdTree_TriOffset_Array);
			}
		}

		/**
		 *	leaf node가 참조하는 triangle의 offset 을 offsetList 마지막에 추가해 넣는다.
		 */
		unsigned* currOffsetList = &g_pKdTree_TriOffset_Array[iTriOffset];

		unsigned leafCount = 0;
		for (unsigned i = 0; i < triangleSize; i++) {
			currOffsetList[leafCount++] = pTriangleInfos[i].offset;
		}

		if (triangleSize == 0)
			g_iKdTree_EmptyNode_Count++;

		g_iKdTree_LeafNode_Count++;
		g_iKdTree_MaxTriInLeafNode_Count = MyMAX(g_iKdTree_MaxTriInLeafNode_Count, triangleSize);

		/**
		 *	더이상 pTriangleInfos 는 필요없으므로 메모리 공간 절약을 위해 없앤다.
		 */
		delete[] pTriangleInfos;

	}
	else
		// ----------------------------------------------------------------------------
		// Inner node 생성
		// ----------------------------------------------------------------------------
	{
		unsigned int nodeNum;
		{
			nodeNum = g_iKdTree_Node_Count;
			setInnerNode(inNode, bestCost.axis, g_iKdTree_Node_Count, bestCost.splitPos);

			g_iKdTree_Node_Count += 2;

			// 메모리 체크 : Node Size
			if (g_iKdTree_Node_Count >= g_iKdTree_Node_CountAlloc) {
				_reAllocKdtreeNodes(MyMAX(2 * g_iKdTree_Node_CountAlloc, 512), g_iKdTree_Node_CountAlloc, &g_pKdTree_Node_Array);
			}
		}

		BoundingBox leftnBounds, rightnBounds;
		leftnBounds = bbox;  leftnBounds.max[bestCost.axis] = bestCost.splitPos;
		rightnBounds = bbox;  rightnBounds.min[bestCost.axis] = bestCost.splitPos;

		TriangleList* pLeftTriangles = new TriangleList[bestCost.n_left];
		TriangleList* pRightTriangles = new TriangleList[bestCost.n_right];
		//	TriangleList *pLeftTriangles = (TriangleList *) malloc(sizeof(TriangleList)*bestCost.n_left);
		//	TriangleList *pRightTriangles = (TriangleList *) malloc(sizeof(TriangleList)*bestCost.n_right); 

		if (1) {
			if (pLeftTriangles == NULL) {
				fprintf(stdout, "b_k_t_r: left pointer NULL\n");
				fprintf(stdout, "b_k_t_r: Point 7 n_left = %d, n_right = %d\n", bestCost.n_left, bestCost.n_right);
			}
			if (pRightTriangles == NULL) {
				fprintf(stdout, "b_k_t_r: right pointer NULL\n");
				fprintf(stdout, "b_k_t_r: Point 7 n_left = %d, n_right = %d\n", bestCost.n_left, bestCost.n_right);
			}
		}

		const unsigned n_bEdge = 2 * triangleSize;

		// TriangleInfo 부터 bEdge 를 생성 및 정렬
		set_bound_edge(bestCost.axis, pTriangleInfos, n_bEdge, bEdge);

		// bEdge 로 부터 pLeftTriangle, pRightTriangle 을 생성
		push_triangles_to_child(n_bEdge, bEdge, pLeftTriangles, pRightTriangles, bestCost);

		// 각 child node 에 맞게 삼각형 clipping
		clip_triangle(bestCost.n_left, bestCost, pLeftTriangles, 0);
		clip_triangle(bestCost.n_right, bestCost, pRightTriangles, 1);

		/**
		 *	더이상 pTriangleInfos 는 필요없으므로 메모리 공간 절약을 위해 없앤다.
		 *	반드시 pushChildTriangles 를 수행한 이후에 없애야 한다.
		 */
		delete[] pTriangleInfos;
		/**
		 *	Left, Right 재귀 탐색.
		 *	pLeftTriangles, pRightTriangles 는 build_kd_tree_recursive 함수 안에서 사용하고 바로 없앤다.
		 */
		build_kd_tree_recursive(bEdge, pLeftTriangles, bestCost.n_left, leftnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum]);
		build_kd_tree_recursive(bEdge, pRightTriangles, bestCost.n_right, rightnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum + 1]);
	}

	if (DEBUG_FLAG) {
		fprintf(stdout, "b_k_t_r: (E)triangleSize = %d, inNodeLevel = %d\n", triangleSize, inNodeLevel);
	}
}

void build_TriAccList(CompositeObject *poly_model, TriAccel*& pTriAcc)
{
	int iTriangleSize  = poly_model->n_triangles;
	printf("build_TriAccList triangle:%d\n", iTriangleSize);

	if (!pTriAcc) {
		pTriAcc = (TriAccel*)_aligned_malloc(sizeof(TriAccel) * iTriangleSize, 16);
		if (!pTriAcc) {
			fprintf(stderr, "ERROR: Failed to allocate memory for TriAccel list! (size: %d)\n", iTriangleSize);
			throw std::bad_alloc();
		}
	}
	memset(pTriAcc, 0, sizeof(TriAccel)* iTriangleSize);

	// pTriAcc가 NULL인지 확인 (메모리 할당 실패 여부 검사)
	//if (pTriAcc == NULL) {
	//	fprintf(stderr, "ERROR: Failed to allocate memory for TriAccel list! (size: %d)\n", iTriangleSize);
	//	return; // 함수를 안전하게 종료
	//}

	float A[3], B[3], C[3];
	float b[3], c[3];
	float N[3];

	int k, u, v;
	float r;
	
	ExtendedVertex *pVertexList = poly_model->extended_vertices;

	unsigned int i;
	for (i = 0; i < iTriangleSize; i++) {
		A[0] = pVertexList[3*i].vertex[0];
		A[1] = pVertexList[3*i].vertex[1];
		A[2] = pVertexList[3*i].vertex[2];
		B[0] = pVertexList[3*i+1].vertex[0];
		B[1] = pVertexList[3*i+1].vertex[1];
		B[2] = pVertexList[3*i+1].vertex[2];
		C[0] = pVertexList[3*i+2].vertex[0];
		C[1] = pVertexList[3*i+2].vertex[1];
		C[2] = pVertexList[3*i+2].vertex[2];

		// calc edges and normal
		b[0] = C[0] - A[0];
		b[1] = C[1] - A[1];
		b[2] = C[2] - A[2];
		c[0] = B[0] - A[0];
		c[1] = B[1] - A[1];
		c[2] = B[2] - A[2];
		fMyVecCrossProduct(b, c, N);
		fMyVecNormalize(N);

		k = (fabsf(N[0]) > fabsf(N[1])) ? ( (fabsf(N[0])>fabsf(N[2]))?0:2 ):( (fabsf(N[1])>fabsf(N[2]))?1:2 );
		u = modulo[k+1];
		v = modulo[k+2];

		pTriAcc[i].N[0] = N[0];
		pTriAcc[i].N[1] = N[1];
		pTriAcc[i].N[2] = N[2];

		// N'
		float fRcp_N_k = 1.0f / N[k];
		N[0] *= fRcp_N_k;
		N[1] *= fRcp_N_k;
		N[2] *= fRcp_N_k;

		pTriAcc[i].paccked_flags = 0; // shyun added: 먼저 모든 비트를 0으로 초기화
		pTriAcc[i].k   = k;
		pTriAcc[i].n_u = N[u];						// N'u
		pTriAcc[i].n_v = N[v];						// N'v
		pTriAcc[i].n_d = fMyVecDotProduct(N, A);	// A dot N'

		r = 1.f / (b[u] * c[v] - b[v] * c[u]);
		// beta
		pTriAcc[i].b_nu = r * -b[v];
		pTriAcc[i].b_nv = r *  b[u];
		pTriAcc[i].b_d  = r * (b[v] * A[u] - b[u] * A[v]);
		// gamma
		pTriAcc[i].c_nu = r *  c[v];
		pTriAcc[i].c_nv = r * -c[u];
		pTriAcc[i].c_d  = r * (c[u] * A[v] - c[v] * A[u]);

		pTriAcc[i].mbox = 0;
		pTriAcc[i].material_ID = pVertexList[3*i].material_ID;

		// Need to be modified
		float fTransparency = 0.0f;
		pTriAcc[i].isTransparent = (fTransparency > 0)?1:0;

		//printf("k(%d): %u\n", i, pTriAcc[i].k);
	}
	//printf("k(%d): %u\n", 315728, pTriAcc[315728].k);
}

//from JJH
std::vector<BoundingBox> extract_leaves_from_kd_tree()
{
	KdTreeNode* node = &g_pKdTree_Node_Array[0];

	struct KdStack {
		KdTreeNode* node;
		BoundingBox box;
	};

	std::stack<KdStack> kd_stack;
	kd_stack.push({ node, g_root_AABB });

	std::vector<BoundingBox> leaf_node_boxes;
	BoundingBox box{};

	while (!kd_stack.empty()) {
		KdStack data = kd_stack.top();
		kd_stack.pop();
		node = data.node;
		box = data.box;
		if (IS_LEAF(*node)) {
			leaf_node_boxes.push_back(box);
		}
		else {
			const float node_split = SPLIT_POS(*node);
			const uint32_t dim = SPLIT_AXIS(*node);

			BoundingBox right_box = box;
			right_box.min[dim] = node_split;
			kd_stack.push({ &g_pKdTree_Node_Array[SECOND_CHILD_OFFSET(*node)], right_box });

			BoundingBox left_box = box;
			left_box.max[dim] = node_split;
			kd_stack.push({ &g_pKdTree_Node_Array[FIRST_CHILD_OFFSET(*node)], left_box });
		}
	}

	return leaf_node_boxes;
}

//std::vector<LeafNodeInfo> extract_all_leaf_data(CompositeObject* c_object) {
//	KdTreeNode* node = &g_pKdTree_Node_Array[0];
//
//	struct KdStack {
//		KdTreeNode* node;
//		BoundingBox box;
//	};
//
//	std::stack<KdStack> kd_stack;
//	kd_stack.push({ node, g_root_AABB });
//
//	// 반환할 데이터 타입 변경
//	std::vector<LeafNodeInfo> all_leaf_info;
//	BoundingBox box{};
//
//	while (!kd_stack.empty()) {
//		KdStack data = kd_stack.top();
//		kd_stack.pop();
//		node = data.node;
//		box = data.box;
//
//		if (IS_LEAF(*node)) {
//			LeafNodeInfo current_leaf;
//			current_leaf.aabb = box; // 1. 바운딩 박스 저장
//
//			// 리프 노드에서 삼각형 오프셋과 개수 가져오기
//			unsigned int offset = OBJECTLIST_OFFSET(*node);
//			unsigned int count = OBJECT_SIZE(*node) + offset;
//
//			// 전역 오프셋 리스트에서 삼각형 인덱스를 가져와 저장
//			current_leaf.triangle_indices.reserve(count); // 메모리 미리 할당
//			for (; offset < count; offset++) {
//				current_leaf.triangle_indices.push_back(c_object->kd_tree->tri_offset_list[offset]);
//			}
//
//			all_leaf_info.push_back(current_leaf);
//			if (OBJECT_SIZE(*node) > 1000) {
//				printf("%dth node size: %d\n", all_leaf_info.size(), OBJECT_SIZE(*node));
//			}
//			// ------------------------------------
//		}
//		else {
//			const float node_split = SPLIT_POS(*node);
//			const uint32_t dim = SPLIT_AXIS(*node);
//
//			BoundingBox right_box = box;
//			right_box.min[dim] = node_split;
//			kd_stack.push({ &g_pKdTree_Node_Array[SECOND_CHILD_OFFSET(*node)], right_box });
//
//			BoundingBox left_box = box;
//			left_box.max[dim] = node_split;
//			kd_stack.push({ &g_pKdTree_Node_Array[FIRST_CHILD_OFFSET(*node)], left_box });
//		}
//	}
//
//	return all_leaf_info;
//}

#include <algorithm>
std::vector<LeafNodeInfo> extract_all_leaf_data(CompositeObject* c_object, int& largest_leaf_index) {
	// Kd-tree가 없으면 빈 벡터 반환
	if (c_object == nullptr || c_object->kd_tree == nullptr || c_object->kd_tree->tree == nullptr) {
		return {};
	}

	// 전역 변수 대신 c_object에서 직접 데이터 가져오기
	KdTreeNode* root_node = &c_object->kd_tree->tree[0];
	unsigned int* tri_offset_list = c_object->kd_tree->tri_offset_list;

	struct KdStack {
		KdTreeNode* node;
		BoundingBox box;
	};

	BoundingBox root_bbox;
	root_bbox.min[0] = c_object->AABB[XMIN];
	root_bbox.max[0] = c_object->AABB[XMAX];
	root_bbox.min[1] = c_object->AABB[YMIN];
	root_bbox.max[1] = c_object->AABB[YMAX];
	root_bbox.min[2] = c_object->AABB[ZMIN];
	root_bbox.max[2] = c_object->AABB[ZMAX];

	std::stack<KdStack> kd_stack;
	kd_stack.push({ root_node, root_bbox });

	std::vector<LeafNodeInfo> all_leaf_info;

	unsigned int max_triangles_found = 0;
	largest_leaf_index = -1;

	while (!kd_stack.empty()) {
		KdStack data = kd_stack.top();
		kd_stack.pop();
		KdTreeNode* current_node = data.node;
		BoundingBox current_box = data.box;

		if (IS_LEAF(*current_node)) {
			LeafNodeInfo leaf;
			leaf.aabb = current_box;

			unsigned int offset = OBJECTLIST_OFFSET(*current_node);
			unsigned int num_triangles = OBJECT_SIZE(*current_node);

			for (unsigned int i = 0; i < num_triangles; ++i) {
				leaf.triangle_indices.push_back(tri_offset_list[offset + i]);
			}

			all_leaf_info.push_back(leaf);
			// --------------------
			if (num_triangles > max_triangles_found) {
				max_triangles_found = num_triangles;
				largest_leaf_index = all_leaf_info.size() - 1;
			}
		}
		else {
			const float node_split = SPLIT_POS(*current_node);
			const uint32_t dim = SPLIT_AXIS(*current_node);

			BoundingBox right_box = current_box;
			right_box.min[dim] = node_split;
			kd_stack.push({ &c_object->kd_tree->tree[SECOND_CHILD_OFFSET(*current_node)], right_box });

			BoundingBox left_box = current_box;
			left_box.max[dim] = node_split;
			kd_stack.push({ &c_object->kd_tree->tree[FIRST_CHILD_OFFSET(*current_node)], left_box });
		}
	}
	if (largest_leaf_index != -1) {
		printf("[INFO] Largest leaf found at index %d with %u triangles.\n",
			largest_leaf_index, max_triangles_found);
	}
	largest_leaf_index = all_leaf_info.size() - 1;
	std::sort(all_leaf_info.begin(), all_leaf_info.end());
	return all_leaf_info;
}