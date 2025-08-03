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

#include "Kd-treeConverter.h"
#include "Kd-treeConstructor.h"
#include "MyMathUtility.h"
//using namespace KDTConverter;
//using namespace KDTConstructor;

static const unsigned int modulo[] = { 0,1,2,0,1 };

// From macros to variables to be able to modify through the configuration file
float v_KD_TREE_TRAVL_COST = 1.0;
float v_KD_TREE_ISECT_COST = 1.5;
unsigned int v_KD_TREE_MAX_LEVEL = 100;
unsigned int v_KD_TREE_MIN_TRIANGLE = 4;
float v_KD_TREE_EMTPY_BONUS = 0.9;

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

void push_triangles_to_child(const int triangleSize, const TriangleList* pTriangleInfos,
	TriangleList* pLeftTriangles, TriangleList* pRightTriangles,
	const SplitCost& bestCost)
{
	int currLeftIndex = 0;
	int currRightIndex = 0;
	int axis = bestCost.axis;
	float splitPos = bestCost.splitPos;

	// 부모 노드의 모든 삼각형을 순회합니다.
	for (int i = 0; i < triangleSize; i++) {
		const BoundingBox& box = pTriangleInfos[i].AABB;
		const float min_val = box.min[axis];
		const float max_val = box.max[axis];

		// 1. 삼각형이 분할 평면의 왼쪽에 걸치거나 왼쪽에 있다면 왼쪽 자식에 추가합니다.
		if (min_val <= splitPos) {
			// 안전장치: 할당된 배열 크기를 절대 넘어가지 않도록 방지합니다.
			if (currLeftIndex < bestCost.n_left) {
				pLeftTriangles[currLeftIndex++] = pTriangleInfos[i];
			}
		}

		// 2. 삼각형이 분할 평면의 오른쪽에 걸치거나 오른쪽에 있다면 오른쪽 자식에 추가합니다.
		if (max_val >= splitPos) {
			// 안전장치: 할당된 배열 크기를 절대 넘어가지 않도록 방지합니다.
			if (currRightIndex < bestCost.n_right) {
				pRightTriangles[currRightIndex++] = pTriangleInfos[i];
			}
		}
	}
}

//void push_triangles_to_child(const unsigned n_bEdge, const BoundEdge *bEdge, 
//                             TriangleList *pLeftTriangles, TriangleList *pRightTriangles,
//                             const SplitCost &bestCost )
//{
//	int currLeftIndex = 0, currRightIndex = 0;
//
//	if (KD_TREE_PLANAR_TRIANGLE_ADD_MODE == BOTH_SIDE) {
//		// NlogN (in RTGPU) 방식
//		for ( unsigned int i = 0; i < n_bEdge; ++i ) {
//			if( !bEdge[i].isPlanar ) {
//
//				if( bEdge[i].t < bestCost.splitPos && bEdge[i].type == BoundEdge::START )
//					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
//				else if(bEdge[i].t > bestCost.splitPos && bEdge[i].type == BoundEdge::END)
//					pRightTriangles[currRightIndex++] = *(bEdge[i].triangleInfo);
//
//			} else if(bEdge[i].type == BoundEdge::START) {
//
//				if(bEdge[i].t < bestCost.splitPos)
//					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
//				else if(bEdge[i].t > bestCost.splitPos)
//					pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
//				else { 
//					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
//					pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
//				}	
//
//			}
//		}
//	} else if (KD_TREE_PLANAR_TRIANGLE_ADD_MODE == MINCOST_SIDE) {
//		// 기존 SGRTx2 방식 (상락&혁 방법)
//		for ( unsigned int i = 0; i < n_bEdge; ++i ) {
//			if( !bEdge[i].isPlanar ) {
//
//				if( bEdge[i].t < bestCost.splitPos && bEdge[i].type == BoundEdge::START )
//					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
//				else if(bEdge[i].t > bestCost.splitPos && bEdge[i].type == BoundEdge::END)
//					pRightTriangles[currRightIndex++] = *(bEdge[i].triangleInfo);
//
//			} else if(bEdge[i].type == BoundEdge::START) {
//
//				if(bEdge[i].t < bestCost.splitPos)
//					pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
//				else if(bEdge[i].t > bestCost.splitPos)
//					pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
//				else { 
//					if(bestCost.planar_side == BoundEdge::START)
//						pLeftTriangles[ currLeftIndex++ ] = *(bEdge[i].triangleInfo);
//					else 
//						pRightTriangles[ currRightIndex++ ] = *(bEdge[i].triangleInfo);
//				}	
//
//			}
//		}
//	}
//}

void try_to_split(const int axis, BoundingBox &inBBox, const TriangleList *pTriangles, const int triangleSize,
                        BoundEdge *bEdge,  SplitCost &bestCost)
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


	for ( unsigned int i = 0; i < n_bEdge; i++ ) {
		// 현재 bEdge[i] 가 자르고자 하는 plane candidate
				BoundEdge curr_bEdge = bEdge[i];

		//planar는 open과 close에 둘 다 포함됨
		// (원래는 2개(min/max)가 planar 한개로 계산 됐으므로 min->open, max->close 로 각각 들어감.)
				open  += local_open  + num_planars;
				close += local_close + num_planars;
				local_open = 0;  local_close = 0; num_planars = 0; // num_planars 도 local 계산임
				num_normalPositive = 0;

		//현재 axis와 side에 대해 포지션구함
				const float cur_position = curr_bEdge.t;

				// ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ --
				// Split plane candidate 와 같은 위치의 edge 들에 대한 처리
				// ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ -- ~~ --
				{
					//똑같은게 여러개 있을 때는 제일 오른쪽에서만 SAH 계산을 한다.
					for ( unsigned int j = i; j < n_bEdge; j++ ) {
						BoundEdge tmp_bEdge = bEdge[j];
						if ( tmp_bEdge.t != cur_position) break;

						const bool
							is_left		= (tmp_bEdge.type == BoundEdge::START),
							is_planar	= tmp_bEdge.isPlanar,
							is_normalPositive = tmp_bEdge.isNormalPositive;

						//카운팅
						if (!is_planar) {
							local_open	+= is_left ? 1 : 0; //!< box 의 왼쪽은 local_open 을 증가
							local_close	+= is_left ? 0 : 1; //!< box 의 오른쪽은 local_close 를 증가
						}
						else {
							//플라나하다면 따로 카운팅
							num_planars += is_left ? 1 : 0;	// only count it once
							num_normalPositive += is_normalPositive ? 1 : 0;
						}

						curr_bEdge = tmp_bEdge;
						i=j;
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
				prob_l = (extent_l*area_mul + area_add)*cell_area_rcp,
				prob_r = (extent_r*area_mul + area_add)*cell_area_rcp;

			const int
				n_leftOnly		= close + local_close,  // close 가 된다면 그 삼각형은 오른쪽에 있지도 않게 됨 (겹치지도 않음)
				n_cross			= open  - n_leftOnly,   // open = n_leftOnly + n_cross (현재 local_open 은 포함하지 않음)
				n_rightOnly		= triangleSize - (n_leftOnly + n_cross + num_planars);

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
							const int tri_num_left  = n_leftOnly +n_cross+num_planars;
							const int tri_num_right = n_rightOnly+n_cross+num_planars;
							const float	emptyBonus  
								= (tri_num_left == 0 || tri_num_right == 0) ? v_KD_TREE_EMTPY_BONUS : 1.0f;
				
				ExpectedCost = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST*( 
					double( tri_num_left )		* prob_l +
					double( tri_num_right )	* prob_r ) * emptyBonus;

				planar_side = 2;
				nTri_left  = tri_num_left;
				nTri_right = tri_num_right;

			// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			// Cost function 의 최소값에 따라
			// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
			} else {

							// 배열[0] 은 planar 가 왼쪽에 들어갔을 경우
							// 배열[1] 은 planar 가 오른쪽에 들어갔을 경우
					
							//최종적인 양쪽 갯수.
							const int tri_num_left[2]	= { n_leftOnly+n_cross+num_planars, n_leftOnly+n_cross };
							const int tri_num_right[2]	= { n_rightOnly+n_cross, n_rightOnly+n_cross+num_planars };

							const float	emptyBonus[2] = { 
								(tri_num_left[0] == 0 || tri_num_right[0] == 0) ? v_KD_TREE_EMTPY_BONUS : 1.0f, 
								(tri_num_left[1] == 0 || tri_num_right[1] == 0) ? v_KD_TREE_EMTPY_BONUS : 1.0f };

							double SAH[2];
							for( int i = 0; i < 2; i++ ) {
								SAH[i] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST*( 
									double( tri_num_left[i] )	* prob_l +
									double( tri_num_right[i] )	* prob_r ) * emptyBonus[i];
							}

				if( SAH[0] <= SAH[1] ) { // planar 를 왼쪽에 넣는 것이 낫다면,
					ExpectedCost	= SAH[0];
					planar_side		= BoundEdge::START;
					nTri_left		= tri_num_left[0];
					nTri_right		= tri_num_right[0];
				} else {
					ExpectedCost	= SAH[1];
					planar_side		= BoundEdge::END;
					nTri_left		= tri_num_left[1];
					nTri_right		= tri_num_right[1];
				}
			}

			if( ExpectedCost < bestCost.cost ) {
				bestCost.cost			= ExpectedCost;
				bestCost.splitPos		= cur_position;
				bestCost.axis			= axis;

				bestCost.n_onlyLeft		= n_leftOnly;
				bestCost.n_onlyRight	= n_rightOnly;
				bestCost.n_cross		= n_cross;
				bestCost.n_planar		= num_planars;
				bestCost.n_left			= nTri_left;
				bestCost.n_right		= nTri_right;

				bestCost.planar_side	= planar_side;
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
				g_pTriangleInfos[i].AABB.min[0] = MyMIN (MyMIN (g_pTriangleInfos[i].point[0].vertex[0], g_pTriangleInfos[i].point[1].vertex[0]), g_pTriangleInfos[i].point[2].vertex[0]);
				g_pTriangleInfos[i].AABB.min[1] = MyMIN (MyMIN (g_pTriangleInfos[i].point[0].vertex[1], g_pTriangleInfos[i].point[1].vertex[1]), g_pTriangleInfos[i].point[2].vertex[1]);
				g_pTriangleInfos[i].AABB.min[2] = MyMIN (MyMIN (g_pTriangleInfos[i].point[0].vertex[2], g_pTriangleInfos[i].point[1].vertex[2]), g_pTriangleInfos[i].point[2].vertex[2]);
				g_pTriangleInfos[i].AABB.max[0] = MyMAX (MyMAX (g_pTriangleInfos[i].point[0].vertex[0], g_pTriangleInfos[i].point[1].vertex[0]), g_pTriangleInfos[i].point[2].vertex[0]);
				g_pTriangleInfos[i].AABB.max[1] = MyMAX (MyMAX (g_pTriangleInfos[i].point[0].vertex[1], g_pTriangleInfos[i].point[1].vertex[1]), g_pTriangleInfos[i].point[2].vertex[1]);
				g_pTriangleInfos[i].AABB.max[2] = MyMAX (MyMAX (g_pTriangleInfos[i].point[0].vertex[2], g_pTriangleInfos[i].point[1].vertex[2]), g_pTriangleInfos[i].point[2].vertex[2]);
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

void build_kd_tree_recursive(BoundEdge *bEdge, const TriangleList *pTriangleInfos, unsigned int triangleSize,
			                 BoundingBox &bbox, unsigned int inNodeLevel, KdTreeNode *inNode)
{
	SplitCost bestCost;
	g_iKdTree_Level = MyMAX( inNodeLevel, g_iKdTree_Level );


	if (DEBUG_FLAG) {
		fprintf(stdout, "b_k_t_r: (S) triangleSize = %d, inNodeLevel = %d\n", triangleSize, inNodeLevel);
	}
	
	// Calculate cost function (in case of no partition)
	bestCost.cost = double( triangleSize ) * v_KD_TREE_ISECT_COST;

	// Calculate cost function (in case of trying to partition)
	if( inNodeLevel < v_KD_TREE_MAX_LEVEL && triangleSize > v_KD_TREE_MIN_TRIANGLE ){
		// (모든 축에 대해 수행)
		for( int axis = 0; axis < 3; axis++ ){
			try_to_split( axis, bbox, pTriangleInfos, triangleSize, bEdge, bestCost );
		}
	}

	// ----------------------------------------------------------------------------
	// Leaf node 생성
	// ----------------------------------------------------------------------------
	if( !bestCost.is_valid() ) {
		unsigned int iTriOffset;

		{
			iTriOffset = g_iKdTree_TriOffset_Count;
			setLeafNode( inNode, triangleSize, g_iKdTree_TriOffset_Count );
			g_iKdTree_TriOffset_Count += triangleSize;

			// 메모리 체크 : Triangle Offset Size
			if( g_iKdTree_TriOffset_Count >= g_iKdTree_TriOffset_CountAlloc ) {
				_reAllocTriangleOffsetList(MyMAX( 2 * g_iKdTree_TriOffset_CountAlloc, 512 ), g_iKdTree_TriOffset_CountAlloc, &g_pKdTree_TriOffset_Array);
			}
		}
	
		/**
		 *	leaf node가 참조하는 triangle의 offset 을 offsetList 마지막에 추가해 넣는다.
		 */
		unsigned *currOffsetList = &g_pKdTree_TriOffset_Array[ iTriOffset ];

		unsigned leafCount = 0;
		for( unsigned i = 0; i < triangleSize; i++) {
			currOffsetList[ leafCount++ ] = pTriangleInfos[ i ].offset;
		}

		if( triangleSize == 0 )
			g_iKdTree_EmptyNode_Count++;

		g_iKdTree_LeafNode_Count++;
		g_iKdTree_MaxTriInLeafNode_Count = MyMAX( g_iKdTree_MaxTriInLeafNode_Count, triangleSize );

		/** 
		 *	더이상 pTriangleInfos 는 필요없으므로 메모리 공간 절약을 위해 없앤다.
		 */
		//delete[] pTriangleInfos;

	} else
	// ----------------------------------------------------------------------------
	// Inner node 생성
	// ----------------------------------------------------------------------------
	{
		unsigned int nodeNum;
		{
			nodeNum = g_iKdTree_Node_Count;
			setInnerNode( inNode, bestCost.axis, g_iKdTree_Node_Count, bestCost.splitPos );

			g_iKdTree_Node_Count += 2;

			// 메모리 체크 : Node Size
			if( g_iKdTree_Node_Count >= g_iKdTree_Node_CountAlloc ) {
				_reAllocKdtreeNodes(MyMAX( 2 * g_iKdTree_Node_CountAlloc, 512 ), g_iKdTree_Node_CountAlloc, &g_pKdTree_Node_Array);
			}
		}

		BoundingBox leftnBounds, rightnBounds;
		leftnBounds  = bbox;  leftnBounds.max[ bestCost.axis ]  = bestCost.splitPos;
		rightnBounds = bbox;  rightnBounds.min[ bestCost.axis ] = bestCost.splitPos;
		//임시 삭제
		/*
		  TriangleList *pLeftTriangles  = new TriangleList[ bestCost.n_left ];
		  TriangleList *pRightTriangles = new TriangleList[ bestCost.n_right ];
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

		const unsigned n_bEdge = 2 * triangleSize;*/

		// 1. bestCost의 부정확한 개수 대신, 실제 필요한 삼각형 개수를 정확히 다시 계산합니다.
		int n_left_actual = 0;
		int n_right_actual = 0;
		int axis = bestCost.axis;
		float splitPos = bestCost.splitPos;

		for (int i = 0; i < triangleSize; i++) {
			const BoundingBox& box = pTriangleInfos[i].AABB;
			if (box.min[axis] <= splitPos) {
				n_left_actual++;
			}
			if (box.max[axis] >= splitPos) {
				n_right_actual++;
			}
		}

		// 2. 새로 계산한 정확한 개수만큼 메모리를 할당합니다.
		TriangleList* pLeftTriangles = new TriangleList[n_left_actual];
		TriangleList* pRightTriangles = new TriangleList[n_right_actual];

		// 3. push_triangles_to_child 함수가 올바른 개수를 참조하도록 bestCost 값을 갱신합니다.
		bestCost.n_left = n_left_actual;
		bestCost.n_right = n_right_actual;

		// TriangleInfo 부터 bEdge 를 생성 및 정렬
		//set_bound_edge( bestCost.axis, pTriangleInfos, n_bEdge, bEdge );

		// bEdge 로 부터 pLeftTriangle, pRightTriangle 을 생성
		//push_triangles_to_child( n_bEdge, bEdge, pLeftTriangles, pRightTriangles, bestCost );
		push_triangles_to_child(triangleSize, pTriangleInfos, pLeftTriangles, pRightTriangles, bestCost);

		// 각 child node 에 맞게 삼각형 clipping
		//clip_triangle( bestCost.n_left,  bestCost, pLeftTriangles,  0 );
		//clip_triangle( bestCost.n_right, bestCost, pRightTriangles, 1 );

		/** 
		 *	더이상 pTriangleInfos 는 필요없으므로 메모리 공간 절약을 위해 없앤다.
		 *	반드시 pushChildTriangles 를 수행한 이후에 없애야 한다.
		 */
		//delete[] pTriangleInfos;
		/**
		 *	Left, Right 재귀 탐색. 
		 *	pLeftTriangles, pRightTriangles 는 build_kd_tree_recursive 함수 안에서 사용하고 바로 없앤다.
		 */
		build_kd_tree_recursive( bEdge, pLeftTriangles,  bestCost.n_left,  leftnBounds,  inNodeLevel + 1, &g_pKdTree_Node_Array[ nodeNum ] );
		build_kd_tree_recursive( bEdge, pRightTriangles, bestCost.n_right, rightnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[ nodeNum + 1 ] );

		// ADD THE FOLLOWING TWO LINES TO FIX THE MEMORY LEAK
		delete[] pLeftTriangles;
		delete[] pRightTriangles;
	}

	if (DEBUG_FLAG) {
		fprintf(stdout, "b_k_t_r: (E)triangleSize = %d, inNodeLevel = %d\n", triangleSize, inNodeLevel);
	}
}

void build_TriAccList(CompositeObject *poly_model, TriAccel*& pTriAcc)
{
	int iTriangleSize  = poly_model->n_triangles;
	printf("build_TriAccList triangle:%d\n", iTriangleSize);

	//pTriAcc = (TriAccel*) _aligned_malloc(iTriangleSize * sizeof(TriAccel), 16);

	// pTriAcc가 NULL인지 확인 (메모리 할당 실패 여부 검사)
	if (pTriAcc == NULL) {
		fprintf(stderr, "ERROR: Failed to allocate memory for TriAccel list! (size: %d)\n", iTriangleSize);
		return; // 함수를 안전하게 종료
	}

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
	printf("build_TriAccList triacc:%d\n", sizeof(pTriAcc)/sizeof(*pTriAcc));
}