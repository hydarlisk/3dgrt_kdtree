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

#include "AABB_Triangle_Clip/AABB_Triangle_Clip.h" 
//#include "Kd-treeCudaKernels.h"
#include <algorithm> // for std::remove_if
#include <array>       // std::array 사용을 위해 추가

#include <cmath>

using namespace std;

static const unsigned int modulo[] = { 0,1,2,0,1 };

// From macros to variables to be able to modify through the configuration file
float v_KD_TREE_TRAVL_COST = TRAVL_COST;
float v_KD_TREE_ISECT_COST = ISCET_COST;
unsigned int v_KD_TREE_MAX_LEVEL = MAX_LEVEL;
float v_KD_TREE_EMTPY_BONUS = EMTPY_BONUS;

BoundingBox   g_root_AABB;
BoundEdge    *g_bEdge = NULL;

#if PRIMITIVE_TYPE == TRI
unsigned int v_KD_TREE_MIN_PRIMITIVE = MIN_TRI;
TriangleList *g_pTriangleInfos = NULL;
unsigned int  g_iTriangleSize;
unsigned long long  g_iKdTree_TriOffset_Count;
unsigned long long  g_iKdTree_TriOffset_CountAlloc;
unsigned int *g_pKdTree_TriOffset_Array = NULL;
unsigned int  g_iKdTree_MaxTriInLeafNode_Count;
#else PRIMITIVE_TYPE == ELLIPSOID
unsigned int v_KD_TREE_MIN_PRIMITIVE = MIN_ELLIPSOID;
TriangleList* g_pEllipsoidInfos = nullptr;
unsigned int g_iEllipsoidSize;
unsigned long long g_iKdTreeEllipsoidOffsetCnt;
unsigned long long g_iKdTreeEllipsoidOffsetCnt_Alloc;
unsigned int* g_pKdTreeEllipsoidOffsetArray = nullptr;
unsigned int g_iKdTreeMaxEllipsoidInLeafNodeCnt;
#endif

unsigned int  g_iKdTree_Level;
unsigned int  g_iKdTree_Node_Count;
unsigned int  g_iKdTree_Node_CountAlloc;
KdTreeNode   *g_pKdTree_Node_Array = NULL;
unsigned int  g_iKdTree_EmptyNode_Count;
unsigned int  g_iKdTree_LeafNode_Count;

extern std::vector<Gaussian> g_gaussians;

#include <iostream>
#include <fstream>

#if BSPT
#include <vector>

vector<BSPNode> g_BSPTNodes;
vector<TriangleList> g_BSPTTris;

/* binary space partitioning tree */
struct Vec3 {
	float x, y, z;
};

inline Vec3 GetPos(const ExtendedVertex& v) {
	return { v.vertex[0], v.vertex[1], v.vertex[2] };
}

inline Vec3 Cross(Vec3 a, Vec3 b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline float Dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 Normalize(Vec3& a) {
	float tmp = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
	a.x /= tmp;
	a.y /= tmp;
	a.z /= tmp;
	return a;
}

// 노드 내 삼각형 정보 출력 함수
void PrintNodeTriangles(BSPNode_Build* node) {
	if (node->onPlaneTriangles.empty()) return;

	std::cout << "--- Node Plane (n: [" << node->n[0] << ", " << node->n[1] << ", " << node->n[2]
		<< "], d: " << node->d << ") ---" << std::endl;

	for (size_t i = 0; i < node->onPlaneTriangles.size(); ++i) {
		const TriangleList& tri = node->onPlaneTriangles[i];
		std::cout << "  Triangle " << i << " [Offset: " << tri.offset << "]" << std::endl;
		for (int j = 0; j < 3; ++j) {
			std::cout << "    V" << j << ": ("
				<< tri.point[j].vertex[0] << ", "
				<< tri.point[j].vertex[1] << ", "
				<< tri.point[j].vertex[2] << ")" << std::endl;
		}
	}
}

// 카메라 위치(eye)에 따른 BSP 트리 순회
void TraverseBSPTree(BSPNode_Build* node, const float eye[3]) {
	if (node == nullptr) return;

	// 1. 현재 노드의 평면과 카메라 사이의 거리 계산 (Plane Equation: Ax + By + Cz - D = 0)
	// dist > 0 이면 카메라가 평면의 앞(Front)에 있음
	float dist = node->n[0] * eye[0] +
		node->n[1] * eye[1] +
		node->n[2] * eye[2] - node->d;

	// 2. 방문 순서 결정 (Back-to-Front)
	if (dist > 0) {
		// 카메라가 앞쪽에 있음: 뒤쪽 자식 -> 현재 평면 -> 앞쪽 자식 순서
		TraverseBSPTree(node->back, eye);
		PrintNodeTriangles(node);
		TraverseBSPTree(node->front, eye);
	}
	else {
		// 카메라가 뒤쪽에 있음: 앞쪽 자식 -> 현재 평면 -> 뒤쪽 자식 순서
		TraverseBSPTree(node->front, eye);
		PrintNodeTriangles(node);
		TraverseBSPTree(node->back, eye);
	}
}

//point-plane signed distance(not normalized)
inline float CalculateDistanceToPlane(const Vec3 n, const float d, const ExtendedVertex& V) {
	return Dot(n, GetPos(V)) - d;
}

Side ClassifyTriangle(const Vec3 n, const float d, const TriangleList& testTri) {
	int frontCnt = 0, backCnt = 0;

	for (int i = 0; i < 3; i++) {
		float dist = CalculateDistanceToPlane(n, d, testTri.point[i]);
		if (dist > EPSILON) frontCnt++;
		else if (dist < -EPSILON) backCnt++;
	}
	if (frontCnt > 0 && backCnt > 0) return STRADDLE;
	if (frontCnt > 0) return FRONT;
	if (backCnt > 0) return BACK;
	return ON_PLANE;
}


//TODO: valid check of n?
void getPlaneFromTri(const TriangleList& tri, Vec3& n, float& d) {
	//get plane
	Vec3 p0 = GetPos(tri.point[0]);
	Vec3 e1 = { tri.point[1].vertex[0] - p0.x, tri.point[1].vertex[1] - p0.y, tri.point[1].vertex[2] - p0.z };
	Vec3 e2 = { tri.point[2].vertex[0] - p0.x, tri.point[2].vertex[1] - p0.y, tri.point[2].vertex[2] - p0.z };
	n = Cross(e1, e2);
	//if (Dot(n, n) < 0.000001f) {
	//	d = FLT_MAX;
	//	return;
	//}
	d = Dot(n, p0);
}

int PickBestSplitter(const vector<TriangleList>& triangles) {
	int bestIndex = -1;
	float bestScore = 1e30f;

	const float SPLIT_WEIGHT = 100.0f;
	const float BALANCE_WEIGHT = 1.0f;

	for (int i = 0; i < triangles.size(); ++i) {
		const TriangleList& candidate = triangles[i];

		Vec3 n;
		float d;
		getPlaneFromTri(candidate, n, d);

		int frontCnt = 0, backCnt = 0, splitCnt = 0;

		for (int j = 0; j < triangles.size(); ++j) {
			if (i == j) continue;

			Side side = ClassifyTriangle(n, d, triangles[j]);
			//Side side = ClassifyTriangleWithAABB(n, d, triangles[j]);
			switch (side) {
			case FRONT:    frontCnt++; break;
			case BACK:     backCnt++;  break;
			case STRADDLE: splitCnt++; break;
			case ON_PLANE: break;
			}
		}

		// 점수 계산 (점수가 낮을수록 좋음)
		float score = (SPLIT_WEIGHT * splitCnt) + (BALANCE_WEIGHT * abs(frontCnt - backCnt));

		if (score < bestScore) {
			bestScore = score;
			bestIndex = i;
		}

		// 만약 쪼개지는 삼각형이 하나도 없는 완벽한 평면을 찾았다면 즉시 반환
		if (splitCnt == 0 && abs(frontCnt - backCnt) < (triangles.size() / 4)) {
			return i;
		}
	}

	return (bestIndex == -1) ? 0 : bestIndex;
}

//get intersection point of plane and line
ExtendedVertex GetIntersect(const ExtendedVertex& a, const ExtendedVertex& b, const Vec3 n, const float d) {
	Vec3 posA = GetPos(a);
	Vec3 posB = GetPos(b);

	float distA = Dot(n, posA) - d;
	float distB = Dot(n, posB) - d;

	float t = distA / (distA - distB);

	ExtendedVertex res = a;
	res.vertex[0] = a.vertex[0] + t * (b.vertex[0] - a.vertex[0]);
	res.vertex[1] = a.vertex[1] + t * (b.vertex[1] - a.vertex[1]);
	res.vertex[2] = a.vertex[2] + t * (b.vertex[2] - a.vertex[2]);
	return res;
}

void SplitTriangle(const TriangleList& tri, const Vec3 n, const float d, std::vector<TriangleList>& frontPart, std::vector<TriangleList>& backPart) {
	std::vector<ExtendedVertex> fPts, bPts;

	for (int i = 0; i < 3; ++i) {
		int next = (i + 1) % 3;
		const ExtendedVertex& A = tri.point[i];
		const ExtendedVertex& B = tri.point[next];

		float distA = Dot(n, GetPos(A)) - d;
		float distB = Dot(n, GetPos(B)) - d;

		if (distA >= -EPSILON) fPts.push_back(A);
		if (distA <= EPSILON) bPts.push_back(A);

		if ((distA > EPSILON && distB < -EPSILON) || (distA < -EPSILON && distB > EPSILON)) {
			ExtendedVertex intersect = GetIntersect(A, B, n, d);
			fPts.push_back(intersect);
			bPts.push_back(intersect);
		}
	}

	auto Finalize = [&](std::vector<ExtendedVertex>& pts, std::vector<TriangleList>& target) {
		if (pts.size() < 3) return;
		for (size_t i = 1; i < pts.size() - 1; ++i) {
			TriangleList nt = tri;
			nt.point[0] = pts[0];
			nt.point[1] = pts[i];
			nt.point[2] = pts[i + 1];
			nt.offset = tri.point[0].material_ID;
			target.push_back(nt);
		}
	};

	Finalize(fPts, frontPart);
	Finalize(bPts, backPart);
}

//side:
//		1: front
//		-1: back
BoundingBox GetClippedAABB(const TriangleList& tri, const Vec3 n, const float d, int side) {
	BoundingBox clippedBox;
	for (int i = 0; i < 3; i++) {
		clippedBox.min[i] = 1e30f;
		clippedBox.max[i] = -1e30f;
	}

	auto UpdateBox = [&](const ExtendedVertex& v) {
		Vec3 p = GetPos(v);
		if (p.x < clippedBox.min[0]) clippedBox.min[0] = p.x;
		if (p.y < clippedBox.min[1]) clippedBox.min[1] = p.y;
		if (p.z < clippedBox.min[2]) clippedBox.min[2] = p.z;
		if (p.x > clippedBox.max[0]) clippedBox.max[0] = p.x;
		if (p.y > clippedBox.max[1]) clippedBox.max[1] = p.y;
		if (p.z > clippedBox.max[2]) clippedBox.max[2] = p.z;
	};

	for (int i = 0; i < 3; ++i) {
		int next = (i + 1) % 3;
		const ExtendedVertex& A = tri.point[i];
		const ExtendedVertex& B = tri.point[next];

		float distA = Dot(n, GetPos(A)) - d;
		float distB = Dot(n, GetPos(B)) - d;

		// 1. 해당 영역(side)에 속한 정점 추가
		if (side == 1) { // Front
			if (distA >= -EPSILON) UpdateBox(A);
		}
		else { // Back
			if (distA <= EPSILON) UpdateBox(A);
		}

		// 2. 평면과 교차하는 지점(Intersect) 추가
		if ((distA > EPSILON && distB < -EPSILON) || (distA < -EPSILON && distB > EPSILON)) {
			ExtendedVertex intersect = GetIntersect(A, B, n, d);
			UpdateBox(intersect);
		}
	}
	BoundingBox result;
	result.min[0] = MyMAX(tri.AABB.min[0], clippedBox.min[0]);
	result.min[1] = MyMAX(tri.AABB.min[1], clippedBox.min[1]);
	result.min[2] = MyMAX(tri.AABB.min[2], clippedBox.min[2]);

	// max는 더 작은 값으로 (조여짐)
	result.max[0] = MyMIN(tri.AABB.max[0], clippedBox.max[0]);
	result.max[1] = MyMIN(tri.AABB.max[1], clippedBox.max[1]);
	result.max[2] = MyMIN(tri.AABB.max[2], clippedBox.max[2]);

	return result;
}

int maxTriInNode = 0;
int maxTriDepth = 0;
int maxDepth = 0;

BSPNode_Build* BuildBSPTree(vector<TriangleList>& triangles, int depth) {
	if (triangles.empty()) return nullptr;
	if (depth > maxDepth) {
		maxDepth = depth;
		printf("maxDepth while building: %d\n", maxDepth);
	}

	BSPNode_Build* node = new BSPNode_Build();

	int bestIdx = PickBestSplitter(triangles);
	TriangleList splitter = triangles[bestIdx];
	//TriangleList splitter = triangles[0];
	
	//get plane
	Vec3 n;
	float d;
	getPlaneFromTri(splitter, n, d);

	node->n[0] = n.x;
	node->n[1] = n.y;
	node->n[2] = n.z;
	node->d = d;

	vector<TriangleList> frontList;
	vector<TriangleList> backList;

	vector<Side> sideList(triangles.size());
	for (int i = 0; i < triangles.size(); i++) {
		Side side;
		if (i == bestIdx) {
			side = ON_PLANE;
		}
		else {
			side = ClassifyTriangle(n, d, triangles[i]);
			//side = ClassifyTriangleWithAABB(n, d, triangles[i]);
		}
		sideList[i] = side;
		if (side == FRONT) {
			frontList.push_back(triangles[i]);
		}
		else if (side == BACK) {
			backList.push_back(triangles[i]);
		}
		else if (side == STRADDLE) {
#if BSPT_NO_SPLIT
			frontList.push_back(triangles[i]);
			backList.push_back(triangles[i]);
			frontList[frontList.size() - 1].AABB = GetClippedAABB(triangles[i], n, d, 1);
			backList[backList.size() - 1].AABB = GetClippedAABB(triangles[i], n, d, -1);
#else
			vector<TriangleList> frontPart, backPart;
			SplitTriangle(triangles[i], n, d, frontPart, backPart);
			frontList.insert(frontList.end(), frontPart.begin(), frontPart.end());
			backList.insert(backList.end(), backPart.begin(), backPart.end());
#endif			
		}
		else if (side == ON_PLANE) {
			node->onPlaneTriangles.push_back(triangles[i]);
		}
	}

	if (node->onPlaneTriangles.size() > maxTriInNode) {
		maxTriInNode = node->onPlaneTriangles.size();
		maxTriDepth = depth;
	}

	if (!frontList.empty()) {
		node->front = BuildBSPTree(frontList, depth+1);
	}
	if (!backList.empty()) {
		node->back = BuildBSPTree(backList, depth+1);
	}
	return node;
}

//dfs traverse
#if BSPT_DUMP_STATISTICS
vector<vector<uint32_t>> onPlaneTriangleCnts;
int onPlaneTriCntsIdx = 0;
#endif
void FlattenBSPTree(BSPNode_Build* nodeB, vector<unsigned int>& triOffsets) {
	int currIdx = g_BSPTNodes.size();
	BSPNode tmp;
	tmp.n[0] = nodeB->n[0];
	tmp.n[1] = nodeB->n[1];
	tmp.n[2] = nodeB->n[2];
	tmp.d = nodeB->d;
	tmp.frontChild = -1;
	tmp.backChild = -1;
	tmp.triStart = g_iKdTree_TriOffset_Count + triOffsets.size();
	tmp.triCnt = nodeB->onPlaneTriangles.size();
#if BSPT_DUMP_STATISTICS
	onPlaneTriangleCnts[onPlaneTriCntsIdx].push_back(tmp.triCnt);
#endif

	for (int i = 0; i < tmp.triCnt; i++) {
		triOffsets.push_back(g_BSPTTris.size());
		g_BSPTTris.push_back(nodeB->onPlaneTriangles[i]);
	}
	g_BSPTNodes.push_back(tmp);

	if (nodeB->front != nullptr) {
		g_BSPTNodes[currIdx].frontChild = g_BSPTNodes.size();
		FlattenBSPTree(nodeB->front, triOffsets);
	}
	if (nodeB->back != nullptr) {
		g_BSPTNodes[currIdx].backChild = g_BSPTNodes.size();
		FlattenBSPTree(nodeB->back, triOffsets);
	}
}
#endif

void _reAllocTriangleOffsetList(unsigned long long _newAllocSize, unsigned long long& _oldAllocSize, unsigned int** _ppTriOffsetArray)
{
	//for debug
	double requested_MB = (double)_newAllocSize * sizeof(unsigned int) / (1024.0 * 1024.0);
	std::cerr << "    Attempted re-alloc. Current Count: " << _oldAllocSize
		<< ", New Count: " << _newAllocSize << " (" << requested_MB << " MB)" << std::endl;

	unsigned int* tmpList = new unsigned int[_newAllocSize];
	memcpy(tmpList, *_ppTriOffsetArray, sizeof(unsigned int) * _oldAllocSize);
	delete[] * _ppTriOffsetArray;
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

#if BSPT
void setLeafNodeBSPT(KdTreeNode* pNode, unsigned int _bsptOffset) {
	pNode->x = 3;
	pNode->y = _bsptOffset;
}
#endif

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
		bEdge[i].type = BoundEdge::START;
		bEdge[i].triangleInfo = &pTriangleInfo[index];
		bEdge[i].t = worldbound.min[axis];
		bEdge[i].isPlanar = ( worldbound.min[axis] == worldbound.max[axis] );
		i++;
		bEdge[i].type = BoundEdge::END;
		bEdge[i].triangleInfo = &pTriangleInfo[index];
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


//quaternion -> rot mat
//local to world
void quaternionWXYZToMatrixTransform(const float* q, float(&R)[3][3]) {
	float r = q[0];
	float x = q[1];
	float y = q[2];
	float z = q[3];

	R[0][0] = 1.0f - 2.0f * (y * y + z * z);
	R[0][1] = 2.0f * (x * y - r * z);
	R[0][2] = 2.0f * (x * z + r * y);

	R[1][0] = 2.0f * (x * y + r * z);
	R[1][1] = 1.0f - 2.0f * (x * x + z * z);
	R[1][2] = 2.0f * (y * z - r * x);

	R[2][0] = 2.0f * (x * z - r * y);
	R[2][1] = 2.0f * (y * z + r * x);
	R[2][2] = 1.0f - 2.0f * (x * x + y * y);
}
//world to local
void quaternionWXYZToMatrix(const float* q, float(&R)[3][3]) {
	float r = q[0];
	float x = q[1];
	float y = q[2];
	float z = q[3];

	R[0][0] = 1.0f - 2.0f * (y * y + z * z);
	R[1][0] = 2.0f * (x * y - r * z);
	R[2][0] = 2.0f * (x * z + r * y);

	R[0][1] = 2.0f * (x * y + r * z);
	R[1][1] = 1.0f - 2.0f * (x * x + z * z);
	R[2][1] = 2.0f * (y * z - r * x);

	R[0][2] = 2.0f * (x * z - r * y);
	R[1][2] = 2.0f * (y * z + r * x);
	R[2][2] = 1.0f - 2.0f * (x * x + y * y);
}

void clip_ellipsoid(const int ellipsoidSize, const SplitCost& bestCost, TriangleList* pEllipsoidInfos, int side) {
	int axis = bestCost.axis;
	float splitPos = bestCost.splitPos;

	for (int gi = 0; gi < ellipsoidSize; gi++) {
		//simple clipping
		BoundingBox& currBBox = pEllipsoidInfos[gi].AABB;
		if (currBBox.min[axis] < splitPos && currBBox.max[axis] > splitPos) {
			if (side == 0) {
				if (currBBox.max[axis] > splitPos) {
					currBBox.max[axis] = splitPos;
				}
			}
			else {
				if (currBBox.min[axis] < splitPos) {
					currBBox.min[axis] = splitPos;
				}
			}
		}

		//tight clipping
		//1. calc covariance
		auto& g = g_gaussians[pEllipsoidInfos[gi].offset];
		float R[3][3];
		quaternionWXYZToMatrixTransform(g.rot, R);
		float s0 = g.scale[0];
		float s1 = g.scale[1];
		float s2 = g.scale[2];

		float M[3][3];
		for (int i = 0; i < 3; i++) {
			M[i][0] = R[i][0] * s0;
			M[i][1] = R[i][1] * s1;
			M[i][2] = R[i][2] * s2;
		}

		float cov[3][3];
		cov[0][0] = M[0][0] * M[0][0] + M[0][1] * M[0][1] + M[0][2] * M[0][2];
		cov[1][1] = M[1][0] * M[1][0] + M[1][1] * M[1][1] + M[1][2] * M[1][2];
		cov[2][2] = M[2][0] * M[2][0] + M[2][1] * M[2][1] + M[2][2] * M[2][2];
		cov[0][1] = M[0][0] * M[1][0] + M[0][1] * M[1][1] + M[0][2] * M[1][2];
		cov[0][2] = M[0][0] * M[2][0] + M[0][1] * M[2][1] + M[0][2] * M[2][2];
		cov[1][2] = M[1][0] * M[2][0] + M[1][1] * M[2][1] + M[1][2] * M[2][2];
		cov[1][0] = cov[0][1];
		cov[2][0] = cov[0][2];
		cov[2][1] = cov[1][2];

		//2. schur complement
		float e[3];
		e[0] = max(EPSILON, sqrt(cov[0][0]));
		e[1] = max(EPSILON, sqrt(cov[1][1]));
		e[2] = max(EPSILON, sqrt(cov[2][2]));

		int i = axis;
		int j = (axis + 1) % 3;
		int k = (axis + 2) % 3;

		//z축 극점
		float jMax = g.pos[i] + cov[i][j] / e[j];
		float jMin = g.pos[i] - cov[i][j] / e[j];
		float kMax = g.pos[i] + cov[i][k] / e[k];
		float kMin = g.pos[i] - cov[i][k] / e[k];

		float invSii = 1.0f / max(cov[i][i], 1e-8f);
		float di = splitPos - g.pos[i];

		float jMu = g.pos[j] + cov[i][j] * invSii * di;
		float kMu = g.pos[k] + cov[i][k] * invSii * di;

		float Sjj_cut = max(0.0f, cov[j][j] - (cov[i][j] * cov[i][j]) * invSii);
		float Skk_cut = max(0.0f, cov[k][k] - (cov[i][k] * cov[i][k]) * invSii);

		float eJ_cut = sqrt(Sjj_cut);
		float eK_cut = sqrt(Skk_cut);
		float sign = side ? 1.0f : -1.0f;

		bool condJMax = (jMax * sign >= splitPos * sign);
		bool condJMin = (jMin * sign >= splitPos * sign);
		currBBox.max[j] = condJMax ? (g.pos[j] + e[j]) : (jMu + eJ_cut);
		currBBox.min[j] = condJMin ? (g.pos[j] - e[j]) : (jMu - eJ_cut);

		bool condKMax = (kMax * sign >= splitPos * sign);
		bool condKMin = (kMin * sign >= splitPos * sign);
		currBBox.max[k] = condKMax ? (g.pos[k] + e[k]) : (kMu + eK_cut);
		currBBox.min[k] = condKMin ? (g.pos[k] - e[k]) : (kMu - eK_cut);
	}
}

void clip_triangle(const int triangleSize, const SplitCost& bestCost, TriangleList* pTriangleInfos, int side)
{
	int axis = bestCost.axis;
	float norm[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };

	float planeNorm[3];
	planeNorm[0] = norm[axis][0];
	planeNorm[1] = norm[axis][1];
	planeNorm[2] = norm[axis][2];

	float planePoint[3];
	planePoint[0] = planeNorm[0] * bestCost.splitPos;
	planePoint[1] = planeNorm[1] * bestCost.splitPos;
	planePoint[2] = planeNorm[2] * bestCost.splitPos;

	for (int i = 0; i < triangleSize; i++) {
		BoundingBox currBBox = pTriangleInfos[i].AABB;

		if ((currBBox.min[axis] < bestCost.splitPos) &&
			(currBBox.max[axis] > bestCost.splitPos)) {

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

			// [수정] 고정 크기 배열 대신 std::vector를 사용하여 버퍼 오버플로우 방지
			std::vector<std::array<float, 3>> leftVec;
			std::vector<std::array<float, 3>> rightVec;

			for (int nPoint = 0; nPoint < 3; nPoint++) {
				if (p[nPoint][axis] < bestCost.splitPos) {
					leftVec.push_back({ p[nPoint][0], p[nPoint][1], p[nPoint][2] });
				}
				else {
					rightVec.push_back({ p[nPoint][0], p[nPoint][1], p[nPoint][2] });
				}
			}

			const size_t leftSize = leftVec.size();
			const size_t rightSize = rightVec.size();
			for (size_t j = 0; j < leftSize; j++)
			{
				for (size_t k = 0; k < rightSize; k++)
				{
					float hitPoint[3];
					if (intersect_edge_plane(leftVec[j].data(), rightVec[k].data(), planePoint, planeNorm, hitPoint)) {
						leftVec.push_back({ hitPoint[0], hitPoint[1], hitPoint[2] });
						rightVec.push_back({ hitPoint[0], hitPoint[1], hitPoint[2] });
					}
				}
			}

			if (side == 0)
			{
				// smaller than splitpos
				BoundingBox reducedBBox = {}; // 0으로 초기화
				if (!leftVec.empty()) {
					reducedBBox.min[0] = reducedBBox.max[0] = leftVec[0][0];
					reducedBBox.min[1] = reducedBBox.max[1] = leftVec[0][1];
					reducedBBox.min[2] = reducedBBox.max[2] = leftVec[0][2];

					for (size_t j = 1; j < leftVec.size(); j++) {
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
				}
				pTriangleInfos[i].AABB = reducedBBox;
			}
			else
			{
				//bigger than splitpos
				BoundingBox reducedBBox = {}; // 0으로 초기화
				if (!rightVec.empty()) {
					reducedBBox.min[0] = reducedBBox.max[0] = rightVec[0][0];
					reducedBBox.min[1] = reducedBBox.max[1] = rightVec[0][1];
					reducedBBox.min[2] = reducedBBox.max[2] = rightVec[0][2];

					for (size_t j = 1; j < rightVec.size(); j++) {
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
				}
				pTriangleInfos[i].AABB = reducedBBox;
			}

#if SAH_OPACITY == 8 | SAH_OPACITY == 9
			pTriangleInfos[i].AABBarea = get_surface_area(pTriangleInfos[i].AABB);
#endif
#if SAH_OPACITY >=2 && CLIP_AREA
			std::vector<ExtendedVertex> clipped_polygon;
			KdTreeClipper::clip_triangle_against_AABB(pTriangleInfos[i].point, pTriangleInfos[i].AABB, clipped_polygon);
			pTriangleInfos[i].area = static_cast<float>(KdTreeClipper::calculate_polygon_area(clipped_polygon));
#endif
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
//shyun added begin
void push_triangles_to_child_vector(const unsigned n_bEdge, const BoundEdge* bEdge,
	std::vector<TriangleList>& pLeftTriangles, std::vector<TriangleList>& pRightTriangles,
	const SplitCost& bestCost)
{
	if (KD_TREE_PLANAR_TRIANGLE_ADD_MODE == BOTH_SIDE) {
		// NlogN (in RTGPU) 방식
		for (unsigned int i = 0; i < n_bEdge; ++i) {
			if (!bEdge[i].isPlanar) {

				if (bEdge[i].t < bestCost.splitPos && bEdge[i].type == BoundEdge::START)
					pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
				else if (bEdge[i].t > bestCost.splitPos && bEdge[i].type == BoundEdge::END)
					pRightTriangles.push_back(*(bEdge[i].triangleInfo));

			}
			else if (bEdge[i].type == BoundEdge::START) {

				if (bEdge[i].t < bestCost.splitPos)
					pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
				else if (bEdge[i].t > bestCost.splitPos)
					pRightTriangles.push_back(*(bEdge[i].triangleInfo));
				else {
					pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
					pRightTriangles.push_back(*(bEdge[i].triangleInfo));
				}

			}
		}
	}
	else if (KD_TREE_PLANAR_TRIANGLE_ADD_MODE == MINCOST_SIDE) {
		// 기존 SGRTx2 방식 (상락&혁 방법)
		for (unsigned int i = 0; i < n_bEdge; ++i) {
			if (!bEdge[i].isPlanar) {

				if (bEdge[i].t < bestCost.splitPos && bEdge[i].type == BoundEdge::START)
					pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
				else if (bEdge[i].t > bestCost.splitPos && bEdge[i].type == BoundEdge::END)
					pRightTriangles.push_back(*(bEdge[i].triangleInfo));

			}
			else if (bEdge[i].type == BoundEdge::START) {

				if (bEdge[i].t < bestCost.splitPos)
					pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
				else if (bEdge[i].t > bestCost.splitPos)
					pRightTriangles.push_back(*(bEdge[i].triangleInfo));
				else {
					if (bestCost.planar_side == BoundEdge::START)
						pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
					else
						pRightTriangles.push_back(*(bEdge[i].triangleInfo));
				}

			}
		}
	}
}
//shyun added end
void try_to_split(const int axis, BoundingBox &inBBox, const TriangleList *pTriangles, const int triangleSize, BoundEdge *bEdge,  SplitCost &bestCost
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

	for (unsigned int i = 0; i < n_bEdge; i++) {
		// 현재 bEdge[i] 가 자르고자 하는 plane candidate
		BoundEdge curr_bEdge = bEdge[i];

		//planar는 open과 close에 둘 다 포함됨
		// (원래는 2개(min/max)가 planar 한개로 계산 됐으므로 min->open, max->close 로 각각 들어감.)
		open += local_open + num_planars;
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
			for (unsigned int j = i; j < n_bEdge; j++) {
				BoundEdge tmp_bEdge = bEdge[j];
				if (tmp_bEdge.t != cur_position) break;

				const bool
					is_left = (tmp_bEdge.type == BoundEdge::START),
					is_planar = tmp_bEdge.isPlanar,
					is_normalPositive = tmp_bEdge.isNormalPositive;
				//카운팅
				if (!is_planar) {
					local_open += is_left ? 1 : 0; //!< box 의 왼쪽은 local_open 을 증가
					local_close += is_left ? 0 : 1; //!< box 의 오른쪽은 local_close 를 증가
				}
				else {
					//플라나하다면 따로 카운팅
					num_planars += is_left ? 1 : 0;	// only count it once
					num_normalPositive += is_normalPositive ? 1 : 0;

				}
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
						double(tri_num_left[side_idx])* prob_l +
						double(tri_num_right[side_idx]) * prob_r
						)* emptyBonus[side_idx];
				}
#if SAH_MAXIMIZE
				if (SAH[0] >= SAH[1]) {
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

#if PRIMITIVE_TYPE == ELLIPSOID
inline float calculateKernelScale(float density, float kernelMinResponse, uint32_t opts = 1, float kernelDegree = 4) {
	const float responseModulation = (opts & 1 /* MOGRenderAdaptiveKernelClamping */) ? density : 1.0f;
	const float minResponse = std::min(kernelMinResponse / responseModulation, 0.97f);

	const float b = kernelDegree;
	const float a = -4.5f / std::pow(3.0f, b);

	// 3. e^{a * r^b} = minResponse 를 만족하는 r(반지름) 계산
	// r = (ln(minResponse) / a)^(1/b)
	return std::pow(std::log(minResponse) / a, 1.0f / b);

}

//TODO: calc more accurate aabb
void calcEllipsoidAABB(Gaussian& g, BoundingBox& b) {
	float k_scale = calculateKernelScale(g.opacity, KERNEL_MIN_RESPONSE);

	float r = g.rot[0];
	float x = g.rot[1];
	float y = g.rot[2];
	float z = g.rot[3];

	float R[3][3];
	quaternionWXYZToMatrixTransform(g.rot, R);

	float s0 = g.scale[0];
	float s1 = g.scale[1];
	float s2 = g.scale[2];

	float sigma_xx = (R[0][0] * s0) * (R[0][0] * s0) + (R[0][1] * s1) * (R[0][1] * s1) + (R[0][2] * s2) * (R[0][2] * s2);
	float sigma_yy = (R[1][0] * s0) * (R[1][0] * s0) + (R[1][1] * s1) * (R[1][1] * s1) + (R[1][2] * s2) * (R[1][2] * s2);
	float sigma_zz = (R[2][0] * s0) * (R[2][0] * s0) + (R[2][1] * s1) * (R[2][1] * s1) + (R[2][2] * s2) * (R[2][2] * s2);

	float half_x = k_scale * std::sqrt(sigma_xx);
	float half_y = k_scale * std::sqrt(sigma_yy);
	float half_z = k_scale * std::sqrt(sigma_zz);

	b.min[0] = g.pos[0] - half_x;
	b.min[1] = g.pos[1] - half_y;
	b.min[2] = g.pos[2] - half_z;
	b.max[0] = g.pos[0] + half_x;
	b.max[1] = g.pos[1] + half_y;
	b.max[2] = g.pos[2] + half_z;
}

//calculate ellipsoid aabb and update scene aabb
void initEllipsoid(CompositeObject& poly_model) {
	poly_model.AABB[XMIN] = poly_model.AABB[YMIN] = poly_model.AABB[ZMIN] = FLT_MAX;
	poly_model.AABB[XMAX] = poly_model.AABB[YMAX] = poly_model.AABB[ZMAX] = -FLT_MAX;
	for (int i = 0; i < g_gaussians.size(); i++) {
		calcEllipsoidAABB(g_gaussians[i], g_pEllipsoidInfos[i].AABB);
		g_pEllipsoidInfos[i].offset = i;
		poly_model.AABB[XMIN] = MyMIN(poly_model.AABB[XMIN], g_pEllipsoidInfos[i].AABB.min[0]);
		poly_model.AABB[YMIN] = MyMIN(poly_model.AABB[YMIN], g_pEllipsoidInfos[i].AABB.min[1]);
		poly_model.AABB[ZMIN] = MyMIN(poly_model.AABB[ZMIN], g_pEllipsoidInfos[i].AABB.min[2]);
		poly_model.AABB[XMAX] = MyMAX(poly_model.AABB[XMAX], g_pEllipsoidInfos[i].AABB.max[0]);
		poly_model.AABB[YMAX] = MyMAX(poly_model.AABB[YMAX], g_pEllipsoidInfos[i].AABB.max[1]);
		poly_model.AABB[ZMAX] = MyMAX(poly_model.AABB[ZMAX], g_pEllipsoidInfos[i].AABB.max[2]);
	}
}

bool initialize_kd_tree(CompositeObject* poly_model) {
	// Returns 1 if kd-tree data was initialized successfully, or 0 otherwise.

	g_pEllipsoidInfos = new TriangleList[g_gaussians.size()];
	initEllipsoid(*poly_model);

	bool bError = false;

	g_iKdTree_Node_Count = 0;
	g_pKdTree_Node_Array = NULL;
	g_iKdTree_Node_CountAlloc = 32 * 1024 * 1024;
	g_iKdTree_Level = 0;
	g_iKdTree_LeafNode_Count = 0;
	g_iKdTree_EmptyNode_Count = 0;
	
	g_iKdTreeEllipsoidOffsetCnt = 0;
	g_pKdTreeEllipsoidOffsetArray = nullptr;
	g_iKdTreeEllipsoidOffsetCnt_Alloc = 64 * 1024 * 1024;
	g_iKdTreeMaxEllipsoidInLeafNodeCnt = 0;
	g_iEllipsoidSize = g_gaussians.size();

	g_root_AABB.min[0] = poly_model->AABB[0];
	g_root_AABB.min[1] = poly_model->AABB[2];
	g_root_AABB.min[2] = poly_model->AABB[4];
	g_root_AABB.max[0] = poly_model->AABB[1];
	g_root_AABB.max[1] = poly_model->AABB[3];
	g_root_AABB.max[2] = poly_model->AABB[5];

	g_bEdge = new BoundEdge[g_iEllipsoidSize * 2];
	if (g_bEdge == NULL) {
		bError |= true;
	}
	else {
		memset(g_bEdge, 0x00, sizeof(BoundEdge) * g_iEllipsoidSize * 2);
	}

	g_pKdTree_Node_Array = new KdTreeNode[g_iKdTree_Node_CountAlloc];
	if (g_pKdTree_Node_Array == NULL) {
		bError |= true;
	}
	else {
		memset(g_pKdTree_Node_Array, 0x00, sizeof(KdTreeNode) * g_iKdTree_Node_CountAlloc);
		g_iKdTree_Node_Count = 1;
	}

	g_pKdTreeEllipsoidOffsetArray = new unsigned int[g_iKdTreeEllipsoidOffsetCnt_Alloc];
	if (g_pKdTreeEllipsoidOffsetArray == NULL) {
		bError |= true;
	}
	else {
		memset(g_pKdTreeEllipsoidOffsetArray, 0x00, sizeof(unsigned int) * g_iKdTreeEllipsoidOffsetCnt_Alloc);
	}

	if (bError) {
		printf("init kd-tree bError");
		uninitialize_kd_tree();
		return 0;
	}
	return 1;
}
#elif PRIMITIVE_TYPE == TRI
bool initialize_kd_tree(CompositeObject *poly_model) {
	// Returns 1 if kd-tree data was initialized successfully, or 0 otherwise.

	bool bError = false;

	g_iKdTree_Node_Count      = 0;
	g_iKdTree_TriOffset_Count = 0;
	g_pKdTree_Node_Array      = NULL;
	g_pKdTree_TriOffset_Array = NULL;
	//g_iKdTree_Node_CountAlloc      =  8 * 1024 * 1024; 
	g_iKdTree_Node_CountAlloc      =  32 * 1024 * 1024; 
	//g_iKdTree_TriOffset_CountAlloc = 16 * 1024 * 1024;
	g_iKdTree_TriOffset_CountAlloc = 64 * 1024 * 1024;

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
	}
	else {
		for( int i = 0; i < g_iTriangleSize; i++ ) {
			g_pTriangleInfos[i].offset = i;
			g_pTriangleInfos[i].point[0] = pVertexList[3*i];
			g_pTriangleInfos[i].point[1] = pVertexList[3*i+1];
			g_pTriangleInfos[i].point[2] = pVertexList[3*i+2];

			//calc triangle aabb
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
#endif

void uninitialize_kd_tree(void) {
#if PRIMITIVE_TYPE == TRI
	if( g_pTriangleInfos != nullptr) {
		delete[] g_pTriangleInfos;
		g_pTriangleInfos = nullptr;
	}
	if( g_pKdTree_TriOffset_Array != nullptr) {
		delete[] g_pKdTree_TriOffset_Array;
		g_pKdTree_TriOffset_Array = nullptr;
	}
#elif PRIMITIVE_TYPE ELLIPSOID
	if (g_pEllipsoidInfos != nullptr) {
		delete[] g_pEllipsoidInfos;
		g_pEllipsoidInfos = nullptr;
	}
	if (g_pKdTreeEllipsoidOffsetArray != nullptr) {
		delete[] g_pKdTreeEllipsoidOffsetArray;
		g_pKdTreeEllipsoidOffsetArray = nullptr;
	}
#endif
	if (g_bEdge != nullptr) {
		delete[] g_bEdge;
		g_bEdge = nullptr;
	}
	if (g_pKdTree_Node_Array != nullptr) {
		delete[] g_pKdTree_Node_Array;
		g_pKdTree_Node_Array = nullptr;
	}
}

#define DEBUG_FLAG 0

ExtendedVertex intersect(const ExtendedVertex& v1, const ExtendedVertex& v2, int axis, float clipVal) {
	ExtendedVertex res = v1;
	float t = (clipVal - v1.vertex[axis]) / (v2.vertex[axis] - v1.vertex[axis]);

	for (int i = 0; i < 3; ++i) {
		res.vertex[i] = v1.vertex[i] + t * (v2.vertex[i] - v1.vertex[i]);
	}
	// material_ID는 보존 (v1의 것을 따름)
	return res;
}

// 특정 평면에 대해 다각형 클리핑
vector<ExtendedVertex> clipWithPlane(const vector<ExtendedVertex>& vertices, int axis, float clipVal, bool isMax) {
	vector<ExtendedVertex> result;
	if (vertices.empty()) return result;

	for (size_t i = 0; i < vertices.size(); ++i) {
		const ExtendedVertex& cur = vertices[i];
		const ExtendedVertex& prev = vertices[(i + vertices.size() - 1) % vertices.size()];

		bool curInside = isMax ? (cur.vertex[axis] <= clipVal) : (cur.vertex[axis] >= clipVal);
		bool prevInside = isMax ? (prev.vertex[axis] <= clipVal) : (prev.vertex[axis] >= clipVal);

		if (curInside) {
			if (!prevInside) {
				result.push_back(intersect(prev, cur, axis, clipVal));
			}
			result.push_back(cur);
		}
		else if (prevInside) {
			result.push_back(intersect(prev, cur, axis, clipVal));
		}
	}
	return result;
}

void clipTriangleToAABB(const TriangleList& inputTri, vector<TriangleList>& outputList) {
	vector<ExtendedVertex> polygon;
	for (int i = 0; i < 3; ++i) polygon.push_back(inputTri.point[i]);

	BoundingBox targetAABB = inputTri.AABB;

	// 6개의 면(x_min, x_max, y_min, y_max, z_min, z_max)에 대해 클리핑
	for (int axis = 0; axis < 3; ++axis) {
		polygon = clipWithPlane(polygon, axis, targetAABB.min[axis], false);
		polygon = clipWithPlane(polygon, axis, targetAABB.max[axis], true);
	}

	if (polygon.size() < 3) return;

	// 결과 다각형(Fan 형태)을 다시 삼각형들로 분할하여 저장
	for (size_t i = 1; i < polygon.size() - 1; ++i) {
		TriangleList tri = inputTri; // 기본값 복사 (offset, side 등)
		tri.AABB = targetAABB;       // 클리핑된 결과이므로 타겟 AABB 할당
		tri.point[0] = polygon[0];
		tri.point[1] = polygon[i];
		tri.point[2] = polygon[i + 1];
		outputList.push_back(tri);
	}
}

#if PRIMITIVE_TYPE == ELLIPSOID
void build_kd_tree_recursive(BoundEdge* bEdge, const TriangleList* pEllipsoidInfos, unsigned int ellipsoidSize, BoundingBox& bbox, unsigned int inNodeLevel, KdTreeNode* inNode){
	SplitCost bestCost;
	g_iKdTree_Level = MyMAX(inNodeLevel, g_iKdTree_Level);

	bestCost.cost = double(ellipsoidSize) * v_KD_TREE_ISECT_COST;

#if FORCE_SPLIT_THRESHOLD
	#define MAX_ELLIPSOID_OFFSET_BUDGET 4294967295
	if (ellipsoidSize > FORCE_SPLIT_THRESHOLD) {
	#if MAX_ELLIPSOID_OFFSET_BUDGET
		if (g_iKdTreeEllipsoidOffsetCnt > MAX_ELLIPSOID_OFFSET_BUDGET) {
			bestCost.cost = double(ellipsoidSize) * v_KD_TREE_ISECT_COST;
			fprintf(stdout, "WARNING: Memory budget exceeded (%u refs). Forcing leaf node at level %u with %u tris.\n",
				g_iKdTreeEllipsoidOffsetCnt, inNodeLevel, ellipsoidSize);
		}
		else {
			bestCost.cost = DBL_MAX;
		}
	#else
		bestCost.cost = DBL_MAX;
	#endif
	}
#endif
	if (inNodeLevel < v_KD_TREE_MAX_LEVEL && ellipsoidSize > v_KD_TREE_MIN_PRIMITIVE) {
		SplitCost axisCosts[3] = { bestCost, bestCost, bestCost };
		#pragma omp parallel for
		for (int axis = 0; axis < 3; axis++) {
			BoundEdge* local_bEdge = new BoundEdge[ellipsoidSize * 2];
			//TODO
			try_to_split(axis, bbox, pEllipsoidInfos, ellipsoidSize, local_bEdge, axisCosts[axis]);
			delete[] local_bEdge;
		}
		for (int axis = 0; axis < 3; axis++) {
			if (axisCosts[axis].cost < bestCost.cost) {
				bestCost = axisCosts[axis];
			}
		}
	}

	if (!bestCost.is_valid()) {
		//leaf node
		unsigned int iEllipsoidOffset;
		{
			iEllipsoidOffset = g_iKdTreeEllipsoidOffsetCnt;
			setLeafNode(inNode, ellipsoidSize, g_iKdTreeEllipsoidOffsetCnt);
			g_iKdTreeEllipsoidOffsetCnt += ellipsoidSize;

			if (g_iKdTreeEllipsoidOffsetCnt >= g_iKdTreeEllipsoidOffsetCnt_Alloc) {
				_reAllocTriangleOffsetList(MyMAX(2 * g_iKdTreeEllipsoidOffsetCnt_Alloc, 512), g_iKdTreeEllipsoidOffsetCnt_Alloc, &g_pKdTreeEllipsoidOffsetArray);
			}
		}

		unsigned* currOffsetList = &g_pKdTreeEllipsoidOffsetArray[iEllipsoidOffset];
		unsigned leafCount = 0;
		for (unsigned i = 0; i < ellipsoidSize; i++) {
			currOffsetList[leafCount++] = pEllipsoidInfos[i].offset;
		}

		if (ellipsoidSize == 0) {
			g_iKdTree_EmptyNode_Count++;
		}

		g_iKdTree_LeafNode_Count++;
		g_iKdTreeMaxEllipsoidInLeafNodeCnt = MyMAX(g_iKdTreeMaxEllipsoidInLeafNodeCnt, ellipsoidSize);

		delete[] pEllipsoidInfos;
	}
	else {
		//Inner node
		unsigned int nodeNum;
		{
			nodeNum = g_iKdTree_Node_Count;
			setInnerNode(inNode, bestCost.axis, g_iKdTree_Node_Count, bestCost.splitPos);

			g_iKdTree_Node_Count += 2;

			if (g_iKdTree_Node_Count >= g_iKdTree_Node_CountAlloc) {
				_reAllocKdtreeNodes(MyMAX(2 * g_iKdTree_Node_CountAlloc, 512), g_iKdTree_Node_CountAlloc, &g_pKdTree_Node_Array);
			}
		}

		BoundingBox leftnBounds, rightnBounds;
		leftnBounds = bbox;  leftnBounds.max[bestCost.axis] = bestCost.splitPos;
		rightnBounds = bbox;  rightnBounds.min[bestCost.axis] = bestCost.splitPos;

		std::vector<TriangleList> leftEllipsoids;
		std::vector<TriangleList> rightEllipsoids;

		const unsigned n_bEdge = 2 * ellipsoidSize;

		set_bound_edge(bestCost.axis, pEllipsoidInfos, n_bEdge, bEdge);

		push_triangles_to_child_vector(n_bEdge, bEdge, leftEllipsoids, rightEllipsoids, bestCost);

		TriangleList* pLeftEllipsoids = new TriangleList[leftEllipsoids.size()];
		TriangleList* pRightEllipsoids = new TriangleList[rightEllipsoids.size()];
		memcpy(pLeftEllipsoids, leftEllipsoids.data(), sizeof(TriangleList) * leftEllipsoids.size());
		memcpy(pRightEllipsoids, rightEllipsoids.data(), sizeof(TriangleList) * rightEllipsoids.size());

		clip_ellipsoid(leftEllipsoids.size(), bestCost, pLeftEllipsoids, 0);
		clip_ellipsoid(rightEllipsoids.size(), bestCost, pRightEllipsoids, 1);

		delete[] pEllipsoidInfos;

		build_kd_tree_recursive(bEdge, pLeftEllipsoids, leftEllipsoids.size(), leftnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum]);
		build_kd_tree_recursive(bEdge, pRightEllipsoids, rightEllipsoids.size(), rightnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum + 1]);
	}

}
#elif PRIMITIVE_TYPE == TRI
void build_kd_tree_recursive(BoundEdge* bEdge, const TriangleList* pTriangleInfos, unsigned int triangleSize,
	BoundingBox& bbox, unsigned int inNodeLevel, KdTreeNode* inNode)
{
	//printf("[Level %2u, Size %u] Processing node...\n", inNodeLevel, triangleSize);

	SplitCost bestCost;
	g_iKdTree_Level = MyMAX(inNodeLevel, g_iKdTree_Level);
	//printf("innodeLevel %d\n", g_iKdTree_Level);

	if (DEBUG_FLAG) {
		fprintf(stdout, "b_k_t_r: (S) triangleSize = %d, inNodeLevel = %d\n", triangleSize, inNodeLevel);
	}

	// Calculate cost function (in case of no partition)
	bestCost.cost = double(triangleSize) * v_KD_TREE_ISECT_COST;

#if FORCE_SPLIT_THRESHOLD
	//#define MAX_TRIANGLE_OFFSET_BUDGET 18446744073709551615
	//#define MAX_TRIANGLE_OFFSET_BUDGET 9223372036854775807
	//#define MAX_TRIANGLE_OFFSET_BUDGET 17179869184
	//#define MAX_TRIANGLE_OFFSET_BUDGET 8589934592
	#define MAX_TRIANGLE_OFFSET_BUDGET 4294967295
	//#define MAX_TRIANGLE_OFFSET_BUDGET 2147483647
	//#define MAX_TRIANGLE_OFFSET_BUDGET 1073741824
	//if (triangleSize > FORCE_SPLIT_THRESHOLD) bestCost.cost = DBL_MAX; //shyun added
	if (triangleSize > FORCE_SPLIT_THRESHOLD) {
	#if MAX_TRIANGLE_OFFSET_BUDGET
			// 강제 분할 전, 메모리 예산을 초과했는지 확인.
		#if SOFT_SPLIT_THRESHOLD
			if ((triangleSize < SOFT_SPLIT_THRESHOLD &&
				g_iKdTree_TriOffset_Count > MAX_TRIANGLE_OFFSET_BUDGET) ||
				(triangleSize < SOFT_SPLIT_THRESHOLD2 &&
					g_iKdTree_TriOffset_Count > MAX_TRIANGLE_OFFSET_BUDGET2)) {
		#else
			if (g_iKdTree_TriOffset_Count > MAX_TRIANGLE_OFFSET_BUDGET) {
		#endif

				// 예산 초과 시: 강제 분할(DBL_MAX)을 하지 않고, 
				// SAH 비용(bestCost.cost)을 그대로 둬서 리프 노드가 되도록 함.
				fprintf(stdout, "WARNING: Memory budget exceeded (%u refs). Forcing leaf node at level %u with %u tris.\n",
					g_iKdTree_TriOffset_Count, inNodeLevel, triangleSize);
			}
			else {
				// 예산 미초과 시: 원래대로 강제 분할 실행
				bestCost.cost = DBL_MAX;
			}
	#else
			bestCost.cost = DBL_MAX;
	#endif
	}
#endif

#if SAH_MAXIMIZE
	bestCost.cost = -1.0;
#endif

	// Calculate cost function (in case of trying to partition)
	if (inNodeLevel < v_KD_TREE_MAX_LEVEL && triangleSize > v_KD_TREE_MIN_PRIMITIVE) {
		SplitCost axisCosts[3] = { bestCost, bestCost, bestCost }; // 각 축의 결과를 저장할 배열
		#pragma omp parallel for
		// (모든 축에 대해 수행)
		for (int axis = 0; axis < 3; axis++) {
			BoundEdge* local_bEdge = new BoundEdge[triangleSize * 2];
			//try_to_split(axis, bbox, pTriangleInfos, triangleSize, bEdge, bestCost
			try_to_split(axis, bbox, pTriangleInfos, triangleSize, local_bEdge, axisCosts[axis]);
			delete[] local_bEdge;
		}

		// 병렬 계산이 끝난 후, 3개 축의 결과 중 가장 좋은 것을 선택
		for (int axis = 0; axis < 3; axis++) {
#if SAH_MAXIMIZE
			if (axisCosts[axis].cost > bestCost.cost) {
#else
			if (axisCosts[axis].cost < bestCost.cost) {
#endif
				bestCost = axisCosts[axis];
			}
		}
	}
	
	if (!bestCost.is_valid()) {
#if BSPT
		if (triangleSize > 0) {
			vector<TriangleList> triangles(triangleSize);
			for (int i = 0; i < triangleSize; i++) {
	#if CLIP_BEFORE_BSPT
				vector<TriangleList> clippedTris;
				clipTriangleToAABB(pTriangleInfos[i], clippedTris);
				triangles.insert(triangles.end(), clippedTris.begin(), clippedTris.end());
	#else
				triangles[i] = pTriangleInfos[i];
	#endif
			}
			int depth = 0;
			BSPNode_Build* root = BuildBSPTree(triangles, depth);
			if (depth > maxDepth) {
				maxDepth = depth;
			}

			vector<unsigned int> triOffsets;
			onPlaneTriangleCnts.resize(onPlaneTriangleCnts.size()+1);
			FlattenBSPTree(root, triOffsets);
			onPlaneTriCntsIdx++;

			unsigned int iTriOffset = g_iKdTree_TriOffset_Count;

			g_iKdTree_TriOffset_Count += triOffsets.size();
			if (g_iKdTree_TriOffset_Count >= g_iKdTree_TriOffset_CountAlloc) {
				_reAllocTriangleOffsetList(MyMAX(2 * g_iKdTree_TriOffset_CountAlloc, 512), g_iKdTree_TriOffset_CountAlloc, &g_pKdTree_TriOffset_Array);
			}
			unsigned* currOffsetList = &g_pKdTree_TriOffset_Array[iTriOffset];

			unsigned leafCount = 0;
			for (unsigned i = 0; i < triOffsets.size(); i++) {
				currOffsetList[leafCount++] = triOffsets[i];
			}
		}
		setLeafNodeBSPT(inNode, g_BSPTNodes.size());
#else
		// Leaf node 생성
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
#endif

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
		//printf("  -> Split at Level %u: [Axis %d, Pos %.3f] -> L: %d tris, R: %d tris\n", inNodeLevel, bestCost.axis, bestCost.splitPos, bestCost.n_left, bestCost.n_right);

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

		// std::vector로 변경하여 메모리 안정성 확보
		std::vector<TriangleList> leftTriangles;
		std::vector<TriangleList> rightTriangles;
		// 예상 크기만큼 미리 예약하여 성능 저하 최소화
		leftTriangles.reserve(bestCost.n_left);
		rightTriangles.reserve(bestCost.n_right);

		const unsigned n_bEdge = 2 * triangleSize;

		// TriangleInfo 부터 bEdge 를 생성 및 정렬
		set_bound_edge(bestCost.axis, pTriangleInfos, n_bEdge, bEdge);

		// bEdge 로 부터 pLeftTriangle, pRightTriangle 을 생성
		push_triangles_to_child_vector(n_bEdge, bEdge, leftTriangles, rightTriangles, bestCost);

		TriangleList* pLeftTriangles = new TriangleList[leftTriangles.size()];
		TriangleList* pRightTriangles = new TriangleList[rightTriangles.size()];
		memcpy(pLeftTriangles, leftTriangles.data(), sizeof(TriangleList)* leftTriangles.size());
		memcpy(pRightTriangles, rightTriangles.data(), sizeof(TriangleList)* rightTriangles.size());

		// 각 child node 에 맞게 삼각형 clipping
		clip_triangle(leftTriangles.size(), bestCost, pLeftTriangles, 0);
		clip_triangle(rightTriangles.size(), bestCost, pRightTriangles, 1);

		/**
		 *	더이상 pTriangleInfos 는 필요없으므로 메모리 공간 절약을 위해 없앤다.
		 *	반드시 pushChildTriangles 를 수행한 이후에 없애야 한다.
		 */
		delete[] pTriangleInfos;
		/**
		 *	Left, Right 재귀 탐색.
		 *	pLeftTriangles, pRightTriangles 는 build_kd_tree_recursive 함수 안에서 사용하고 바로 없앤다.
		 */
		build_kd_tree_recursive(bEdge, pLeftTriangles, leftTriangles.size(), leftnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum]);
		build_kd_tree_recursive(bEdge, pRightTriangles, rightTriangles.size(), rightnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum + 1]);
	}

#if BSPT
	if (inNodeLevel < 10) {
		printf("bspt triangle size: %d\n", g_BSPTTris.size());
	}
	if (inNodeLevel == 0) {
		printf("bspt max depth : %d\n", maxDepth);
		printf("bspt max tri cnt : %d\n", maxTriInNode);

	#if BSPT_DUMP_STATISTICS
		std::ofstream bspt_file("bspt_onplane_stats.csv");
		bspt_file << "BSPT_Idx,ManyOnPlane_Count,AvgOnPlane_Count,MaxOnPlane_Count\n";

		// 전체적인 분포를 보기 위한 히스토그램 (예: 0~255 범위)
		std::vector<int> bspt_hist_max(256, 0);

		for (int i = 0; i < onPlaneTriangleCnts.size(); i++) {
			float avgOnPlaneCnt = 0;
			int manyOnPlaneTrisCnt = 0;
			int maxOnPlaneCnt = 0;

			for (int j = 0; j < onPlaneTriangleCnts[i].size(); j++) {
				int current_cnt = onPlaneTriangleCnts[i][j];
				if (current_cnt > 1) {
					manyOnPlaneTrisCnt++;
					avgOnPlaneCnt += current_cnt;
					if (current_cnt > maxOnPlaneCnt) {
						maxOnPlaneCnt = current_cnt;
					}
				}
			}

			if (manyOnPlaneTrisCnt > 0) {
				avgOnPlaneCnt /= manyOnPlaneTrisCnt;
			}

			// 2. CSV 파일 기록
			bspt_file << i << "," << manyOnPlaneTrisCnt << "," << avgOnPlaneCnt << "," << maxOnPlaneCnt << "\n";

			// 3. 히스토그램 수집 (최댓값 분포 확인용)
			if (maxOnPlaneCnt < bspt_hist_max.size()) {
				bspt_hist_max[maxOnPlaneCnt]++;
			}
			else {
				// 범위를 벗어나면 확장
				bspt_hist_max.resize(maxOnPlaneCnt + 1, 0);
				bspt_hist_max[maxOnPlaneCnt]++;
			}
		}
		bspt_file.close();
	#endif
	}
#endif

	if (DEBUG_FLAG) {
		fprintf(stdout, "b_k_t_r: (E)triangleSize = %d, inNodeLevel = %d\n", triangleSize, inNodeLevel);
	}
}
#endif

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

//for debug
std::vector<LeafNodeInfo> extract_all_leaf_data(CompositeObject* c_object, int& largest_leaf_index) {
	// Kd-tree가 없으면 빈 벡터 반환
#if PRIMITIVE_TYPE == ELLIPSOID
	return {};
#endif
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

inline double get_surface_area(const BoundingBox& box) {
	// AABB의 세 변의 길이를 계산. 유효하지 않은 (찌그러진) 박스를 고려하여 0보다 작아지지 않도록
	double dx = MyMAX(0.0, (double)box.max[0] - box.min[0]);
	double dy = MyMAX(0.0, (double)box.max[1] - box.min[1]);
	double dz = MyMAX(0.0, (double)box.max[2] - box.min[2]);

	// 6면의 넓이를 합산하여 반환
	return 2.0 * (dx * dy + dx * dz + dy * dz);
}
inline double get_surface_volume(const BoundingBox& box) {
	// AABB의 세 변의 길이를 계산. 유효하지 않은 (찌그러진) 박스를 고려하여 0보다 작아지지 않도록
	double dx = MyMAX(0.0, (double)box.max[0] - box.min[0]);
	double dy = MyMAX(0.0, (double)box.max[1] - box.min[1]);
	double dz = MyMAX(0.0, (double)box.max[2] - box.min[2]);

	// 부피 반환
	return dx * dy * dz;
}
