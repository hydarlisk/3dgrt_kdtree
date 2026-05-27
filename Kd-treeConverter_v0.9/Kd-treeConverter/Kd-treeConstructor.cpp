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
#include "Gaussian.h"

#include "AABB_Triangle_Clip/AABB_Triangle_Clip.h" 
//#include "Kd-treeCudaKernels.h"
#include <algorithm> // for std::remove_if
#include <array>       // std::array 사용을 위해 추가

#include <cmath>

#if PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
#include <set>
#include <unordered_map>
#endif

using namespace std;

static const unsigned int modulo[] = { 0,1,2,0,1 };

// From macros to variables to be able to modify through the configuration file
float v_KD_TREE_TRAVL_COST = TRAVL_COST;
float v_KD_TREE_ISECT_COST = ISCET_COST;
unsigned int v_KD_TREE_MAX_LEVEL = MAX_LEVEL;
float v_KD_TREE_EMTPY_BONUS = EMTPY_BONUS;

BoundingBox   g_root_AABB;
BoundEdge    *g_bEdge = NULL;

#if PRIMITIVE_TYPE == TRI || PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
unsigned int v_KD_TREE_MIN_PRIMITIVE = MIN_TRI;
#elif PRIMITIVE_TYPE == ELLIPSOID
unsigned int v_KD_TREE_MIN_PRIMITIVE = MIN_ELLIPSOID;
#endif
PrimList *g_pPrimInfos = nullptr;
unsigned int  g_iPrimSize;
unsigned long long  g_iKdTreePrimOffsetCnt;
unsigned long long  g_iKdTreePrimOffsetCnt_Alloc;
unsigned int *g_pKdTreePrimOffsetArray = NULL;
unsigned int  g_iKdTreeMaxPrimInLeafNodeCnt;

unsigned int  g_iKdTree_Level;
unsigned int  g_iKdTree_Node_Count;
unsigned int  g_iKdTree_Node_CountAlloc;
KdTreeNode   *g_pKdTree_Node_Array = NULL;
unsigned int  g_iKdTree_EmptyNode_Count;
unsigned int  g_iKdTree_LeafNode_Count;

extern std::vector<Gaussian> g_gaussians;
extern std::vector<bool> g_isValidG;
#if !QUATERNION
extern std::vector<float> g_kScales;
#endif

std::vector<std::vector<PrimList>> g_leafDebug;
#if PRIMITIVE_TYPE == ELLIPSOID
std::vector<PrimList> g_ellipsoidAabbDebug;
std::vector<std::vector<std::vector<PrimList>>> g_ellipsoidClipAabbDebug;
std::vector<std::vector<std::vector<PrimList>>> g_ellipsoidInternalDebug;
#endif

#include <iostream>
#include <fstream>
#include <string>

#if DUMP_LEAF_CSV
bool exportToCSV(const PrimList* ellipsoidInfo, const int ellipsoidSize) {
	static int fileIdx = 0;
	int dir = 0;
	if (ellipsoidSize < 32) dir = 32;
	else if (ellipsoidSize < 64) dir = 64;
	else if (ellipsoidSize < 96) dir = 96;
	else if (ellipsoidSize < 128) dir = 128;
	else if (ellipsoidSize < 192) dir = 192;
	else if (ellipsoidSize <= 256) dir = 256;
	string filename = DUMP_DIR_PATH + string("leaves/") + to_string(dir) + string("/leaf") + std::to_string(fileIdx++) + ".csv";
	std::ofstream outFile(filename);

	vector<Gaussian> leaves;
	for (int i = 0; i < ellipsoidSize; i++) {
		leaves.push_back(g_gaussians[ellipsoidInfo[i].offset]);
	}

	if (!outFile.is_open()) {
		std::cerr << "cannot open file: " << filename << std::endl;
		return false;
	}

	outFile << "pos_x,pos_y,pos_z,scale_x,scale_y,scale_z,rot_w,rot_x,rot_y,rot_z,opacity,k_scale,f_dc_r,f_dc_g,f_dc_b";

	for (int i = 0; i < 45; ++i) {
		outFile << ",f_rest_" << i;
	}

	for (const auto& g : leaves) {
		outFile << g.pos[0] << "," << g.pos[1] << "," << g.pos[2] << ","
			<< g.scale[0] << "," << g.scale[1] << "," << g.scale[2] << ","
			<< g.rot[0] << "," << g.rot[1] << "," << g.rot[2] << "," << g.rot[3] << ","
			<< g.opacity << ","
			<< g.k_scale << ","
			<< g.f_dc[0] << "," << g.f_dc[1] << "," << g.f_dc[2];

		for (int i = 0; i < 45; ++i) {
			outFile << "," << g.f_rest[i];
		}

		outFile << "\n";
	}

	outFile.close();
	return true;
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

int compare_bound_edge(const void *elem0, const void *elem1)
{
	const BoundEdge* obj0 = (const BoundEdge *)elem0;
	const BoundEdge* obj1 = (const BoundEdge *)elem1;

	return obj0->t == obj1->t ?
		(obj0->triangleInfo->offset > obj1->triangleInfo->offset ? 1 : -1)
		:
		(obj0->t > obj1->t ? 1 : -1);
}

void set_bound_edge(const int axis, const PrimList *pTriangleInfo, const unsigned int n_bEdge, BoundEdge *bEdge)
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
#if PRIMITIVE_TYPE == ELLIPSOID
bool isGaussianIntersectingAABB(const BoundingBox& currBBox, const float pos[3], const MyMat33& covInv) {
	float x_coords[2] = { currBBox.min[0], currBBox.max[0] };
	float y_coords[2] = { currBBox.min[1], currBBox.max[1] };
	float z_coords[2] = { currBBox.min[2], currBBox.max[2] };
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				MyVec3 v = { x_coords[i], y_coords[j], z_coords[k] };
				MyVec3 gPos = { pos[0], pos[1], pos[2] };
				MyVec3 p = v - gPos;
				float e = p * covInv * p - 1;
				if (e < 0) {
					return true;
				}
			}
		}
	}
	return false;
}

bool isGaussianIntersectingAABB(const float aabbMin[3], const float aabbMax[3], const float pos[3], const float cov[3][3]) {


	// 1. AABB 내에서 가우시안 중심(pos)과 가장 가까운 점(closestPoint) 계산
	float closestPoint[3];
	for (int i = 0; i < 3; ++i) {
		closestPoint[i] = std::max(aabbMin[i], std::min(pos[i], aabbMax[i]));
	}

	// 2. 가우시안 중심으로부터의 변위 벡터 (v = x - mu) 계산
	float v[3];
	v[0] = closestPoint[0] - pos[0];
	v[1] = closestPoint[1] - pos[1];
	v[2] = closestPoint[2] - pos[2];

	// 3. 3x3 공분산 행렬의 행렬식(Determinant) 계산
	float det = cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[1][2]) -
		cov[0][1] * (cov[0][1] * cov[2][2] - cov[1][2] * cov[0][2]) +
		cov[0][2] * (cov[0][1] * cov[1][2] - cov[1][1] * cov[0][2]);

	// 행렬식이 0에 가까우면 역행렬 존재 불가 (에러 방지)
	if (std::abs(det) < 1e-9f) return false;

	float invDet = 1.0f / det;

	// 4. 역행렬(Inverse Covariance)의 상삼각 성분 계산 (대칭성 활용)
	float m00 = (cov[1][1] * cov[2][2] - cov[1][2] * cov[1][2]) * invDet;
	float m01 = (cov[0][2] * cov[1][2] - cov[0][1] * cov[2][2]) * invDet;
	float m02 = (cov[0][1] * cov[1][2] - cov[0][2] * cov[1][1]) * invDet;
	float m11 = (cov[0][0] * cov[2][2] - cov[0][2] * cov[0][2]) * invDet;
	float m12 = (cov[0][2] * cov[0][1] - cov[0][0] * cov[1][2]) * invDet;
	float m22 = (cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]) * invDet;

	// 5. 마할라노비스 거리 제곱 계산 (d^2 = v^T * Sigma^-1 * v)
	// d^2 = v0*(m00*v0 + m01*v1 + m02*v2) + v1*(m01*v0 + m11*v1 + m12*v2) + ...
	float d2 = v[0] * (m00 * v[0] + m01 * v[1] + m02 * v[2]) +
		v[1] * (m01 * v[0] + m11 * v[1] + m12 * v[2]) +
		v[2] * (m02 * v[0] + m12 * v[1] + m22 * v[2]);

	// 6. 최종 판정 (1.0f는 1-sigma, 4.0f는 2-sigma, 9.0f는 3-sigma)
	return d2 <= 9.0f;
}

void clip_ellipsoid(std::vector<PrimList>& ellipsoidInfos, const SplitCost& bestCost, int side) {
	int axis = bestCost.axis;
	float splitPos = bestCost.splitPos;

	for (int gi = 0; gi < ellipsoidInfos.size(); gi++) {
		//simple clipping
		BoundingBox& currBBox = ellipsoidInfos[gi].AABB;
		if (currBBox.min[axis] < splitPos && currBBox.max[axis] > splitPos) {
			bool remove = false;
			if (side == 0) {
				if (currBBox.max[axis] >= splitPos) {
					if (abs(currBBox.max[axis] - splitPos) / (currBBox.max[axis] - currBBox.min[axis]) < IGNORE_THRESHOLD)
						remove = true;
					currBBox.max[axis] = splitPos;
				}
			}
			else {
				if (currBBox.min[axis] < splitPos) {
					if (abs(currBBox.max[axis] - splitPos) / (currBBox.max[axis] - currBBox.min[axis]) < IGNORE_THRESHOLD)
						remove = true;
					currBBox.min[axis] = splitPos;
				}
			}
			if (remove) {
				ellipsoidInfos.erase(ellipsoidInfos.begin() + gi);
				gi--;
				continue;
			}

		//tight clipping
		//1. calc covariance
			auto& g = g_gaussians[ellipsoidInfos[gi].offset];
			float k_scale;
			float R[3][3];
#if QUATERNION
			quaternionWXYZToMatrix(g.rot, R);
			k_scale = g.k_scale;
#else
			memcpy(R, &g.rotMat, sizeof(float3x3));
			k_scale = calcKernelScale(g.opacity);
#endif
			float s0 = g.scale[0];
			float s1 = g.scale[1];
			float s2 = g.scale[2];

			MyMat33 M;
			for (int i = 0; i < 3; i++) {
				M[i][0] = R[i][0] * s0;
				M[i][1] = R[i][1] * s1;
				M[i][2] = R[i][2] * s2;
			}
			MyMat33 cov;
			cov = M.multTranspose();
			cov = cov * (k_scale * k_scale);

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

			float invSii = 1.0f / max(cov[i][i], 1e-5f);
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
			currBBox.max[j] = min(currBBox.max[j], condJMax * (g.pos[j] + e[j]) + (1 - condJMax) * (jMu + eJ_cut));
			currBBox.min[j] = max(currBBox.min[j], condJMin * (g.pos[j] - e[j]) + (1 - condJMin) * (jMu - eJ_cut));

			bool condKMax = (kMax * sign >= splitPos * sign);
			bool condKMin = (kMin * sign >= splitPos * sign);
			currBBox.max[k] = min(currBBox.max[k], condKMax * (g.pos[k] + e[k]) + (1 - condKMax) * (kMu + eK_cut));
			currBBox.min[k] = max(currBBox.min[k], condKMin * (g.pos[k] - e[k]) + (1 - condKMin) * (kMu - eK_cut));

			////ellipsoid aabb side test
			//for (int i = 0; i < 3; i++) {
			//    M[i][0] = R[i][0] / s0;
			//    M[i][1] = R[i][1] / s1;
			//    M[i][2] = R[i][2] / s2;
			//}

			//MyMat33 covInv;
			//covInv = M.multTranspose();
			//covInv = covInv / k_scale / k_scale;
			//bool isInside = isGaussianIntersectingAABB(currBBox, g.pos, covInv);
			//if (!isInside) {
			//	ellipsoidInfos.erase(ellipsoidInfos.begin() + gi);
			//	gi--;
			//	continue;
			//}

			ellipsoidInfos[gi].cutAxis = axis;
			ellipsoidInfos[gi].cutCenter[i] = splitPos;
			ellipsoidInfos[gi].cutCenter[j] = jMu;
			ellipsoidInfos[gi].cutCenter[k] = kMu;
			ellipsoidInfos[gi].ejCut = eJ_cut;
			ellipsoidInfos[gi].ekCut = eK_cut;
		}
	}
}
#else
void clip_triangle(const int triangleSize, const SplitCost& bestCost, PrimList* pTriangleInfos, int side)
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
#endif
void push_triangles_to_child(const unsigned n_bEdge, const BoundEdge *bEdge, 
                             PrimList *pLeftTriangles, PrimList *pRightTriangles,
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
	std::vector<PrimList>& pLeftTriangles, std::vector<PrimList>& pRightTriangles,
	const SplitCost& bestCost)
{
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
#if KD_TREE_PLANAR_TRIANGLE_ADD_MODE == BOTH_SIDE
				pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
				pRightTriangles.push_back(*(bEdge[i].triangleInfo));
#elif KD_TREE_PLANAR_TRIANGLE_ADD_MODE == MINCOST_SIDE
				if (bestCost.planar_side == BoundEdge::START)
					pLeftTriangles.push_back(*(bEdge[i].triangleInfo));
				else
					pRightTriangles.push_back(*(bEdge[i].triangleInfo));
#endif
			}
		}
	}
}
//shyun added end
void try_to_split(const int axis, BoundingBox &inBBox, const PrimList *pTriangles, const int triangleSize, BoundEdge *bEdge,  SplitCost &bestCost
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
	set_bound_edge(axis, pTriangles, n_bEdge, bEdge);

#if COUNT_BY_GID
	std::unordered_map<int, int> total_ends;
	for (unsigned int k = 0; k < n_bEdge; k++) {
		if (bEdge[k].type == BoundEdge::END) {
			total_ends[bEdge[k].triangleInfo->offset]++;
		}
	}
	std::set<int> seen_starts;
	std::unordered_map<int, int> seen_ends;
#endif

	for (unsigned int i = 0; i < n_bEdge; i++) {
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
#if COUNT_BY_GID
			// 현재 위치(cur_position)에서의 gId별 상태를 추적합니다.
			// 1: START만 있음, 2: END만 있음, 3: START와 END가 동시에 있음 (즉, Planar)
			std::unordered_map<int, int> gId_status;
			std::unordered_map<int, bool> gId_normal; // 평면일 경우 법선 방향 저장용
#endif

			//똑같은게 여러개 있을 때는 제일 오른쪽에서만 SAH 계산을 한다.
			for (unsigned int j = i; j < n_bEdge; j++) {
				BoundEdge tmp_bEdge = bEdge[j];
				if (tmp_bEdge.t != cur_position) break;

				const bool
					is_left = (tmp_bEdge.type == BoundEdge::START),
					is_planar = tmp_bEdge.isPlanar,
					is_normalPositive = tmp_bEdge.isNormalPositive;

#if COUNT_BY_GID
				int gId = tmp_bEdge.triangleInfo->offset;

				if (is_left) {
					if (seen_starts.insert(gId).second) {
						gId_status[gId] |= 1; // 비트 연산으로 START(1) 상태 추가
					}
				}
				else {
					seen_ends[gId]++;
					if (seen_ends[gId] == total_ends[gId]) {
						gId_status[gId] |= 2; // 비트 연산으로 END(2) 상태 추가
					}
				}

				// 혹시 내부 서브 삼각형 중에 평면이 있다면 그 법선을 임시로 저장해 둡니다.
				if (is_planar) {
					gId_normal[gId] = is_normalPositive;
				}
#else
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
#endif
				curr_bEdge = tmp_bEdge;
				i = j;
			}

#if COUNT_BY_GID
			for (const auto& pair : gId_status) {
				int status = pair.second;
				int gId = pair.first;

				if (status == 3) {
					// 한 위치(cur_position)에서 START와 END가 모두 발생!
					// -> 전체 Primitive 기준으로 이 물체는 이 축에서 완벽한 평면(Planar)입니다.
					num_planars += 1;
					num_normalPositive += gId_normal[gId] ? 1 : 0;
				}
				else if (status == 1) {
					// 이 위치에서 시작만 함
					local_open += 1;
				}
				else if (status == 2) {
					// 이 위치에서 끝나기만 함
					local_close += 1;
				}
			}
#endif
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
#if COUNT_BY_GID
				n_rightOnly = total_ends.size() - (n_leftOnly + n_cross + num_planars);
#else
				n_rightOnly = triangleSize - (n_leftOnly + n_cross + num_planars);
#endif

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

#if SAH_MODE == 2
				auto get_batched_count = [](int n) -> int {
					if (n == 0) return 0;
					return (n + 7) & ~7;
					};
				const int eff_num_left[2] = {
					get_batched_count(tri_num_left[0]),
					get_batched_count(tri_num_left[1])
				};
				const int eff_num_right[2] = {
					get_batched_count(tri_num_right[0]),
					get_batched_count(tri_num_right[1])
				};
#endif

				double SAH[2];
				for (int side_idx = 0; side_idx < 2; side_idx++) {
#if SAH_MODE == 0
					SAH[side_idx] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
						double(tri_num_left[side_idx])* prob_l +
						double(tri_num_right[side_idx]) * prob_r
						)* emptyBonus[side_idx];
#elif SAH_MODE == 1	//balanced
					const double N_total = double(tri_num_left[side_idx] + tri_num_right[side_idx]);
					const double diff = std::abs(double(tri_num_left[side_idx]) - double(tri_num_right[side_idx]));

					const double balanceRatio = (N_total > 0.0) ? (diff / N_total) : 0.0;

					// 0.05 ~ 0.2
					const double BALANCE_WEIGHT = 0.1;
					const double balancePenalty = 1.0 + (balanceRatio * BALANCE_WEIGHT);
					SAH[side_idx] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
						double(tri_num_left[side_idx]) * prob_l +
						double(tri_num_right[side_idx]) * prob_r
						) * emptyBonus[side_idx];
#elif SAH_MODE == 2
					SAH[side_idx] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
						double(eff_num_left[side_idx]) * prob_l +
						double(eff_num_right[side_idx]) * prob_r
						) * emptyBonus[side_idx];
#elif SAH_MODE == 3
					SAH[side_idx] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
						double(sqrt(tri_num_left[side_idx])) * prob_l +
						double(sqrt(tri_num_right[side_idx])) * prob_r
						) * emptyBonus[side_idx];
#elif SAH_MODE == 4
					auto scale_count_power = [](int n) -> double {
						if (n == 0) return 0.0; // 빈 공간 보존
						const double T = 16.0;  // 기준점
						const double k = 0.5;   // 휘어지는 정도 (0.1 ~ 1.0)
						double x = double(n);
						return x * std::pow(x / T, k);
						};
					SAH[side_idx] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
						scale_count_power(tri_num_left[side_idx]) * prob_l +
						scale_count_power(tri_num_right[side_idx]) * prob_r
						) * emptyBonus[side_idx];
#elif SAH_MODE == 5
					auto scale_count_quad = [](int n) -> double {
						if (n == 0) return 0.0; // 빈 공간 보존
						const double T = 16.0;  // 기준점
						const double alpha = 0.05;// 가파름 조정 (0.01 ~ 0.1)
						double x = double(n);
						return std::max(0.0, x + alpha * x * (x - T)); // 음수 방지
						};
					SAH[side_idx] = v_KD_TREE_TRAVL_COST + v_KD_TREE_ISECT_COST * (
						scale_count_quad(tri_num_left[side_idx]) * prob_l +
						scale_count_quad(tri_num_right[side_idx]) * prob_r
						) * emptyBonus[side_idx];
#endif
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
//TODO: calc more accurate aabb
void calcEllipsoidAABB(Gaussian& g, BoundingBox& b
#if !QUATERNION
	,int idx
#endif
) {
	float R[3][3];
	float k_scale;
#if QUATERNION
	quaternionWXYZToMatrix(g.rot, R);
	k_scale = g.k_scale;
#else
	memcpy(R, g.rotMat.m, sizeof(float3x3));
	k_scale = g_kScales[idx];
#endif

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
uint32_t initEllipsoid(CompositeObject& poly_model) {
	poly_model.AABB[XMIN] = poly_model.AABB[YMIN] = poly_model.AABB[ZMIN] = FLT_MAX;
	poly_model.AABB[XMAX] = poly_model.AABB[YMAX] = poly_model.AABB[ZMAX] = -FLT_MAX;
	g_ellipsoidAabbDebug.resize(g_gaussians.size());
	uint32_t validEllipsoid = 0;
	for (int i = 0; i < g_gaussians.size(); i++) {
		if (g_isValidG[i] == 0) continue;
#if OCCLUDE_MIN_OPACITY
		float sigma = g_gaussians[i].opacity;
		if (sigma < KERNEL_MIN_RESPONSE || sigma < SIGMA_THRESHOLD_MODE / 255.0f) continue;
#endif
		calcEllipsoidAABB(g_gaussians[i], g_pPrimInfos[validEllipsoid].AABB
#if !QUATERNION
			,i
#endif
		);
		g_pPrimInfos[validEllipsoid].offset = i;
		poly_model.AABB[XMIN] = MyMIN(poly_model.AABB[XMIN], g_pPrimInfos[validEllipsoid].AABB.min[0]);
		poly_model.AABB[YMIN] = MyMIN(poly_model.AABB[YMIN], g_pPrimInfos[validEllipsoid].AABB.min[1]);
		poly_model.AABB[ZMIN] = MyMIN(poly_model.AABB[ZMIN], g_pPrimInfos[validEllipsoid].AABB.min[2]);
		poly_model.AABB[XMAX] = MyMAX(poly_model.AABB[XMAX], g_pPrimInfos[validEllipsoid].AABB.max[0]);
		poly_model.AABB[YMAX] = MyMAX(poly_model.AABB[YMAX], g_pPrimInfos[validEllipsoid].AABB.max[1]);
		poly_model.AABB[ZMAX] = MyMAX(poly_model.AABB[ZMAX], g_pPrimInfos[validEllipsoid].AABB.max[2]);
		g_ellipsoidAabbDebug[validEllipsoid] = g_pPrimInfos[validEllipsoid];
		validEllipsoid++;
	}
	g_ellipsoidAabbDebug.resize(validEllipsoid);
	return validEllipsoid;
}

bool initialize_kd_tree(CompositeObject* poly_model) {
	// Returns 1 if kd-tree data was initialized successfully, or 0 otherwise.

	g_pPrimInfos = new PrimList[g_gaussians.size()];
#if OCCLUDE_MIN_OPACITY
	uint32_t ellipsoidCnt = initEllipsoid(*poly_model);
#else
	initEllipsoid(*poly_model);
	uint32_t ellipsoidCnt = g_gaussians.size();
#endif
	
	bool bError = false;

	g_iKdTree_Node_Count = 0;
	g_pKdTree_Node_Array = NULL;
	g_iKdTree_Node_CountAlloc = 32 * 1024 * 1024;
	g_iKdTree_Level = 0;
	g_iKdTree_LeafNode_Count = 0;
	g_iKdTree_EmptyNode_Count = 0;
	
	g_iKdTreePrimOffsetCnt = 0;
	g_pKdTreePrimOffsetArray = nullptr;
	g_iKdTreePrimOffsetCnt_Alloc = 64 * 1024 * 1024;
	g_iKdTreeMaxPrimInLeafNodeCnt = 0;
	g_iPrimSize = ellipsoidCnt;

	g_root_AABB.min[0] = poly_model->AABB[0];
	g_root_AABB.min[1] = poly_model->AABB[2];
	g_root_AABB.min[2] = poly_model->AABB[4];
	g_root_AABB.max[0] = poly_model->AABB[1];
	g_root_AABB.max[1] = poly_model->AABB[3];
	g_root_AABB.max[2] = poly_model->AABB[5];

	g_bEdge = new BoundEdge[g_iPrimSize * 2];
	if (g_bEdge == NULL) {
		bError |= true;
	}
	else {
		memset(g_bEdge, 0x00, sizeof(BoundEdge) * g_iPrimSize * 2);
	}

	g_pKdTree_Node_Array = new KdTreeNode[g_iKdTree_Node_CountAlloc];
	if (g_pKdTree_Node_Array == NULL) {
		bError |= true;
	}
	else {
		memset(g_pKdTree_Node_Array, 0x00, sizeof(KdTreeNode) * g_iKdTree_Node_CountAlloc);
		g_iKdTree_Node_Count = 1;
	}

	g_pKdTreePrimOffsetArray = new unsigned int[g_iKdTreePrimOffsetCnt_Alloc];
	if (g_pKdTreePrimOffsetArray == NULL) {
		bError |= true;
	}
	else {
		memset(g_pKdTreePrimOffsetArray, 0x00, sizeof(unsigned int) * g_iKdTreePrimOffsetCnt_Alloc);
	}

	if (bError) {
		printf("init kd-tree bError");
		uninitialize_kd_tree();
		return 0;
	}
	return 1;
}
#elif PRIMITIVE_TYPE == TRI || PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
bool initialize_kd_tree(CompositeObject *poly_model) {
	// Returns 1 if kd-tree data was initialized successfully, or 0 otherwise.

	bool bError = false;

	g_iKdTree_Node_Count      = 0;
	g_iKdTreePrimOffsetCnt = 0;
	g_pKdTree_Node_Array      = NULL;
	g_pKdTreePrimOffsetArray = NULL;
	//g_iKdTree_Node_CountAlloc      =  8 * 1024 * 1024; 
	g_iKdTree_Node_CountAlloc      =  32 * 1024 * 1024; 
	//g_iKdTreePrimOffsetCnt_Alloc = 16 * 1024 * 1024;
	g_iKdTreePrimOffsetCnt_Alloc = 64 * 1024 * 1024;

	g_iKdTree_Level = 0;
	g_iKdTree_LeafNode_Count  = 0;
	g_iKdTree_EmptyNode_Count = 0;
	g_iKdTreeMaxPrimInLeafNodeCnt  = 0;

	g_iPrimSize = poly_model->n_triangles;

	g_root_AABB.min[0] = poly_model->AABB[0];
	g_root_AABB.min[1] = poly_model->AABB[2];
	g_root_AABB.min[2] = poly_model->AABB[4];
	g_root_AABB.max[0] = poly_model->AABB[1];
	g_root_AABB.max[1] = poly_model->AABB[3];
	g_root_AABB.max[2] = poly_model->AABB[5];

	ExtendedVertex *pVertexList = poly_model->extended_vertices;

	g_pPrimInfos = new PrimList[g_iPrimSize];
	if( g_pPrimInfos == NULL ) {
		bError |= true;
	}
	else {
		for( int i = 0; i < g_iPrimSize; i++ ) {
#if PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
			g_pPrimInfos[i].offset = pVertexList[3 * i].material_ID;
#else
			g_pPrimInfos[i].offset = i;
#endif
			g_pPrimInfos[i].point[0] = pVertexList[3*i];
			g_pPrimInfos[i].point[1] = pVertexList[3*i+1];
			g_pPrimInfos[i].point[2] = pVertexList[3*i+2];

			//calc triangle aabb
			g_pPrimInfos[i].AABB.min[0] = MyMIN (MyMIN (g_pPrimInfos[i].point[0].vertex[0], g_pPrimInfos[i].point[1].vertex[0]), g_pPrimInfos[i].point[2].vertex[0]);
			g_pPrimInfos[i].AABB.min[1] = MyMIN (MyMIN (g_pPrimInfos[i].point[0].vertex[1], g_pPrimInfos[i].point[1].vertex[1]), g_pPrimInfos[i].point[2].vertex[1]);
			g_pPrimInfos[i].AABB.min[2] = MyMIN (MyMIN (g_pPrimInfos[i].point[0].vertex[2], g_pPrimInfos[i].point[1].vertex[2]), g_pPrimInfos[i].point[2].vertex[2]);
			g_pPrimInfos[i].AABB.max[0] = MyMAX (MyMAX (g_pPrimInfos[i].point[0].vertex[0], g_pPrimInfos[i].point[1].vertex[0]), g_pPrimInfos[i].point[2].vertex[0]);
			g_pPrimInfos[i].AABB.max[1] = MyMAX (MyMAX (g_pPrimInfos[i].point[0].vertex[1], g_pPrimInfos[i].point[1].vertex[1]), g_pPrimInfos[i].point[2].vertex[1]);
			g_pPrimInfos[i].AABB.max[2] = MyMAX (MyMAX (g_pPrimInfos[i].point[0].vertex[2], g_pPrimInfos[i].point[1].vertex[2]), g_pPrimInfos[i].point[2].vertex[2]);
		}
	}

	g_bEdge = new BoundEdge[g_iPrimSize * 2];
	if( g_bEdge == NULL ) {
		bError |= true;
	} else {
		memset( g_bEdge, 0x00, sizeof( BoundEdge ) * g_iPrimSize * 2 );
	}
		
	g_pKdTree_Node_Array = new KdTreeNode[ g_iKdTree_Node_CountAlloc ];
	if( g_pKdTree_Node_Array == NULL ) {
		bError |= true;
	} else {
		memset( g_pKdTree_Node_Array, 0x00, sizeof( KdTreeNode ) * g_iKdTree_Node_CountAlloc );
		g_iKdTree_Node_Count = 1; 
	}
		
	g_pKdTreePrimOffsetArray = new unsigned int [ g_iKdTreePrimOffsetCnt_Alloc ];
	if( g_pKdTreePrimOffsetArray == NULL )	{
		bError |= true;
	} else {
		memset( g_pKdTreePrimOffsetArray, 0x00, sizeof( unsigned int ) * g_iKdTreePrimOffsetCnt_Alloc );
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
	if( g_pPrimInfos != nullptr) {
		delete[] g_pPrimInfos;
		g_pPrimInfos = nullptr;
	}
	if( g_pKdTreePrimOffsetArray != nullptr) {
		delete[] g_pKdTreePrimOffsetArray;
		g_pKdTreePrimOffsetArray = nullptr;
	}
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

void clipTriangleToAABB(const PrimList& inputTri, vector<PrimList>& outputList) {
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
		PrimList tri = inputTri; // 기본값 복사 (offset, side 등)
		tri.AABB = targetAABB;       // 클리핑된 결과이므로 타겟 AABB 할당
		tri.point[0] = polygon[0];
		tri.point[1] = polygon[i];
		tri.point[2] = polygon[i + 1];
		outputList.push_back(tri);
	}
}

#if PRIMITIVE_TYPE == ELLIPSOID
bool desc(const PrimList& a, const PrimList& b) {
	return g_gaussians[a.offset].opacity > g_gaussians[b.offset].opacity;
}

void build_kd_tree_recursive(BoundEdge* bEdge, const PrimList* pEllipsoidInfos, unsigned int ellipsoidSize, BoundingBox& bbox, unsigned int inNodeLevel, KdTreeNode* inNode){
	SplitCost bestCost;
	g_iKdTree_Level = MyMAX(inNodeLevel, g_iKdTree_Level);

	bestCost.cost = double(ellipsoidSize) * v_KD_TREE_ISECT_COST;

#if FORCE_SPLIT_THRESHOLD
	#define MAX_ELLIPSOID_OFFSET_BUDGET 4294967295
	if (ellipsoidSize > FORCE_SPLIT_THRESHOLD) {
	#if MAX_ELLIPSOID_OFFSET_BUDGET
		if (g_iKdTreePrimOffsetCnt > MAX_ELLIPSOID_OFFSET_BUDGET) {
			bestCost.cost = double(ellipsoidSize) * v_KD_TREE_ISECT_COST;
			fprintf(stdout, "WARNING: Memory budget exceeded (%u refs). Forcing leaf node at level %u with %u tris.\n",
				g_iKdTreePrimOffsetCnt, inNodeLevel, ellipsoidSize);
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
#if REMOVE_SMALL_PRIM
		std::vector<PrimList> leaf;
		for (int i = 0; i < ellipsoidSize; i++) {
			leaf.push_back(pEllipsoidInfos[i]);
		}
		sort(leaf.begin(), leaf.end(), desc);
		//ellipsoidSize = ellipsoidSize - ellipsoidSize / REMOVE_SMALL_PRIM;
		if(ellipsoidSize > 8)
			ellipsoidSize = ellipsoidSize / 2;
		else if(ellipsoidSize > 16){
			ellipsoidSize = ellipsoidSize / 4;
		}
		else if (ellipsoidSize > 32) {
			ellipsoidSize = ellipsoidSize / 8;
		}
#endif
#if DUMP_LEAF_CSV
		if(ellipsoidSize > 0)
			exportToCSV(pEllipsoidInfos, ellipsoidSize);
#endif
		unsigned int iEllipsoidOffset;
		{
			iEllipsoidOffset = g_iKdTreePrimOffsetCnt;
			setLeafNode(inNode, ellipsoidSize, g_iKdTreePrimOffsetCnt);
			g_iKdTreePrimOffsetCnt += ellipsoidSize;

			if (g_iKdTreePrimOffsetCnt >= g_iKdTreePrimOffsetCnt_Alloc) {
				_reAllocTriangleOffsetList(MyMAX(2 * g_iKdTreePrimOffsetCnt_Alloc, 512), g_iKdTreePrimOffsetCnt_Alloc, &g_pKdTreePrimOffsetArray);
			}
		}

		unsigned* currOffsetList = &g_pKdTreePrimOffsetArray[iEllipsoidOffset];
		unsigned leafCount = 0;
		for (unsigned i = 0; i < ellipsoidSize; i++) {
#if REMOVE_SMALL_PRIM
			currOffsetList[leafCount++] = leaf[i].offset;
#else
			currOffsetList[leafCount++] = pEllipsoidInfos[i].offset;
#endif
		}
#if DEBUG_LEAF_GL
		if (ellipsoidSize > 0) {
			std::vector<PrimList> leafDebug;
			leafDebug.push_back(pEllipsoidInfos[0]);
			leafDebug[0].AABB = bbox;
			for (int i = 0; i < ellipsoidSize; i++) {
				leafDebug.push_back(pEllipsoidInfos[i]);
				//leafDebug[i].AABB = bbox;
			}
			g_leafDebug.push_back(leafDebug);
		}
#endif

		if (ellipsoidSize == 0) {
			g_iKdTree_EmptyNode_Count++;
		}

		g_iKdTree_LeafNode_Count++;
		g_iKdTreeMaxPrimInLeafNodeCnt = MyMAX(g_iKdTreeMaxPrimInLeafNodeCnt, ellipsoidSize);

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

		std::vector<PrimList> leftEllipsoids;
		std::vector<PrimList> rightEllipsoids;

		const unsigned n_bEdge = 2 * ellipsoidSize;

		set_bound_edge(bestCost.axis, pEllipsoidInfos, n_bEdge, bEdge);

		push_triangles_to_child_vector(n_bEdge, bEdge, leftEllipsoids, rightEllipsoids, bestCost);

		PrimList* pLeftEllipsoids = new PrimList[leftEllipsoids.size()];
		PrimList* pRightEllipsoids = new PrimList[rightEllipsoids.size()];

		clip_ellipsoid(leftEllipsoids, bestCost, 0);
		clip_ellipsoid(rightEllipsoids, bestCost, 1);

		memcpy(pLeftEllipsoids, leftEllipsoids.data(), sizeof(PrimList) * leftEllipsoids.size());
		memcpy(pRightEllipsoids, rightEllipsoids.data(), sizeof(PrimList) * rightEllipsoids.size());

#if DEBUG_ELLIPSOID_CLIP_AABB
		if (ellipsoidSize > 0) {
			if (inNodeLevel >= g_ellipsoidClipAabbDebug.size()) {
				g_ellipsoidClipAabbDebug.resize(g_ellipsoidClipAabbDebug.size() + 1);
			}
			std::vector<PrimList> forDebug;
			for (int i = 0; i < leftEllipsoids.size(); i++) {
				forDebug.push_back(leftEllipsoids[i]);
			}
			for (int i = 0; i < rightEllipsoids.size(); i++) {
				forDebug.push_back(rightEllipsoids[i]);
			}
			g_ellipsoidClipAabbDebug[inNodeLevel].push_back(forDebug);
		}
#endif
#if DEBUG_ELLIPSOID_INTERNAL
		if (ellipsoidSize > 0) {
			if (inNodeLevel >= g_ellipsoidInternalDebug.size()) {
				g_ellipsoidInternalDebug.resize(g_ellipsoidInternalDebug.size() + 1);
			}
			vector<PrimList> forDebug;
			forDebug.push_back(pEllipsoidInfos[0]);
			forDebug[0].AABB = bbox;
			for (int i = 0; i < ellipsoidSize; i++) {
				forDebug.push_back(pEllipsoidInfos[i]);
			}
			g_ellipsoidInternalDebug[inNodeLevel].push_back(forDebug);
		}
#endif

		delete[] pEllipsoidInfos;

		build_kd_tree_recursive(bEdge, pLeftEllipsoids, leftEllipsoids.size(), leftnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum]);
		build_kd_tree_recursive(bEdge, pRightEllipsoids, rightEllipsoids.size(), rightnBounds, inNodeLevel + 1, &g_pKdTree_Node_Array[nodeNum + 1]);
	}

}
#elif PRIMITIVE_TYPE == TRI || PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
	#if PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
static int compLeafCnt = 0;
static int compCnt = 0;
static int ampledLeafCnt = 0;
	#endif
void build_kd_tree_recursive(BoundEdge* bEdge, const PrimList* pTriangleInfos, unsigned int triangleSize,
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
#if COUNT_BY_GID
	std::set<int> gIdsTris;
	for (int i = 0; i < triangleSize; i++) {
		gIdsTris.insert(pTriangleInfos[i].offset);
	}
	bestCost.cost = double(gIdsTris.size()) * v_KD_TREE_ISECT_COST;
#else
	bestCost.cost = double(triangleSize) * v_KD_TREE_ISECT_COST;
#endif

#if FORCE_SPLIT_THRESHOLD
	//#define MAX_TRIANGLE_OFFSET_BUDGET 18446744073709551615
	//#define MAX_TRIANGLE_OFFSET_BUDGET 9223372036854775807
	//#define MAX_TRIANGLE_OFFSET_BUDGET 17179869184
	//#define MAX_TRIANGLE_OFFSET_BUDGET 8589934592
	#define MAX_TRIANGLE_OFFSET_BUDGET2 4294967295
	//#define MAX_TRIANGLE_OFFSET_BUDGET 2147483647
	//#define MAX_TRIANGLE_OFFSET_BUDGET 1073741824
	//if (triangleSize > FORCE_SPLIT_THRESHOLD) bestCost.cost = DBL_MAX; //shyun added
	if (triangleSize > FORCE_SPLIT_THRESHOLD) {
	#if MAX_TRIANGLE_OFFSET_BUDGET
			// 강제 분할 전, 메모리 예산을 초과했는지 확인.
		#if SOFT_SPLIT_THRESHOLD
			if ((triangleSize < SOFT_SPLIT_THRESHOLD &&
				g_iKdTreePrimOffsetCnt > MAX_TRIANGLE_OFFSET_BUDGET) ||
				(triangleSize < SOFT_SPLIT_THRESHOLD2 &&
					g_iKdTreePrimOffsetCnt > MAX_TRIANGLE_OFFSET_BUDGET2)) {
		#else
			if (g_iKdTreePrimOffsetCnt > MAX_TRIANGLE_OFFSET_BUDGET) {
		#endif

				// 예산 초과 시: 강제 분할(DBL_MAX)을 하지 않고, 
				// SAH 비용(bestCost.cost)을 그대로 둬서 리프 노드가 되도록 함.
				fprintf(stdout, "WARNING: Memory budget exceeded (%u refs). Forcing leaf node at level %u with %u tris.\n",
					g_iKdTreePrimOffsetCnt, inNodeLevel, triangleSize);
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
#if COUNT_BY_GID
	if (inNodeLevel < v_KD_TREE_MAX_LEVEL && gIdsTris.size() > v_KD_TREE_MIN_PRIMITIVE) {
#else
	if (inNodeLevel < v_KD_TREE_MAX_LEVEL && triangleSize > v_KD_TREE_MIN_PRIMITIVE) {
#endif
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
		// Leaf node 생성

#if DEBUG_LEAF_GL
		if (triangleSize > 0) {
			std::vector<PrimList> leafDebug;
			leafDebug.push_back(pTriangleInfos[0]);
			leafDebug[0].AABB = bbox;
			for (int i = 0; i < triangleSize; i++) {
				leafDebug.push_back(pTriangleInfos[i]);
			}
			g_leafDebug.push_back(leafDebug);
		}
#endif
#if PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
		std::vector<int> leafOffset;
		std::set<int> gIds;
		for (int i = 0; i < triangleSize; i++) {
			gIds.insert(pTriangleInfos[i].offset);
		}
		leafOffset.insert(leafOffset.end(), gIds.begin(), gIds.end());
		if (leafOffset.size() < triangleSize) {
			compLeafCnt++;
			compCnt += triangleSize - leafOffset.size();
		}
		else if (leafOffset.size() > triangleSize) {
			ampledLeafCnt++;
		}
		triangleSize = leafOffset.size();
#endif
		unsigned int iTriOffset;
		{
			iTriOffset = g_iKdTreePrimOffsetCnt;
			setLeafNode(inNode, triangleSize, g_iKdTreePrimOffsetCnt);
			g_iKdTreePrimOffsetCnt += triangleSize;

			// 메모리 체크 : Triangle Offset Size
			if (g_iKdTreePrimOffsetCnt >= g_iKdTreePrimOffsetCnt_Alloc) {
				_reAllocTriangleOffsetList(MyMAX(2 * g_iKdTreePrimOffsetCnt_Alloc, 512), g_iKdTreePrimOffsetCnt_Alloc, &g_pKdTreePrimOffsetArray);
			}
		}

		/**
		 *	leaf node가 참조하는 triangle의 offset 을 offsetList 마지막에 추가해 넣는다.
		 */
		unsigned* currOffsetList = &g_pKdTreePrimOffsetArray[iTriOffset];

		unsigned leafCount = 0;
		for (unsigned i = 0; i < triangleSize; i++) {
#if PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
			currOffsetList[leafCount++] = leafOffset[i];
#else
			currOffsetList[leafCount++] = pTriangleInfos[i].offset;
#endif
		}

		if (triangleSize == 0)
			g_iKdTree_EmptyNode_Count++;

		g_iKdTree_LeafNode_Count++;
		g_iKdTreeMaxPrimInLeafNodeCnt = MyMAX(g_iKdTreeMaxPrimInLeafNodeCnt, triangleSize);

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
		std::vector<PrimList> leftTriangles;
		std::vector<PrimList> rightTriangles;
		// 예상 크기만큼 미리 예약하여 성능 저하 최소화
		leftTriangles.reserve(bestCost.n_left);
		rightTriangles.reserve(bestCost.n_right);

		const unsigned n_bEdge = 2 * triangleSize;

		// TriangleInfo 부터 bEdge 를 생성 및 정렬
		set_bound_edge(bestCost.axis, pTriangleInfos, n_bEdge, bEdge);

		// bEdge 로 부터 pLeftTriangle, pRightTriangle 을 생성
		push_triangles_to_child_vector(n_bEdge, bEdge, leftTriangles, rightTriangles, bestCost);

		PrimList* pLeftTriangles = new PrimList[leftTriangles.size()];
		PrimList* pRightTriangles = new PrimList[rightTriangles.size()];
		memcpy(pLeftTriangles, leftTriangles.data(), sizeof(PrimList)* leftTriangles.size());
		memcpy(pRightTriangles, rightTriangles.data(), sizeof(PrimList)* rightTriangles.size());

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

	if (DEBUG_FLAG) {
		fprintf(stdout, "b_k_t_r: (E)triangleSize = %d, inNodeLevel = %d\n", triangleSize, inNodeLevel);
	}
#if PRIMITIVE_TYPE == ELLIPSOID_BY_TRI
	if (inNodeLevel == 0) {
		printf("compressed by ellipsoid leaf count: %d\n", compLeafCnt);
		printf("compressed by ellipsoid  count: %d\n", compCnt);
		printf("ampled by ellipsoid leaf count(error): %d\n", ampledLeafCnt);
	}
#endif
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
	unsigned int* prim_offset_list = c_object->kd_tree->prim_offset_list;

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

	unsigned int max_prims_found = 0;
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
			if (num_triangles <= 0) continue;

			for (unsigned int i = 0; i < num_triangles; ++i) {
				leaf.primIndices.push_back(prim_offset_list[offset + i]);
			}

			all_leaf_info.push_back(leaf);
			// --------------------
			if (num_triangles > max_prims_found) {
				max_prims_found = num_triangles;
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
		printf("[INFO] Largest leaf found at index %d with %u primitives.\n",
			largest_leaf_index, max_prims_found);
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
