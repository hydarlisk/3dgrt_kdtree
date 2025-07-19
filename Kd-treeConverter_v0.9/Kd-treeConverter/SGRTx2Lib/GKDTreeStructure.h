#pragma once

#include "GBase.h"
#include "GScene.h"
#include "GSpatialStructure.h"
#include "GBoundingBox.h"
#include "GTriangleWrapper.h"
#include "GTriangleWrapperList.h"
#include "GKDTreeNode.h"
#include "GKDTreeOption.h"
#include "cudaRenderPipeline.h"
//#include "SSERenderPipeline.h"
#include "SSERenderData.h"
#include <float.h>

/**
 *	KD Tree 를 이용한 공간 구조체.
 *	KD Tree 생성. 탐색은 오상락군의 코드를 기부 받아
 *	수정함.
 *
 *	by graphicsian.
 */

struct SplitCost {

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

	SplitCost( void ): 
		cost(DBL_MAX), splitPos(FLT_MAX), 
		axis(-1), n_onlyLeft(0), n_onlyRight(0), n_cross(0), n_planar(0), n_left(0), n_right(0), planar_side(-1){}
	bool is_valid() const { return axis >= 0; }

};

class  GKDTreeStructure : public GSpatialStructure
{
private:
	GBoundingBox m_SceneBBox;
	GScene *m_pScene;
	int m_iSceneTriangleCount;
	GTriangleWrapperList *m_pSceneTriangleList;
	
	kdtreeNode	*m_pKDTreeNodes;
	unsigned int m_iKDTreeNodeCount;
	unsigned int m_iAllocatedkdNodeCount;

	unsigned int *m_pTriangleOffsetList;
	unsigned int m_iAllocatedTriangleOffsetSize;
	unsigned int m_iCurrentTriangleOffset;

	unsigned int m_iMaxTreeLevel;
	unsigned int m_iTreeLevel;
	unsigned int m_iMinObjPerLeafNode;

	unsigned int m_iEmptyLeafCount, m_iLeafNodeCount;
	unsigned int m_iLeafMaxTriangleCount;

	float m_fEmptyBonus;
	float m_fTraversalCost;
	float m_fIntersectionCost;
	float m_fExtraTraversalCost;

	GKDTreeOption *m_pKDTreeOption;

public:
	GKDTreeStructure( GScene *pScene );
	virtual ~GKDTreeStructure(void);

	void setKDTreeOption( GKDTreeOption *pKDTreeOption );

	virtual GError initialize();
	virtual GBoundingBox getBoundingBox();

	virtual GError makeCudaRenderStructureInfo( cudaRenderPipeline *pCudaPipeline );
	virtual GError makeSSERenderStructureInfo( SSESceneData *pSSEData );

	virtual GTriangleWrapper *getTriangleWrapper( const unsigned int triIndex );
	virtual GTriangleWrapperList *getTriangleWrapperList();
	virtual int getTriangleCount();

	kdtreeNode* getRootNode() { return &m_pKDTreeNodes[0]; }

	typedef struct spbean{
		float t;
		int flag;
	}spbean;

	bool loadStructureFromFile( const char *filename );
	bool saveStructureToFile( const char *filename );
	bool loadfromFile_SAH     ( const char *filename );			// load kdtree from file (SAH)
	bool loadfromFile_EmptySAH( const char *filename );			// load kdtree from file (Empty SAH)

	//shyun
	unsigned int getKdTreeNodeCount() { return m_iKDTreeNodeCount; }
	void setKdTreeNodeCount(unsigned int cnt) { m_iKDTreeNodeCount = cnt; }
	kdtreeNode* getKdTreeNode() { return m_pKDTreeNodes; }
	void setKdTreeNode(kdtreeNode* kdtree) { m_pKDTreeNodes = kdtree; }
	unsigned int getTriangleOffset() { return m_iCurrentTriangleOffset; }
	void setTriangleOffset(unsigned int triangleOffset) { m_iCurrentTriangleOffset = triangleOffset; }
	unsigned int* getTriangleOffsetList() { return m_pTriangleOffsetList; }
	void setTriangleOffsetList(unsigned int* triangleOffsetList) { m_pTriangleOffsetList = triangleOffsetList; }
	void setSceneTriangleCount(int cnt) { m_iSceneTriangleCount = cnt; }
	void setSceneTriangleList(GTriangleWrapperList* triList) { m_pSceneTriangleList = triList; }

	void setBBoxMin(const GVector& vmin) { m_SceneBBox.setMin(vmin); }
	void setBBoxMax(const GVector& vmax) { m_SceneBBox.setMax(vmax); }
	//shyun end

protected:
	GError uninitialize();
	void buildKDTree(
			BoundEdge *bEdge, const TriangleInfo *pTriangleInfos, unsigned int triangleSize,
			GBoundingBox bbox, unsigned int inNodeLevel, kdtreeNode *inNode );

	inline cuPlueckerTriangleInfo toCuPlueckerTriangleInfo( GTriangleWrapper &tri );
	inline cuWaldTriangleInfo toCuWaldTriangleInfo( GTriangleWrapper &tri );
	inline void setInnerNode( kdtreeNode* pNode, int _splitAxis, unsigned int _firstChildOffset, float _splitPos );
	inline void setLeafNode( kdtreeNode* pNode, unsigned int _objectSize, unsigned int _objectListOffset );

	void setBoundEdgeList( const int axis, const TriangleInfo *pTriangleInfo, 
						   const unsigned int n_bEdge, BoundEdge *bEdge );
	void setBoundEdgeList2( const int axis, const TriangleInfo *pTriangleInfo, 
						   const unsigned int n_bEdge, BoundEdge *bEdge, spbean *bean );

	//! Not function pointer as parameter
	void setSplitFunction( SPLIT_FUNCTION SplitFunctionEnum );
	void splitWithSAH( const int axis, GBoundingBox inBBox, 
								 const TriangleInfo *pTriangles, const int triangleSize, 
								 BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly = false );
	void splitWithSAH_ExtraCost( const int axis, GBoundingBox inBBox, 
								 const TriangleInfo *pTriangles, const int triangleSize, 
								 BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly = false );
	void splitWithVisibility( const int axis, GBoundingBox inBBox, 
								 const TriangleInfo *pTriangles, const int triangleSize, 
								 BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly = false );
	void tryEmptySplit( const int axis, GBoundingBox inBBox, 
				   const TriangleInfo *pTriangleInfos, 
				   const int triangleSize, BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly = false );

	//! Function Pointer
	void (GKDTreeStructure::*splitFunction)( const int axis, GBoundingBox inBBox, 
				   const TriangleInfo *pTriangleInfos, 
				   const int triangleSize, BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly );

	bool interSectEdgePlane( GPoint &p0,	
						     GPoint &p1, 
							 GPoint &planePoint,
							 GVector &planeNorm,
							 GPoint &hitPoint );

	void splitClipping( const int triangleSize, 
					    const SplitCost &bestCost, 
					    TriangleInfo *pTriangleInfos, 
					    int side );

	inline void pushChildTriangles( const unsigned n_bEdge, 
								    const BoundEdge *bEdge, 
								    TriangleInfo *pLeftTriangles, 
								    TriangleInfo *pRightTriangles, 
								    const SplitCost &bestCost );
	inline void reAllocTriangleOffsetList(  unsigned int _newAllocSize );
	inline void reAllocKdtreeNodes(  unsigned int _newAllocSize );

	static int compare( const void *elem0, const void *elem1 );

};
