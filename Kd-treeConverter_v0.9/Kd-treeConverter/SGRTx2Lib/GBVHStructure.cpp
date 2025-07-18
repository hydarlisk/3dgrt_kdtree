#include "GBVHStructure.h"
#include <stack>
#include <algorithm>


GBVHStructure::GBVHStructure(const GBoundingBox& aabbScene)
	:m_pnodeRoot(0), m_totTriCount(0)
{
	// NOP 
}

GBVHStructure::GBVHStructure(GScene *pScene)
	:m_pnodeRoot(0), m_totTriCount(0)
{
	m_pScene = pScene;
}

GBVHStructure::~GBVHStructure()
{
	delete m_pnodeRoot;
}

// BVH construct
GError 
GBVHStructure::initialize()
{
	/*
	const unsigned int nPoly = m_totTriCount;// = m_Scenedata->m_TriObjCnt;
		
	// precalc. centroid and AABB of all polys
	GVector* aryvCentroid  = new GVector[nPoly];
	GBoundingBox* aryGAABB = new GBoundingBox[nPoly];
	
	pre_calculate(aryvCentroid, aryGAABB);	

	// construct BVH
	constructBVH(aryvCentroid, aryGAABB);

	delete[] aryvCentroid;
	delete[] aryGAABB;

	return errorNo;
*/

	//! Initialization (reset)	
	m_pSceneTriangleList = NULL;
	//! Start Timer
	GTimer timer;
	timer.start();

	GLogManager::logging( LOG_INFO, "------------------------ BVH Spatial Structure -------------------" );
	GLogManager::logging( LOG_INFO, " -> BVH build started..." );

	m_pSceneTriangleList = m_pScene->createSceneTriangleList( m_aabbScene );
	m_totTriCount = m_pSceneTriangleList->size();

	GVector* aryvCentroid = new GVector     [m_totTriCount];
	GBoundingBox* aryAABB = new GBoundingBox[m_totTriCount];
	
	pre_calculate(aryvCentroid, aryAABB);

	// construct BVH
	constructBVH(aryvCentroid, aryAABB);

	delete[] aryvCentroid;
	delete[] aryAABB;

	timer.end();

	GLogManager::logging( LOG_INFO, " -> BVH build end." );
	GLogManager::logging( LOG_INFO, " -> BVH Construction Time : %f sec", timer.getElapsedTime() );
	GLogManager::logging( LOG_INFO, "--------------------------------------------------------------------" );
/*
	//! Initialization (reset)
	m_iSceneTriangleCount = 0;
	m_iTreeLevel = 0;
	m_pKDTreeNodes = NULL;
	m_pTriangleOffsetList = NULL;
	m_pSceneTriangleList = NULL;

	m_iLeafNodeCount = 0;
	m_iCurrentTriangleOffset = 0;
	m_iTreeLevel = 0;
	m_iEmptyLeafCount = 0;
	m_iLeafMaxTriangleCount = 0;

	//! Start Timer
	GTimer timer;
	timer.start();

	GLogManager::logging( LOG_INFO, "------------------------ KDTree Spatial Structure -------------------" );
	GLogManager::logging( LOG_INFO, " -> KDTree build started..." );

	
	// *	Scene 전체의 삼각형 list 를 구성해 온다.
	m_pSceneTriangleList = m_pScene->precalcBVHTriangleList( aryvCentroid, aryGAABB, m_aabbScene );
	m_iSceneTriangleCount = m_pSceneTriangleList->size();

	 //*	KD-Tree 를 위한 데이터 구성. 
	 //*	모든 삼각형의 정렬을 위한 공간.offset 정보는 
	 //*	m_SceneTriangleList vector 안의 index 와 동일하다.
	 //
	TriangleInfo *pTriangleInfos = new TriangleInfo[ m_iSceneTriangleCount ];
	for( int i = 0; i < m_iSceneTriangleCount; i++ ) {
		pTriangleInfos[ i ].offset = i;
		pTriangleInfos[ i ].pTriangleWrapper = m_pSceneTriangleList->getTriangleWrapper( i );
		// bounding box 는 실제 triangle bouding box 과는 다를수 있으므로 따로 저장관리 
		pTriangleInfos[ i ].boundingBox = m_pSceneTriangleList->getTriangleWrapper( i )->m_BBox;
	}

	BoundEdge *bEdge = new BoundEdge[ m_iSceneTriangleCount * 2 ];
	memset( bEdge, 0x00, sizeof( BoundEdge ) * m_iSceneTriangleCount * 2 );
	
	m_iAllocatedkdNodeCount = 524288;
	m_pKDTreeNodes = new kdtreeNode[ m_iAllocatedkdNodeCount ];
	m_iKDTreeNodeCount = 1; 

	m_iAllocatedTriangleOffsetSize = 1048576;
	m_pTriangleOffsetList = new unsigned int [ m_iAllocatedTriangleOffsetSize ];
	memset( m_pTriangleOffsetList, 0x00, sizeof( unsigned int ) * m_iAllocatedTriangleOffsetSize );

	 //*	pTraangleInfos 는 buildKDTree 안에서 사용하고 없앤다.
	 
	buildKDTree( bEdge, pTriangleInfos, m_iSceneTriangleCount, m_SceneBBox, 0, &(m_pKDTreeNodes[0]) );
	
	delete[] bEdge;

	timer.end();

	GLogManager::logging( LOG_INFO, " -> KDTree build end." );
	GLogManager::logging( LOG_INFO, " -> KDTree Construction Time : %f sec", timer.getElapsedTime() );
	GLogManager::logging( LOG_INFO, " -> Scene Bounding Box : ( %f, %f, %f ) - ( %f, %f, %f )", 
										m_SceneBBox.getMin().x,  m_SceneBBox.getMin().y, m_SceneBBox.getMin().z, 
										m_SceneBBox.getMax().x,  m_SceneBBox.getMax().y, m_SceneBBox.getMax().z );
	GLogManager::logging( LOG_INFO, " -> ObjectOffsetCount: %d (%fMB)", 
									m_iCurrentTriangleOffset, 
									sizeof(unsigned)*m_iCurrentTriangleOffset / ( 1024.f * 1024.f ) );
	GLogManager::logging( LOG_INFO, " -> KDTree Node Count: %d (%fMB)", 
									m_iKDTreeNodeCount, sizeof(kdtreeNode)*m_iKDTreeNodeCount/(1024.f*1024.f));
	GLogManager::logging( LOG_INFO, " -> n_leafNode: %d", m_iLeafNodeCount);
	GLogManager::logging( LOG_INFO, " -> treeLevel: %d", m_iTreeLevel);
	GLogManager::logging( LOG_INFO, " -> maxLeafSize: %d", m_iLeafMaxTriangleCount);
	GLogManager::logging( LOG_INFO, " -> n_emptyLeaf: %d (%f %%%%%%%)", m_iEmptyLeafCount, 
										double(m_iEmptyLeafCount)/double(m_iLeafNodeCount)*100.0);
	GLogManager::logging( LOG_INFO, "--------------------------------------------------------------------" );
*/
	return errorNo;
}

void
GBVHStructure::pre_calculate(GVector* aryvCentroid, GBoundingBox* aryGAABB)
{
	GTriangleWrapper* TriData; 
		
	for(unsigned int i = 0; i< m_totTriCount; ++i)
	{
		TriData = m_pSceneTriangleList->getTriangleWrapper(i);
		
		// 삼각형마다 AABB를 구해서 저장.
		aryGAABB[i] = TriData->m_BBox;
		
		// 삼각형 중심(무게 중심) 좌표 저장.
		TriData->calCentroid(aryvCentroid[i]);		
	}
}

#define AXIS_X    0
#define AXIS_Y    1
#define AXIS_Z    2
#define NUM_AXIS  3

#define AABB_COST 0.3f
#define TRI_COST  0.7f

#define BVH_MAX_DEPTH 100

typedef enum
{
	SPLIT_X   = 0,
	SPLIT_Y   = 1,
	SPLIT_Z   = 2,
	MAKE_LEAF = 3
} BVHEVENT;

class GBVHStructure::compareCentroidX
{
public:
	compareCentroidX(const GVector* aryvCentroid)
		: m_aryvCentroid(aryvCentroid)
	{ /* NOP */ }

	bool
	operator()(const int a, const int b)
	{
		return (m_aryvCentroid[a].x < m_aryvCentroid[b].x);
	}

private:
	const GVector* const m_aryvCentroid;
};

class GBVHStructure::compareCentroidY
{
public:
	compareCentroidY(const GVector* aryvCentroid)
		: m_aryvCentroid(aryvCentroid)
	{ /* NOP */ }

	bool
	operator()(const int a, const int b)
	{
		return (m_aryvCentroid[a].y < m_aryvCentroid[b].y);
	}

private:
	const GVector* const m_aryvCentroid;
};

class GBVHStructure::compareCentroidZ
{
public:
	compareCentroidZ(const GVector* aryvCentroid)
		: m_aryvCentroid(aryvCentroid)
	{ /* NOP */ }

	bool
	operator()(const int a, const int b)
	{
		return (m_aryvCentroid[a].z < m_aryvCentroid[b].z);
	}

private:
	const GVector* const m_aryvCentroid;
};

inline
float
GBVHStructure::SAH(
	const unsigned int nLeft, const float fLeftArea,
	const unsigned int nRight, const float fRightArea,
	const float fAABBArea)
{
	return
		  2.0f * AABB_COST
		+ (fLeftArea * (float)nLeft + fRightArea * (float)nRight) / fAABBArea * TRI_COST;
}

void
GBVHStructure::constructBVH(const GVector* aryvCentroid, const GBoundingBox* aryAABB)
{
	const unsigned int nPoly = m_totTriCount;

	// ** create temporary buffers
	SVPOLYINDEX* temparyPoly[NUM_AXIS];
	
	// 축별로 polygon 수 만큼 저장 장소 생성.
	for(unsigned int axis = 0; axis < NUM_AXIS; ++ axis)
	{
		temparyPoly[axis] = new SVPOLYINDEX(nPoly);
	}
	// leftArea를 삼각형 개수만큼 생성. aryfLeftArea = S1
	float* aryfLeftArea = new float[nPoly];


	// push root node onto stack
	{
		//x 축 인덱스 설정.
		SVPOLYINDEX* pbaseset = temparyPoly[0];
		for(unsigned int i = 0; i < nPoly; ++ i)
		{			
			(*pbaseset)[i] = i;
		}		
		// X축 중심으로 polygon들 sorting
		sort(temparyPoly[0]->begin(), temparyPoly[0]->end(), compareCentroidX(aryvCentroid));			
	}	

	std::stack<GBVHNode*> stackpNode;
	std::stack<SVPOLYINDEX*> stackpBaseSet;
	std::stack<int> stacknStart;
	std::stack<int> stacknEnd;
	
	m_pnodeRoot = new GBVHNode(m_aabbScene); // 전체 AABB
	
	stackpNode.push(m_pnodeRoot);
	stackpBaseSet.push(temparyPoly[0]);
	stacknStart.push(0);
	stacknEnd.push(nPoly);

	// main loop
	while( !stackpNode.empty() )
	{
		// pop next work from stack
		GBVHNode* pNode = stackpNode.top(); stackpNode.pop();
		SVPOLYINDEX* pbaseset = stackpBaseSet.top(); stackpBaseSet.pop();
		int nStart = stacknStart.top(); stacknStart.pop();
		int nEnd = stacknEnd.top(); stacknEnd.pop();

		// determine what type of node this node should be.
		// -- first candidate is making a leaf
		float fBestCost = (nEnd - nStart) * TRI_COST;
		BVHEVENT eventBest = MAKE_LEAF;
		int nBestSplitPos = 0;

		// if #tri > 2
		if(nEnd - nStart > 2 && stackpNode.size() < BVH_MAX_DEPTH)
		{
			const float fAreaNodeAABB = pNode->getAABB().calcSurfaceArea();			
			// -- try splitting axis
			for(unsigned int axis = 0; axis < NUM_AXIS; ++ axis)
			{
				SVPOLYINDEX& currset = *(temparyPoly[axis]);

				// prepare currset
				{
					for(int i = nStart; i < nEnd; ++ i)
					{
						currset[i] = (*pbaseset)[i];
					}

					switch( axis )
					{
					case AXIS_X:
						sort(currset.begin() + nStart, currset.begin() + nEnd, compareCentroidX(aryvCentroid));
						break;
					case AXIS_Y:
						sort(currset.begin() + nStart, currset.begin() + nEnd, compareCentroidY(aryvCentroid));
						break;
					case AXIS_Z:
						sort(currset.begin() + nStart, currset.begin() + nEnd, compareCentroidZ(aryvCentroid));
						break;
					}
				}

				// setup aryfLeftArea array [nStart, nEnd)
				GBoundingBox aabbLeft;
				aabbLeft = aryAABB[currset[nStart]];
				aryfLeftArea[nStart] = aabbLeft.calcSurfaceArea();
				for(int i=nStart+1; i<nEnd; ++i)
				{
					// update aabbLeft
					const GBoundingBox& aabb = aryAABB[currset[i]];
					aabbLeft.tryupdateMax(aabb.m_Max);
					aabbLeft.tryupdateMin(aabb.m_Min);

					aryfLeftArea[i] = aabbLeft.calcSurfaceArea();
				}				

				// try partitioning from nEnd-2 -> nStart
				// ( We know that nEnd-1 will not work, don't we? ;-) )
				int index = (nEnd-2);				
				GBoundingBox aabbRight;
				aabbRight = aryAABB[currset[index+1]];
				const float fRightArea = aabbRight.calcSurfaceArea();

				const float fCost = SAH( index-nStart, aryfLeftArea[index], 1, fRightArea, fAreaNodeAABB );
				
				if(fCost < fBestCost)
				{
					fBestCost = fCost;
					eventBest = BVHEVENT(axis);
					nBestSplitPos = index;
				}
				for(int i = nEnd - 3; i >= nStart; -- i)
				{					
					// update aabbRight
					const GBoundingBox& aabb = aryAABB[currset[i+1]];
					aabbRight.tryupdateMin(aabb.m_Min);
					aabbRight.tryupdateMax(aabb.m_Max);

					const float fRightArea = aabbRight.calcSurfaceArea();

					const float fCost = SAH( i-nStart, aryfLeftArea[i],	nEnd-i-1, fRightArea, fAreaNodeAABB	);

					if(fCost < fBestCost)
					{
						fBestCost = fCost;
						eventBest = BVHEVENT(axis);
						nBestSplitPos = i;
					}
				}
			}
		}

		// execute the "best event"
		if(eventBest == MAKE_LEAF)
		{
			// make the node leaf node
			const unsigned int nPolyLeaf = nEnd - nStart;
			SVPOLYINDEX& svpolyidx = pNode->leafNode(nPolyLeaf);
			
			for(int i = nStart; i < nEnd; ++ i)
			{
				svpolyidx[i-nStart] = (*pbaseset)[i];
			}
		}
		
		else
		{
			SVPOLYINDEX& currset = *(temparyPoly[eventBest]);

			// make child nodes
			// -- determine FAABBs of child nodes
			GBoundingBox aabbLeftA = aryAABB[currset[nStart]];
			for(int i = nStart + 1; i < (nBestSplitPos + 1); ++ i)
			{
				// update aabbLeft
				const GBoundingBox& aabb = aryAABB[currset[i]];
				aabbLeftA.tryupdateMin(aabb.m_Min);
				aabbLeftA.tryupdateMax(aabb.m_Max);
			}

			GBoundingBox aabbRightB = aryAABB[currset[nBestSplitPos+1]];
			for(int i = nBestSplitPos + 2; i < nEnd; ++ i)
			{
				// update aabbRight
				const GBoundingBox& aabb = aryAABB[currset[i]];
				aabbRightB.tryupdateMin(aabb.m_Min);
				aabbRightB.tryupdateMax(aabb.m_Max);
			}

			GBVHNode* pChildNode = pNode->createChilds((unsigned char)eventBest, aabbLeftA, aabbRightB);
			
			// push childs onto the stack
			// -- second child. second child will be right side: [nBestSplitPos + 1, nEnd)
			stackpNode.push(pChildNode + 1);
			stackpBaseSet.push(&currset);
			stacknStart.push(nBestSplitPos + 1);
			stacknEnd.push(nEnd);

			// -- first child. first child will be left side : [nStart, nBestSplitPos] -> [nStart, nBestSplitPos + 1)
			stackpNode.push(pChildNode);
			stackpBaseSet.push(&currset);
			stacknStart.push(nStart);
			stacknEnd.push(nBestSplitPos + 1);
		}
	};

	// ** destroy temporary buffer
	for(unsigned int axis = 0; axis < NUM_AXIS; ++ axis)
	{
		delete temparyPoly[axis];
	}
	delete[] aryfLeftArea;

	GLogManager::logging( LOG_DEBUG, "bvh construction complete" );
}


/**
 *	SSE 클래스 에서 참조할 BVH 포인터 복사.
 */
GError 
GBVHStructure::makeSSERenderStructureInfo( SSESceneData *pSSEData )
{
	GError error;

	if ( m_pScene->getObjectCount() <= 0 || m_totTriCount <= 0 )
		return errorUnknown;

	 //	BVH 세팅.
	error = pSSEData->setBVHNodeData( m_pnodeRoot, m_aabbScene );
	if ( error != errorNo )
		return error;
	
	 //Triangle Object List 세팅
	error = pSSEData->setTriangleObjectList( m_pSceneTriangleList );
	if ( error != errorNo )
		return error;

	 //	Triangle Accel List 세팅
	error = pSSEData->buildTriAccList_Barycentric();
	//error = pSSEData->buildTriAccList_Pluecker();
	if ( error != errorNo )
		return error;

	return errorNo;
}