/**
 *	Traversal 및 Intersection Test 관련 가속화용 자료구조
 *	등등.
 *	
 *	by oipini.
 */
#ifndef _SSE_RENDER_DATA_H_
#define _SSE_RENDER_DATA_H_

#include "GBase.h"
#include "SSERenderCommon.h"
#include "GCriticalSection.h"

class  GBVHNode;

class SSESceneData
{
public:
	SSESceneData(Scene* a_Scene);
	~SSESceneData();

	GError setTriangleObjectList( GTriangleWrapperList *pSceneTriangleList );
	GError buildTriAccList_Barycentric( void );
	GError buildTriAccList_Pluecker( void );

	unsigned int		m_TriObjCnt;		// 전체 삼각형 개수
	GTriangleWrapper**	m_TriObjList;		// 전체 삼각형 리스트
	TriAccel*			m_TriAccList;		// Barycentric Triangle Accel (for intersection test)
	TriAccel_P*			m_TriAccList_P;		// Pluecker    Triangle Accel (for intersection test)
	
	GBoundingBox		m_SceneBBox;

public:
	//-------------------------------------------------------------------------------------
	// Kd-Tree
	//-------------------------------------------------------------------------------------
	GError setKDTreeNodeData( kdtreeNode *pRootNode, int nodeCount, GBoundingBox sceneBox );
	GError setTriangleOffsetList( unsigned int *pTriangleOffsetList, int offsetCount );
	KdTreeNode*			m_pKDTreeNodes;		// Pointer of Kd-Tree root node 
	unsigned int*		m_TriOffList;		// Kd-Tree 의 모든 leaf node 에 있는 Triangle ID list

	//-------------------------------------------------------------------------------------
	// BVH
	//-------------------------------------------------------------------------------------
	GError setBVHNodeData( GBVHNode* pRootNode, GBoundingBox sceneBox );
	GBVHNode*           m_pBVHNodes;

	//-------------------------------------------------------------------------------------
	// Grid
	//-------------------------------------------------------------------------------------


public:
	GError calSceneProperty( void );
	int m_TriCnt_Refl;
	int m_TriCnt_Refr;
	int m_TriCnt_Tex;
	float m_TriAreaSum_Refl;
	float m_TriAreaSum_Refr;
	float m_TriAreaSum_Tex;
	float m_TriAreaSum_All;

private:
	Scene*				m_Scene;
	int					m_nKDTreeNodeCount;
};

#endif
