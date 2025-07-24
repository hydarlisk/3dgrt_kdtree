// -----------------------------------------------------------
// raytracer.cpp
// 2008 - oipini
// -----------------------------------------------------------

//#include "raytracer.h"
//#include "scene.h"
//#include "kdtree.h"
//#include "texture.h"	// Texture t->init
//#include "light.h"
//#include "memory.h"
#include "GScene.h"
#include "GRenderSystem.h"


#include <stdio.h>
#include "SSE_math.h"
#include "SSERenderData.h"

#pragma warning ( disable : 4068 )
#pragma warning ( disable : 949 )

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

const unsigned int modulo[] =  {0,1,2,0,1};

// ------------------------------------------------------------------------------------------------
// Engine::Engine
//		Allocate memory & Initialize values
// ------------------------------------------------------------------------------------------------
SSESceneData::SSESceneData(Scene* a_Scene) : m_Scene(a_Scene)
{
	// Memory Allocation
	m_nKDTreeNodeCount	= 0;
	m_pKDTreeNodes		= NULL;

	m_TriObjCnt			= 0;
	m_TriObjList		= NULL;
	m_TriOffList		= NULL;
	m_TriAccList		= NULL;
	m_TriAccList_P		= NULL;
}

SSESceneData::~SSESceneData()
{
	if (m_TriObjList)   _aligned_free(m_TriObjList);
	if (m_TriAccList)   _aligned_free(m_TriAccList);
	if (m_TriAccList_P) _aligned_free(m_TriAccList_P);
}

// *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
// Preparation for rendering
//
//		TriAcc precomputation
//		KdTree & Triangle Object data setting
// *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
GError SSESceneData::setKDTreeNodeData( KdTreeNode2 *pRootNode, int nodeCount, GBoundingBox sceneBox  )
{
	m_pKDTreeNodes		= pRootNode;
	m_nKDTreeNodeCount	= nodeCount;
	m_SceneBBox			= sceneBox;
	return errorNo;
}

GError SSESceneData::setTriangleOffsetList( unsigned int *pTriangleOffsetList, int offsetCount )
{
	m_TriOffList = pTriangleOffsetList;
	return errorNo;
}

GError SSESceneData::setTriangleObjectList( GTriangleWrapperList *pSceneTriangleList )
{
	m_TriObjCnt		= pSceneTriangleList->size();
	m_TriObjList	= (GTriangleWrapper**)	_aligned_malloc(m_TriObjCnt	* sizeof(GTriangleWrapper*), 16);
	m_TriAccList	= (TriAccel2*)			_aligned_malloc(m_TriObjCnt	* sizeof(TriAccel2), 16);
//	m_TriAccList_P	= (TriAccel_P*)			_aligned_malloc(m_TriObjCnt	* sizeof(TriAccel_P), 16);

	unsigned int i;
	for (i = 0; i < m_TriObjCnt; i++) {
		m_TriObjList[i] = (*pSceneTriangleList)[i];
	}
	return errorNo;
}

GError SSESceneData::buildTriAccList_Barycentric ( void )
{
	GPoint	A, B, C;
	GPoint	b, c;
	GPoint	N;
	int k, u, v;
	float r;

	unsigned int i;
	for (i = 0; i < m_TriObjCnt; i++) {
		m_TriObjList[i]->getPoint(A, B, C);

		// calc edges and normal
		b = C-A; c = B-A; N = b.outerProduct(c);

		N = N.normalize();
		m_TriAccList[i].N = N;

		k = (fabsf(N[0]) > fabsf(N[1])) ? ( (fabsf(N[0])>fabsf(N[2]))?0:2 ):( (fabsf(N[1])>fabsf(N[2]))?1:2 );
		u = modulo[k+1];
		v = modulo[k+2];

		N /= N[k];										// N'
		m_TriAccList[i].k   = k;
		m_TriAccList[i].n_u = N[u];						// N'u
		m_TriAccList[i].n_v = N[v];						// N'v
		m_TriAccList[i].n_d = N.innerProduct(A);		// A dot N'

		r = 1.f / (b[u] * c[v] - b[v] * c[u]);
		// beta
		m_TriAccList[i].b_nu = r * -b[v];
		m_TriAccList[i].b_nv = r *  b[u];
		m_TriAccList[i].b_d  = r * (b[v] * A[u] - b[u] * A[v]);
		// gamma
		m_TriAccList[i].c_nu = r *  c[v];
		m_TriAccList[i].c_nv = r * -c[u];
		m_TriAccList[i].c_d  = r * (c[u] * A[v] - c[v] * A[u]);

		m_TriAccList[i].mbox = 0;
		m_TriAccList[i].isTransparent = (m_TriObjList[i]->m_pObject->getMaterial()->getTransparency() > 0)?1:0;

		//m_TriAccList[i].k = 2;
		//m_TriAccList[i].isTransparent = 1;
		//m_TriAccList[i].mbox = 0;

		//int h = m_TriAccList[i].k;
		//int g = m_TriAccList[i].mbox;
		//int f = m_TriAccList[i].isTransparent;

		//m_TriAccList[i].k = 1;
		//m_TriAccList[i].isTransparent = 0;
		//m_TriAccList[i].mbox = 30;

		// h = m_TriAccList[i].k;
		// g = m_TriAccList[i].mbox;
		// f = m_TriAccList[i].isTransparent;
sizeof (_sse_vec);

		// 프로파일링용
		//m_TriAccList[i].area = m_TriObjList[i]->calArea();
		m_TriAccList[i].pObject = m_TriObjList[i]->m_pObject;
		m_TriAccList[i].indexInObject = m_TriObjList[i]->indexInObject;
	}

	return errorNo;
}

GError SSESceneData::buildTriAccList_Pluecker ( void )
{
	GPoint	A, B, C;
	GPoint	b, c;
	GPoint	N;
	int k, u, v;
	float w;

	unsigned int i;
	for (i = 0; i < m_TriObjCnt; i++) {
		m_TriObjList[i]->getPoint(A, B, C);

		// calc edges and normal
		b = C-A; c = B-A; N = b.outerProduct(c);

		N = N.normalize();
		m_TriAccList_P[i].N = N;

		k = (fabsf(N[0]) > fabsf(N[1])) ? ( (fabsf(N[0])>fabsf(N[2]))?0:2 ):( (fabsf(N[1])>fabsf(N[2]))?1:2 );
		u = modulo[k+1];
		v = modulo[k+2];
//		if  (u > v) { int t = u; u = v; v = t; }

		//w = fabsf(N[k]);
		w = N[k];
		N /= w;											// N'
		m_TriAccList_P[i].k   = k;
		m_TriAccList_P[i].n_u = N[u];					// N'u
		m_TriAccList_P[i].n_v = N[v];					// N'v
		m_TriAccList_P[i].n_d = N.innerProduct(A);		// P dot N'

		m_TriAccList_P[i].p_u = A[u];					// P_u
		m_TriAccList_P[i].p_v = A[v];					// P_v

		m_TriAccList_P[i].e0u = b[u]/w;// * ((k == 1)?-1.0f:1.0f);
		m_TriAccList_P[i].e0v = b[v]/w;// * ((k == 1)?-1.0f:1.0f);
		m_TriAccList_P[i].e1u = c[u]/w;// * ((k == 1)?-1.0f:1.0f);
		m_TriAccList_P[i].e1v = c[v]/w;// * ((k == 1)?-1.0f:1.0f);

		m_TriAccList_P[i].rcp_area = fabsf(1.0f /  (m_TriAccList_P[i].e0u * m_TriAccList_P[i].e1v -
													m_TriAccList_P[i].e0v * m_TriAccList_P[i].e1u));

		m_TriAccList_P[i].isTransparent = (m_TriObjList[i]->m_pObject->getMaterial()->getTransparency() > 0)?1:0;
	}

	return errorNo;
}


GError SSESceneData::calSceneProperty( void )
{

	m_TriAreaSum_Refl = 0;
	m_TriAreaSum_Refr = 0;
	m_TriAreaSum_Tex  = 0;
	m_TriAreaSum_All  = 0;

	unsigned int i;
	for (i = 0; i < m_TriObjCnt; i++) {
		float     fArea      = m_TriObjList[i]->calArea();
		GObject   *pObject   = m_TriObjList[i]->m_pObject;
		GMaterial *pMaterial = pObject->getMaterial();

		float mat_fRefl   = pMaterial->getReflection();
		float mat_fRefr   = pMaterial->getTransparency();
		bool  has_Texture = pObject->hasTexture();


		if (mat_fRefl > 0) {
			m_TriAreaSum_Refl += fArea;
			m_TriCnt_Refl ++;
		}

		if (mat_fRefr > 0) {
			m_TriAreaSum_Refr += fArea;
			m_TriCnt_Refr ++;
		}

		if (has_Texture) {
			m_TriAreaSum_Tex += fArea;
			m_TriCnt_Tex ++;
		}

		m_TriAreaSum_All += fArea;
	}

	return errorNo;

}

// *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
// Preparation for rendering
//
//		TriAcc precomputation
//		BVH & Triangle Object data setting
// *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
GError SSESceneData::setBVHNodeData( GBVHNode *pRootNode, GBoundingBox sceneBox  )
{
	m_pBVHNodes		= pRootNode;
	m_SceneBBox		= sceneBox;
	return errorNo;
}
