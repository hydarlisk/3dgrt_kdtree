#include "GKDTreeStructure.h"
#include "GPolygonObject.h"
#include <math.h>
#include "assert.h"
#include "cudaRenderPipeline.h"
#include "GTextureManager.h"
#include "GlobalOption.h"

/**
 *	KD Tree 를 이용한 공간 구조체.
 *	KD Tree 생성. 탐색은 오상락군의 코드를 기부 받아
 *	수정함.
 *
 *	by graphicsian.
 */

#define G_RAY_EPSILON 0.001f
#define SYS_EPSILON_FLT 0.00001f

inline int Log2Int(float v) {
	return ( (*(int *) &v) >> 23) - 127;
}

GKDTreeStructure::GKDTreeStructure( GScene* pScene ) : m_pKDTreeOption(NULL)
{
	m_pScene = pScene;
	m_iSceneTriangleCount = 0;

	m_fEmptyBonus = 0.15f;
	m_fTraversalCost = 1.5f;
	m_fIntersectionCost = 20.0f;
	m_fExtraTraversalCost = 10.0f;

	m_iMaxTreeLevel = 1024;
	m_iTreeLevel = 0;
	m_iMinObjPerLeafNode = 4;

	m_pKDTreeNodes = NULL;
	m_pTriangleOffsetList = NULL;
	m_pSceneTriangleList = NULL;

	m_iEmptyLeafCount = 0;
	m_iLeafNodeCount = 0;
	m_iLeafMaxTriangleCount = 0;
}

GKDTreeStructure::~GKDTreeStructure(void)
{
	uninitialize();
}

void GKDTreeStructure::setKDTreeOption( GKDTreeOption *pKDTreeOption )
{
	m_pKDTreeOption = pKDTreeOption;
	if( m_pKDTreeOption == NULL )
		return;

	setSplitFunction( m_pKDTreeOption->getSplitFunction() );

	if( m_pKDTreeOption->isValidFloat( "EmptyBonus" ) )
		m_fEmptyBonus = m_pKDTreeOption->getParameterFloat( "EmptyBonus" );
	else
		m_fEmptyBonus = 0.15f;

	m_fTraversalCost = m_pKDTreeOption->getParameterFloat( "TraversalCost" );
	m_fIntersectionCost = m_pKDTreeOption->getParameterFloat( "IntersectionCost" );

	m_iMaxTreeLevel = m_pKDTreeOption->getParameterInteger( "MaximumTreeDepth" );
	m_iMinObjPerLeafNode = m_pKDTreeOption->getParameterInteger( "MaximumObjectCount" );
}

/**
 *	KD-Tree 를 만든다.
 */
GError GKDTreeStructure::initialize()
{
	if( m_pKDTreeOption == NULL )
	{
		GLogManager::logging( LOG_INFO, "[Error] KD-Tree Option must be set." );
		return errorUnknown;
	}

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

	/** 
	 *	Scene 전체의 삼각형 list 를 구성해 온다.
	 */
	m_pSceneTriangleList = m_pScene->createSceneTriangleList( m_SceneBBox );
	m_iSceneTriangleCount = m_pSceneTriangleList->size();

	/**
	 *	KD-Tree 를 위한 데이터 구성. 
	 *	모든 삼각형의 정렬을 위한 공간.offset 정보는 
	 *	m_SceneTriangleList vector 안의 index 와 동일하다.
	 */
	TriangleInfo *pTriangleInfos = new TriangleInfo[ m_iSceneTriangleCount ];
	for( int i = 0; i < m_iSceneTriangleCount; i++ ) {
		pTriangleInfos[ i ].offset = i;
		pTriangleInfos[ i ].pTriangleWrapper = m_pSceneTriangleList->getTriangleWrapper( i );
		/** bounding box 는 실제 triangle bouding box 과는 다를수 있으므로 따로 저장관리 */
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

	/**
	 *	pTraangleInfos 는 buildKDTree 안에서 사용하고 없앤다.
	 */
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

	return errorNo;
}

GError GKDTreeStructure::uninitialize()
{
	if ( m_pSceneTriangleList )
		delete m_pSceneTriangleList;

	if ( m_pTriangleOffsetList ) {
		delete[] m_pTriangleOffsetList;
	}

	if ( m_pKDTreeNodes ) {
		delete[] m_pKDTreeNodes;
	}

	return errorNo;
}

inline cuPlueckerTriangleInfo GKDTreeStructure::toCuPlueckerTriangleInfo( GTriangleWrapper &tri )
{
	cuPlueckerTriangleInfo t;
	GVector v0( tri.p0[ 0 ], tri.p0[ 1 ], tri.p0[ 2 ] );
	GVector v1( tri.p1[ 0 ], tri.p1[ 1 ], tri.p1[ 2 ] );
	GVector v2( tri.p2[ 0 ], tri.p2[ 1 ], tri.p2[ 2 ] );
	GVector b( v2 - v0 ), c( v1 - v0 ), N( b.outerProduct( c ) );

	t.p0.x = tri.p0[ 0 ]; t.p0.y = tri.p0[ 1 ]; t.p0.z = tri.p0[ 2 ];
	t.p1.x = tri.p1[ 0 ]; t.p1.y = tri.p1[ 1 ]; t.p1.z = tri.p1[ 2 ];
	t.p2.x = tri.p2[ 0 ]; t.p2.y = tri.p2[ 1 ]; t.p2.z = tri.p2[ 2 ];

	/** normal 저장 */
	t.normal.x = N[0]; t.normal.y = N[1]; t.normal.z = N[2];

	t.attrib.x = unsigned_as_float( tri.objectIndexInScene );
	t.attrib.y = unsigned_as_float( tri.m_pObject->getMaterial()->m_fTransparency > 0.0f );

	return t;
}

inline cuWaldTriangleInfo GKDTreeStructure::toCuWaldTriangleInfo( GTriangleWrapper &tri )
{
	cuWaldTriangleInfo w;

	GVector v0( tri.p0[ 0 ], tri.p0[ 1 ], tri.p0[ 2 ] );
	GVector v1( tri.p1[ 0 ], tri.p1[ 1 ], tri.p1[ 2 ] );
	GVector v2( tri.p2[ 0 ], tri.p2[ 1 ], tri.p2[ 2 ] );
	GVector A( v0 );
	GVector b( v2 - v0 ), c( v1 - v0 ), N( b.outerProduct( c ).normalize() );
	unsigned k = 0;
	int flag = 0;

	/** projection axis 를 선택 */
	for (unsigned i = 1; i < 3; ++i) 
		k = fabsf(N[i]) > fabsf(N[k]) ? i : k;

	const unsigned u = ( k + 1 ) % 3, v = ( k + 2 ) % 3;
	const double denom = ( (double) b[ u ] * (double) c[ v ] - (double) b[ v ] * (double) c[ u ] );
	const double krec = (double) N[ k ];
	const double nu = (double) N[ u ] / krec, nv = (double) N[ v ] / krec, nd = (double) A.innerProduct( N ) / krec;
	const double bnu =  (double) b[ u ] / denom, bnv = (double) -b[ v ] / denom;
	const double cnu =  (double) c[ v ] / denom, cnv = (double) -c[ u ] / denom;

	w.internal0.x = unsigned_as_float( k );
	w.internal0.y = float( nu );
	w.internal0.z = float( nv );
	w.internal0.w = float( nd );

	w.internal1.x = float( A[u]);
	w.internal1.y = float( A[v]);
	w.internal1.z = float( bnu );
	w.internal1.w = float( bnv );

	w.internal2.x = float( cnu );
	w.internal2.y = float( cnv );

	/** 
	 *	z 를 이용해서 투명한지 여부와 krec 가 음수인지 양수인지 저장 
	 *	나중에 back face 인지를 체크하기 위해서 normal 의 방향을 계산할때 사용.
	 */
	/** 
	 * 투명한 물체인지 여부.
	 */
	if ( tri.m_pObject->getMaterial()->m_fTransparency > 0.0f )
		flag = 1;
	else
		flag = 2;

	/** 
	 *	krec 가 양수/음수인지 여부를 그대로 적용.
	 */
	if ( krec < 0.0f ) 
		flag = -flag;

	w.internal2.z = unsigned_as_float( flag );

	// 상위1bytes 는 삼각형이 선택되었는지 여부. 하위3bytes 는 물체 id
	w.internal2.w = unsigned_as_float( (tri.bSelected << 24) | tri.objectIndexInScene );

	return w;
}

/**
 *	CUDA KERNEL 에 올릴 KDTree 데이터를 복사.
 */
GError GKDTreeStructure::makeCudaRenderStructureInfo( cudaRenderPipeline *pCudaPipeline )
{
	GError error;

	if ( m_pScene->getObjectCount() <= 0 || m_iSceneTriangleCount <= 0 )
		return errorUnknown;

	cuBoundingBox sceneBox;
	
	//sceneBox.setMin( make_float4( m_SceneBBox.m_Min.x, m_SceneBBox.m_Min.y, m_SceneBBox.m_Min.z, 0.0f ) );
	sceneBox.min_max[0] = make_float4(m_SceneBBox.m_Min.x, m_SceneBBox.m_Min.y, m_SceneBBox.m_Min.z, 0.0f);
	//sceneBox.setMax( make_float4( m_SceneBBox.m_Max.x, m_SceneBBox.m_Max.y, m_SceneBBox.m_Max.z, 0.0f ) );
	sceneBox.min_max[1] = make_float4(m_SceneBBox.m_Max.x, m_SceneBBox.m_Max.y, m_SceneBBox.m_Max.z, 0.0f);

	/**
	 *	KDTree 세팅.
	 */
	error = pCudaPipeline->setKDTreeNodeData( m_pKDTreeNodes, m_iKDTreeNodeCount, sceneBox );
	if ( error != errorNo )
		return error;

	/**
	 *	Triangle Offset List 세팅. 총 메모리중 m_iCurrentTriangleOffset 만 세팅하면 됨.
	 */
	error = pCudaPipeline->setTriangleOffsetList( m_pTriangleOffsetList, m_iCurrentTriangleOffset );
	if ( error != errorNo )
		return error;
	
	/**
	 *	cuda 로 올릴 Object Material 정보 구성. 순서는 scene 의 object list 순서를 따른다.
	 *	cu_tri 구조체의 internal2.w 에 각 삼각형이 포함된 object 의	material offset 을 기록한다.
	 */
	GObject *pObject = NULL;
	GMaterial *pMaterial = NULL;
	int objectSize = m_pScene->getObjectCount();
	cuObjectMaterial *pcuObjectMaterial = new cuObjectMaterial[ objectSize ];

	for ( int i = 0; i < objectSize; ++i ) {

		pObject = m_pScene->getObject( i );
		pMaterial = pObject->getMaterial();

		// 광원이면 1.0 을 세팅. 
		if ( pObject->isLight() )
			pcuObjectMaterial[ i ].light = 1.0f;
		else
			pcuObjectMaterial[ i ].light = 0.0f;
	
		pcuObjectMaterial[ i ].iObjectID = int_as_float_H( pObject->getObjectNumber() );
/*		
		pcuObjectMaterial[ i ].ambient = make_float3( pMaterial->m_Ambient.r, 
													pMaterial->m_Ambient.g, 
													pMaterial->m_Ambient.b );
		pcuObjectMaterial[ i ].emission = make_float3( pMaterial->m_Emission.r, 
													pMaterial->m_Emission.g, 
													pMaterial->m_Emission.b );
*/
		
		pcuObjectMaterial[ i ].ambient_emission = make_float3(m_pScene->getGlobalAmbient().r * pMaterial->m_Ambient.r + pMaterial->m_Emission.r,
															m_pScene->getGlobalAmbient().g * pMaterial->m_Ambient.g + pMaterial->m_Emission.g,
															m_pScene->getGlobalAmbient().b * pMaterial->m_Ambient.b + pMaterial->m_Emission.b);
		pcuObjectMaterial[ i ].diffuse = make_float3( pMaterial->m_Diffuse.r, 
													pMaterial->m_Diffuse.g, 
													pMaterial->m_Diffuse.b );
		pcuObjectMaterial[ i ].specular = make_float3( pMaterial->m_Specular.r, 
													 pMaterial->m_Specular.g, 
													 pMaterial->m_Specular.b );
		pcuObjectMaterial[ i ].transparency = pMaterial->m_fTransparency;
		pcuObjectMaterial[ i ].reflection = pMaterial->m_fReflection;

		pcuObjectMaterial[ i ].roughness = pMaterial->m_fRoughness;
		pcuObjectMaterial[ i ].refractionIndex = pMaterial->m_fRefractionIndex;
		
		/**
		 *	texture id 세팅. texture 가 loading 이 되었을때만.
		 */
		if ( pObject->hasTexture() && GTextureManager::getInstance()->getTexture( pObject->getTextureID() )->isLoaded() ) {
			pcuObjectMaterial[ i ].textureNumber = int_as_float_H( pObject->getTextureID() );
		} else {
			pcuObjectMaterial[ i ].textureNumber = int_as_float_H( -1 );
		}

	}

	/**
	 *	Texture Manager 로부터 texture 들을 가져와서 gpu 로 올린다.
	 */
	int textureCount = GTextureManager::getInstance()->getTextureCount();
	if ( textureCount > 0 ) {

		cuTexture *pTextureData = new cuTexture[ textureCount ];
		GTexture* pTexture = NULL;

		for ( int i = 0; i < (int) textureCount; ++i ) {

			pTexture = GTextureManager::getInstance()->getTexture( i );
			if ( pTexture->isLoaded() ) {
				pTextureData[ i ].width = pTexture->getWidth();
				pTextureData[ i ].height = pTexture->getHeight();
				pTextureData[ i ].pData = pTexture->getTextureData();
			} else {
				pTextureData[ i ].width = 0;
				pTextureData[ i ].height = 0;
				pTextureData[ i ].pData = NULL;
			}

		}

		error = pCudaPipeline->setTextureData( pTextureData, textureCount );

		delete[] pTextureData;

		if ( error != errorNo ) {
			return error;
		}

	}

	/**
	 *	GPU 로 OBJECT 정보들을 올리고 HOST 는 delete
	 */
	error = pCudaPipeline->setObjectMaterial( pcuObjectMaterial, objectSize );
	delete[] pcuObjectMaterial;
	if ( error != errorNo ) {
		return error;
	}

	/**
	 *	cuda 로 올릴 삼각형 정보 구성.
	 */
	#if INTERSECTION_METHOD == 0
		cuWaldTriangleInfo *pcuTriangleInfo = new cuWaldTriangleInfo[ m_iSceneTriangleCount ];
	#elif INTERSECTION_METHOD == 1
		cuPlueckerTriangleInfo *pcuTriangleInfo = new cuPlueckerTriangleInfo[ m_iSceneTriangleCount ];
	#endif

	cuTriangleGeometry *pcuTriangleGeometry = new cuTriangleGeometry[ m_iSceneTriangleCount ];

	assert( pcuTriangleInfo != NULL );
	assert( pcuTriangleGeometry != NULL );

	float *n0, *n1, *n2;
	float *uv0, *uv1, *uv2;

	/**
	 *	cuda 로 올릴 삼각형 정보 구성.
	 */
	for( int i = 0; i < m_iSceneTriangleCount; i++ ) {

		#if INTERSECTION_METHOD == 0 
			pcuTriangleInfo[i] = toCuWaldTriangleInfo( *(*m_pSceneTriangleList)[i] );
		#elif INTERSECTION_METHOD == 1
			pcuTriangleInfo[i] = toCuPlueckerTriangleInfo( *(*m_pSceneTriangleList)[i] );
		#endif

		/** 
		 *	삼각형 geometry 정보 구성. normal 만올리면 된다.
		 */
		n0 = (*m_pSceneTriangleList)[i]->n0;
		n1 = (*m_pSceneTriangleList)[i]->n1;
		n2 = (*m_pSceneTriangleList)[i]->n2;

		uv0 = (*m_pSceneTriangleList)[i]->uv0;
		uv1 = (*m_pSceneTriangleList)[i]->uv1;
		uv2 = (*m_pSceneTriangleList)[i]->uv2;

		pcuTriangleGeometry[i].n0 = make_float3( n0[0], n0[1], n0[2] );
		pcuTriangleGeometry[i].n1 = make_float3( n1[0], n1[1], n1[2] );
		pcuTriangleGeometry[i].n2 = make_float3( n2[0], n2[1], n2[2] );

		if ( uv0 != NULL && uv1 != NULL && uv2 != NULL ) {
			pcuTriangleGeometry[i].u0 = uv0[0];
			pcuTriangleGeometry[i].v0 = uv0[1];
			pcuTriangleGeometry[i].u1 = uv1[0];
			pcuTriangleGeometry[i].v1 = uv1[1];
			pcuTriangleGeometry[i].u2 = uv2[0];
			pcuTriangleGeometry[i].v2 = uv2[1];
		} else {
			pcuTriangleGeometry[i].u0 = 0.0f;
			pcuTriangleGeometry[i].v0 = 0.0f;
			pcuTriangleGeometry[i].u1 = 0.0f;
			pcuTriangleGeometry[i].v1 = 0.0f;
			pcuTriangleGeometry[i].u2 = 0.0f;
			pcuTriangleGeometry[i].v2 = 0.0f;
		}

	}

	
	/**
	 *	GPU 로 삼각형 정보들을 올리고 HOST 는 delete
	 */
	GLogManager::logging( LOG_DEBUG, "Triangles = %d, pcuTriangleInfo = %p "
					"pcuTriangleGeometry = %p, triangleinfo type sizeof=%d, trianglegeometry type sizeof=%d",
					 m_iSceneTriangleCount, pcuTriangleInfo, pcuTriangleGeometry, 
					 sizeof( cuWaldTriangleInfo ),
					 sizeof( cuTriangleGeometry ) );

	#if INTERSECTION_METHOD == 0 
		error = pCudaPipeline->setWaldTriangleInfo( pcuTriangleInfo, m_iSceneTriangleCount );
		if ( error != errorNo ) {
			delete[] pcuTriangleInfo;
			delete[] pcuTriangleGeometry;
			return error;
		}
	#elif INTERSECTION_METHOD == 1
		error = pCudaPipeline->setPlueckerTriangleInfo( pcuTriangleInfo, m_iSceneTriangleCount );
		if ( error != errorNo ) {
			delete[] pcuTriangleInfo;
			delete[] pcuTriangleGeometry;
			return error;
		}
	#endif

	error = pCudaPipeline->setTriangleGeometry( pcuTriangleGeometry, m_iSceneTriangleCount );

	delete[] pcuTriangleInfo;
	delete[] pcuTriangleGeometry;
	if ( error != errorNo )
		return error;

	return errorNo;
}

/**
 *	SSE 클래스 에서 참조할 KDTree 포인터 복사.
 */
GError GKDTreeStructure::makeSSERenderStructureInfo( SSESceneData *pSSEData )
{
	GError error;

	if ( m_pScene->getObjectCount() <= 0 || m_iSceneTriangleCount <= 0 )
		return errorUnknown;

	/**
	 *	KDTree 세팅.
	 */
	error = pSSEData->setKDTreeNodeData( m_pKDTreeNodes, m_iKDTreeNodeCount, m_SceneBBox );
	if ( error != errorNo )
		return error;

	/**
	 *	Triangle Offset List 세팅. 총 메모리중 m_iCurrentTriangleOffset 만 세팅하면 됨.
	 */
	error = pSSEData->setTriangleOffsetList( m_pTriangleOffsetList, m_iCurrentTriangleOffset );
	if ( error != errorNo )
		return error;

	/**
	 *	Triangle Object List 세팅
	 */
	error = pSSEData->setTriangleObjectList( m_pSceneTriangleList );
	if ( error != errorNo )
		return error;

	/**
	 *	Triangle Accel List 세팅
	 */
	error = pSSEData->buildTriAccList_Barycentric();
	//error = pSSEData->buildTriAccList_Pluecker();
	if ( error != errorNo )
		return error;

	return errorNo;
}

/**
 *	삼각형 index 정보를 가지고 TriangleWrapper 정보를 찾아서 리턴.
 */
GTriangleWrapper *GKDTreeStructure::getTriangleWrapper( const unsigned int triIndex )
{
	return (*m_pSceneTriangleList)[ triIndex ];
}

int GKDTreeStructure::getTriangleCount()
{
	return m_iSceneTriangleCount;
}

void GKDTreeStructure::buildKDTree(	BoundEdge *bEdge, const TriangleInfo *pTriangleInfos, 
								    unsigned int triangleSize, GBoundingBox bbox, 
									unsigned int inNodeLevel, kdtreeNode *inNode )
{
	SplitCost bestCost;
	m_iTreeLevel = max( inNodeLevel, m_iTreeLevel );

	/**
	 *	split 하지 않았을 때의 cost를 세팅한다.
	 */
	bestCost.cost = 1. * double( triangleSize ) * m_fIntersectionCost;

	/** 
	 *	만약 tree level과 object 수가 threashold를 넘지 않는다면 
	 *	split을 시도한다.
	 */
	if( inNodeLevel < m_iMaxTreeLevel && triangleSize > m_iMinObjPerLeafNode ){
		for( int axis = 0; axis < 3; axis++ ){
#if SPLIT_FUNCTION_POINTER == ON
			(this->*splitFunction)( axis, bbox, pTriangleInfos, triangleSize, bEdge, bestCost );
#elif SPLIT_FUNCTION_POINTER == OFF
			splitWithSAH( axis, bbox, pTriangleInfos, triangleSize, bEdge, bestCost );
#endif

			//tryEmptySplit( axis, bbox, pTriangleInfos, triangleSize, bEdge, bestCost );
		}
	}



	/**
	 *	split 할 필요가 없다면 leaf node 로 만든다.
	 *  (split 된 후의 left/right 의 삼각형 개수가 minimum 보다 작으면 split 하지 않음.)
	 */
	if( !bestCost.is_valid() )
	{
#if FRUSTUM_CULLING == ON
		const int bbox_data_size = 6;		// bbox 의 min/max 좌표값을 저장할 공간 마련
#else
		const int bbox_data_size = 0;		// 기존의 경우 bbox 정보 저장하지 않음
#endif
		/**
		 *	leaf노드가 많아서 objectOffsetList가 부족하면 메모리를 더 할당한다.
		 */
		if( m_iCurrentTriangleOffset + triangleSize + bbox_data_size >= m_iAllocatedTriangleOffsetSize )
			reAllocTriangleOffsetList( max( 2 * m_iAllocatedTriangleOffsetSize, 512 ) );

		/**
		 *	leaf node가 참조하는 triangle의 offset 을 offsetList 마지막에 추가해 넣는다.
		 */
		unsigned *currOffsetList = &m_pTriangleOffsetList[ m_iCurrentTriangleOffset ];

		unsigned leafCount = 0;
		for( unsigned i = 0; i < triangleSize; i++)
		{
			currOffsetList[ bbox_data_size + leafCount++ ] = pTriangleInfos[ i ].offset;
		}

#if FRUSTUM_CULLING == ON
		currOffsetList[0] = (*(int *) &bbox.m_Min.x);
		currOffsetList[1] = (*(int *) &bbox.m_Min.y);
		currOffsetList[2] = (*(int *) &bbox.m_Min.z);
		currOffsetList[3] = (*(int *) &bbox.m_Max.x);
		currOffsetList[4] = (*(int *) &bbox.m_Max.y);
		currOffsetList[5] = (*(int *) &bbox.m_Max.z);
#endif

		if( triangleSize == 0 )
			m_iEmptyLeafCount++;

		m_iLeafNodeCount++;
		m_iLeafMaxTriangleCount = max( m_iLeafMaxTriangleCount, leafCount );

		setLeafNode( inNode, leafCount, m_iCurrentTriangleOffset );
		m_iCurrentTriangleOffset += (bbox_data_size + leafCount);

		/** 
		 *	더이상 pTriangleInfos 는 필요없으므로 메모리 공간 절약을 위해 없앤다.
		 */
		delete[] pTriangleInfos;

	} else {

		GBoundingBox leftnBounds, rightnBounds;
		leftnBounds = bbox;	leftnBounds.m_Max[ bestCost.axis ] = bestCost.splitPos;
		rightnBounds = bbox;  rightnBounds.m_Min[ bestCost.axis ] = bestCost.splitPos;

		TriangleInfo *pLeftTriangles = new TriangleInfo[ bestCost.n_left ];
		TriangleInfo *pRightTriangles = new TriangleInfo[ bestCost.n_right ];

		const unsigned n_bEdge = 2 * triangleSize;
		setBoundEdgeList( bestCost.axis, pTriangleInfos, n_bEdge, bEdge );						// TriangleInfo 부터 bEdge 를 생성 및 정렬

		pushChildTriangles( n_bEdge, bEdge, pLeftTriangles, pRightTriangles, bestCost );		// bEdge 로 부터 pLeftTriangle, pRightTriangle 을 생성
		splitClipping( bestCost.n_left, bestCost, pLeftTriangles, 0 );
		splitClipping( bestCost.n_right, bestCost, pRightTriangles, 1 );

		/** 
		 *	더이상 pTriangleInfos 는 필요없으므로 메모리 공간 절약을 위해 없앤다.
		 *	반드시 pushChildTriangles 를 수행한 이후에 없애야 한다.
		 */
		delete[] pTriangleInfos;

		/**
		 *	left, child node 세팅.
		 */
		const unsigned int nodeNum = m_iKDTreeNodeCount;
		m_iKDTreeNodeCount += 2;

		setInnerNode( inNode, bestCost.axis, nodeNum, bestCost.splitPos );

		/**
		 *	노드가 많아서 m_iKDTreeNodeCount가 부족하면 메모리를 더 할당한다.
		 */
		if( m_iKDTreeNodeCount >= m_iAllocatedkdNodeCount )
			reAllocKdtreeNodes( max( 2 * m_iAllocatedkdNodeCount, 512 ) );

		/**
		 *	Left, Right 재귀 탐색. 
		 *	pLeftTriangles, pRightTriangles 는 buildKDTree 함수 안에서
		 *	사용하고 바로 없앤다.
		 */
		buildKDTree( bEdge, pLeftTriangles, bestCost.n_left, leftnBounds, inNodeLevel + 1, &m_pKDTreeNodes[ nodeNum ] );
		buildKDTree( bEdge, pRightTriangles, bestCost.n_right, rightnBounds, inNodeLevel + 1, &m_pKDTreeNodes[ nodeNum + 1 ] );

	}
}

inline void GKDTreeStructure::setInnerNode( kdtreeNode* pNode, int _splitAxis, unsigned int _firstChildOffset, float _splitPos )
{
	pNode->x = _firstChildOffset << 3;
	pNode->x |= _splitAxis;
	pNode->y = float_as_unsigned( _splitPos );
}

inline void GKDTreeStructure::setLeafNode( kdtreeNode* pNode, unsigned int _objectSize, unsigned int _objectListOffset )
{
	pNode->x = _objectSize << 3;
	pNode->x |= 3;
	pNode->y = _objectListOffset;
}


int GKDTreeStructure::compare(const void *elem0, const void *elem1)
{
	const BoundEdge* obj0 = (const BoundEdge *)elem0;
	const BoundEdge* obj1 = (const BoundEdge *)elem1;

	return obj0->t == obj1->t ?
		(obj0->triangleInfo->offset > obj1->triangleInfo->offset ? 1 : -1)
		:
		(obj0->t > obj1->t ? 1 : -1);
}

void GKDTreeStructure::setBoundEdgeList( const int axis, 
										 const TriangleInfo *pTriangleInfo, 
										 const unsigned int n_bEdge, BoundEdge *bEdge )
{
	int index = 0;

	for( unsigned int i = 0; i < n_bEdge; ) {
		GBoundingBox worldbound = pTriangleInfo[index].boundingBox;
		bEdge[i].type = BoundEdge::START;	bEdge[i].triangleInfo = &pTriangleInfo[index];
		bEdge[i].t = worldbound.m_Min[axis];
		bEdge[i].isPlanar = ( worldbound.m_Min[axis] == worldbound.m_Max[axis] );
		i++;
		bEdge[i].type = BoundEdge::END;		bEdge[i].triangleInfo = &pTriangleInfo[index];
		bEdge[i].t = worldbound.m_Max[axis];
		bEdge[i].isPlanar = ( worldbound.m_Min[axis] == worldbound.m_Max[axis] );
		i++; index++;
	}
	
	/**
	 *	bound edge sorting
	 */
	qsort( &( bEdge[0] ), n_bEdge, sizeof( BoundEdge ), compare );
}

void GKDTreeStructure::setBoundEdgeList2( const int axis, 
										 const TriangleInfo *pTriangleInfo, 
										 const unsigned int n_bEdge, BoundEdge *bEdge, spbean *bean )
{
	int index = 0, i, j;

	for( i = 0; i < (int)n_bEdge; ) {
		GBoundingBox worldbound = pTriangleInfo[index].boundingBox;
		bEdge[i].type = BoundEdge::START;	bEdge[i].triangleInfo = &pTriangleInfo[index];
		bEdge[i].t = worldbound.m_Min[axis];
		bEdge[i].isPlanar = ( worldbound.m_Min[axis] == worldbound.m_Max[axis] );
		i++;
		bEdge[i].type = BoundEdge::END;		bEdge[i].triangleInfo = &pTriangleInfo[index];
		bEdge[i].t = worldbound.m_Max[axis];
		bEdge[i].isPlanar = ( worldbound.m_Min[axis] == worldbound.m_Max[axis] );
		i++; index++;
	}
	
	/**
	 *	bound edge sorting
	 */
	qsort( &( bEdge[0] ), n_bEdge, sizeof( BoundEdge ), compare );

	for( i = 0 ; i < (int)n_bEdge ; i++ )
		bean[i].flag = -1;

	for( i = 0 ; i < (int)n_bEdge; i++){
		if(bEdge[i].type == BoundEdge::START){
			for( j = i+1 ; j < (int)n_bEdge ; j++){
				bean[j - 1].flag = 1;
				bean[j - 1].t = bEdge[j-1].t;
				if(bEdge[j].triangleInfo == bEdge[i].triangleInfo)
					break;				
			}
		}
	}
}


void GKDTreeStructure::tryEmptySplit( const int axis, GBoundingBox inBBox, 
								 const TriangleInfo *pTriangles, const int triangleSize, 
								 BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly )
{
	const int axis1 = ( axis + 1 ) % 3, axis2 = ( axis + 2 ) % 3;
	GVector cell_extent( inBBox.m_Max - inBBox.m_Min );

	//다른 축에 비해 너무 작은 축을 자르려면 스킵하는 루틴을 추가할 수도.
	//전체 씬에 대해서 너무 큰 리프를 만들려고 하면 메디안 스플릿?(큰 empty node를 방해할지도)
	const double
		cell_area = cell_extent.x * cell_extent.y + 
					cell_extent.y * cell_extent.z + 
					cell_extent.x * cell_extent.z;
	const double cell_area_rcp = 1 / cell_area,

		area_mul = double( cell_extent.getElement( axis1 ) ) + double( cell_extent.getElement( axis2 ) ),
		area_add = double( cell_extent.getElement( axis1 ) ) * double( cell_extent.getElement( axis2 ) ),

		cost_term_add = m_fTraversalCost,
		cost_term_mul = cell_area_rcp * m_fIntersectionCost;

	const float
		cell_min = inBBox.m_Min.getElement( axis ),
		cell_max = inBBox.m_Max.getElement( axis ),
		cell_length = cell_max - cell_min;

	/**
	 *	모든 split candidate 에 대해서 Cost 를 계산한다.
	 */
	{
		int open = 0, close = 0, num_planars = 0, local_open = 0, local_close = 0, j;
		
		const unsigned n_bEdge = triangleSize * 2;
		spbean *bean = new spbean [triangleSize * 2];
		
		setBoundEdgeList2( axis, pTriangles, n_bEdge, bEdge, bean );

		for ( unsigned int i = 0; i < n_bEdge; i++ ) {

			BoundEdge curr_bEdge = bEdge[i];

			//planar는 open과 close에 둘 다 포함됨
			open += local_open + num_planars;
			close += local_close + num_planars;
			local_open = 0;  local_close = 0; num_planars = 0;			

			//현재 axis와 side에 대해 포지션구함
			float cur_position = curr_bEdge.t;
			{
				//똑같은게 여러개 있을 때는 제일 오른쪽에서만 SAH 계산을 한다.
				for ( j = i; j < (int)n_bEdge; j++ )
				{
					BoundEdge tmp_bEdge = bEdge[j];
					if ( tmp_bEdge.t != cur_position) break;

					const bool
						is_left		= (tmp_bEdge.type == BoundEdge::START),
						is_planar	= tmp_bEdge.isPlanar;//(axis);

					//카운팅
					if (!is_planar) {
						local_open	+= is_left ? 1 : 0;
						local_close	+= is_left ? 0 : 1;
					}
					else//플라나하다면 따로 카운팅
						num_planars += is_left ? 1 : 0;	// only count it once

					curr_bEdge = tmp_bEdge;
					i=j;
					assert( cur_position == tmp_bEdge.t );
				}
			}

			//이런 경우가 실제로 생길려나? ==의 경우는 당연히 있겠지.
			if (cur_position <= cell_min + SYS_EPSILON_FLT) continue;//
			if (cur_position >= cell_max - SYS_EPSILON_FLT) break;//

			
			{
				float cur_position2 = cur_position;
/*
				if(curr_bEdge.type == BoundEdge::START ){
					j = i;
					while(  j >= 0 && fabs(curr_bEdge.t - bean[j].t) < 0.000001 ) j--;					
					for( ; j >= 0 ; j--){
						if( bean[j].flag == 1 ){							
							cur_position2 = cur_position;
							cur_position = bean[j].t;
							break;
						}
					}
				}

*/
				if(curr_bEdge.type == BoundEdge::END ){
					j = i + 1;
					while(  j < (int)n_bEdge -1 && fabs(curr_bEdge.t - bean[j].t) < 0.000001 ) j++;
					j--;
					for( ; j < (int)n_bEdge-1 ; j++){
						if( bean[j].flag == 1 ){							
							cur_position2 = bEdge[j].t;
							break;
						}
					}
				}

				const double
					extent_l = double(cur_position) - cell_min,
					extent_r = cell_max - double(cur_position2),
					// area = x*(y+z) + y*z = x*area_mul + area_add
					area_l = extent_l*area_mul + area_add,
					area_r = extent_r*area_mul + area_add;

				const int
					n_leftOnly		= close + local_close,
					n_cross			= open - n_leftOnly,
					n_rightOnly		= triangleSize - (n_leftOnly + n_cross + num_planars);

				//작은 셀이거나, 한쪽이 비어 있다면 다른쪽 셀로 planar가 간다.

				int planar_side = area_l < area_r ? BoundEdge::START : BoundEdge::END;
				if ((n_leftOnly+n_cross == 0) | (n_rightOnly+n_cross == 0))
					planar_side = n_leftOnly == 0 ? BoundEdge::END : BoundEdge::START;
				//플라나를 한쪽으로 다 밀었으므로 없는 셀은 갯수를 0으로 세팅
				const int
					num_planar_left		= planar_side == BoundEdge::START ? num_planars : 0,
					num_planar_right	= planar_side == BoundEdge::END ? num_planars : 0;

				//최종적인 양쪽 갯수.
				const int
					cost_num_left	= n_leftOnly+n_cross+num_planar_left,
					cost_num_right	= n_rightOnly+n_cross+num_planar_right;

				// 현재 m_fEmptyBonus = 0.15
				//! \ref <1> "Fast Ray Tracing for Modern General Purpose CPU"
				const float
					emptyBonus = 
					((cost_num_left == 0 || cost_num_right == 0))  ?	m_fEmptyBonus : 1;

				const double
					factor = 1, // .85, //.5, //
					//아마 cost_num_left를 그대로 안 쓰는 이유는 적당한 factor를 주기 위해서인듯.
					nl = double(n_leftOnly+num_planar_left) + factor*(n_cross),
					nr = double(n_rightOnly+num_planar_right) + factor*(n_cross),

					//이런 흔적을 보니 다양한 시도를 하며 삽질한게 보이는구나.
					//mul = ((nl == 0.) | (nr == 0.)) ? cost_term_mul*factor : cost_term_mul,
					//score = cost_term_add + mul*(nl*area_l + nr*area_r);
					score = (cost_term_add + cost_term_mul*(nl*area_l + nr*area_r))*emptyBonus;
				if (
					((cost_num_left > 0) | (cost_num_right > 0)) &//안 이런 경우도 있나?;;;;
					//((cost_num_left > 0) & (cost_num_right > 0)) &	// shouldn't happen, and in any case should be produced by the space cut heuristic earlier.
					(score < bestCost.cost))
				{
					bestCost.cost			= score;
					bestCost.splitPos		= cur_position;
					bestCost.axis			= axis;	

					bestCost.n_onlyLeft		= n_leftOnly;			
					bestCost.n_onlyRight	= n_rightOnly;			
					bestCost.n_cross		= n_cross;	
					bestCost.n_planar		= num_planars;	
					bestCost.n_left			= cost_num_left;	
					bestCost.n_right		= cost_num_right;

					bestCost.planar_side	= planar_side;					
				}
			} // scoring
		} // for all split candidates.
		delete bean;
	}
}

void GKDTreeStructure::setSplitFunction( SPLIT_FUNCTION SplitFunctionEnum )
{
	switch( SplitFunctionEnum )
	{
	case EMPTY_SPLIT_FUNCTION:
		splitFunction = &GKDTreeStructure::tryEmptySplit;
		break;
	case VISIBILITY_SPLIT_FUNCTION:
		splitFunction = &GKDTreeStructure::splitWithVisibility;
		break;
	case SAH_WITH_EXTRA_COST_SPLIT_FUNCTION:
		splitFunction = &GKDTreeStructure::splitWithSAH_ExtraCost;
		break;
	default:
		splitFunction = &GKDTreeStructure::splitWithSAH;
		break;
	}
}

void GKDTreeStructure::splitWithSAH( const int axis, GBoundingBox inBBox, 
								 const TriangleInfo *pTriangles, const int triangleSize, 
								 BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly )
{
	/**
	 * 2009. 12. 02.
	 * boolean 형의 연산을 | -> || 로 & -> && 로 고쳤음
	 * 그 외 개인 적인 바램은 double 을 없애고, edge 를 포인터로 바꾸고 싶음.
	*/
	/**
	 * 2010. 3. 8.
	 * score 라는 단어를 모두 cost 로 바꿨음. (score 와 cost 는 정 반대의 의미)
	 * coord 라는 단어를 모두 cur_position 으로 바꿨음.
	 * Build time 을 줄일 수 있는 요소.
	 *	1. Edge 를 포인터로 바꿔서 사용함.
	 *	2. Edge 정렬을 매번 하지 말고, 한번에 함.
	 *	3. Edge 의 Type 을 고려해서 정렬에 사용하거나 할 수 있는 것으로 알고 있음.
	*/
	const int axis1 = ( axis + 1 ) % 3, axis2 = ( axis + 2 ) % 3;

	// 실제 벡터가 아님. 3축의 길이.
	GVector cell_extent( inBBox.m_Max - inBBox.m_Min );

	/**
	 * 여기서 모든 넓이는 실제 넓이가 아니라 1/2 된 넓이임.
	*/

	//다른 축에 비해 너무 작은 축을 자르려면 스킵하는 루틴을 추가할 수도.
	//전체 씬에 대해서 너무 큰 리프를 만들려고 하면 메디안 스플릿?(큰 empty node를 방해할지도)

	// 전체 Cell 넓이의 1/2
	const double
		cell_area = cell_extent.x * cell_extent.y + 
					cell_extent.y * cell_extent.z + 
					cell_extent.x * cell_extent.z;
	const double cell_area_rcp = 1 / cell_area,

		area_mul = double( cell_extent.getElement( axis1 ) ) + double( cell_extent.getElement( axis2 ) ),
		area_add = double( cell_extent.getElement( axis1 ) ) * double( cell_extent.getElement( axis2 ) ),

		cost_term_add = m_fTraversalCost,
		cost_term_mul = cell_area_rcp * m_fIntersectionCost;

	const float
		cell_min = inBBox.m_Min.getElement( axis ),
		cell_max = inBBox.m_Max.getElement( axis ),
		cell_length = cell_max - cell_min;

	/**
	 *	모든 split candidate 에 대해서 Cost 를 계산한다.
	 */
	{
		/**
		 * open			: triangle box 기준으로 min 에 해당하는 개수
		 * close		: triangle box 기준으로 max 에 해당하는 개수
		 * num_planars	: triangle box 가 min = max 가 되는 것들의 개수(local 임)
		 * local_open / local_close : 현재 curr_bEdge.t 인 split position 에 해당하는 개수
		*/
		int open = 0, close = 0, num_planars = 0, local_open = 0, local_close = 0;
		
		const unsigned n_bEdge = triangleSize * 2;
		// nlogn 방식처럼 정렬하지 않음, edge 의 type 은 고려하지 않고 순수히 위치로만 정렬함
		setBoundEdgeList( axis, pTriangles, n_bEdge, bEdge );

		for ( unsigned int i = 0; i < n_bEdge; i++ ) {

			BoundEdge curr_bEdge = bEdge[i];

			//planar는 open과 close에 둘 다 포함됨
			// (원래는 2개(min/max)가 planar 한개로 계산 됐으므로 min->open, max->close 로 각각 들어감.)
			open += local_open + num_planars;
			close += local_close + num_planars;
			local_open = 0;  local_close = 0; num_planars = 0; // num_planars 도 local 계산임

			//현재 axis와 side에 대해 포지션구함
			const float cur_position = curr_bEdge.t;
			{
				//똑같은게 여러개 있을 때는 제일 오른쪽에서만 SAH 계산을 한다.
				for ( unsigned int j = i; j < n_bEdge; j++ )
				{
					BoundEdge tmp_bEdge = bEdge[j];
					if ( tmp_bEdge.t != cur_position) break;

					const bool
						is_left		= (tmp_bEdge.type == BoundEdge::START),
						is_planar	= tmp_bEdge.isPlanar;//(axis);

					//카운팅
					if (!is_planar) {
						local_open	+= is_left ? 1 : 0; //!< box 의 왼쪽은 local_open 을 증가
						local_close	+= is_left ? 0 : 1; //!< box 의 오른쪽은 local_close 를 증가
					}
					else//플라나하다면 따로 카운팅
						num_planars += is_left ? 1 : 0;	// only count it once

					curr_bEdge = tmp_bEdge;
					i=j;
					assert( cur_position == tmp_bEdge.t );
				}
			}

			//이런 경우가 실제로 생길려나? ==의 경우는 당연히 있겠지.
			if (cur_position <= cell_min + SYS_EPSILON_FLT) continue;//
			if (cur_position >= cell_max - SYS_EPSILON_FLT) break;//

			
			{
				const double
					extent_l = double(cur_position) - cell_min,
					extent_r = cell_max - double(cur_position),
					// area = x*(y+z) + y*z = x*area_mul + area_add
					area_l = extent_l*area_mul + area_add,
					area_r = extent_r*area_mul + area_add;

				const int
					n_leftOnly		= close + local_close,	// close 가 된다면 그 삼각형은 오른쪽에 있지도 않게 됨 (겹치지도 않음)
					n_cross			= open - n_leftOnly,	// open = n_leftOnly + n_cross (현재 local_open 은 포함하지 않음)
					n_rightOnly		= triangleSize - (n_leftOnly + n_cross + num_planars);

				//////작은 셀이거나, 한쪽이 비어 있다면 다른쪽 셀로 planar가 간다.

				/**
				 * 2010. 3. 8. 기존 코드는 planar 를 넣을 때 
				 *	1. 기본적으로 작은 크기의 박스로 들어감.
				 *	2. 만약 큰 크기의 박스에 넣어봤는데, 작은 박스가 empty 가 될 경우 큰 박스에 넣음.
				 * 수정 : 양쪽다 넣어보고, 실질적으로 cost 를 계산함.
				*/
				{
					//// 기본적으로 planar 삼각형들은 작은 곳으로 들어감 
					//int planar_side = area_l < area_r ? BoundEdge::START : BoundEdge::END;
					//// 하지만 만약 planar 가 더 큰 곳으로 들어가서 다른 곳이 empty 가 될 경우는 더 큰 곳으로 들어감
					//if ((n_leftOnly+n_cross == 0) || (n_rightOnly+n_cross == 0))
					//	planar_side = n_leftOnly == 0 ? BoundEdge::END : BoundEdge::START;

					////플라나를 한쪽으로 다 밀었으므로 없는 셀은 갯수를 0으로 세팅
					//const int
					//	num_planar_left		= planar_side == BoundEdge::START ? num_planars : 0,
					//	num_planar_right	= planar_side == BoundEdge::END ? num_planars : 0;

					////최종적인 양쪽 갯수.
					//const int
					//	cost_num_left	= n_leftOnly+n_cross+num_planar_left,
					//	cost_num_right	= n_rightOnly+n_cross+num_planar_right;

					//const float					
					//	emptyBonus = 
					//	((cost_num_left == 0 || cost_num_right == 0))  ?	m_fEmptyBonus : 1;

					//const double
					//	factor = 1, // .85, //.5, //
					//	//아마 cost_num_left를 그대로 안 쓰는 이유는 적당한 factor를 주기 위해서인듯.
					//	nl = double(n_leftOnly+num_planar_left) + factor*(n_cross),
					//	nr = double(n_rightOnly+num_planar_right) + factor*(n_cross),

					//	//이런 흔적을 보니 다양한 시도를 하며 삽질한게 보이는구나.
					//	//mul = ((nl == 0.) | (nr == 0.)) ? cost_term_mul*factor : cost_term_mul,
					//	//cost = cost_term_add + mul*(nl*area_l + nr*area_r);
					//	cost = (cost_term_add + cost_term_mul*(nl*area_l + nr*area_r))*emptyBonus;
					////if (
					////	((cost_num_left > 0) || (cost_num_right > 0)) &&//안 이런 경우도 있나?;;;;
					////	//((cost_num_left > 0) && (cost_num_right > 0)) &&	// shouldn't happen, and in any case should be produced by the space cut heuristic earlier.
				}

				// 배열[0] 은 plnar 가 왼쪽에 들어갔을 경우
				// 배열[1] 은 plnar 가 오른쪽에 들어갔을 경우

				//최종적인 양쪽 갯수.
				const int cost_num_left[2]		= { n_leftOnly+n_cross+num_planars, n_leftOnly+n_cross };
				const int cost_num_right[2]	= { n_rightOnly+n_cross, n_rightOnly+n_cross+num_planars };

				double prob_l = (extent_l*area_mul + area_add)*cell_area_rcp;
				double prob_r = (extent_r*area_mul + area_add)*cell_area_rcp;

				const float	emptyBonus[2] = { 
					(cost_num_left[0] == 0 || cost_num_right[0] == 0) ? m_fEmptyBonus : 1.0f, 
					(cost_num_left[1] == 0 || cost_num_right[1] == 0) ? m_fEmptyBonus : 1.0f };

					// 이 코드는 n_cross 에 대한 factor 를 고려하지 않았음
					// factor = 1 로 되어 있기 때문임. 만약 factor 가 1 말고 다른 값이 들어갈 경우 코드가 살짝 수정되어야함.
					// unrolling 해줄 것이라고 가정함
					double SAH[2]; // SAH cost
					for( int i = 0; i < 2; i++ )
					{
						SAH[i] = bestCost.cost + 100.0f; // for skip

						if( emptyTestOnly == true )
						{
							// Empty Test Only 일 경우 Empty Test 만 수행하고 best cost 와 비교하지 않음.
							// 즉, 둘 중 하나가 Empty 일 경우만 자름.
							if( cost_num_left[i] != 0 && cost_num_right[i] != 0 )
								continue;
						}
						else
						{
							SAH[i] = m_fTraversalCost + m_fIntersectionCost*( 
								double( cost_num_left[i] )		* prob_l +
								double( cost_num_right[i] )	* prob_r ) * emptyBonus[i];
						}
					}

					int planar_side;
					double cost;
					int num_left;
					int num_right;

					if( SAH[0] <= SAH[1] ) //! planar 를 왼쪽에 넣는 것이 낫다면,
					{
						cost = SAH[0];
						planar_side = BoundEdge::START;

						num_left = cost_num_left[0];
						num_right = cost_num_right[0];
					}
					else
					{
						cost = SAH[1];
						planar_side = BoundEdge::END;
						num_left = cost_num_left[1];
						num_right = cost_num_right[1];
					}


				if( cost < bestCost.cost )
				{
					bestCost.cost			= cost;
					bestCost.splitPos		= cur_position;
					bestCost.axis			= axis;

					bestCost.n_onlyLeft		= n_leftOnly;
					bestCost.n_onlyRight	= n_rightOnly;
					bestCost.n_cross		= n_cross;
					bestCost.n_planar		= num_planars;
					bestCost.n_left			= num_left;
					bestCost.n_right		= num_right;

					bestCost.planar_side	= planar_side;
				}
			} // scoring
		} // for all split candidates.
	}
}

void GKDTreeStructure::splitWithSAH_ExtraCost( const int axis, GBoundingBox inBBox, 
								 const TriangleInfo *pTriangles, const int triangleSize, 
								 BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly )
{
	const int axis1 = ( axis + 1 ) % 3, axis2 = ( axis + 2 ) % 3;
	GVector cell_extent( inBBox.m_Max - inBBox.m_Min );

	//다른 축에 비해 너무 작은 축을 자르려면 스킵하는 루틴을 추가할 수도.
	//전체 씬에 대해서 너무 큰 리프를 만들려고 하면 메디안 스플릿?(큰 empty node를 방해할지도)
	const double
		cell_area = cell_extent.x * cell_extent.y + 
					cell_extent.y * cell_extent.z + 
					cell_extent.x * cell_extent.z;
	const double cell_area_rcp = 1 / cell_area,

		area_mul = double( cell_extent.getElement( axis1 ) ) + double( cell_extent.getElement( axis2 ) ),
		area_add = double( cell_extent.getElement( axis1 ) ) * double( cell_extent.getElement( axis2 ) ),

		cost_term_add = m_fTraversalCost,
		cost_term_mul = cell_area_rcp * m_fIntersectionCost;

	const float
		cell_min = inBBox.m_Min.getElement( axis ),
		cell_max = inBBox.m_Max.getElement( axis ),
		cell_length = cell_max - cell_min;

	/**
	 *	모든 split candidate 에 대해서 Cost 를 계산한다.
	 */
	{
		int open = 0, close = 0, num_planars = 0, local_open = 0, local_close = 0;
		
		const unsigned n_bEdge = triangleSize * 2;
		setBoundEdgeList( axis, pTriangles, n_bEdge, bEdge );

		for ( unsigned int i = 0; i < n_bEdge; i++ ) {

			BoundEdge curr_bEdge = bEdge[i];

			//planar는 open과 close에 둘 다 포함됨
			open += local_open + num_planars;
			close += local_close + num_planars;
			local_open = 0;  local_close = 0; num_planars = 0;			

			//현재 axis와 side에 대해 포지션구함
			const float cur_position = curr_bEdge.t;
			{
				//똑같은게 여러개 있을 때는 제일 오른쪽에서만 SAH 계산을 한다.
				for ( unsigned int j = i; j < n_bEdge; j++ )
				{
					BoundEdge tmp_bEdge = bEdge[j];
					if ( tmp_bEdge.t != cur_position) break;

					const bool
						is_left		= (tmp_bEdge.type == BoundEdge::START),
						is_planar	= tmp_bEdge.isPlanar;//(axis);

					//카운팅
					if (!is_planar) {
						local_open	+= is_left ? 1 : 0;
						local_close	+= is_left ? 0 : 1;
					}
					else//플라나하다면 따로 카운팅
						num_planars += is_left ? 1 : 0;	// only count it once

					curr_bEdge = tmp_bEdge;
					i=j;
					assert( cur_position == tmp_bEdge.t );
				}
			}

			//이런 경우가 실제로 생길려나? ==의 경우는 당연히 있겠지.
			if (cur_position <= cell_min + SYS_EPSILON_FLT) continue;//
			if (cur_position >= cell_max - SYS_EPSILON_FLT) break;//

			
			{
				const double
					extent_l = double(cur_position) - cell_min,
					extent_r = cell_max - double(cur_position),
					// area = x*(y+z) + y*z = x*area_mul + area_add
					area_l = extent_l*area_mul + area_add,
					area_r = extent_r*area_mul + area_add;

				const int
					n_leftOnly		= close + local_close,
					n_cross			= open - n_leftOnly,
					n_rightOnly		= triangleSize - (n_leftOnly + n_cross + num_planars);

				//작은 셀이거나, 한쪽이 비어 있다면 다른쪽 셀로 planar가 간다.

				int planar_side = area_l < area_r ? BoundEdge::START : BoundEdge::END;	
				if ((n_leftOnly+n_cross == 0) | (n_rightOnly+n_cross == 0))
					planar_side = n_leftOnly == 0 ? BoundEdge::END : BoundEdge::START;
				//플라나를 한쪽으로 다 밀었으므로 없는 셀은 갯수를 0으로 세팅
				const int
					num_planar_left		= planar_side == BoundEdge::START ? num_planars : 0,
					num_planar_right	= planar_side == BoundEdge::END ? num_planars : 0;

				//최종적인 양쪽 갯수.
				const int
					cost_num_left	= n_leftOnly+n_cross+num_planar_left,
					cost_num_right	= n_rightOnly+n_cross+num_planar_right;

				// 둘 중 하나가 empty 일 경우
				const bool emptyBoxed = (cost_num_left == 0 || cost_num_right == 0);


				const float
					emptyBonus = emptyBoxed ? m_fEmptyBonus : 1.0f;

				const float bothProbability = emptyBoxed ? 0.0f : float( cell_area_rcp * area_add );

				const double
					factor = 1, // .85, //.5, //
					//아마 cost_num_left를 그대로 안 쓰는 이유는 적당한 factor를 주기 위해서인듯.
					nl = double(n_leftOnly+num_planar_left) + factor*(n_cross),
					nr = double(n_rightOnly+num_planar_right) + factor*(n_cross),

					//이런 흔적을 보니 다양한 시도를 하며 삽질한게 보이는구나.
					//mul = ((nl == 0.) | (nr == 0.)) ? cost_term_mul*factor : cost_term_mul,
					//score = cost_term_add + mul*(nl*area_l + nr*area_r);

					// 기존 방법
					//score = (cost_term_add + cost_term_mul*(nl*area_l + nr*area_r))*emptyBonus;

					// with extra cost
					// extra cost 는 empty box 가 아닐 경우만 적용됨
					score = (cost_term_add + cost_term_mul*(nl*area_l + nr*area_r) + 
						bothProbability * m_fExtraTraversalCost)*emptyBonus;

				if (
					((cost_num_left > 0) | (cost_num_right > 0)) &//안 이런 경우도 있나?;;;;
					//((cost_num_left > 0) & (cost_num_right > 0)) &	// shouldn't happen, and in any case should be produced by the space cut heuristic earlier.
					(score < bestCost.cost))
				{
					bestCost.cost			= score;
					bestCost.splitPos		= cur_position;
					bestCost.axis			= axis;

					bestCost.n_onlyLeft		= n_leftOnly;			
					bestCost.n_onlyRight	= n_rightOnly;			
					bestCost.n_cross		= n_cross;	
					bestCost.n_planar		= num_planars;	
					bestCost.n_left			= cost_num_left;	
					bestCost.n_right		= cost_num_right;

					bestCost.planar_side	= planar_side;					
				}
			} // scoring
		} // for all split candidates.
	}
}

void GKDTreeStructure::splitWithVisibility( const int axis, GBoundingBox inBBox, 
								 const TriangleInfo *pTriangles, const int triangleSize, 
								 BoundEdge *bEdge,  SplitCost &bestCost, bool emptyTestOnly )
{
	const int axis1 = ( axis + 1 ) % 3, axis2 = ( axis + 2 ) % 3;
	GVector cell_extent( inBBox.m_Max - inBBox.m_Min );

	//다른 축에 비해 너무 작은 축을 자르려면 스킵하는 루틴을 추가할 수도.
	//전체 씬에 대해서 너무 큰 리프를 만들려고 하면 메디안 스플릿?(큰 empty node를 방해할지도)
	const double
		cell_area = cell_extent.x * cell_extent.y + 
					cell_extent.y * cell_extent.z + 
					cell_extent.x * cell_extent.z;
	const double cell_area_rcp = 1 / cell_area,

		area_mul = double( cell_extent.getElement( axis1 ) ) + double( cell_extent.getElement( axis2 ) ),
		area_add = double( cell_extent.getElement( axis1 ) ) * double( cell_extent.getElement( axis2 ) ),

		cost_term_add = m_fTraversalCost,
		cost_term_mul = cell_area_rcp * m_fIntersectionCost;

	const float
		cell_min = inBBox.m_Min.getElement( axis ),
		cell_max = inBBox.m_Max.getElement( axis ),
		cell_length = cell_max - cell_min;

	/**
	 *	모든 split candidate 에 대해서 Cost 를 계산한다.
	 */
	{
		int open = 0, close = 0, num_planars = 0, local_open = 0, local_close = 0;
		
		const unsigned n_bEdge = triangleSize * 2;
		setBoundEdgeList( axis, pTriangles, n_bEdge, bEdge );

		for ( unsigned int i = 0; i < n_bEdge; i++ ) {

			BoundEdge curr_bEdge = bEdge[i];

			//planar는 open과 close에 둘 다 포함됨
			open += local_open + num_planars;
			close += local_close + num_planars;
			local_open = 0;  local_close = 0; num_planars = 0;			

			//현재 axis와 side에 대해 포지션구함
			const float cur_position = curr_bEdge.t;
			{
				//똑같은게 여러개 있을 때는 제일 오른쪽에서만 SAH 계산을 한다.
				for ( unsigned int j = i; j < n_bEdge; j++ )
				{
					BoundEdge tmp_bEdge = bEdge[j];
					if ( tmp_bEdge.t != cur_position) break;

					const bool
						is_left		= (tmp_bEdge.type == BoundEdge::START),
						is_planar	= tmp_bEdge.isPlanar;//(axis);

					//카운팅
					if (!is_planar) {
						local_open	+= is_left ? 1 : 0;
						local_close	+= is_left ? 0 : 1;
					}
					else//플라나하다면 따로 카운팅
						num_planars += is_left ? 1 : 0;	// only count it once

					curr_bEdge = tmp_bEdge;
					i=j;
					assert( cur_position == tmp_bEdge.t );
				}
			}

			//이런 경우가 실제로 생길려나? ==의 경우는 당연히 있겠지.
			if (cur_position <= cell_min + SYS_EPSILON_FLT) continue;//
			if (cur_position >= cell_max - SYS_EPSILON_FLT) break;//

			
			{
				const double
					extent_l = double(cur_position) - cell_min,
					extent_r = cell_max - double(cur_position),
					// area = x*(y+z) + y*z = x*area_mul + area_add
					area_l = extent_l*area_mul + area_add,
					area_r = extent_r*area_mul + area_add;

				const int
					n_leftOnly		= close + local_close,
					n_cross			= open - n_leftOnly,
					n_rightOnly		= triangleSize - (n_leftOnly + n_cross + num_planars);

				//작은 셀이거나, 한쪽이 비어 있다면 다른쪽 셀로 planar가 간다.

				int planar_side = area_l < area_r ? BoundEdge::START : BoundEdge::END;	
				if ((n_leftOnly+n_cross == 0) | (n_rightOnly+n_cross == 0))
					planar_side = n_leftOnly == 0 ? BoundEdge::END : BoundEdge::START;
				//플라나를 한쪽으로 다 밀었으므로 없는 셀은 갯수를 0으로 세팅
				const int
					num_planar_left		= planar_side == BoundEdge::START ? num_planars : 0,
					num_planar_right	= planar_side == BoundEdge::END ? num_planars : 0;

				//최종적인 양쪽 갯수.
				const int
					cost_num_left	= n_leftOnly+n_cross+num_planar_left,
					cost_num_right	= n_rightOnly+n_cross+num_planar_right;

				const float					
					emptyBonus = 
					((cost_num_left == 0 || cost_num_right == 0))  ?	m_fEmptyBonus : 1;

				const double
					factor = 1, // .85, //.5, //
					//아마 cost_num_left를 그대로 안 쓰는 이유는 적당한 factor를 주기 위해서인듯.
					nl = double(n_leftOnly+num_planar_left) + factor*(n_cross),
					nr = double(n_rightOnly+num_planar_right) + factor*(n_cross),

					//이런 흔적을 보니 다양한 시도를 하며 삽질한게 보이는구나.
					//mul = ((nl == 0.) | (nr == 0.)) ? cost_term_mul*factor : cost_term_mul,
					//score = cost_term_add + mul*(nl*area_l + nr*area_r);
					score = (cost_term_add + cost_term_mul*(nl*area_l + nr*area_r))*emptyBonus;
				if (
					((cost_num_left > 0) | (cost_num_right > 0)) &//안 이런 경우도 있나?;;;;
					//((cost_num_left > 0) & (cost_num_right > 0)) &	// shouldn't happen, and in any case should be produced by the space cut heuristic earlier.
					(score < bestCost.cost))
				{
					bestCost.cost			= score;
					bestCost.splitPos		= cur_position;
					bestCost.axis			= axis;

					bestCost.n_onlyLeft		= n_leftOnly;			
					bestCost.n_onlyRight	= n_rightOnly;			
					bestCost.n_cross		= n_cross;	
					bestCost.n_planar		= num_planars;	
					bestCost.n_left			= cost_num_left;	
					bestCost.n_right		= cost_num_right;

					bestCost.planar_side	= planar_side;					
				}
			} // scoring
		} // for all split candidates.
	}
}

inline void GKDTreeStructure::pushChildTriangles( const unsigned n_bEdge, 
												  const BoundEdge *bEdge, 
												  TriangleInfo *pLeftTriangles, 
												  TriangleInfo *pRightTriangles, 
												  const SplitCost &bestCost )
{
	int currLeftIndex = 0, currRightIndex = 0;

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

	assert( currLeftIndex == bestCost.n_left && currRightIndex == bestCost.n_right );
}

bool GKDTreeStructure::interSectEdgePlane( GPoint &p0,	
										   GPoint &p1, 
										   GPoint &planePoint,
										   GVector &planeNorm,
										   GPoint &hitPoint)
{
	GVector u = p1 - p0;
	GVector w = p0 - planePoint;

	float D = planeNorm.innerProduct( u );
	float N = -planeNorm.innerProduct( w );

	// N==0이면 plane에 엣지가 붙어 있는거고, 아니면 intersection안 한거.
	if ( fabs( D ) < SYS_EPSILON_FLT )
		return false;

	float sI = N / D;
	if( sI < 0.f || sI > 1.f )
		return false;

	hitPoint = p0 + u * sI;

	return true;
}

void GKDTreeStructure::splitClipping( const int triangleSize, 
									  const SplitCost &bestCost, 
									  TriangleInfo *pTriangleInfos, 
									  int side )
{
	int axis = bestCost.axis;
	float norm[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
	GVector planeNorm( norm[ axis ][ 0 ], norm[ axis ][ 1 ], norm[ axis ][ 2 ] );

	GPoint planePoint( planeNorm.x*bestCost.splitPos, 
					   planeNorm.y*bestCost.splitPos, 
					   planeNorm.z*bestCost.splitPos );

	for( int i = 0; i < triangleSize; i++ ) {

		GBoundingBox currBBox = pTriangleInfos[i].boundingBox;

		/**
		 *	만약 삼각형이 Split Plane과 교차 한다면,
		 */
		if( ( currBBox.m_Min[ axis ] < bestCost.splitPos ) && 
			( currBBox.m_Max[ axis ] > bestCost.splitPos ) ) {

			GPoint p[3];
			pTriangleInfos[i].pTriangleWrapper->getPoint( p[0], p[1], p[2] );

			int nLeft = 0,	nRight = 0;
			GPoint leftVec[4], rightVec[4];

			//왼쪽 subBox에 있는건 왼쪽에, 오른쪽도 마찬가지로.
			for( int nPoint = 0; nPoint < 3; nPoint++ ) {
				if( p[ nPoint ][ axis ] < bestCost.splitPos )
					leftVec[ nLeft++ ] = p[ nPoint ];
				else
					rightVec[ nRight++ ] = p[ nPoint ];
			}
			assert( nLeft + nRight == 3 );
			{
				const int leftSize = nLeft,
					rightSize = nRight;
				for(int j=0; j<leftSize; j++)
				{
					for(int k=0; k<rightSize; k++)
					{
						GPoint hitPoint;
						if( interSectEdgePlane( leftVec[j], rightVec[k], planePoint, planeNorm, hitPoint ) )
						{
							leftVec[nLeft++] = rightVec[nRight++] = hitPoint;
						}					
					}
				}
			}

			GBoundingBox reducedBBox;

			if( side == 0 )
			{
				for(int j=0; j<nLeft; j++) {
					reducedBBox += leftVec[j];
				}
				reducedBBox.shrink( currBBox );
				assert( currBBox.fullyCover( reducedBBox ) );
				pTriangleInfos[i].boundingBox = reducedBBox;				
			}
			else
			{
				for(int j=0; j<nRight; j++)	{
					reducedBBox += rightVec[j];
				}
				reducedBBox.shrink( currBBox );
				assert( currBBox.fullyCover( reducedBBox ) );
				pTriangleInfos[i].boundingBox = reducedBBox;
			}
		}
	}
}

inline void GKDTreeStructure::reAllocTriangleOffsetList( unsigned int _newAllocSize )
{
	assert( _newAllocSize > m_iAllocatedTriangleOffsetSize );
	unsigned int *tmpList = new unsigned int [ _newAllocSize ];
	memcpy( tmpList, m_pTriangleOffsetList, sizeof( unsigned int ) * m_iAllocatedTriangleOffsetSize );
	delete[] m_pTriangleOffsetList; 
	m_pTriangleOffsetList = tmpList;
	m_iAllocatedTriangleOffsetSize = _newAllocSize;

	GLogManager::logging( LOG_INFO, " ####objectOffset reAlloced = %d", m_iAllocatedTriangleOffsetSize );
}

inline void GKDTreeStructure::reAllocKdtreeNodes(  unsigned int _newAllocSize )
{
	assert( _newAllocSize > m_iAllocatedkdNodeCount );
	kdtreeNode *tmpList = new kdtreeNode[_newAllocSize];
	memcpy( tmpList, m_pKDTreeNodes, sizeof( kdtreeNode ) * m_iAllocatedkdNodeCount );
	delete[] m_pKDTreeNodes;
	m_pKDTreeNodes = tmpList;	
	m_iAllocatedkdNodeCount = _newAllocSize;

	GLogManager::logging( LOG_INFO, " ####kdnode reAlloced = %d", m_iAllocatedkdNodeCount );
}

GTriangleWrapperList *GKDTreeStructure::getTriangleWrapperList()
{
	return m_pSceneTriangleList;
}

GBoundingBox GKDTreeStructure::getBoundingBox()
{
	return m_SceneBBox;
}

bool GKDTreeStructure::loadStructureFromFile( const char *filename )
{
	if (m_pScene->getKdTreeFileType() == SAH) {
		loadfromFile_SAH(filename);
	} else {
		loadfromFile_EmptySAH(filename);
	}
	return true;
}

bool GKDTreeStructure::loadfromFile_EmptySAH( const char *filename )
{
	FILE* f = fopen( filename, "r" );
	if (f == NULL) return false;

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
	GLogManager::logging( LOG_INFO, " -> KDTree load started..." );

	/** 
	 *	Scene 전체의 삼각형 list 를 구성해 온다.
	 */
	m_pSceneTriangleList = m_pScene->createSceneTriangleList( m_SceneBBox );
	m_iSceneTriangleCount = m_pSceneTriangleList->size();

	/**
	 *	KD-Tree 를 위한 데이터 구성. 
	 *	모든 삼각형의 정렬을 위한 공간.offset 정보는 
	 *	m_SceneTriangleList vector 안의 index 와 동일하다.
	 */
	TriangleInfo *pTriangleInfos = new TriangleInfo[ m_iSceneTriangleCount ];
	for( int i = 0; i < m_iSceneTriangleCount; i++ ) {
		pTriangleInfos[ i ].offset = i;
		pTriangleInfos[ i ].pTriangleWrapper = m_pSceneTriangleList->getTriangleWrapper( i );
		/** bounding box 는 실제 triangle bouding box 과는 다를수 있으므로 따로 저장관리 */
		pTriangleInfos[ i ].boundingBox = m_pSceneTriangleList->getTriangleWrapper( i )->m_BBox;
	}

	BoundEdge *bEdge = new BoundEdge[ m_iSceneTriangleCount * 2 ];
	memset( bEdge, 0x00, sizeof( BoundEdge ) * m_iSceneTriangleCount * 2 );

	int i;
	char data[1024] = { 0x00, };
	char data_a[64], data_b[64];

	// ------------------------------------------------------------
	// Load kd-tree node info
	// ------------------------------------------------------------
	int nLeafSize_MIN = 100000;		// 통계용
	int nLeafSize_AVG = 0;			// 통계용
	int nLeafSize_MAX = 0;			// 통계용
	int nTreeEmptyNodeCount = 0;	// 통계용
	int nTreeNodeCount = 0;
	fgets( data, 1024, f );
	sscanf(data+1,"%d", &nTreeNodeCount);

	m_iKDTreeNodeCount      = nTreeNodeCount;
	m_iAllocatedkdNodeCount = 524288;
	while (m_iKDTreeNodeCount >= m_iAllocatedkdNodeCount) {
		m_iAllocatedkdNodeCount *= 2;
	}
	m_pKDTreeNodes = new kdtreeNode[ m_iAllocatedkdNodeCount ];

	KdTreeNode* n = (KdTreeNode*)m_pKDTreeNodes;
	for ( i = 0; i < nTreeNodeCount; i++, n++ )	{
		fgets( data, 1024, f );
		if (data[0] == 'L') {
			int offset;
			int leaftris;
			sscanf(data+2, "%s %s", data_a, data_b);
			sscanf(data_a+4, "%d", &leaftris);
			sscanf(data_b+4, "%d", &offset);
			setLeafNode( n, leaftris, offset );

			if (nLeafSize_MIN > leaftris) nLeafSize_MIN = leaftris;
			if (nLeafSize_MAX < leaftris) nLeafSize_MAX = leaftris;
			nLeafSize_AVG += leaftris;
			if (leaftris == 0) nTreeEmptyNodeCount++;
		} else {
			int   left_child;
			int   split_axis;
			float split_pos;
			split_axis = (data[0] - 'X');
			sscanf(data+2, "%f %d", &split_pos, &left_child);
			setInnerNode( n, split_axis, left_child, split_pos );
		}
	}

	// ------------------------------------------------------------
	// Load kd-tree triangle offset in leafnode
	// ------------------------------------------------------------
	int nTriOffCount;
	fgets( data, 1024, f );
	sscanf(data+1, "%d", &nTriOffCount);
	
	nTriOffCount;
	unsigned int* pOldTriangleOffsetList = new unsigned int [ nTriOffCount ];

	for ( i = 0; i < nTriOffCount; i++ ) {
		fgets( data, 1024, f );
		int triID;
		sscanf(data, "%d", &triID);
		pOldTriangleOffsetList[i] = triID;
	}

	fclose( f );

	// ------------------------------------------------------------
	// Rebuild kd-tree
	// ------------------------------------------------------------
	int nTreeLevel_MIN = 100000;	// 통계용
	int nTreeLevel_AVG = 0;			// 통계용
	int nTreeLevel_MAX = 0;			// 통계용
	int curDepth = 0;				// 통계용
	m_iAllocatedTriangleOffsetSize = 1048576;
	m_pTriangleOffsetList = new unsigned int [ m_iAllocatedTriangleOffsetSize ];
	memset( m_pTriangleOffsetList, 0x00, sizeof( unsigned int ) * m_iAllocatedTriangleOffsetSize );

	const int    stackmax   = (nTreeNodeCount + 1)/2;		// 이미 구성된 Empty SAH tree 의 최대 가능한 level 크기
	unsigned int stackIndex = 0;
	GBoundingBox*	pStack_BBox  = (GBoundingBox*) malloc (stackmax * sizeof (GBoundingBox));
	KdTreeNode**	pStack_Node  = (kdtreeNode**)  malloc (stackmax * sizeof (kdtreeNode*));
	int*			pStack_Depth = (int*)  malloc (stackmax * sizeof (int));
	GBoundingBox	curBBox		 = m_SceneBBox;
	KdTreeNode*		curNode		 = (KdTreeNode*)m_pKDTreeNodes;

	while (1) {
		while (IS_LEAF(*curNode) == 0) {
			const float node_split = SPLIT_POS(*curNode);
			const unsigned int dim = SPLIT_AXIS(*curNode);
			KdTreeNode *FrontSideSon	= &m_pKDTreeNodes[FIRST_CHILD_OFFSET(*curNode)];
			KdTreeNode *BackSideSon		= &m_pKDTreeNodes[FIRST_CHILD_OFFSET(*curNode) + 1];

			GBoundingBox FrontSideBBox	= curBBox;	FrontSideBBox.m_Max[ dim ] = node_split;
			GBoundingBox BackSideBBox	= curBBox;	BackSideBBox.m_Min[ dim ]  = node_split;

			curNode = FrontSideSon;
			curBBox = FrontSideBBox;
			pStack_Depth[stackIndex] = curDepth+1;
			pStack_Node[stackIndex]  = BackSideSon;
			pStack_BBox[stackIndex]  = BackSideBBox;
			stackIndex++;
			curDepth++;
		}

		if (nTreeLevel_MIN > curDepth) nTreeLevel_MIN = curDepth;
		if (nTreeLevel_MAX < curDepth) nTreeLevel_MAX = curDepth;
		nTreeLevel_AVG += curDepth;

		{
			// pTriangles 구성
			const int baseOffset = OBJECTLIST_OFFSET(*curNode);
			const int nObjs		 = OBJECT_SIZE(*curNode);

			int currIndex = 0;
			TriangleInfo *pTriangles = new TriangleInfo[ nObjs ];

			for (i = baseOffset; i < baseOffset+nObjs; i++) {
				int     triID = pOldTriangleOffsetList[i];
				pTriangles[ currIndex++ ] = pTriangleInfos[triID];
			}

			//buildKDTree( bEdge, pTriangles, nObjs, curBBox, stackIndex, &m_pKDTreeNodes[ m_iKDTreeNodeCount ] );
			buildKDTree( bEdge, pTriangles, nObjs, curBBox, stackIndex, curNode );

			//delete[] pTriangles; // buildKDTree() 안에서 메모리가 해제됨
		}
		
		if (stackIndex == 0) break;

		--stackIndex;
		curDepth = pStack_Depth[stackIndex];
		curNode  = pStack_Node[stackIndex];
		curBBox  = pStack_BBox[stackIndex];
	}


	free(pStack_Depth);
	free(pStack_Node);
	free(pStack_BBox);
	delete[] pOldTriangleOffsetList;
	delete[] bEdge;

	timer.end();

	GLogManager::logging( LOG_INFO, "--------------------------------------------------------------------" );
	GLogManager::logging( LOG_INFO, " -> EMPTY TREE All          Node Count : %d", nTreeNodeCount );
	GLogManager::logging( LOG_INFO, " -> EMPTY TREE Empty / Leaf Node Count : %d %d", nTreeEmptyNodeCount, (nTreeNodeCount+1)/2 );
	GLogManager::logging( LOG_INFO, " -> EMPTY TREE Min/Avg/Max treeLevel  : %7d %7.1f %7d", nTreeLevel_MIN, nTreeLevel_AVG*2.0f/(nTreeNodeCount+1), nTreeLevel_MAX );
	GLogManager::logging( LOG_INFO, " -> EMPTY TREE Min/Avg/Max LeafSize   : %7d %7.1f %7d", nLeafSize_MIN, nLeafSize_AVG*2.0f/(nTreeNodeCount+1), nLeafSize_MAX );

	GLogManager::logging( LOG_INFO, "--------------------------------------------------------------------" );
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


	return true;
}



#define BINARY_KDTREE_FILE 0

bool GKDTreeStructure::loadfromFile_SAH( const char *filename )
{
	FILE* f = fopen( filename, "r" );
	if (f == NULL) return false;

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


//	GLogManager::logging( LOG_INFO, "------------------------ KDTree Spatial Structure -------------------" );
//	GLogManager::logging( LOG_INFO, " -> KDTree load started..." );

	/** 
	 *	Scene 전체의 삼각형 list 를 구성해 온다.
	 */
	m_pSceneTriangleList = m_pScene->createSceneTriangleList( m_SceneBBox );
	m_iSceneTriangleCount = m_pSceneTriangleList->size();

#if 0
	/**
	 *	KD-Tree 를 위한 데이터 구성. 
	 *	모든 삼각형의 정렬을 위한 공간.offset 정보는 
	 *	m_SceneTriangleList vector 안의 index 와 동일하다.
	 */
	TriangleInfo *pTriangleInfos = new TriangleInfo[ m_iSceneTriangleCount ];
	for( int i = 0; i < m_iSceneTriangleCount; i++ ) {
		pTriangleInfos[ i ].offset = i;
		pTriangleInfos[ i ].pTriangleWrapper = m_pSceneTriangleList->getTriangleWrapper( i );
		/** bounding box 는 실제 triangle bouding box 과는 다를수 있으므로 따로 저장관리 */
		pTriangleInfos[ i ].boundingBox = m_pSceneTriangleList->getTriangleWrapper( i )->m_BBox;
	}

	BoundEdge *bEdge = new BoundEdge[ m_iSceneTriangleCount * 2 ];
	memset( bEdge, 0x00, sizeof( BoundEdge ) * m_iSceneTriangleCount * 2 );
#endif


	int i;
	char data[1024] = { 0x00, };
	char data_a[64], data_b[64];

	// ------------------------------------------------------------
	// Load kd-tree node info
	// ------------------------------------------------------------
	int nTreeNodeCount = 0;
	fgets( data, 1024, f );
	sscanf(data+1,"%d", &nTreeNodeCount);
	m_iAllocatedkdNodeCount = m_iKDTreeNodeCount = nTreeNodeCount;
	m_pKDTreeNodes = new kdtreeNode[ m_iAllocatedkdNodeCount ];

	KdTreeNode* n = (KdTreeNode*)m_pKDTreeNodes;
	for ( i = 0; i < nTreeNodeCount; i++, n++ )	{
		fgets( data, 1024, f );
		if (data[0] == 'L') {
			int offset;
			int leaftris;
			sscanf(data+2, "%s %s", data_a, data_b);
			sscanf(data_a+4, "%d", &leaftris);
			sscanf(data_b+4, "%d", &offset);
			setLeafNode( n, leaftris, offset );
		} else {
			int   left_child;
			int   split_axis;
			float split_pos;
			split_axis = (data[0] - 'X');
			sscanf(data+2, "%f %d", &split_pos, &left_child);
			setInnerNode( n, split_axis, left_child, split_pos );
		}
	}

	// ------------------------------------------------------------
	// Load kd-tree triangle offset in leafnode
	// ------------------------------------------------------------
	int nTriOffCount;
	fgets( data, 1024, f );
	sscanf(data+1, "%d", &nTriOffCount);
	
	m_iAllocatedTriangleOffsetSize = m_iCurrentTriangleOffset = nTriOffCount;
	m_pTriangleOffsetList = new unsigned int [ m_iAllocatedTriangleOffsetSize ];

	for ( i = 0; i < nTriOffCount; i++ ) {
		fgets( data, 1024, f );
		int triID;
		sscanf(data, "%d", &triID);
		m_pTriangleOffsetList[i] = triID;
	}

	fclose( f );

	return true;
}

bool GKDTreeStructure::saveStructureToFile( const char *filename )
{
	const char axis_label[3][2] = { "X", "Y", "Z" };

	FILE* f = fopen( filename, "w" );
	int i;

#if BINARY_KDTREE_FILE
	// ------------------------------------------------------------
	// Dump kd-tree node info
	// ------------------------------------------------------------
	int nTreeNodeCount = m_iKDTreeNodeCount;
	fwrite( &nTreeNodeCount, 4, 1, f );						// 전체 node 개수 write(4)

	KdTreeNode* node = &m_pKDTreeNodes[0];

	for ( i = 0; i < nTreeNodeCount; i++, node++ ) {
		if (IS_LEAF(*node) == 0) {
			fwrite( node, 8, 1, f );					// internal node 정보 write (8)
		} else {
			fwrite( node, 8, 1, f );
		}
	}

	// ------------------------------------------------------------
	// Dump kd-tree triangle offset in leafnode
	// ------------------------------------------------------------
	int nTriOffCount = m_iCurrentTriangleOffset;
	fwrite( &nTriOffCount, 4, 1, f );

	for ( i = 0; i < nTriOffCount; i++ ) {
		fwrite( &offset, 4, 1, f );	
	}
#else

	// ------------------------------------------------------------
	// Dump kd-tree node info
	// ------------------------------------------------------------
	int nTreeNodeCount = m_iKDTreeNodeCount;
	fprintf(f, "n%d\n", nTreeNodeCount);

	KdTreeNode* node = &m_pKDTreeNodes[0];

	for ( i = 0; i < nTreeNodeCount; i++, node++ ) {
		if (IS_LEAF(*node) == 0) {
			int   left_child = FIRST_CHILD_OFFSET(*node);
			int   split_axis = SPLIT_AXIS(*node);
			float split_pos  = SPLIT_POS(*node);
			fprintf(f, "%s|%f %d<>%d\n", axis_label[split_axis], split_pos, left_child, left_child+1);
		} else {
			int offset   = OBJECTLIST_OFFSET(*node);
			int leaftris = OBJECT_SIZE(*node);
			fprintf(f, "L cnt:%d off:%d\n", leaftris, offset);
		}
	}

	// ------------------------------------------------------------
	// Dump kd-tree triangle offset in leafnode
	// ------------------------------------------------------------
	int nTriOffCount = m_iCurrentTriangleOffset;
	fprintf(f, "o%d\n", nTriOffCount);

	for ( i = 0; i < nTriOffCount; i++ ) {
		int triID = m_pTriangleOffsetList[i];
		fprintf(f, "%d\n", triID);
	}
#endif

	fclose( f );

	return true;
}