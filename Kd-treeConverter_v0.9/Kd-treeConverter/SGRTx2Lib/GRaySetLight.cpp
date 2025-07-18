#include "GRaySetLight.h"

GRaySetLight::GRaySetLight(void)
{
	m_pRaySetData = NULL;
	m_iRaySetDataCount = 0;
	m_bRandomMode = true;
}

GRaySetLight::~GRaySetLight(void)
{
	if ( m_pRaySetData )
		delete m_pRaySetData;
}

void GRaySetLight::setRaySetData( GRaySet *pRaySet, int count )
{
	m_pRaySetData = pRaySet;
	m_iRaySetDataCount = count;

	makeDebugInfoObject();
}

GRaySet* GRaySetLight::getRaySetData()
{
	return m_pRaySetData;
}

int GRaySetLight::getRaySetDataCount()
{
	return m_iRaySetDataCount;
}

/**
 *	지점 pos 와 normal 을 가진 현재지점으로 들어오는 radiance 를 구한다.
 */
GColor GRaySetLight::getRadiance( GPoint &pos, GVector &normal )
{
	return GColor( 0.0f, 0.0f, 0.0f, 1.0f );
}

/**
 *	polygon 데이터를 변경한다음에,
 *	rayset 데이터도 변환한다.
 *	변환하고, matrix 를 identity 로 초기화 한다.
 */
GError GRaySetLight::convertToWorldObject()
{
	if ( m_pRaySetData == NULL )
		return errorNo;

	GPoint point, normal;

	for ( int i = 0; i < m_iRaySetDataCount; ++i ) {

		point.x = m_pRaySetData[ i ].pos[ 0 ];
		point.y = m_pRaySetData[ i ].pos[ 1 ];
		point.z = m_pRaySetData[ i ].pos[ 2 ];

		normal.x = m_pRaySetData[ i ].dir[ 0 ];
		normal.y = m_pRaySetData[ i ].dir[ 1 ];
		normal.z = m_pRaySetData[ i ].dir[ 2 ];
		
		point =	this->m_Matrix * point;
		normal = this->m_NormalMatrix * normal;

		m_pRaySetData[ i ].pos[ 0 ] = point.x;
		m_pRaySetData[ i ].pos[ 1 ] = point.y;
		m_pRaySetData[ i ].pos[ 2 ] = point.z;

		m_pRaySetData[ i ].dir[ 0 ] = normal.x;
		m_pRaySetData[ i ].dir[ 1 ] = normal.y;
		m_pRaySetData[ i ].dir[ 2 ] = normal.z;

	}

	makeDebugInfoObject();

	/** 
	 *	반드시 마지막에 현재 polygon 데이터도 convert 시켜야 한다. 
	 */
	return GLight::convertToWorldObject();
}

GError GRaySetLight::validObject()
{
	return errorNo;
}

void GRaySetLight::makeDebugInfoObject()
{
	/**
	 *	화면에 light 를 점으로 보여주기 위해서, debug info 
	 *	object 를 구성한다.
	 *	RaySet 정보에서 point 와 normal 을 떼어내서
	 *	point set 을 위한 geometry 정보로 구성한다. point 정보로 구성.
	 */
	float *pVertexArray = (float*) malloc( sizeof( float ) * m_iRaySetDataCount * 3 );
	float *pNormalArray = (float*) malloc( sizeof( float ) * m_iRaySetDataCount * 3 );
	int *pIndexArray = (int*) malloc( sizeof( int ) * m_iRaySetDataCount );

	for ( int i = 0; i < m_iRaySetDataCount; ++i ) {

		pVertexArray[ i * 3 + 0 ] = m_pRaySetData[ i ].pos[ 0 ];
		pVertexArray[ i * 3 + 1 ] = m_pRaySetData[ i ].pos[ 1 ];
		pVertexArray[ i * 3 + 2 ] = m_pRaySetData[ i ].pos[ 2 ];

		pNormalArray[ i * 3 + 0 ] = m_pRaySetData[ i ].dir[ 0 ];
		pNormalArray[ i * 3 + 1 ] = m_pRaySetData[ i ].dir[ 1 ];
		pNormalArray[ i * 3 + 2 ] = m_pRaySetData[ i ].dir[ 2 ];

		pIndexArray[ i ] = i;

	}

	GPolygonObject *pDebugObject = new GPolygonObject();

	pDebugObject->getMaterial()->m_Diffuse = getMaterial()->m_Diffuse;

	pDebugObject->setVertexCount( m_iRaySetDataCount );
	pDebugObject->setVertexArray( pVertexArray );
	pDebugObject->setNormalArray( pNormalArray );

	pDebugObject->setTriangleCount( m_iRaySetDataCount );
	pDebugObject->setIndexArray( pIndexArray );
	pDebugObject->setPolygonType( typePolygonPoint );
	pDebugObject->setIntersection( false );

	this->setDebugInfoObject( pDebugObject );
}