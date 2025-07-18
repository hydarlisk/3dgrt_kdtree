#include ".\gintersectionpointmap.h"

GIntersectionPointMap::GIntersectionPointMap( 
						GDimension resolution, 
						GDimension sampling, int maxDepth )
{
	m_Resolution = resolution;
	m_Sampling = sampling;

	m_iCurrentCount = 0;
	m_iMaxCount = resolution.x * resolution.y * sampling.x * sampling.y * maxDepth + 5;

	m_pPointData = ( cuIntersectionPoint* ) malloc( sizeof( cuIntersectionPoint ) * m_iMaxCount );
	memset( m_pPointData, -1, sizeof( cuIntersectionPoint ) * m_iMaxCount );
}

GIntersectionPointMap::~GIntersectionPointMap(void)
{
	if ( m_pPointData )
		free( m_pPointData );
}

/**
 *	Intersection point clear.
 */
void GIntersectionPointMap::clear()
{
	m_iCurrentCount = 0;

	// memset 하지말자. fps 가 1/3 로 준다. 메모리가 너무 커서.
	// memset( m_pPointData, -1, sizeof( cuIntersectionPoint ) * m_iMaxCount );
}

/**
 */
GError GIntersectionPointMap::insertIntersectionPoint( cuIntersectionPoint *point, int count )
{
	if ( m_iCurrentCount + count >= m_iMaxCount )
		return errorOverflowMaxDepth;

	memcpy( m_pPointData + m_iCurrentCount, point, sizeof( cuIntersectionPoint ) * count );
	m_iCurrentCount += count;

	return errorNo;
}

const cuIntersectionPoint* GIntersectionPointMap::getIntersectionPoint( int index )
{
	return ( m_pPointData + index );
}

const int GIntersectionPointMap::getSize()
{
	return m_iCurrentCount;
}
