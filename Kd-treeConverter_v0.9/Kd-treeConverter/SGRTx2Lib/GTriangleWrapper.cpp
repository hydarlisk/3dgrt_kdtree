#include "GTriangleWrapper.h"
#include "GPolygonObject.h"

#include "SSE_math.h"

GTriangleWrapper::GTriangleWrapper(void)
{
	p0 = NULL;
	p1 = NULL;
	p2 = NULL;
	n0 = NULL;
	n1 = NULL;
	n2 = NULL;
	uv0 = NULL;
	uv1 = NULL;
	uv2 = NULL;
	visibility = 0.0;

	m_mailBoxId = -1;
}

GTriangleWrapper::~GTriangleWrapper(void)
{
}

float GTriangleWrapper::calArea()
{
	GVector a, b;
	
	a.x = p1[0] - p0[0]; a.y = p1[1] - p0[1]; a.z = p1[2] - p0[2];
	b.x = p2[0] - p0[0]; b.y = p2[1] - p0[1]; b.z = p2[2] - p0[2];

	return 0.5f * a.outerProduct( b ).length();
}

void GTriangleWrapper::getPoint( GPoint &point0, GPoint &point1, GPoint &point2 )
{
	point0.setPoint( p0[0], p0[1], p0[2] );
	point1.setPoint( p1[0], p1[1], p1[2] );
	point2.setPoint( p2[0], p2[1], p2[2] );
}
//shyun
void GTriangleWrapper::setPoint(GPoint& point0, GPoint& point1, GPoint& point2)
{
	//p0 = point0.getPoint();
	p0[0] = point0.getPoint()[0];
	p0[1] = point0.getPoint()[1];
	p0[2] = point0.getPoint()[2];
	p1[0] = point1.getPoint()[0];
	p1[1] = point1.getPoint()[1];
	p1[2] = point1.getPoint()[2];
	p2[0] = point2.getPoint()[0];
	p2[1] = point2.getPoint()[1];
	p2[2] = point2.getPoint()[2];
}
//shyun end
GVector GTriangleWrapper::calBarycentricNormal( float alpha, float beta, float gamma )
{
	return m_pObject->calBarycentricNormal( indexInObject, alpha, beta, gamma );
}

GPoint GTriangleWrapper::calBarycentricPosition( float alpha, float beta, float gamma )
{
	return m_pObject->calBarycentricPosition( indexInObject, alpha, beta, gamma );
}

GPoint GTriangleWrapper::calBarycentricUV( float alpha, float beta, float gamma )
{
	return m_pObject->calBarycentricUV( indexInObject, alpha, beta, gamma );
}

void GTriangleWrapper::calBarycentricNormal( __m128 alpha, __m128 beta, __m128 gamma, _sse_vec *normal )
{
	return m_pObject->calBarycentricNormal( indexInObject, alpha, beta, gamma, normal );
}

void GTriangleWrapper::calBarycentricUV( __m128 alpha, __m128 beta, __m128 gamma, __m128 *uv )
{
	m_pObject->calBarycentricUV( indexInObject, alpha, beta, gamma, uv );
}

void GTriangleWrapper::calCentroid(GVector& cen)
{
	cen.x = (p0[0] + p1[0] + p2[0]) / 3.0f;
	cen.y = (p0[1] + p1[1] + p2[1]) / 3.0f;
	cen.z = (p0[2] + p1[2] + p2[2]) / 3.0f;
}
