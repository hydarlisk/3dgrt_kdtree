#include ".\GPolygonObject.h"
#include "GClassMacro.h"

#include "SSE_math.h"

GPolygonObject::GPolygonObject(void)
{
	m_pVertexArray = NULL;
	m_pNormalArray = NULL;
	m_pColorArray = NULL;
	m_pUVArray = NULL;
	m_pIndexArray = NULL;
	m_iVertexCount = 0;
	m_iTriangleCount = 0;
	m_pVisibilityArray = NULL;
}

GPolygonObject::~GPolygonObject(void)
{
	if ( m_pVertexArray )
		free( m_pVertexArray );
	if ( m_pNormalArray )
		free( m_pNormalArray );
	if ( m_pColorArray )
		free( m_pColorArray );
	if ( m_pUVArray )
		free( m_pUVArray );
	if ( m_pIndexArray )
		free( m_pIndexArray );
	if( m_pVisibilityArray )
		free( m_pVisibilityArray );
}

GError GPolygonObject::validObject()
{
	if ( m_iVertexCount == 0 || m_iTriangleCount == 0 )
		return errorInvalidObject;

	return errorNo;
}

/**
 *	현재 삼각형 정보에 transform matrix 를
 *	적용해서 데이터를 월드좌표계로 모두 옮긴후에
 *	transform matrix 는 identity 로 만든다.
 */
GError GPolygonObject::convertToWorldObject()
{
	applyTransform( &this->m_Matrix, &this->m_NormalMatrix );
	identityTransform();
	updateBoundingBox();

	return errorNo;
}

/**
 *	현재 삼각형 정보를 가지고 bounding box 를 다시 계산한다.
 */
void GPolygonObject::updateBoundingBox()
{
	float x, y, z;
	GVector bmin, bmax;

	for ( int i = 0; i < m_iVertexCount; ++i ) {

		x = *( m_pVertexArray + i * 3 + 0 );
		y = *( m_pVertexArray + i * 3 + 1 );
		z = *( m_pVertexArray + i * 3 + 2 );

		if ( i == 0 ) {
			bmin.setVector( x, y, z );
			bmax.setVector( x, y, z );
		} else {
			bmin.x = min( bmin.x, x ); bmin.y = min( bmin.y, y ); bmin.z = min( bmin.z, z );
			bmax.x = max( bmax.x, x ); bmax.y = max( bmax.y, y ); bmax.z = max( bmax.z, z );
		}
	}

	setBoundingBox( GBoundingBox( bmin, bmax ) );
}

/**
 *	현재 삼각형 데이터에 transform matrix 를 적용해서
 *	데이터 자체를 변경한다.
 */
void GPolygonObject::applyTransform( GMatrix4 *pTransformMatrix, 
									  GMatrix4 *pNormalTransformMatrix )
{
	float *x, *y, *z;
	float *nx, *ny, *nz;
	GPoint point;
	GVector normal;

	/**
	 *	삼각형의 point, normal 을 transform 한다.
	 */
	for ( int i = 0; i < m_iVertexCount; ++i ) {

		x = ( m_pVertexArray + i * 3 + 0 );
		y = ( m_pVertexArray + i * 3 + 1 );
		z = ( m_pVertexArray + i * 3 + 2 );
		nx = ( m_pNormalArray + i * 3 + 0 );
		ny = ( m_pNormalArray + i * 3 + 1 );
		nz = ( m_pNormalArray + i * 3 + 2 );

		point.setPoint( *x, *y, *z );
		normal.setVector( *nx, *ny, *nz );

		point = (*pTransformMatrix) * point;
		normal = ( (*pNormalTransformMatrix) * normal ).normalize();

		*x = point.x; *y = point.y; *z = point.z;
		*nx = normal.x; *ny = normal.y; *nz = normal.z;

	}
}

void GPolygonObject::setVertexArray( float *array )
{
	if ( m_pVertexArray )
		free( m_pVertexArray );
	m_pVertexArray = array;
}

/** 
 * vertex 배열의 pointer 를 리턴한다. pCount 에는 vertex 갯수 리턴 
 */
const float* GPolygonObject::getVertexArray()
{
	return m_pVertexArray;
}

void GPolygonObject::setNormalArray( float *array )
{
	if ( m_pNormalArray )
		free( m_pNormalArray );
	m_pNormalArray = array;
}

/** 
 * normal 배열의 pointer 를 리턴한다. count 에는 normal 갯수 리턴 
 */
const float* GPolygonObject::getNormalArray()
{
	return m_pNormalArray;
}


void GPolygonObject::setIndexArray( int *array )
{
	if ( m_pIndexArray )
		free( m_pIndexArray );
	m_pIndexArray = array;
}

/** 
 * triangle index 배열의 pointer 를 리턴한다. count 에는 triangle 갯수 리턴 
 */
const int* GPolygonObject::getIndexArray()
{
	return m_pIndexArray;
}

void GPolygonObject::setUVArray( float *array )
{
	if ( m_pUVArray )
		free( m_pUVArray );
	m_pUVArray = array;
}

/** 
 * UV 데이터 배열의 pointer 를 리턴한다. count 에는 UV 갯수 리턴 
 */
const float* GPolygonObject::getUVArray()
{
	return m_pUVArray;
}

void GPolygonObject::setColorArray( float *array )
{
	if ( m_pColorArray )
		free( m_pColorArray );
	m_pColorArray = array;
}
/** 
 * Color 데이터 배열의 pointer 를 리턴한다. count 에는 UV 갯수 리턴 
 */
const float* GPolygonObject::getColorArray()
{
	return m_pColorArray;
}

void GPolygonObject::setVisibilityArray( float *array )
{
	if( m_pVisibilityArray )
		free( m_pVisibilityArray );
	m_pVisibilityArray = array;
}

const float* GPolygonObject::getVisibilityArray()
{
	return m_pVisibilityArray;
}

int GPolygonObject::getTriangleCount()
{
	return m_iTriangleCount;
}

/**  
 *	현재 Object 의 삼각형 정보를 list 에 담아서 리턴한다.
 *	baseOffset 은 이 object 안의 삼각형이 가질 offset 의 시작번호.
 *	전체 Scene 에 걸쳐서 각 삼각형은 고유한 offset 을 가져야 하기 때문에
 *	baseOffset 은 이전 삼각형들이 가지고 있던 최대 offset 이다.
 */
//#define FOR_FAIRY

void GPolygonObject::getTriangleList( GTriangleWrapperList *pList, int objectIndexInScene, int baseIndex )
{
	/**
	 *	TriangleWrapper 를 구성해서 가지게 한다.
	 *	index 3개가 삼각형 하나를 이룸을 명심하라.
	 */
	GVector bmin, bmax;

	for ( int i = 0; i < m_iTriangleCount; ++i ) {

		GTriangleWrapper *ptWrapper = new GTriangleWrapper();

		ptWrapper->p0 = ( m_pVertexArray + m_pIndexArray[ i * 3 + 0 ] * 3 );
		ptWrapper->p1 = ( m_pVertexArray + m_pIndexArray[ i * 3 + 1 ] * 3 );
		ptWrapper->p2 = ( m_pVertexArray + m_pIndexArray[ i * 3 + 2 ] * 3 );
		ptWrapper->n0 = ( m_pNormalArray + m_pIndexArray[ i * 3 + 0 ] * 3 );
		ptWrapper->n1 = ( m_pNormalArray + m_pIndexArray[ i * 3 + 1 ] * 3 );
		ptWrapper->n2 = ( m_pNormalArray + m_pIndexArray[ i * 3 + 2 ] * 3 );

		if ( m_pUVArray == NULL ) {
			ptWrapper->uv0 = NULL;
			ptWrapper->uv1 = NULL;
			ptWrapper->uv2 = NULL;
		} else {
			ptWrapper->uv0 = ( m_pUVArray + m_pIndexArray[ i * 3 + 0 ] * 2 );
			ptWrapper->uv1 = ( m_pUVArray + m_pIndexArray[ i * 3 + 1 ] * 2 );
			ptWrapper->uv2 = ( m_pUVArray + m_pIndexArray[ i * 3 + 2 ] * 2 );
		}

		ptWrapper->m_pObject = this;

		bmin.x = min( min( ptWrapper->p0[ 0 ], ptWrapper->p1[ 0 ] ), ptWrapper->p2[ 0 ] );
		bmin.y = min( min( ptWrapper->p0[ 1 ], ptWrapper->p1[ 1 ] ), ptWrapper->p2[ 1 ] );
		bmin.z = min( min( ptWrapper->p0[ 2 ], ptWrapper->p1[ 2 ] ), ptWrapper->p2[ 2 ] );
		bmax.x = max( max( ptWrapper->p0[ 0 ], ptWrapper->p1[ 0 ] ), ptWrapper->p2[ 0 ] );
		bmax.y = max( max( ptWrapper->p0[ 1 ], ptWrapper->p1[ 1 ] ), ptWrapper->p2[ 1 ] );
		bmax.z = max( max( ptWrapper->p0[ 2 ], ptWrapper->p1[ 2 ] ), ptWrapper->p2[ 2 ] );

		ptWrapper->m_BBox.setMax( bmax );
		ptWrapper->m_BBox.setMin( bmin );
		ptWrapper->indexInObject = i;
		ptWrapper->index = baseIndex + i;
		ptWrapper->objectIndexInScene = objectIndexInScene;

		pList->addTriangleWrapper( ptWrapper );

		/******** selection adaptive supersampling 을 위한 정보. 현재 삼각형이 선택되었는지를 체크 */

#ifdef FOR_FAIRY 

		ptWrapper->bSelected = false;

		/** fairy scene 을 위한 하드코딩 세팅 */
			
		static char name[27][100] = { "wing3in", "wand", "wandballs", "spirals", "skinhand", "skinforearm",
							   "skintorso", "skinarm", "straps", "bodice", "overskirt",
							   "bodicebottom", "skinhip", "skinleg", "skinfeet", "antennae", "basemess",
							   "bangs", "skinhead", "eyebrows", "eyelashes", "burnmess",
							   "skinneck", "bunmess", "wings", "dbody", "grass" };
		for ( int i = 0; i < 27; ++i ) {
			if ( stricmp( name[i], m_szObjectName ) == 0 )
				ptWrapper->bSelected = true;
		}
#else#

		ptWrapper->bSelected = true;

#endif

	}
}

void GPolygonObject::setTriangleCount( int count )
{
	m_iTriangleCount = count;
}

int GPolygonObject::getVertexCount()
{
	return m_iVertexCount;
}

void GPolygonObject::setVertexCount( int count )
{
	m_iVertexCount = count;
}

GVector GPolygonObject::calBarycentricNormal( int indexInObject, float alpha, float beta, float gamma )
{
	GVector n0, n1, n2;
	float* temp;

	temp = ( m_pNormalArray + m_pIndexArray[ indexInObject * 3 + 0 ] * 3 + 0 );
	n0.x = temp[ 0 ];
	n0.y = temp[ 1 ];
	n0.z = temp[ 2 ];

	temp = ( m_pNormalArray + m_pIndexArray[ indexInObject * 3 + 1 ] * 3 + 0 );
	n1.x = temp[ 0 ];
	n1.y = temp[ 1 ];
	n1.z = temp[ 2 ];

	temp = ( m_pNormalArray + m_pIndexArray[ indexInObject * 3 + 2 ] * 3 + 0 );
	n2.x = temp[ 0 ];
	n2.y = temp[ 1 ];
	n2.z = temp[ 2 ];

	//return ( n0 * alpha + n1 * beta + n2 * gamma ).normalize();

	_GVEC_vMUL(n0,n0,alpha);
	_GVEC_vMUL(n1,n1,beta);
	_GVEC_vMUL(n2,n2,gamma);
	_GVEC_vADD(n0,n0,n1);
	_GVEC_vADD(n0,n0,n2);
	_GVEC_vNORMAL(n0);
	return n0;
}

GPoint GPolygonObject::calBarycentricPosition(  int indexInObject, float alpha, float beta, float gamma )
{
	GPoint p0, p1, p2;
	float* temp;

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 0 ] * 3 + 0 );
	p0.x = temp[ 0 ];
	p0.y = temp[ 1 ];
	p0.z = temp[ 2 ];

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 1 ] * 3 + 0 );
	p1.x = temp[ 0 ];
	p1.y = temp[ 1 ];
	p1.z = temp[ 2 ];

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 2 ] * 3 + 0 );
	p2.x = temp[ 0 ];
	p2.y = temp[ 1 ];
	p2.z = temp[ 2 ];

	return p0 * alpha + p1 * beta + p2 * gamma;
}

GPoint GPolygonObject::calBarycentricUV( int indexInObject, float alpha, float beta, float gamma )
{
	GPoint p0, p1, p2;
	float* temp;

	temp = ( m_pUVArray + m_pIndexArray[ indexInObject * 3 + 0 ] * 2 + 0 );
	p0.x = temp[ 0 ];
	p0.y = temp[ 1 ];
	p0.z = 0.0f;

	temp = ( m_pUVArray + m_pIndexArray[ indexInObject * 3 + 1 ] * 2 + 0 );
	p1.x = temp[ 0 ];
	p1.y = temp[ 1 ];
	p1.z = 0.0f;

	temp = ( m_pUVArray + m_pIndexArray[ indexInObject * 3 + 2 ] * 2 + 0 );
	p2.x = temp[ 0 ];
	p2.y = temp[ 1 ];
	p2.z = 0.0f;

	//return p0 * alpha + p1 * beta + p2 * gamma;

	_GPNT_vMUL(p0,p0,alpha);
	_GPNT_vMUL(p1,p1,beta);
	_GPNT_vMUL(p2,p2,gamma);
	_GPNT_vADD(p0,p0,p1);
	_GPNT_vADD(p0,p0,p2);
	return p0;
}

void GPolygonObject::calBarycentricNormal( int indexInObject, __m128 alpha, __m128 beta, __m128 gamma, _sse_vec* normal )
{
	GVector n0, n1, n2;
	float* temp;

	temp = ( m_pNormalArray + m_pIndexArray[ indexInObject * 3 + 0 ] * 3 + 0 );
	n0.x = temp[ 0 ];
	n0.y = temp[ 1 ];
	n0.z = temp[ 2 ];

	temp = ( m_pNormalArray + m_pIndexArray[ indexInObject * 3 + 1 ] * 3 + 0 );
	n1.x = temp[ 0 ];
	n1.y = temp[ 1 ];
	n1.z = temp[ 2 ];

	temp = ( m_pNormalArray + m_pIndexArray[ indexInObject * 3 + 2 ] * 3 + 0 );
	n2.x = temp[ 0 ];
	n2.y = temp[ 1 ];
	n2.z = temp[ 2 ];

	normal->x4 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(n0.x), alpha),
					_mm_mul_ps(_mm_set1_ps(n1.x), beta)), _mm_mul_ps(_mm_set1_ps(n2.x), gamma));
	normal->y4 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(n0.y), alpha),
					_mm_mul_ps(_mm_set1_ps(n1.y), beta)), _mm_mul_ps(_mm_set1_ps(n2.y), gamma));
	normal->z4 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(n0.z), alpha),
					_mm_mul_ps(_mm_set1_ps(n1.z), beta)), _mm_mul_ps(_mm_set1_ps(n2.z), gamma));

	// normalize
	const __m128 v1 = sse_rsqrt( _mm_add_ps( _mm_add_ps( _mm_mul_ps( normal->x4, normal->x4 ), 
					  _mm_mul_ps( normal->y4, normal->y4 ) ), _mm_mul_ps( normal->z4, normal->z4 ) ) );
	normal->x4 = _mm_mul_ps( normal->x4, v1 );
	normal->y4 = _mm_mul_ps( normal->y4, v1 );
	normal->z4 = _mm_mul_ps( normal->z4, v1 );
}

void GPolygonObject::calBarycentricUV(  int indexInObject, __m128 alpha, __m128 beta, __m128 gamma, __m128 *uv )
{
	GPoint p0, p1, p2;
	float* temp;

	temp = ( m_pUVArray + m_pIndexArray[ indexInObject * 3 + 0 ] * 2 + 0 );
	p0.x = temp[ 0 ];
	p0.y = temp[ 1 ];
	p0.z = 0.0f;

	temp = ( m_pUVArray + m_pIndexArray[ indexInObject * 3 + 1 ] * 2 + 0 );
	p1.x = temp[ 0 ];
	p1.y = temp[ 1 ];
	p1.z = 0.0f;

	temp = ( m_pUVArray + m_pIndexArray[ indexInObject * 3 + 2 ] * 2 + 0 );
	p2.x = temp[ 0 ];
	p2.y = temp[ 1 ];
	p2.z = 0.0f;

	uv[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(p0.x), alpha),
					_mm_mul_ps(_mm_set1_ps(p1.x), beta)), _mm_mul_ps(_mm_set1_ps(p2.x), gamma));
	uv[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(p0.y), alpha),
					_mm_mul_ps(_mm_set1_ps(p1.y), beta)), _mm_mul_ps(_mm_set1_ps(p2.y), gamma));
}

GVector GPolygonObject::getCenter( int indexInObject )
{
	GVector n0, n1, n2;
	float* temp;

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 0 ] * 3 + 0 );
	n0.x = temp[ 0 ];
	n0.y = temp[ 1 ];
	n0.z = temp[ 2 ];

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 1 ] * 3 + 0 );
	n1.x = temp[ 0 ];
	n1.y = temp[ 1 ];
	n1.z = temp[ 2 ];

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 2 ] * 3 + 0 );
	n2.x = temp[ 0 ];
	n2.y = temp[ 1 ];
	n2.z = temp[ 2 ];

	return ( n0 + n1 + n2 ) / 3.0f;
}

GVector GPolygonObject::getNormal( int indexInObject )
{
	GVector n0, n1, n2;
	float* temp;

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 0 ] * 3 + 0 );
	n0.x = temp[ 0 ];
	n0.y = temp[ 1 ];
	n0.z = temp[ 2 ];

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 1 ] * 3 + 0 );
	n1.x = temp[ 0 ];
	n1.y = temp[ 1 ];
	n1.z = temp[ 2 ];

	temp = ( m_pVertexArray + m_pIndexArray[ indexInObject * 3 + 2 ] * 3 + 0 );
	n2.x = temp[ 0 ];
	n2.y = temp[ 1 ];
	n2.z = temp[ 2 ];

	return (n1-n0).outerProduct( (n2-n0) ).normalize();
}

GVector GPolygonObject::getVertex( int indexInObject, int sub_index )
{
	float* temp = m_pVertexArray + m_pIndexArray[ indexInObject * 3 + sub_index ] * 3 + 0;
	return GVector( temp[0], temp[1], temp[2], 0.0f );
}