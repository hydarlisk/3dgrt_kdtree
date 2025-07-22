#pragma once

#include "GBase.h"
#include "GObject.h"
#include "GTriangleWrapper.h"
#include "GTriangleWrapperList.h"

/**
 *	Triangle 로 구성된 데이터를 지원하는 Object 클래스.
 *	추상객체이다. 현재 모든 Light 및 Object 는 이 TriangleObject 를
 *	상속받아야 한다.
 *
 *	by graphicsian.
 */
class  GPolygonObject : public GObject
{
private:
	float *m_pVertexArray;
	float *m_pNormalArray;
	float *m_pColorArray;
	float *m_pUVArray;
	int *m_pIndexArray;

	int m_iVertexCount;
	int m_iTriangleCount;

	float *m_pVisibilityArray;

public:
	GPolygonObject(void);
	virtual ~GPolygonObject(void);

	virtual GError validObject();
	virtual GError convertToWorldObject();

	virtual void setTriangleCount( int count );
	virtual int getVertexCount();
	virtual void setVertexCount( int count );

	/**
	 *	현재 삼각형 데이터에 transform matrix 를 적용해서
	 *	데이터 자체를 변경한다.
	 */
	virtual void applyTransform( GMatrix4 *pTransformMatrix, 
								 GMatrix4 *pNormalTransformMatrix );

	/** 
	 * vertex 배열의 pointer 를 리턴한다. pCount 에는 vertex 갯수 리턴 
	 */
	virtual void setVertexArray( float *array );
	virtual const float* getVertexArray();

	/** 
	 * normal 배열의 pointer 를 리턴한다. count 에는 normal 갯수 리턴 
	 */
	virtual void setNormalArray( float *array );
	virtual const float* getNormalArray();

	/** 
	 * triangle index 배열의 pointer 를 리턴한다. count 에는 triangle 갯수 리턴 
	 */
	virtual void setIndexArray( int *array );
	virtual const int* getIndexArray();

	/** 
	 * UV 데이터 배열의 pointer 를 리턴한다. count 에는 UV 갯수 리턴 
	 */
	virtual void setUVArray( float *array );
	virtual const float* getUVArray();

	/** 
	 * Color 데이터 배열의 pointer 를 리턴한다. 
	 * count 에는 UV 갯수 리턴 
	 */
	virtual void setColorArray( float *array );
	virtual const float* getColorArray();

	// Visibility
	virtual void setVisibilityArray( float *array );
	virtual const float* getVisibilityArray();


	virtual int getTriangleCount();

	virtual void getTriangleList( GTriangleWrapperList* pList, int objectIndexInScene, int baseOffset );

	virtual GVector calBarycentricNormal( int indexInObject, float alpha, float beta, float gamma );
	virtual GPoint calBarycentricPosition(  int indexInObject, float alpha, float beta, float gamma );
	virtual GPoint calBarycentricUV(  int indexInObject, float alpha, float beta, float gamma );
	virtual void calBarycentricNormal( int indexInObject, __m128 alpha, __m128 beta, __m128 gamma, _sse_vec *normal );
	virtual void calBarycentricUV(  int indexInObject, __m128 alpha, __m128 beta, __m128 gamma, __m128 *uv );

	// 기타 함수
	GVector getCenter( int indexInObject );
	GVector getNormal( int indexInObject );
	GVector getVertex( int indexInObject, int sub_index );

private:
	void updateBoundingBox();

};
