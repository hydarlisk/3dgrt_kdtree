#pragma once

#include "GBase.h"
#include "GBoundingBox.h"
#include "GVector.h"
#include "GPoint.h"

#include "SSE_common.h"

/**
 *	Triangle Wrapper 클래스.
 *	메모리에 저장되어 있는 vertex data 를 p0, p1, p2 가
 *	가리킨다. 이 클래스는 반드시 Wrapper 의 Target이 되는
 *	클래스가 살아있어야 한다.
 *
 *	by graphicsian.
 */
class GPolygonObject;
class GTriangleWrapper
{
public:
	float *p0, *p1, *p2;			//	Triangle position 의 메모리 위치를 가리킴.
	float *n0, *n1, *n2;			//	Triangle normal 의 메모리 위치를 가리킴.
	float *uv0, *uv1, *uv2;			//	texture 좌표 s, t 의 메모리 위치를 가리킴.

	GPolygonObject* m_pObject;		//	Triangle 이 속한 Triangle Object.
	GBoundingBox m_BBox;			//	삼각형의 Bounding Box.
	int indexInObject;				//	물체내에서의 삼각형 index.
	int index;						//	Scene 전체 삼각형 List 데이터에서 현재 삼각형의 index.
	int objectIndexInScene;			//	삼각형이 포함된 object 의 scene 안에서의 index.
	bool bSelected;					//	삼각형이 선택되었는지 여부.

	float visibility;

	int m_mailBoxId;

public:
	GTriangleWrapper(void);
	~GTriangleWrapper(void);

	float calArea();
	void  calCentroid(GVector& cen);

	void getPoint( GPoint &p0, GPoint &p1, GPoint &p2 );

	GVector calBarycentricNormal( float alpha, float beta, float gamma );
	GPoint  calBarycentricPosition( float alpha, float beta, float gamma );
	GPoint  calBarycentricUV( float alpha, float beta, float gamma );
	void    calBarycentricNormal( __m128 alpha, __m128 beta, __m128 gamma, _sse_vec *normal );
	void    calBarycentricUV( __m128 alpha, __m128 beta, __m128 gamma, __m128 *uv );
};