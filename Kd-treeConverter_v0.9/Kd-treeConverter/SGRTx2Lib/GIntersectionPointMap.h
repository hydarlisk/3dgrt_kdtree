#pragma once

#include "GBase.h"
#include "GDimension.h"
#include "GPoint.h"
#include "GVector.h"
#include "GPolygonObject.h"
#include "cudaRenderPipeline.h"

///**
// *	Itersection Point 정보를 관리.
// *	by graphicsian.
// */
//typedef struct _GIntersectionPoint_
//{
//	GPoint m_Position;				//	position.
//	GVector m_Dir;					//	ray direction. intersection point 로부터 밖을 향하는 방향.
//	GVector m_Normal;				//	normal.
//	GPolygonObject *m_pObject;		//	Object.
//	float m_fU, m_fV;				//	Texture 좌표.
//	int m_iImageIndex;				//	이 point 가 영향을 줄 이미지 내 index.
//} GIntersectionPoint;

/**
 *	Intersection Point 를 관리하는 클래스.
 *	Scene 의 resolution 과 supersambling 개수
 *	에 따라서 달라진다. 
 *	 3차원으로 (resolution.x,resolution.y, samping개수) 만큼 depth
 *	하나에 대해서 표현하고 이러한 3차원맵이 depth 개수 만큼 있다.
 *
 *	by graphicsian.
 */

class GIntersectionPointMap
{
private:
	GDimension m_Resolution;									//	해상도.
	GDimension m_Sampling;										//	Super Sampling.

	int m_iCurrentCount;
	int m_iMaxCount;

	cuIntersectionPoint *m_pPointData;

public:
	GIntersectionPointMap( GDimension resolution, GDimension sampling, int iDepth );
	~GIntersectionPointMap(void);

	void clear();

	GError insertIntersectionPoint( cuIntersectionPoint *point, int size );
	const cuIntersectionPoint* getIntersectionPoint( int index );
	const int getSize();
};



