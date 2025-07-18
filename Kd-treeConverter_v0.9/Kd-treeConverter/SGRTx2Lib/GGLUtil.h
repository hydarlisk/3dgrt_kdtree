//--------------------------------------------------------------------------//
//																			//
//	공통으로 사용할 UTIL 클래스												//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#pragma once

#include "GBase.h"
#include "GMatrix4.h"
#include "GVector.h"

class  GGLUtil
{
public:
	GGLUtil(void);
	~GGLUtil(void);

public:
	/**
	 *	쿼터니언을 이용해서 단위회전축 e 를 중심으로 arc 만큼 회전한
	 *	결과를 리턴한다. arc 의 단위는 degree
	 */
	static GMatrix4 getQuaternianMatrix( float arc, GVector e );

	/**
	 *	vec 가 x축에 일치하기 위해서 x, y, z 축으로 얼마만큼 rotate 해야 하는지를
	 *	계산해서 리턴한다. return 값은 degree 단위로 (xarc,yarc,zarc) 를 의미
	 */
	static GVector getRotateArcFromXAxis( const GVector &vec );

	/**
	 *	평면위의 세점을 주면, normal 벡터를 계산해서 리턴. 오른손 좌표계 기준.
	 */
	static GVector calNormal( GVector &v1, GVector &v2, GVector &v3 );
};
