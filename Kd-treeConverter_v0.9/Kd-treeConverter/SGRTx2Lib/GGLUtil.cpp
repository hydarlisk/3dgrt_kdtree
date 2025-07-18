//--------------------------------------------------------------------------//
//																			//
//	공통으로 사용할 UTIL 클래스												//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#include "GGLUtil.h"
#include <math.h>

GGLUtil::GGLUtil(void)
{
}

GGLUtil::~GGLUtil(void)
{
}

/**
 *	쿼터니언을 이용해서 단위회전축 e 를 중심으로 arc 만큼 회전한
 *	결과를 리턴한다. arc 의 단위는 degree
 */
GMatrix4 GGLUtil::getQuaternianMatrix( float arc, GVector e )
{
	GMatrix4 mat;

	/** 쿼터니언의 q0, q1, q2, q3 를 구한다. */
	float q0 = (float) cos( arc / 2.0 * G_TO_RADIAN );
	float q1 = e.x * (float) sin( arc / 2.0 * G_TO_RADIAN );
	float q2 = e.y * (float) sin( arc / 2.0 * G_TO_RADIAN );
	float q3 = e.z * (float) sin( arc / 2.0 * G_TO_RADIAN );

	/** 쿼터니언을 이용한 회전 Matrix 를 구한다. */
	mat.SetMatrix( 2.0f * ( q0 * q0 + q1 * q1 ) - 1.0f, 2.0f * ( q1 * q2 - q0 * q3 ), 2.0f * ( q1 * q3 + q0 * q2 ), 0.0f,
				   2.0f * ( q1 * q2 + q0 * q3 ), 2.0f * ( q0 * q0 + q2 * q2 ) - 1.0f, 2.0f * ( q2 * q3 - q0 * q1 ), 0.0f,
				   2.0f * ( q1 * q3 - q0 * q2 ), 2.0f * ( q3 * q2 + q0 * q1 ), 2.0f * ( q0 * q0 + q3 * q3 ) - 1.0f, 0.0f,
				   0.0f, 0.0f, 0.0f, 1.0f );

	return mat;
}

/**
 *	vec 가 x축에 일치하기 위해서 x, y, z 축으로 얼마만큼 rotate 해야 하는지를
 *	계산해서 리턴한다. return 값은 degree 단위로 (xarc,yarc,zarc) 를 의미
 */
GVector GGLUtil::getRotateArcFromXAxis( const GVector &vec )
{
	GVector arc;

	/** x 축 기준으로 rotate 된 arc 를 구한다. */
	float xlength = sqrt( vec.y * vec.y + vec.z * vec.z );
	float ylength = sqrt( vec.x * vec.x + vec.z * vec.z );

	if ( xlength == 0.0f ) arc.x = 0.0f;
	else arc.x = float(acos( vec.y / xlength  ) * G_TO_DEGREE);

	if ( ylength == 0.0f ) arc.y = 0.0f;
	else arc.y = float(acos( vec.x / ylength  ) * G_TO_DEGREE);

	return arc;
}

/**
 *	평면위의 세점을 주면, normal 벡터를 계산해서 리턴
 */	
GVector GGLUtil::calNormal( GVector &v1, GVector &v2, GVector &v3 )
{
	GVector vec1 = ( v2 - v1 );
	GVector vec2 = ( v3 - v1 );

	return vec1.outerProduct( vec2 );
}
