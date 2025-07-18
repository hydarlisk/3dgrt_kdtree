//--------------------------------------------------------------------------//
//																			//
//	카메라 변환에 관한 클래스												//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#pragma once

#include "GBase.h"
#include "GVector.h"
#include "GMatrix4.h"
#include "GGLUtil.h"
#include "GDimension.h"

#define AXIS_X	0
#define AXIS_Y	1
#define AXIS_Z	2

class  GCamera
{
private:
	GMatrix4 m_matrix;				//	View 변환에 사용될 카메라 변환 Matrix
	GMatrix4 m_rotateMatrix;		//  u,v,n 축으로의 rotation 만계산하는 matrix
	GMatrix4 m_transMatrix;			//	눈 위치의 translate 만 계산하는 matrix

	GVector m_eye;					//	카메라 위치

	GVector m_uInitVec, m_uVec;		//	초기 카메라 u 좌표축과, 현재 카메라 u 좌표축
	GVector m_vInitVec, m_vVec;		//	초기 카메라 v 좌표축과, 현재 카메라 v 좌표축
	GVector m_nInitVec, m_nVec;		//	초기 카메라 n 좌표축과, 현재 카메라 n 좌표축

	GVector m_transformedWorldX;		//	x, y, z 월드 좌표계가 카메라 좌표축에서 변환된 vector
	GVector m_transformedWorldY;
	GVector m_transformedWorldZ;

	float	m_left;
	float	m_right;
	float	m_bottom;
	float	m_top;
	float	m_near;			
	float	m_far;
	float	m_aspect;
	float	m_fovy;

	BOOL	m_bOrtho;
	int		m_UpAxis;

public:
	GCamera(void);
	virtual ~GCamera(void);

	BOOL operator== ( const GCamera &camera );		// 카메라의 현재 프레임 축과 눈위치가 같으면 true
	void operator= ( const GCamera &camera );

	/**
	 * pivot 포인트를 중심으로 카메라 돌리기
	 */
	void rotateByPivot( GVector pivot, float harc, float varc, float narc = 0 );

	/**
	 *	카메라 축을 기준으로 회전하기
	 */
	void rotateUVN( float uarc, float varc, float narc );

	/**
	 *	World 좌표계 x,y,z 기준으로 회전하기
	 */
	void rotateWorldXYZ( float xarc, float yarc, float zarc );

	/**
	 *	카메라 zoom in, out
	 */
	void zooming( float zoom );

	int getUpAxis();

	/**
	 *	카메라의 수평, 수직 움직임.
	 */
	void moveHorizonalVertical( float h, float v );

	/**
	 *	카메라 위치 및 투영방법 설정 함수들.
	 */
	void setCameraPos( GVector eye, GVector view, GVector up );
	void setOrtho( float left, float right, float bottom, float top, float n, float f );
	void setPerspective( float povy, float aspect, float n, float f );
	void setOrtho( float left, float right, float bottom, float top );
	void setPerspective( float povy, float aspect );
	void setViewDistance( float n, float f );
	void setAspect( float aspect );

	GMatrix4* const getMatrix();
	GMatrix4* const getRotateMatrix();
	GMatrix4* const getTranslateMatrix();

	BOOL isOrtho();

	float getLeft();
	float getRight();
	float getBottom();
	float getTop();
	float getNear();
	float getFar();
	float getAspect();
	float getFovy();

	GVector getUVec();
	GVector getVVec();
	GVector getNVec();

	GVector getTransformedWorldX();
	GVector getTransformedWorldY();
	GVector getTransformedWorldZ();

	GVector getEye();
	//GVector calRayDir( float x, float y );

protected:
	/**
	 *	카메라 축방향으로 기준으로 이동하기
	 */
	void moveCameraUVNAxis( float du, float dv, float dn );

	void calViewMatrix();

};
