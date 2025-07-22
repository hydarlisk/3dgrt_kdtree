//--------------------------------------------------------------------------//
//																			//
//	카메라 변환에 관한 클래스												//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#include "GCamera.h"
#include <gl/gl.h>
#include <gl/glu.h>

GCamera::GCamera(void)
{
	m_bOrtho = FALSE;
	m_near = 1.0f;
	m_far = 500.0f;
	m_UpAxis = AXIS_Z;
}

GCamera::~GCamera(void)
{
}

void GCamera::operator= ( const GCamera& camera )
{
	m_eye = camera.m_eye;
	m_aspect = camera.m_aspect;
	m_bOrtho = camera.m_bOrtho;
	m_left = camera.m_left; m_right = camera.m_right; m_bottom = camera.m_bottom; m_top = camera.m_top;
	m_near = camera.m_near;	m_far = camera.m_far; m_aspect = camera.m_aspect; m_fovy = camera.m_fovy;
	m_nVec = camera.m_nVec;	m_uVec = camera.m_uVec;	m_vVec = camera.m_vVec;
	m_nInitVec = camera.m_nInitVec;	m_uInitVec = camera.m_uInitVec;	m_vInitVec = camera.m_vInitVec;
	m_UpAxis = camera.m_UpAxis;

	calViewMatrix();
}	

BOOL GCamera::operator== ( const GCamera& camera )
{
	if ( m_uVec == camera.m_uVec &&
		 m_vVec == camera.m_vVec &&
		 m_nVec == camera.m_nVec &&
		 m_matrix == camera.m_matrix ) return TRUE;
	return FALSE;
}

int GCamera::getUpAxis()
{
	return m_UpAxis;
}

void GCamera::setCameraPos( GVector eye, GVector view, GVector up )
{
	m_eye = eye;

	m_nVec = ( m_eye - view ).normalize();
	m_uVec = up.outerProduct( m_nVec ).normalize();
	m_vVec = m_nVec.outerProduct( m_uVec ).normalize();

	// 초기 좌표축을 기억
	m_uInitVec = m_uVec;
	m_vInitVec = m_vVec;
	m_nInitVec = m_nVec;

	/** 
	 *	up 벡터중 가장 큰 긴 축 카메라 upAxis 로 설정. 카메라 회전축 기준을 위해서.
	 */
	if ( m_vInitVec.x >= m_vInitVec.y && m_vInitVec.x >= m_vInitVec.z )
		m_UpAxis = AXIS_X;
	if ( m_vInitVec.y >= m_vInitVec.z && m_vInitVec.y >= m_vInitVec.x )
		m_UpAxis = AXIS_Y;
	if ( m_vInitVec.z >= m_vInitVec.x && m_vInitVec.z >= m_vInitVec.y )
		m_UpAxis = AXIS_Z;

	calViewMatrix();
}

// U, V, N 와 eye 좌표를 이용해서, 카메라 변환 Matrix 를 계산한다.
void GCamera::calViewMatrix()
{
	m_rotateMatrix.SetMatrix( m_uVec.x, m_uVec.y, m_uVec.z, 0.0f,
						m_vVec.x, m_vVec.y, m_vVec.z, 0.0f,
						m_nVec.x, m_nVec.y, m_nVec.z, 0.0f,
						0.0f, 0.0f, 0.0f, 1.0f );

	m_transMatrix.SetMatrix( 1.0f, 0.0f, 0.0f, -m_eye.x,
				   0.0f, 1.0f, 0.0f, -m_eye.y,
				   0.0f, 0.0f, 1.0f, -m_eye.z,
				   0.0f, 0.0f, 0.0f, 1.0f );

	m_matrix = m_rotateMatrix * m_transMatrix;

	/** world 좌표축이 어떻게 바뀌었는지 정보 기록해둠 */
	m_transformedWorldX = m_rotateMatrix * GVector( 1.0f, 0.0f, 0.0f );
	m_transformedWorldY = m_rotateMatrix * GVector( 0.0f, 1.0f, 0.0f );
	m_transformedWorldZ = m_rotateMatrix * GVector( 0.0f, 0.0f, 1.0f );
}

//----------------------------------------------------------------------//
// Pivot 포인트 ( 특정물체중심 ) 으로 카메라를 회전시킨다.				//
// 만약 카메라가 바라보는 방향에 Pivot 포인트가 없다면,					//
// 카메라가 바라보는 방향벡터위에 이 pivot 포인트가 수직으로 만나는		//
// 위치까지의 거리가 카메라회전 반경의 기준이 된다.						//
//	이 값을 distance 라고 하자.	이 distance 가 회전 반경이다.			//
// 그리고 변환행렬은 다음과 같이 구한다.								//
// 먼저 Camera 를 -n 방향으로 distance 만큼 진행시켜서 pivot point 에	//
// 위치시키고, camera 회전을 한후 다시 n 방향으로 distance 만큼			//
//	진행시킨다.															//
//																		//
//	이와 같은 카메라움직임은 Maya 의 카메라 움직임을 구현하기 위한것.	//
//----------------------------------------------------------------------//
void GCamera::rotateByPivot( GVector pivot, float harc, float varc, float narc )
{
	/** 
	 *	현재 카메라가 바라보는 방향벡터에, 
	 *	pivot 점이 수직으로 만나는 점까지의
	 *	거리를 구한다.
	 */
	GVector toPivot = ( pivot - m_eye ).normalize();
	float cost = -m_nVec.innerProduct( toPivot );

	float length = ( pivot - m_eye ).length();
	float distance = length * cost;
	//distance = 10.0f;
	//harc = 0.0f;

	GVector translation = *getMatrix() * pivot;
	moveCameraUVNAxis( translation.x, translation.y, -translation.z );

	//moveCameraUVNAxis( 0.0f, 0.0f, distance );

	rotateUVN( -varc, 0.0f, -narc );

	/** 
	 *	카메라의 up 방향이 x, y, z 축 어디인지에 따라서 회전결정
	 */
	//if ( m_UpAxis == AXIS_X )
	//	rotateWorldXYZ( -harc, 0.0, 0.0f );
	//if ( m_UpAxis == AXIS_Y )
		rotateWorldXYZ( 0.0f, -harc, 0.0f );
//	if ( m_UpAxis == AXIS_Z )
//		rotateWorldXYZ( 0.0f, 0.0f, -harc );

	//moveCameraUVNAxis( 0.0f, 0.0f, -distance );
	//m_eye -= translation;
	moveCameraUVNAxis( -translation.x, -translation.y, translation.z );
	calViewMatrix();
}

/**
 *	카메라 초기 u, v, n 축으로부터 얼마나 회전했는지를 누적. 
 *	-360~360 사이로 정규화한다.
 */
void GCamera::rotateUVN( float uarc, float varc, float narc )
{
	GMatrix4 rotate;

	rotate.identity();

	rotate = GGLUtil::getQuaternianMatrix( uarc, m_uVec ) * rotate;
	rotate = GGLUtil::getQuaternianMatrix( varc, m_vVec ) * rotate;
	rotate = GGLUtil::getQuaternianMatrix( narc, m_nVec ) * rotate;

	m_uVec = rotate * m_uVec;
	m_vVec = rotate * m_vVec;
	m_nVec = rotate * m_nVec;

	// 새로운 View Matrix 행렬을 구한다.
	calViewMatrix();
}

/**
 *	World 좌표계 x, y, z 축을 기준으로 얼마나 회전했는지를 누적. 
 *	-360~360 사이로 정규화한다.
 */
void GCamera::rotateWorldXYZ( float xarc, float yarc, float zarc )
{
	GMatrix4 rotate;

	rotate.identity();

	rotate = GGLUtil::getQuaternianMatrix( xarc, GVector( 1.0f, 0.0f, 0.0f ) ) * rotate;
	rotate = GGLUtil::getQuaternianMatrix( yarc, GVector( 0.0f, 1.0f, 0.0f ) ) * rotate;
	rotate = GGLUtil::getQuaternianMatrix( zarc, GVector( 0.0f, 0.0f, 1.0f ) ) * rotate;

	m_uVec = rotate * m_uVec;
	m_vVec = rotate * m_vVec;
	m_nVec = rotate * m_nVec;

	// 새로운 View Matrix 행렬을 구한다.
	calViewMatrix();
}

/**
 *	카메라가 바라보는 방향(z) 으로 zoom in, out 수행
 */
void GCamera::zooming( float zoom )
{
	moveCameraUVNAxis( 0.0f, 0.0f, zoom );
}

void GCamera::moveHorizonalVertical( float h, float v )
{
	moveCameraUVNAxis( h, v, 0.0f );
}

void GCamera::moveCameraUVNAxis( float du, float dv, float dn )
{
	GVector vec = m_nVec * (-dn);
	m_eye = m_eye + vec; 

	vec = m_uVec * du;
	m_eye = m_eye + vec; 

	vec = m_vVec * dv;
	m_eye = m_eye + vec; 

	calViewMatrix();
}

void GCamera::setOrtho( float left, float right, float bottom, float top, 
						float n, float f )
{
	m_bOrtho = TRUE;
	m_left = left;
	m_right = right;
	m_bottom = bottom;
	m_top = top;
	m_near = n;
	m_far = f;
}

void GCamera::setPerspective( float fovy, float aspect, float n, float f )
{
	m_bOrtho = FALSE;
	m_fovy = fovy;
	m_aspect = aspect;
	m_near = n;
	m_far = f;
}

void GCamera::setAspect( float aspect )
{
	m_aspect = aspect;
}

void GCamera::setOrtho( float left, float right, float bottom, float top )
{
	m_bOrtho = TRUE;
	m_left = left;
	m_right = right;
	m_bottom = bottom;
	m_top = top;
}

void GCamera::setPerspective( float fovy, float aspect )
{
	m_bOrtho = FALSE;
	m_fovy = fovy;
	m_aspect = aspect;
}

void GCamera::setViewDistance( float n, float f )
{
	m_near = n;
	m_far = f;
}

GMatrix4* const GCamera::getMatrix()
{
	return &m_matrix;
}

BOOL GCamera::isOrtho()
{
	return m_bOrtho;
}

float GCamera::getLeft()
{
	return m_left;
}

float GCamera::getRight()
{
	return m_right;
}

float GCamera::getBottom()
{
	return m_bottom;
}

float GCamera::getTop()
{
	return m_top;
}

float GCamera::getNear()
{
	return m_near;
}

float GCamera::getFar()
{
	return m_far;
}

float GCamera::getAspect()
{
	return m_aspect;
}

float GCamera::getFovy()
{
	return m_fovy;
}

GVector GCamera::getUVec()
{
	return m_uVec;
}

GVector GCamera::getVVec()
{
	return m_vVec;
}

GVector GCamera::getNVec()
{
	return m_nVec;
}

GMatrix4* const GCamera::getRotateMatrix()
{
	return &m_rotateMatrix;
}

GMatrix4* const GCamera::getTranslateMatrix()
{
	return &m_transMatrix;
}

GVector GCamera::getTransformedWorldX()
{
	return m_transformedWorldX;
}

GVector GCamera::getTransformedWorldY()
{
	return m_transformedWorldY;
}

GVector GCamera::getTransformedWorldZ()
{
	return m_transformedWorldZ;
}

GVector GCamera::getEye()
{
	return m_eye;
}
