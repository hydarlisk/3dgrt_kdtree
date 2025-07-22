#include "GObject.h"
#include "GObjectNumberGenerator.h"
#include "GGLUtil.h"

GObject::GObject(void)
{
	m_TextureID = -1;
	m_BumpTextureID = -1;

	m_PolygonType = typePolygonNone;
	m_bDebugObject = false;
	m_bSelected = false;
	m_iObjectNumber = GObjectNumberGenerator::generateObjectNumber();
	strcpy( m_szObjectName, "Undefined" );
	m_Translate.setPoint( 0.0f, 0.0f, 0.0f );
	m_Scale.setVector( 1.0f, 1.0f, 1.0f );
	m_RotateArc.setVector( 0.0f, 0.0f, 0.0f );
	m_BoundingBox.setMin( GVector( 0.0f, 0.0f, 0.0f ) );
	m_BoundingBox.setMax( GVector( 1.0f, 1.0f, 1.0f ) );
	m_rotateMatrix.identity();
	m_bIntersection = true;
	m_bVisible = true;
	m_bLight = false;

	calMatrix();
}

GObject::~GObject(void)
{
}

bool GObject::isDebugObject()
{
	return m_bDebugObject;
}

void GObject::setDebugObject( bool flag )
{
	m_bDebugObject = flag;
}

void GObject::setPivot( const GPoint &pivot )
{
	m_Pivot = pivot;
}

GPoint GObject::getPivot()
{
	return m_Pivot;
}

void GObject::setBoundingBox( const GBoundingBox &bbox )
{
	m_BoundingBox = bbox;
}

GBoundingBox* const GObject::getBoundingBox()
{
	return &m_BoundingBox;
}

bool GObject::isSelected()
{
	return m_bSelected;
}

void GObject::setSelected( const bool &flag )
{
	m_bSelected = flag;
}

UINT GObject::getObjectNumber()
{
	return m_iObjectNumber;
}

void GObject::setName( const char* objectName )
{
	strncpy( m_szObjectName, objectName, 250 );
	m_szObjectName[250] = 0x00;
}

const char* GObject::getName()
{
	return m_szObjectName;
}

void GObject::setTranslate( const GPoint &t )
{
	m_Translate = t;
	calMatrix();
}

void GObject::addTranslate( const GPoint &offset )
{
	m_Translate += offset;
	calMatrix();
}

GPoint GObject::getTranslate()
{
	return m_Translate;
}

GVector GObject::getRotate()
{
	return m_RotateArc;
}

void GObject::setRotate( const GVector &arc )
{
	m_rotateMatrix = GGLUtil::getQuaternianMatrix( arc.x, GVector( 1.0f, 0.0f, 0.0f ) );
	m_rotateMatrix = GGLUtil::getQuaternianMatrix( arc.y, GVector( 0.0f, 1.0f, 0.0f ) ) * m_rotateMatrix;
	m_rotateMatrix = GGLUtil::getQuaternianMatrix( arc.z, GVector( 0.0f, 0.0f, 1.0f ) ) * m_rotateMatrix;
	calMatrix();
}

/**
 *	임의의축 vector 를 기준으로 회전은 다음과 같은 순서로 볼 수 있다.
 *	1. 먼저 x 축을 기준으로 arc 만큼 회전한다.
 *	2. 그리고 x 축이 vector 와 일치하게 만들기 위해서 
 *     x, y, z 축으로 얼마만큼 회전해야 하는지를 체크한다.
 */
void GObject::rotateByVector( const GVector &vector, const float &arc )
{
	m_rotateMatrix = GGLUtil::getQuaternianMatrix( arc, vector ) * m_rotateMatrix;
	calMatrix();
}

void GObject::addRotate( const GVector &arc )
{
	m_rotateMatrix = GGLUtil::getQuaternianMatrix( arc.x, GVector( 1.0f, 0.0f, 0.0f ) ) * m_rotateMatrix;
	m_rotateMatrix = GGLUtil::getQuaternianMatrix( arc.y, GVector( 0.0f, 1.0f, 0.0f ) ) * m_rotateMatrix;
	m_rotateMatrix = GGLUtil::getQuaternianMatrix( arc.z, GVector( 0.0f, 0.0f, 1.0f ) ) * m_rotateMatrix;
	calMatrix();
}

void GObject::setScale( const GVector &scale )
{
	m_Scale = scale;
	calMatrix();
}

void GObject::addScale( const GVector &scale )
{
	m_Scale = m_Scale + scale;
	calMatrix();
}

GVector GObject::getScale()
{
	return m_Scale;
}

/**
 *	모든 transform 을 identity 로 만든다.
 */
void GObject::identityTransform()
{
	m_Scale.setVector( 1.0f, 1.0f, 1.0f );
	m_Translate.setPoint( 0.0f, 0.0f, 0.0f );
	m_rotateMatrix.identity();
	m_InvMatrix.identity();
	m_Matrix.identity();
	m_NormalMatrix.identity();
	calMatrix();
}

/**
 *	Object 의 변환행렬을 계산한다.
 *	회전/스케일 계산후 이동행렬을 계산한다.
 *	행렬곱으로는 거꾸로 해야하므로 이동*회전*스케일
 */
void GObject::calMatrix()
{
	GMatrix4 matrix, scaleMatrix, pivotMatrix;
	GMatrix4 scaleInvMatrix, pivotInvMatrix, rotateInvMatrix, moveInvMatrix;
	
	pivotMatrix.SetMatrix( 1.0f, 0.0f, 0.0f, -m_Pivot.x,
					  0.0f, 1.0f, 0.0f, -m_Pivot.y,
					  0.0f, 0.0f, 1.0f, -m_Pivot.z,
					  0.0f, 0.0f, 0.0f, 1.0f );

	pivotInvMatrix.SetMatrix( 1.0f, 0.0f, 0.0f, m_Pivot.x,
					  0.0f, 1.0f, 0.0f, m_Pivot.y,
					  0.0f, 0.0f, 1.0f, m_Pivot.z,
					  0.0f, 0.0f, 0.0f, 1.0f );

	matrix.SetMatrix( 1.0f, 0.0f, 0.0f, m_Translate.x,
		0.0f, 1.0f, 0.0f, m_Translate.y,
		0.0f, 0.0f, 1.0f, m_Translate.z,
		0.0f, 0.0f, 0.0f, 1.0f );

	moveInvMatrix.SetMatrix( 1.0f, 0.0f, 0.0f, -m_Translate.x,
		0.0f, 1.0f, 0.0f, -m_Translate.y,
		0.0f, 0.0f, 1.0f, -m_Translate.z,
		0.0f, 0.0f, 0.0f, 1.0f );


	/** x,y,z 회전 */
	matrix = matrix * m_rotateMatrix;

	/** 회전행렬 역행렬. 내부 3x3 만 transpose 한 행렬 */
	rotateInvMatrix = m_rotateMatrix.transpose();
	rotateInvMatrix.SetElement( 3, 0, 0.0f );
	rotateInvMatrix.SetElement( 3, 1, 0.0f );
	rotateInvMatrix.SetElement( 3, 2, 0.0f );
	rotateInvMatrix.SetElement( 3, 3, 1.0f );
	rotateInvMatrix.SetElement( 0, 3, 0.0f );
	rotateInvMatrix.SetElement( 1, 3, 0.0f );
	rotateInvMatrix.SetElement( 2, 3, 0.0f );

	/** x,y,z scale */
	scaleMatrix.SetMatrix( m_Scale.x, 0.0f, 0.0f, 0.0f,
					  0.0f, m_Scale.y, 0.0f, 0.0f,
					  0.0f, 0.0f, m_Scale.z, 0.0f,
					  0.0f, 0.0f, 0.0f, 1.0f );
	scaleInvMatrix.SetMatrix( 1.0f / m_Scale.x, 0.0f, 0.0f, 0.0f,
					  0.0f, 1.0f / m_Scale.y, 0.0f, 0.0f,
					  0.0f, 0.0f, 1.0f / m_Scale.z, 0.0f,
					  0.0f, 0.0f, 0.0f, 1.0f );
	
	matrix = matrix * scaleMatrix;
	matrix = matrix * pivotMatrix;

	m_Matrix = matrix;

	/**
	 *	Normal 변환 행렬 계산.
	 *	회전행렬의 3x3 부분의 역행렬의 transpose 이고,
	 *	원래 행렬은 강체변환이므로, 회전행렬의 3x3 을 그대로 옮긴다.
	 *	대신 나머지 부분은 모두 0 으로 만든다. 원래
	 *	matrix 에서 3,3 만 0.0f 으로 만들면 된다.
	 */
	m_NormalMatrix = m_rotateMatrix;
	m_NormalMatrix.SetElement( 3, 3, 0.0f );

	/**
	 *	역행렬 계산.
	 */
	m_InvMatrix.identity();
	m_InvMatrix = m_InvMatrix * pivotInvMatrix;
	m_InvMatrix = m_InvMatrix * scaleInvMatrix;
	m_InvMatrix = m_InvMatrix * m_rotateMatrix.transpose();
	m_InvMatrix = m_InvMatrix * moveInvMatrix;

}

GMatrix4* const GObject::getMatrix()
{
	return &m_Matrix;
}

GMatrix4* const GObject::getNormalMatrix()
{
	return &m_NormalMatrix;
}

GMatrix4* const GObject::getInvMatrix()
{
	return &m_InvMatrix;
}

GMatrix4* const GObject::getRotateMatrix()
{
	return &m_rotateMatrix;
}

GMaterial* GObject::getMaterial()
{
	return &m_Material;
}

void GObject::setTextureID( int id )
{
	m_TextureID = id;
}

int GObject::getTextureID()
{
	return m_TextureID;
}

bool GObject::hasTexture()
{
	return m_TextureID != -1;
}

void GObject::setBumpTextureID( int id )
{
	m_BumpTextureID = id;
}

int GObject::getBumpTextureID()
{
	return m_BumpTextureID;
}

bool GObject::hasBumpTexture()
{
	return m_BumpTextureID != -1;
}

void GObject::setVisible( bool flag )
{
	m_bVisible = flag;
}

bool GObject::isVisible()
{
	return m_bVisible;
}

