#pragma once

#include "GBase.h"
#include "GBoundingBox.h"
#include "GVector.h"
#include "GPoint.h"
#include "GMatrix4.h"
#include "GColor.h"
#include "GObject.h"
#include "GMaterial.h"
#include "GTexture.h"
#include <vector>

using namespace std;

typedef enum _type_polygon_ {
	typePolygonNone,
	typePolygonPoint,
	typePolygonTriangle,
} typePolygon;

class  GObject
{
public:
	/**
	 *	물체에게 붙여질 고유번호. Object 가 생성될때
	 *	자동으로 번호가 붙여지며, 프로그램실행시 마다 바뀐다.
	 */
	UINT m_iObjectNumber;

protected:
	/**
	 * 물체가 선택되었는지 여부.
	 */
	bool m_bSelected;

	/**
	 * Object 이름
	 */
	char m_szObjectName[255];

	/** 
     *	물체내의 pivot 위치
	 */
	GPoint m_Pivot;

	/**
	 *	물체의 Translate 위치
	 */
	GPoint m_Translate;

	/**
	 *	x,y,z 축으로 회전한 각 ( degree )
	 */
	GVector m_RotateArc;
	/**
	 *	물체의 x, y, z 축 scale
	 */
	GVector m_Scale;

	/**
	 *	회전변환행렬
	 */
	GMatrix4 m_rotateMatrix;

	/**
	 *	변환행렬
	 */
	GMatrix4 m_Matrix;

	/**
	 *	Normal 변환행렬
	 */
	GMatrix4 m_NormalMatrix;

	/**
	 *	변환역행렬
	 */
	GMatrix4 m_InvMatrix;

	/**
	 *	바운딩 박스
	 */
	GBoundingBox m_BoundingBox;

	/**
	 *	Material 정보.
	 */
	GMaterial m_Material;

	/**
	 *	Texture 정보.
	 */
	int m_TextureID;
	
	/**
	 *	Bump Texture 정보.
	 */
	int m_BumpTextureID;

	/**
	 *	DEBUG OBJECT 인지 여부.
	 */
	bool m_bDebugObject;

	/**
	 *	intersection test 처리할 object 인지.
	 */
	bool m_bIntersection;		
	
	bool m_bVisible;
	bool m_bLight;

	typePolygon m_PolygonType;

public:
	GObject(void);
	virtual ~GObject(void);

	/**
	 *	Object Valid 를 체크한다.
	 */
	virtual GError validObject() = 0;
	/**
	 *	현재 설정된 matrix 변환을 object 의 geometry 에 적용해서
	 *	물체 자체를 world object 데이터로 변환시키고,
	 *	matrix 는 identity 로 만드는 기능을 수행해야 한다.
	 */
	virtual GError convertToWorldObject() = 0;

	/**
	 *	Light 인지 여부.
	 */
	virtual bool isLight() { return m_bLight; }
	virtual void setLight( bool flag ) { m_bLight = flag; }

	/**
	 *	Intersection 을 처리할 물체일지 여부.
	 */
	virtual bool isIntersection() { return m_bIntersection; }
	virtual void setIntersection( bool flag ) { m_bIntersection = flag; }

	/**
	 *	Triangle Data 제공 여부.
	 */
	virtual typePolygon getPolygonType() { return m_PolygonType; }
	virtual void setPolygonType( typePolygon type ) { m_PolygonType = type; }
	
	/**
	 *	Debug 용 Object 인지 여부.
	 */
	bool isDebugObject();
	void setDebugObject( bool flag );

	virtual void setBoundingBox( const GBoundingBox &bbox );
	virtual GBoundingBox* const getBoundingBox();

	virtual void setPivot( const GPoint &pivot );
	virtual GPoint getPivot();

	virtual void calMatrix();
	virtual GMatrix4* const getMatrix();
	virtual GMatrix4* const getNormalMatrix();
	virtual GMatrix4* const getInvMatrix();
	virtual GMatrix4* const getRotateMatrix();

	virtual bool isSelected();
	virtual void setSelected( const bool &flag );
	virtual UINT getObjectNumber();
	virtual void setName( const char* objectName );
	virtual const char* getName();

	virtual void setTranslate( const GPoint &position );
	virtual GPoint getTranslate();
	virtual void addTranslate( const GPoint &offset );

	virtual void setVisible( bool flag );
	virtual bool isVisible();

	/**
	 *	물체를 임의의 vector 기준으로 회전
	 */
	virtual void rotateByVector( const GVector &vector, const float &arc );

	virtual void addRotate( const GVector &arc );
	virtual void setRotate( const GVector &arc );
	virtual GVector getRotate();

	virtual void setScale( const GVector &scale );
	virtual GVector getScale();
	virtual void addScale( const GVector &scale );

	virtual GMaterial* getMaterial();
	virtual void setTextureID( int id );
	virtual int getTextureID();

	virtual void setBumpTextureID( int id );
	virtual int getBumpTextureID();

	virtual bool hasTexture();
	virtual bool hasBumpTexture();

	virtual void identityTransform();
	
};
