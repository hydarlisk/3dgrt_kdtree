#pragma once

#include "GBase.h"
#include "GPolygonObject.h"
#include "GColor.h"

typedef enum {
	typePointLight,
	typeRaySetLight,
	typeAreaLight,
	typeRectSpotLight,
	typeVirtualLight,
} lightType;

class GLight : public GPolygonObject
{
protected:
	GPoint m_Position;
	float m_fIntensity;
	bool m_bUsePhoton;						// 이 light 가 photon 을 emit 하는지 여부.
	bool m_Enabled;

	GPolygonObject *m_pDebugInfoObject;		// debugging 을 위한 light geometry info.

public:
	GLight(void);
	virtual ~GLight(void);

	virtual lightType getLightType() = 0;

	virtual void setUsePhoton( bool flag );
	virtual bool isUsePhoton();

	/** light color 는 material 의 모든 컬러를 동일하게 세팅 */
	virtual void setLightColor( GColor &color );
	virtual GColor getLightColor();

	virtual void setIntensity( float value );
	virtual float getIntensity();

	virtual GPolygonObject *getDebugInfoObject();
	virtual void setDebugInfoObject( GPolygonObject *pObject );

	virtual void setPosition( GPoint &position );
	virtual GPoint getPosition();

	virtual	GError convertToWorldObject();
	virtual GError validObject();

	virtual bool isEnabled();
	virtual void setEnable( bool flag = true );

	/**
	 *	지점 pos 와 normal 을 가진 현재지점으로 들어오는 radiance 를 구한다.
	 */
	virtual GColor getRadiance( GPoint &pos, GVector &normal ) = 0;
};
