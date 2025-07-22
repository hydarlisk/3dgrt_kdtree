#pragma once

#include "GBase.h"
#include "GLight.h"
#include "GPoint.h"

/**
 *	ray 에 대한 origin, dir, power sampling 로
 *	이루어진 Light
 */
typedef struct _rayset_ {
	float pos[3];
	float dir[3];
	float power;
} GRaySet;

class GRaySetLight : public GLight
{
private:
	GRaySet *m_pRaySetData;
	int m_iRaySetDataCount;
	bool m_bRandomMode;

public:
	GRaySetLight(void);
	virtual ~GRaySetLight(void);

	virtual void setRaySetData( GRaySet *pRaySet, int count );
	virtual GRaySet *getRaySetData();
	virtual int getRaySetDataCount();

	virtual lightType getLightType() { return typeRaySetLight; }

	void setRandomMode( bool flag ) { m_bRandomMode = flag; }
	bool isRandomMode() { return m_bRandomMode; }

	virtual GError convertToWorldObject();
	virtual GError validObject();
	void makeDebugInfoObject();

	/**
	 *	지점 pos 와 normal 을 가진 현재지점으로 들어오는 radiance 를 구한다.
	 */
	virtual GColor getRadiance( GPoint &pos, GVector &normal );

};
