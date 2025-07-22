#pragma once

#include "GLight.h"
/**
 *	Halogen Light 를 표현하기 위한 클래스.
 *	Polygon 으로 표현.
 */
class GVirtualLight : public GLight
{
public:
	GVirtualLight(void);
	~GVirtualLight(void);

	virtual lightType getLightType() { return typeVirtualLight; }
	virtual GColor getRadiance( GPoint &pos, GVector &normal );

};
