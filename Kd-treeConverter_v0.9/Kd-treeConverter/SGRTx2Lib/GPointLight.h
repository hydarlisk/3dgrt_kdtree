#pragma once

#include "GBase.h"
#include "GLight.h"
#include "GPoint.h"

/**
 *	PointLight Å¬·¡½º.
 *
 *	by graphicsian
 */

class  GPointLight : public GLight
{
private:

public:
	GPointLight(void);
	~GPointLight(void);

	virtual lightType getLightType() { return typePointLight; }
	bool isPointSet() { return true; }

	virtual GColor getRadiance( GPoint &pos, GVector &normal );
};
