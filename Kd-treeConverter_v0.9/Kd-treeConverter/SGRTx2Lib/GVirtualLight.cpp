#include "GVirtualLight.h"

GVirtualLight::GVirtualLight(void)
{
}

GVirtualLight::~GVirtualLight(void)
{
}

/**
 *	지점 pos 와 normal 을 가진 현재지점으로 들어오는 radiance 를 구한다.
 */
GColor GVirtualLight::getRadiance( GPoint &pos, GVector &normal )
{
	return GColor( 0.0f, 0.0f, 0.0f, 1.0f );
}
