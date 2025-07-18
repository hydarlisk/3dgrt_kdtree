#include ".\gpointlight.h"

GPointLight::GPointLight(void)
{
}

GPointLight::~GPointLight(void)
{
}

GColor GPointLight::getRadiance( GPoint &pos, GVector &normal )
{
	GPoint p = ( m_Position - pos );
	GVector L( p.x, p.y, p.z );

	L = L.normalize();
	float dot = (float) max( 0.0, normal.innerProduct( L ) );

	return GColor( m_Material.m_Diffuse.r * dot, m_Material.m_Diffuse.g * dot, m_Material.m_Diffuse.b * dot, 1.0f );
}
