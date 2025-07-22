#include "GLight.h"

GLight::GLight(void)
	: m_Enabled(true)
{
	m_fIntensity = 1.0f;
	m_bUsePhoton = false;
	m_bIntersection = false;
	m_pDebugInfoObject = NULL;
	m_bLight = true;
}

GLight::~GLight(void)
{
	if ( m_pDebugInfoObject )
		delete m_pDebugInfoObject;
}

void GLight::setLightColor( GColor &color )
{
	m_Material.m_Emission = color;
}

GColor GLight::getLightColor()
{
	return m_Material.m_Emission;
}

void GLight::setIntensity( float value )
{
	m_fIntensity = value;
}

float GLight::getIntensity()
{
	return m_fIntensity;
}

void GLight::setUsePhoton( bool flag )
{
	m_bUsePhoton = flag;
}

bool GLight::isUsePhoton()
{
	return m_bUsePhoton;
}
	
GPolygonObject *GLight::getDebugInfoObject()
{
	return m_pDebugInfoObject;
}

void GLight::setDebugInfoObject( GPolygonObject *pObject )
{
	if ( m_pDebugInfoObject )
		delete m_pDebugInfoObject;
	m_pDebugInfoObject = pObject;
}

void GLight::setPosition( GPoint &position )
{
	m_Position = position;
}

GPoint GLight::getPosition()
{
	return m_Position;
}

bool GLight::isEnabled()
{
	return m_Enabled;
}

void GLight::setEnable( bool flag )
{
	m_Enabled = flag;
}

/**
 *	데이터를 world 좌표계상의 데이터로 
 *	변환하고, matrix 를 identity 로 초기화 한다.
 */
GError GLight::convertToWorldObject()
{
	m_Position = this->m_Matrix * m_Position;

	return GPolygonObject::convertToWorldObject();
}

GError GLight::validObject()
{
	return errorNo;
}
