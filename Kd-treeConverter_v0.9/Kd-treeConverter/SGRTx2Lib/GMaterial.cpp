#include ".\gmaterial.h"

GMaterial::GMaterial(void)
{
	m_fRoughness = 1.0f;
	m_fTransparency = 0.0f;
	m_fReflection = 0.0f;
	m_Diffuse.r = 0.8f; 
	m_Diffuse.g = 0.8f; 
	m_Diffuse.b = 0.8f;
	m_fRefractionIndex = 1.0f;
}

GMaterial::~GMaterial(void)
{
}
void GMaterial::setDiffuse( const GColor &diffuse )
{
	m_Diffuse = diffuse;
}
void GMaterial::setAmbient( const GColor &ambient )
{
	m_Ambient = ambient;
}
void GMaterial::setSpecular( const GColor &specular )
{
	m_Specular = specular;
}
void GMaterial::setEmission( const GColor &emission )
{
	m_Emission = emission;
}
void GMaterial::setRoughness( const float &roughness )
{
	m_fRoughness = roughness;
}

void GMaterial::setReflection( const float &r )
{
	m_fReflection = r;
}

void GMaterial::setTransparency( const float &r )
{
	m_fTransparency = r;
}
