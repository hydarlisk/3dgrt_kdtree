#pragma once

#include "GBase.h"
#include "GColor.h"

/**
 *	물체의 Material.
 *	물체는 다음과 같은 material 로 표현한다.
 *
 *	by graphicsian
 */
class  GMaterial
{
public:
	GColor m_Ambient;				//	ambient color
	GColor m_Diffuse;				//	diffuse color
	GColor m_Specular;				//	specular color
	GColor m_Emission;				//	emission color

	//	m_fReflection + m_fTransparency + m_fLocalShading 은 1 이어야 한다.
	//	따라서 m_fLocalShading 비율은 = m_fReflection - m_fTransparency;
	float m_fReflection;			//	reflection 확률.
	float m_fTransparency;			//	refraction 확률.

	float m_fRoughness;				//	roughness. 
	float m_fRefractionIndex;		//  굴절률

public:
	GMaterial(void);
	virtual ~GMaterial(void);

	void setDiffuse( const GColor &diffuse );
	void setAmbient( const GColor &ambient );
	void setSpecular( const GColor &specular );
	void setTransmission( const GColor &trans );
	void setEmission( const GColor &emission );
	void setRoughness( const float &roughness );
	void setReflection( const float &r );
	void setTransparency( const float &r );

	float getReflection()        { return m_fReflection; }
	GColor getDiffuse()          { return m_Diffuse; }
	GColor getSpecular()         { return m_Specular; }
	GColor getTransmission();
	GColor getAmbient()          { return m_Ambient; }
	GColor getEmission()         { return m_Emission; }
	float getRoughness()         { return m_fRoughness; }
	float getRefractionIndex()   { return m_fRefractionIndex; }
	float getTransparency()      { return m_fTransparency; }

	bool validMaterialProperty();
	bool isTransparent();
};