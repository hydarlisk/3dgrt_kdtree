#pragma once

/**
 *	Photon Mapping 包访 可记
 *
 *	by graphicsian.
 */
#include "GRenderOption.h"

// photon density area 甫 备窍绰 规过.
typedef enum {
	
	densityProjectedCircle,				//	projected circle.
	densityAreaPhoton,					//	area photon 捞侩.

} enumDensityMethod;

class GPhotonMappingOption : public GRenderOption
{
public:
	int m_iEmitPhotonPerIteration;
	int m_iMaxBound;
	int m_iIteration;

	float m_fSearchRadius;

	float m_fTotalSceneLightPower;
	float m_fGridUnitLength;

	bool m_bDirectIllumByPhotonMap;
	bool m_bSaveDirectPhoton;
	bool m_bGlossyEffectPhotonMap;
	enumDensityMethod m_eDensityMethod;

public:
	GPhotonMappingOption(void);
	~GPhotonMappingOption(void);
};
