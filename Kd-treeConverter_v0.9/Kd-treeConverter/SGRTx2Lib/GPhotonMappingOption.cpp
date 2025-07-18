#include ".\gphotonmappingoption.h"

GPhotonMappingOption::GPhotonMappingOption(void)
{
	m_iEmitPhotonPerIteration = 10000;
	m_iIteration = 1;
	m_iMaxBound = 10;
	m_fSearchRadius = 1.0f;
	m_fGridUnitLength = 1.0f;
	m_fTotalSceneLightPower = 500.0f;
	m_bDirectIllumByPhotonMap = false;
	m_bSaveDirectPhoton = true;
	m_bGlossyEffectPhotonMap = false;
	m_eDensityMethod = densityProjectedCircle;
}

GPhotonMappingOption::~GPhotonMappingOption(void)
{
}
