#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GScene.h"
#include "GDimension.h"
#include "GSpatialStructure.h"
#include "GIntersectionPointMap.h"
#include "GKDTreeStructure.h"
#include "GGridBox.h"
#include "cudaRenderPipeline.h"
#include "cudaPhotonMapping.h"
#include "GPhotonMappingOption.h"
#include "GRenderCommon.h"

/**
 *	Grid Box 에서 x, y, z 각각의 최대 grid 개수.
 */
#define MAX_GRID_COUNT	250		

/**
 *	Ray Tracing 과 Photon Mapping 으로 GI 를 구현한 Renderer
 *
 *	by graphicsian.
 */
class GPhotonMapRayTracer : public GRenderer
{
private:
	GScene *m_pScene;
	
	bool m_bDebug;

	GPhotonMappingOption m_Option;

	int m_iRandomSeed;
	int m_iMaxPhotonSize;
	float m_fSceneLightPowerPerIteration;
	float m_fOnePhotonPower;

	int m_iOldSceneNumber;
	int m_iSceneTimestamp;
	GDimension m_oldResolution;

	cudaRenderPipeline *m_pCudaRenderPipeline;
	cudaPhotonMapping *m_pCudaPhotonMapping;

	GIntersectionPointMap *m_pIntersectionPointMap;
	cuPhoton *m_pHostPhotonMem;

	GGridBox<cuPhoton, char> *m_pGlobalPhotonGridBox;

	GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *m_pIPointGridBox;
	bool m_bRunTracing;

	/** 통계데이터 */
	int m_iLogTotalTracedPhoton;
	float m_iLogTotalIsectGridMakingTime;
	float m_iLogTotalIsectAreaDensityTime;
	float m_iLogTotalTracingTime;
	float m_iLogTotalGatheringTime;
	float m_iLogTotalIndexGridTime;
	float m_iLogTotalPhotonGridMakingTime;
	float m_iLogTotalAccumulateTime;
	float m_iLogTotalPhotonMappingTime;

public:
	GPhotonMapRayTracer( GPhotonMappingOption *pOption );
	virtual ~GPhotonMapRayTracer(void);

	virtual bool isOpenGLPipeline() { return false; }
	virtual bool isDistributed() { return false; }
	GError rendering( GScene* pScene, bool isDebug );
	GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *getIPointGridBox();
	void accumulateRadiance( GImageBuffer *pImageBuffer,
				 GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox );

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
	GError initPhotonMapping( GScene *pScene );
	GError makePrimaryRaySet_BlockGrouping( GScene *pScene, int *generatedCount, int currentSampleX, int currentSampleY );
	GError makePrimaryRaySet( GScene *pScene, int *generatedRayCount, int currentSampleX, int currentSampleY );
	GError constructPhotonEmitLightInfo( cuLight* plightList, int icuLightCount );
	GError backupIntersectionResult( int count );
	GError photonMapIteration( bool m_bRunTracing, int iteration, GImageBuffer *pIndirectImageBuffer );
	GError photonMapOneIteration( int randomSeed,
			GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox,	
			bool m_bRunTracing, int iterationId, int maxIteration );

	GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *makeIPointGridBox( 
		GIntersectionPointMap *pIntersectionPointMap );
	GGridBox<cuPhoton, char> *makeAreaPhotonGridBox( const vector<cuPhoton*> &list );
	GGridBox<cuPhoton, char> *makePhotonGridBox( cuPhoton *pPhotons, int size );
	cuPhotonIndex* makeIPointVsPhotonIndexData( 
		GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox,
		GGridBox<cuPhoton, char> *pPhotonGridBox, int *pOutPhotonIndexCount );

	GError estimateAreaByAreaPhoton( 
				GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox );
	GError estimateDensityArea( 
				GGridBox<cuIntersectionPoint, cuPMIntersectionPoint>* pIGridBox,
								enumDensityMethod method );
	inline int toImageIndex( int rayIndex, int *x, int *y );

};
