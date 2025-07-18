#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GSpatialStructure.h"
#include "GImageBuffer.h"
#include "GIntersectionPointMap.h"
#include "GKDTreeStructure.h"
#include "cudaRenderPipeline.h"

/**
 *	기본적으로 Path Tracing 을 사용하는 렌더러
 *	by graphicsian.
 */
class  GGPURayTracer : public GRenderer
{
private:
	int m_iOldSceneNumber;
	int m_iSceneTimestamp;
	GDimension m_oldResolution;
	GScene *m_pScene;

	cudaRenderPipeline *m_pCudaRenderPipeline;

public:
	GGPURayTracer();
	virtual ~GGPURayTracer(void);

	virtual bool isOpenGLPipeline() { return false; }
	virtual bool isDistributed() { return false; }
	GError rendering( GScene* pScene, bool isDebug );

	static int toImageIndex( GScene *pScene, int rayIndex, int *x, int *y );

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
	GError makePrimaryRaySet( GScene *pScene, int *generatedRayCount, int currentSampleX, int currentSampleY );
	GError makePrimaryRaySet_BlockGrouping( GScene *pScene, int *generatedCount, int currentSampleX, int currentSampleY );

	cuCamera calCameraInfo( GScene *pScene );
};
