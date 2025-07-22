#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GSpatialStructure.h"
#include "GImageBuffer.h"
#include "GIntersectionPointMap.h"
#include "GKDTreeStructure.h"
#include "cudaRenderPipeline.h"

/**
 *	여러가지 속도 테스트를 위한 CUDA GPU RayTracer
 */
class GGPUExperimentalRayTracer : public GRenderer
{
private:
	int m_iOldSceneNumber;
	int m_iSceneTimestamp;
	GDimension m_oldResolution;
	GScene *m_pScene;

	cudaRenderPipeline *m_pCudaRenderPipeline;

public:
public:
	GGPUExperimentalRayTracer(void);
	virtual ~GGPUExperimentalRayTracer(void);

	virtual bool isOpenGLPipeline() { return false; }
	virtual bool isDistributed() { return false; }
	GError rendering( GScene* pScene, bool isDebug );
	void enableShadow( bool flag );

	static int toImageIndex( GScene *pScene, int rayIndex, int *x, int *y );

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
	cuCamera calCameraInfo( GScene *pScene );
};
