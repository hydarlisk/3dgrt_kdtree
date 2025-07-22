#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GSpatialStructure.h"
#include "GImageBuffer.h"
#include "GIntersectionPointMap.h"
#include "GKDTreeStructure.h"
#include "cudaRenderPipeline.h"

/**
 *	다른 논문과의 속도 비교를 위한 GPU 버전 클래스.
 *	기능과 확장성 보다는 불필요한 기능들을 빼고 Primary Ray 만을 체크하거나
 *	Shading 을 포함한 속도등 속도위주의 체크를 위한 버전.
 */
class GGPURayTracerForPaper : public GRenderer
{
private:
	int m_iOldSceneNumber;
	int m_iSceneTimestamp;
	GDimension m_oldResolution;
	GScene *m_pScene;

	cudaRenderPipeline *m_pCudaRenderPipeline;

public:
public:
	GGPURayTracerForPaper(void);
	virtual ~GGPURayTracerForPaper(void);

	virtual bool isOpenGLPipeline() { return false; }
	virtual bool isDistributed() { return false; }
	GError rendering( GScene* pScene, bool isDebug );

	static int toImageIndex( GScene *pScene, int rayIndex, int *x, int *y );

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
	cuCamera calCameraInfo( GScene *pScene );
};
