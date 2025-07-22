#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GSpatialStructure.h"
#include "GImageBuffer.h"
#include "GIntersectionPointMap.h"
#include "GKDTreeStructure.h"
#include "SSERenderPipelineQ.h"
#include "SSERayQueue.h"
#include "SSERenderData.h"

/**
 *	기본적으로 Path Tracing 을 사용하는 렌더러
 *	by graphicsian.
 */
class  GSSERayTracerForQueue : public GRenderer
{
private:
	int m_iOldSceneNumber;
	int m_iSceneTimestamp;
	GDimension m_oldResolution;
	GScene *m_pScene;
	int m_iThreadCount;

	SSERenderPipelineQ  *m_pSSERenderPipeline;
	SSESceneData        *m_pSSESceneData;

public:
	GSSERayTracerForQueue();
	virtual ~GSSERayTracerForQueue(void);

	virtual bool isDistributed() { return false; }
	GError rendering( GScene* pScene, bool isDebug );

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
};
