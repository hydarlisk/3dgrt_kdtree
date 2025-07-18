#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GSpatialStructure.h"
#include "GImageBuffer.h"
#include "GIntersectionPointMap.h"
#include "GKDTreeStructure.h"
#include "SSERenderPipeline.h"
#include "SSERenderData.h"

#include "GThreadWork.h"
#include "GThreadingOption.h"
#include "GCriticalSection.h"

/**
 *	기본적으로 Path Tracing 을 사용하는 렌더러
 *	by graphicsian.
 */
class  GSSERayTracerForPaper : public GRenderer, public GThreadWork
{
public:
	// thread work 를 위한 함수
	void work( GThreadContext *pThreadContext );
	void stop();

private:
	int m_iOldSceneNumber;
	int m_iSceneTimestamp;
	GDimension m_oldResolution;
	GScene *m_pScene;
	int m_iThreadCount;
	int m_iThreadStep;
	int m_iThreadJobSize;

	vector<SSERenderPipeline*>	m_SSERenderPipelineList;
	SSERenderPipeline			*m_pSSERenderPipeline;
	SSESceneData				*m_pSSESceneData;

	Detect_ADPSS_measure	*m_ADPSS_data;
	SSERayQueue4x4			*m_RayQ4x4;
	SSERayTable2x2			*m_RayT2x2;
	SSERayTable1x1			*m_RayT1x1;

public:
	GSSERayTracerForPaper();
	virtual ~GSSERayTracerForPaper(void);

	virtual bool isOpenGLPipeline() { return false; }
	virtual bool isDistributed() { return false; }
	GError rendering( GScene* pScene, bool isDebug );
	void Prepare_Render(void);
	void Execute_Render(void);

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
};