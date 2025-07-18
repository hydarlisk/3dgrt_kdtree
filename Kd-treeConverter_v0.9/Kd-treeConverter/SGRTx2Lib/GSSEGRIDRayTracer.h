#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GSpatialStructure.h"
#include "GImageBuffer.h"
#include "GIntersectionPointMap.h"

#include "GGridStructure.h"

#include "SSERenderPipeline.h"
#include "SSERenderData.h"

#include "GThreadWork.h"
#include "GThreadingOption.h"

/**
 *	기본적으로 Path Tracing 을 사용하는 렌더러
 *	by graphicsian.
 */
class  GSSEGRIDRayTracer : public GRenderer, public GThreadWork
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

	vector<SSERenderPipeline*> m_SSERenderPipelineList;
	SSERenderPipeline *m_pSSERenderPipeline;
	SSESceneData     *m_pSSESceneData;

public:
	GSSEGRIDRayTracer();
	virtual ~GSSEGRIDRayTracer(void);

	virtual bool isOpenGLPipeline() { return false; }
	virtual bool isDistributed() { return false; }
	GError rendering( GScene* pScene, bool isDebug );
	void logging_init();
	void logging_out();

	GVector m_OldEye;
	int m_RunStatics;
	int m_RunProfile;

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
};
