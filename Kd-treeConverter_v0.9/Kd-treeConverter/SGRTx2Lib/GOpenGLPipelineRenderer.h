#pragma once

//#include "GBase.h"
#include "GRenderer.h"
#include "GSpatialStructure.h"
//#include "GImageBuffer.h"
//#include "GIntersectionPointMap.h"
#include "GKDTreeStructure.h"
//#include "SSERenderPipeline.h"
//#include "SSERenderData.h"

#include "GThreadWork.h"
#include "GThreadingOption.h"

class  GOpenGLPipelineRenderer : public GRenderer
{
private:
	GScene *m_pScene;

public:
	GOpenGLPipelineRenderer();
	virtual ~GOpenGLPipelineRenderer(void);

	GError rendering( GScene* pScene, bool isDebug );

	bool isOpenGLPipeline() { return true; }
	bool isDistributed() { return false; }

protected:
	void uninitialize();
	GError initialize( GScene *pScene );
};
