#include "GSSERayTracerForQueue.h"
#include "GSpatialStructure.h"
#include "GKDTreeStructure.h"
#include "GPointLight.h"
#include "GRenderCommon.h"
		#include "SSERenderPipeline.h"				//yet GRenderCommon.h 에 넣을것
#include "math.h"

#include "GThreadManager.h"
#include "GThreadingOption.h"


// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

GSSERayTracerForQueue::GSSERayTracerForQueue()
{
	m_iOldSceneNumber = 0;
	m_pSSERenderPipeline = NULL;
	m_pSSESceneData     = NULL;
	m_iSceneTimestamp = -1;
	m_pScene = NULL;
	m_iThreadCount = 0;
}

GSSERayTracerForQueue::~GSSERayTracerForQueue(void)
{
	uninitialize();
}

// ------------------------------------------------------------------------------------------------
// uninitialize()
// ------------------------------------------------------------------------------------------------
void GSSERayTracerForQueue::uninitialize()
{
	if ( m_pSSERenderPipeline ) {
		delete m_pSSERenderPipeline;
		m_pSSERenderPipeline = NULL;
	}

	if ( m_pSSESceneData ) {
		delete m_pSSESceneData;
		m_pSSESceneData = NULL;
	}
}

// ------------------------------------------------------------------------------------------------
// initialize()		: scene 정보를 재구성해야할때 초기화한다.
// ------------------------------------------------------------------------------------------------
GError GSSERayTracerForQueue::initialize( GScene *pScene )
{
	GError error;
	m_pScene = pScene;

	bool bModifiedScene  = false;
	bool bModifiedThread = false;

	if ( m_iOldSceneNumber != pScene->getSceneNumber() || 
		 m_iSceneTimestamp != pScene->getGeometryChangeTimestamp() ||
		 m_oldResolution != pScene->getResolution() ) {

		// ----------------------------------------------------------
		// KdTree 및 triangle 정보를 포인팅하는 공유 구조체 생성
		// ----------------------------------------------------------
		if ( m_pSSESceneData ) {
			delete m_pSSESceneData;
		}
		m_pSSESceneData = new SSESceneData(pScene);

		error = pScene->getKDTreeStructure()->makeSSERenderStructureInfo( m_pSSESceneData );
		if ( error != errorNo )
			return error;

		// ----------------------------------------------------------
		// 현재 Renderer 가 처리한 Scene 을 기억한다.
		// ----------------------------------------------------------
		m_iOldSceneNumber = pScene->getSceneNumber();
		m_iSceneTimestamp = pScene->getGeometryChangeTimestamp();
		m_oldResolution = pScene->getResolution();
		bModifiedScene = true;
	}

	if (m_iThreadCount != m_pScene->getCPUThreadCount()) {
		m_iThreadCount = pScene->getCPUThreadCount();
		bModifiedThread = true;
	}

	if (bModifiedScene) {
		// ----------------------------------------------------------
		// Render Pipeline 생성
		// ----------------------------------------------------------
		if ( m_pSSERenderPipeline ) {
			delete m_pSSERenderPipeline;
		}
		m_pSSERenderPipeline = new SSERenderPipelineQ(pScene, m_pSSESceneData);
	}

	return errorNo;
}

// ------------------------------------------------------------------------------------------------
// rendering()
// ------------------------------------------------------------------------------------------------
GError GSSERayTracerForQueue::rendering( GScene *pScene, bool isDebug )
{
	GError error;

	// ----------------------------------------------------------
	//	Scene 이 이전 geometry 상태에서 변한게 있는지 체크해서 있다면
	//	SpatialStructure 를 재구성한다.
	// ----------------------------------------------------------
	error = initialize( pScene );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't initialize rendering\n" );
		return error;
	}

	// ----------------------------------------------------------
	// PrepareRender 호출 : Screen 및 Ray 등 셋팅
	// ----------------------------------------------------------
	m_pSSERenderPipeline->PrepareRender(m_iThreadCount);
	m_pSSERenderPipeline->m_TraceQ4x4->ClearIndex();		// 현재는 안쓰는데.. 나중은 모름 에러나면 NULL 이라서 그럴수 있음-.-

	GTimer timer1;
	timer1.start();

	// ----------------------------------------------------------
	// Render()
	// ----------------------------------------------------------
	m_pSSERenderPipeline->Render4x4Q();

	if ( error != errorNo ) {
		GLogManager::logging( LOG_FATAL, "Rendering Error, %s\n", GErrorManager::getGErrorString( error ) );
		return errorNo;
	}

	timer1.end();
	m_pScene->setFPS( 1.0f / timer1.getElapsedTime() );

	GLogManager::logging( LOG_DEBUG, "traceRayFullBounce time : %f sec.\n", timer1.getElapsedTime() );

	return errorNo;
}