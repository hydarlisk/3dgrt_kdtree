#include "GSSERayTracerForPaper.h"
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

#define GOThread 1

// ------------------------------------------------------------------------------------------------
// work
// ------------------------------------------------------------------------------------------------
void GSSERayTracerForPaper::work( GThreadContext *pThreadContext ) 
{
	int nTID = pThreadContext->getWorkNumber();

	// PrePare Renderer
	if (m_iThreadStep == -1) {
		m_SSERenderPipelineList[nTID]->PrepareRender(m_iThreadCount);
		return;
	}

	int nPacketSize = m_pScene->getCPUPacketSize().x;
	switch (nPacketSize) {
		case 1:
			for (int i = nTID; i < m_iThreadJobSize; i+=m_iThreadCount) {
			switch (m_iThreadStep) {
				case 0 :	{ m_SSERenderPipelineList[nTID]->Render1x1_ADPSS_OnePass(i);		break;	}
				case 1 :	{ m_SSERenderPipelineList[nTID]->Render1x1_ADPSS_Detection(i);		break;	}
				case 2 :	{ m_SSERenderPipelineList[nTID]->Render1x1_ADPSS_TwoPass(i);		break;	}
			}}
			break;
		case 4:
			for (int i = nTID; i < m_iThreadJobSize; i+=m_iThreadCount) {
			switch (m_iThreadStep) {
				case 0 :	{ m_SSERenderPipelineList[nTID]->Render4x4_ADPSS_OnePass(i);		break;	}
				case 1 :	{ m_SSERenderPipelineList[nTID]->Render4x4_ADPSS_Detection(i);		break;	}
				case 2 :	{ m_SSERenderPipelineList[nTID]->Render4x4_ADPSS_TwoPass(i);		break;	}
			}}
			break;
	}
}

// ------------------------------------------------------------------------------------------------
// stop
// ------------------------------------------------------------------------------------------------
void GSSERayTracerForPaper::stop()
{
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

GSSERayTracerForPaper::GSSERayTracerForPaper()
{
	m_iOldSceneNumber = 0;
	m_pSSERenderPipeline = NULL;
	m_pSSESceneData     = NULL;
	m_iSceneTimestamp = -1;
	m_pScene = NULL;
	m_iThreadCount = 0;

	m_ADPSS_data = (Detect_ADPSS_measure*) _aligned_malloc(1280*1280* sizeof(Detect_ADPSS_measure), 16);
	//m_RayQ4x4  = new SSERayQueue4x4(102400);		// 1280*1280/16  = 102400   (All  Primary Rays, 1x1 subsampling, packetsize = 16)
	m_RayT2x2  = new SSERayTable2x2(1280*1280*4);	// 1280*1280*4   = 6553600  (Some Primary Rays, Max 4packets per 1pix)
	m_RayT1x1  = new SSERayTable1x1(1280*1280*16);	// 1280*1280*16  = ?        (Some Primary Rays, Max 16rays per 1pix)
}

GSSERayTracerForPaper::~GSSERayTracerForPaper(void)
{
	uninitialize();

	_aligned_free(m_ADPSS_data);
	//delete(m_RayQ4x4);
	delete(m_RayT2x2);
	delete(m_RayT1x1);
}

// ------------------------------------------------------------------------------------------------
// uninitialize()
// ------------------------------------------------------------------------------------------------
void GSSERayTracerForPaper::uninitialize()
{
#if GOThread
	for ( int i = 0; i < (int)m_SSERenderPipelineList.size(); ++i ) {
		delete m_SSERenderPipelineList[i];
	}
	m_SSERenderPipelineList.clear();
#else
	if ( m_pSSERenderPipeline ) {
		delete m_pSSERenderPipeline;
		m_pSSERenderPipeline = NULL;
	}
#endif

	if ( m_pSSESceneData ) {
		delete m_pSSESceneData;
		m_pSSESceneData = NULL;
	}
}

// ------------------------------------------------------------------------------------------------
// initialize()		: scene 정보를 재구성해야할때 초기화한다.
// ------------------------------------------------------------------------------------------------
GError GSSERayTracerForPaper::initialize( GScene *pScene )
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

	if (bModifiedScene || bModifiedThread) {
#if GOThread
		// ----------------------------------------------------------
		// Render Pipeline 생성
		// ----------------------------------------------------------
		for ( int i = 0; i < (int)m_SSERenderPipelineList.size(); ++i ) {
			delete m_SSERenderPipelineList[i];
		}
		m_SSERenderPipelineList.clear();

		for ( int i = 0; i < m_iThreadCount; ++i ) {
			SSERenderPipeline *pSSERenderPipeline = new SSERenderPipeline(pScene, m_pSSESceneData, m_ADPSS_data, m_RayQ4x4, m_RayT2x2, m_RayT1x1);
			m_SSERenderPipelineList.push_back( pSSERenderPipeline );
		}
#else
		// ----------------------------------------------------------
		// Render Pipeline 생성
		// ----------------------------------------------------------
		if ( m_pSSERenderPipeline ) {
			delete m_pSSERenderPipeline;
		}
		m_pSSERenderPipeline = new SSERenderPipeline(pScene, m_pSSESceneData, m_ADPSS_data, m_RayQ4x4, m_RayT2x2, m_RayT1x1);
#endif
	}

	return errorNo;
}


// ------------------------------------------------------------------------------------------------
// PrepareRender() : Screen 및 Ray 등 셋팅
// ------------------------------------------------------------------------------------------------
void GSSERayTracerForPaper::Prepare_Render( void )
{
	#if GOThread
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//                USE Thread
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		m_iThreadStep = -1;
		GThreadManager::startThreadWork( this, m_iThreadCount );
		GThreadManager::waitThreadWork();
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//                USE Thread
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


	#else
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//               NO use Thread
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		m_pSSERenderPipeline->PrepareRender(1);
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//               NO use Thread
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	#endif
}

// ------------------------------------------------------------------------------------------------
// Execute_Render() : Render-Pipeline 실행
// ------------------------------------------------------------------------------------------------
void GSSERayTracerForPaper::Execute_Render( void )
{
	// ----------------------------------------------------------
	// Render()
	// ----------------------------------------------------------

	#if GOThread
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//                USE Thread
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

		int nThreadStepSize = 3;
		m_iThreadJobSize	= m_iThreadCount * THREADING_JITTER_SIZE;

		for (m_iThreadStep = 0; m_iThreadStep < nThreadStepSize; m_iThreadStep++) {
			GThreadManager::startThreadWork( this, m_iThreadCount );
			GThreadManager::waitThreadWork();
		}

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//                USE Thread
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	#else
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//               NO use Thread
	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

		int nPacketSize = m_pScene->getCPUPacketSize().x;
		switch (nPacketSize) {
			case 1:
				m_pSSERenderPipeline->Render1x1_ADPSS_OnePass();
				m_pSSERenderPipeline->Render1x1_ADPSS_Detection();
				m_pSSERenderPipeline->Render1x1_ADPSS_TwoPass();
				break;
			case 4:
				m_pSSERenderPipeline->Render4x4_ADPSS_OnePass();
				m_pSSERenderPipeline->Render4x4_ADPSS_Detection();
				m_pSSERenderPipeline->Render4x4_ADPSS_TwoPass();
				break;
		}

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//               NO use Thread
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	#endif
}

// ------------------------------------------------------------------------------------------------
// rendering()
// ------------------------------------------------------------------------------------------------
GError GSSERayTracerForPaper::rendering( GScene *pScene, bool isDebug )
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

	Prepare_Render();

	//m_RayQ4x4->Clear();
	m_RayT2x2->Clear(m_oldResolution.x*m_oldResolution.y*4);
	m_RayT1x1->Clear(m_oldResolution.x*m_oldResolution.y*16);

	GTimer timer1;
	timer1.start();

	Execute_Render();

	timer1.end();
	m_pScene->setFPS( 1.0f / timer1.getElapsedTime() );

	GLogManager::logging( LOG_DEBUG, "traceRayFullBounce time : %f sec.\n", timer1.getElapsedTime() );

	if ( error != errorNo ) {
		GLogManager::logging( LOG_FATAL, "Rendering Error, %s\n", GErrorManager::getGErrorString( error ) );
		return errorNo;
	}

	return errorNo;
}
