
#include "GSSEBVHRayTracer.h"
#include "GSpatialStructure.h"
#include "GKDTreeStructure.h"
#include "GPointLight.h"
#include "GRenderCommon.h"
		#include "SSERenderPipeline.h"				//yet GRenderCommon.h 에 넣을것
#include "math.h"

#include "GThreadManager.h"
#include "GThreadingOption.h"

#include <algorithm>


// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

#define GOThread 0

// ------------------------------------------------------------------------------------------------
// work
// ------------------------------------------------------------------------------------------------
void GSSEBVHRayTracer::work( GThreadContext *pThreadContext ) 
{
	int nJob = pThreadContext->getWorkNumber();

	if (m_pScene->getCPUPacketSize().x == 4) {
		//m_SSERenderPipelineList[nJob]->Render4x4(nJob);
		m_SSERenderPipelineList[nJob]->Split_Render4x4__PriRay(nJob);
	} else if (m_pScene->getCPUPacketSize().x == 2) {
		m_SSERenderPipelineList[nJob]->Render2x2(nJob);
	} else {
		m_SSERenderPipelineList[nJob]->Render1x1(nJob);
	}
}

// ------------------------------------------------------------------------------------------------
// stop
// ------------------------------------------------------------------------------------------------
void GSSEBVHRayTracer::stop()
{
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

GSSEBVHRayTracer::GSSEBVHRayTracer()
{
	m_iOldSceneNumber = 0;
	m_pSSERenderPipeline = NULL;
	m_pSSESceneData     = NULL;
	m_iSceneTimestamp = -1;
	m_pScene = NULL;
	m_iThreadCount = 0;

	m_RunStatics = 0;
	m_RunProfile = 0;
}

GSSEBVHRayTracer::~GSSEBVHRayTracer(void)
{
	uninitialize();
}

// ------------------------------------------------------------------------------------------------
// uninitialize()
// ------------------------------------------------------------------------------------------------
void GSSEBVHRayTracer::uninitialize()
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
GError GSSEBVHRayTracer::initialize( GScene *pScene )
{
	GError error;
	m_pScene = pScene;

	bool bModifiedScene  = false;
	bool bModifiedThread = false;

	bool bProfileFlag = pScene->IsProfileFlag();

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

		error = pScene->getBVHStructure()->makeSSERenderStructureInfo( m_pSSESceneData );		
		if ( error != errorNo )
			return error;

		// ----------------------------------------------------------
		// 여러가지 수 계산
		// ----------------------------------------------------------
		m_pSSESceneData->calSceneProperty();

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
			SSERenderPipeline *pSSERenderPipeline = new SSERenderPipeline(pScene, m_pSSESceneData);
			m_SSERenderPipelineList.push_back( pSSERenderPipeline );
		}

		if (bProfileFlag) {
			m_RunStatics = 1;
			m_RunProfile = 1;
		}
#else
		// ----------------------------------------------------------
		// Render Pipeline 생성
		// ----------------------------------------------------------
		if ( m_pSSERenderPipeline ) {
			delete m_pSSERenderPipeline;
		}
		m_pSSERenderPipeline = new SSERenderPipeline(pScene, m_pSSESceneData);
		if (bProfileFlag) {
			m_RunStatics = 1;
			m_RunProfile = 1;
		}
#endif
	}

	GCamera* pCamera = pScene->getRenderCamera();
	GVector m_NewEye = pCamera->getEye();
	if (!(m_OldEye == m_NewEye)) {
		if (bProfileFlag) {
			m_RunStatics = 1;
			m_RunProfile = 1;
		}
		m_OldEye = m_NewEye;
	}
	return errorNo;
}

// ------------------------------------------------------------------------------------------------
// rendering()
// ------------------------------------------------------------------------------------------------
GError GSSEBVHRayTracer::rendering( GScene *pScene, bool isDebug )
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
	#if GOThread
	for ( int i = 0; i < m_iThreadCount; ++i ) {
		m_SSERenderPipelineList[i]->PrepareRender(m_iThreadCount);
	}
	#else
	m_pSSERenderPipeline->PrepareRender(1);
	#endif

	// ----------------------------------------------------------
	// logging_init()
	// ----------------------------------------------------------
	if (m_RunProfile == 1 && (m_pScene->getCPUPacketSize().x == 1 || m_pScene->getCPUPacketSize().x == 2 || m_pScene->getCPUPacketSize().x == 4)) {
		logging_init();
	}

	if (m_pScene->IsRun10Times() == true) {
		GTimer timer1;
		timer1.start();
		for(int i = 0; i < 10; i++) {
			#if GOThread
			GThreadManager::startThreadWork( this, m_iThreadCount );
			GThreadManager::waitThreadWork();
			#else
			if (m_pScene->getCPUPacketSize().x == 4) {
				m_pSSERenderPipeline->Split_Render4x4__PriRay();
				//m_pSSERenderPipeline->Render4x4();
			} else if (m_pScene->getCPUPacketSize().x == 2) {
				//m_pSSERenderPipeline->Split_Render2x2__PriRay();
				m_pSSERenderPipeline->Split_Render2x2__PriRay();
			} else {
				m_pSSERenderPipeline->Split_Render1x1__PriRay();
				//m_pSSERenderPipeline->Render1x1();
			}
			#endif
		}
		timer1.end();

		float fAvgTime = timer1.getElapsedTime() / 10.0f;

		if (m_RunProfile == 1) {
			GLogManager::logging( LOG_INFO, "       , 10 avg times  (second), %10.3f", fAvgTime );
			GLogManager::logging( LOG_INFO, "CMDO-1 , Frame per second      , %10.3f", 1/fAvgTime);
			FILE *fp = fopen( m_pScene->getProfileResultPath(), "at" );
			fprintf(fp, "       , 10 avg times  (second), %10.3f\n", fAvgTime );
			fprintf(fp, "CMDO-1 , Frame per second      , %10.3f\n", 1/fAvgTime);
			fclose(fp);
			m_RunProfile = 0;
		} else {
			GLogManager::logging( LOG_INFO, "         10 avg times  (second), %10.3f", fAvgTime );
		}
		m_pScene->setRun10Times( false );

		return errorNo;
	}


	GTimer timer1;
	timer1.start();

	// ----------------------------------------------------------
	// Render()
	// ----------------------------------------------------------
	#if GOThread
	GThreadManager::startThreadWork( this, m_iThreadCount );
	GThreadManager::waitThreadWork();
	#else
	if (m_pScene->getCPUPacketSize().x == 4) {
		//m_pSSERenderPipeline->Render4x4();
		m_pSSERenderPipeline->Split_Render4x4__PriRay();
		//m_pSSERenderPipeline->Split_Render4x4__PriRayBVH();
		//m_pSSERenderPipeline->Split_Render1x1__PriRay();
	} else if (m_pScene->getCPUPacketSize().x == 2) {
		//m_pSSERenderPipeline->Render2x2();
		m_pSSERenderPipeline->Split_Render2x2__PriRay();
		//m_pSSERenderPipeline->Split_Render1x1__PriRayBVH();
	} else {
		//m_pSSERenderPipeline->Render1x1();
		//m_pSSERenderPipeline->Split_Render1x1__PriRay();
		//m_pSSERenderPipeline->Split_Render1x1__PriRayBVH_PACKET();		
		m_pSSERenderPipeline->Render1x1_BVHPacketTraversal();
		//m_pSSERenderPipeline->Render_SSE_BVHPacketTraversal();
		//m_pSSERenderPipeline->Render1x1_BVHTraversal();
	}
	#endif

	if ( error != errorNo ) {
		GLogManager::logging( LOG_FATAL, "Rendering Error, %s\n", GErrorManager::getGErrorString( error ) );
		return errorNo;
	}

	timer1.end();
	m_pScene->setFPS( 1.0f / timer1.getElapsedTime() );

	GLogManager::logging( LOG_DEBUG, "traceRayFullBounce time : %f sec.\n", timer1.getElapsedTime() );

	// ----------------------------------------------------------
	// logging_out()
	// ----------------------------------------------------------
	if (m_RunProfile == 1 && (m_pScene->getCPUPacketSize().x == 1 || m_pScene->getCPUPacketSize().x == 2 || m_pScene->getCPUPacketSize().x == 4)) {
		logging_out();
		m_RunStatics = 0;
	}

	return errorNo;

}

// ------------------------------------------------------------------------------------------------
// logging()
// ------------------------------------------------------------------------------------------------
void GSSEBVHRayTracer::logging_init()
{
	#if GOThread
		if ( m_iThreadCount > 0) {
			for ( int i = 0; i < m_iThreadCount; ++i ) {
				m_SSERenderPipelineList[i]->m_RunStatics = 1;

				m_SSERenderPipelineList[i]->m_pf_Hit_DiffPnt_PR  = 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_DiffPnt_RR  = 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_SpecPnt_PR  = 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_SpecPnt_RR  = 0;
				m_SSERenderPipelineList[i]->m_pf_TexRef_PR          = 0;
				m_SSERenderPipelineList[i]->m_pf_TexRef_RR  = 0;

				m_SSERenderPipelineList[i]->m_pf_Hit_ShwPnt_ALL	= 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_ShwCnt_PR	= 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_ShwCnt_RR	= 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_ShadCnt_PR	= 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_ShadCnt_RR	= 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_ShadPnt_PR	= 0;
				m_SSERenderPipelineList[i]->m_pf_Hit_ShadPnt_RR	= 0;
			}
		}
	#else
	m_pSSERenderPipeline->m_RunStatics = 1;

	m_pSSERenderPipeline->m_pf_Hit_DiffPnt_PR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_DiffPnt_RR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_SpecPnt_PR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_SpecPnt_RR	= 0;

	m_pSSERenderPipeline->m_pf_TexRef_PR	= 0;
	m_pSSERenderPipeline->m_pf_TexRef_RR	= 0;

	m_pSSERenderPipeline->m_pf_Hit_ShwPnt_ALL	= 0;
	m_pSSERenderPipeline->m_pf_Hit_ShwCnt_PR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_ShwCnt_RR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_ShadCnt_PR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_ShadCnt_RR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_ShadPnt_PR	= 0;
	m_pSSERenderPipeline->m_pf_Hit_ShadPnt_RR	= 0;

	m_pSSERenderPipeline->vTriID.clear();
	#endif
}


void GSSEBVHRayTracer::logging_out()
{
	FILE *fp = fopen( m_pScene->getProfileResultPath(), "wt" );

	GDimension Resolution = m_pScene->getResolution();
	GDimension SuperSampling = m_pScene->getSuperSampling();
	bool bIsEnableShadow = m_pScene->isEnableShadow();
	const vector<GLight*>* pLightList = m_pScene->getLightList();
	int iMaxReflectionDepth = m_pScene->getMaxReflectionDepth();

	GLogManager::logging( LOG_INFO, "PRE-1  , Image Resolution      , %dx%d", Resolution.x, Resolution.y);
	GLogManager::logging( LOG_INFO, "PRE-2  , Sampling              , %dx%d", SuperSampling.x, SuperSampling.y);
	GLogManager::logging( LOG_INFO, "PRE-3  , Shadow                , %s", bIsEnableShadow?"on":"off");
	GLogManager::logging( LOG_INFO, "PRE-4  , # of Light Sources    , %d", pLightList->size());
	GLogManager::logging( LOG_INFO, "PRE-5  , Ray Bounce Depth      , %d", iMaxReflectionDepth);

	fprintf(fp, "PRE-1  , Image Resolution      , %dx%d\n", Resolution.x, Resolution.y);
	fprintf(fp, "PRE-2  , Sampling              , %dx%d\n", SuperSampling.x, SuperSampling.y);
	fprintf(fp, "PRE-3  , Shadow                , %s\n", bIsEnableShadow?"on":"off");
	fprintf(fp, "PRE-4  , # of Light Sources    , %d\n", pLightList->size());
	fprintf(fp, "PRE-5  , Ray Bounce Depth      , %d\n", iMaxReflectionDepth);


	// 프로파일링용 --------------------------------------------------------------------------
	float fRatio;
	unsigned int nDupTri = 0;				// 보이는 삼각형 개수 (중복허용)
	unsigned int nUniTri = 1;				// 보이는 삼각형 개수 (unique)

	// PRI-6 : Spec/Diff 지점
	unsigned int _pf_nHit_DiffPnt_ALL, _pf_nHit_DiffPnt_PR, _pf_nHit_DiffPnt_RR; _pf_nHit_DiffPnt_ALL = _pf_nHit_DiffPnt_PR = _pf_nHit_DiffPnt_RR = 0;
	unsigned int _pf_nHit_SpecPnt_ALL, _pf_nHit_SpecPnt_PR, _pf_nHit_SpecPnt_RR; _pf_nHit_SpecPnt_ALL = _pf_nHit_SpecPnt_PR = _pf_nHit_SpecPnt_RR = 0;
	// PRI-9 : 그림자 지는 지점
	unsigned int _pf_nHit_ShwPnt_ALL    = 0;
	unsigned int _pf_nHit_ShwCnt_PR, _pf_nHit_ShwCnt_RR;	_pf_nHit_ShwCnt_PR = _pf_nHit_ShwCnt_RR = 0;
	// CMI-7 : 텍스쳐 access
	unsigned int _pf_nTexRef_ALL, _pf_nTexRef_PR, _pf_nTexRef_RR; _pf_nTexRef_ALL = _pf_nTexRef_PR = _pf_nTexRef_RR = 0;

	// CMI-6 : Shading 처리 횟수
	unsigned int _pf_nHit_ShadPnt_PR, _pf_nHit_ShadPnt_RR; _pf_nHit_ShadPnt_PR = _pf_nHit_ShadPnt_RR = 0;
	unsigned int _pf_nHit_ShadCnt_PR, _pf_nHit_ShadCnt_RR; _pf_nHit_ShadCnt_PR = _pf_nHit_ShadCnt_RR = 0;
	unsigned int _pf_nGen_RaysCnt_PR, _pf_nGen_RaysCnt_RR, _pf_nGen_RaysCnt_SR;  _pf_nGen_RaysCnt_PR = _pf_nGen_RaysCnt_RR = _pf_nGen_RaysCnt_SR = 0;

	#if GOThread
		if ( m_iThreadCount > 0) {
			for ( int i = 0; i < m_iThreadCount; ++i ) {
				m_SSERenderPipelineList[i]->m_RunStatics = 0;
				_pf_nHit_DiffPnt_PR		+= m_SSERenderPipelineList[i]->m_pf_Hit_DiffPnt_PR;
				_pf_nHit_DiffPnt_RR		+= m_SSERenderPipelineList[i]->m_pf_Hit_DiffPnt_RR;
				_pf_nHit_SpecPnt_PR		+= m_SSERenderPipelineList[i]->m_pf_Hit_SpecPnt_PR;
				_pf_nHit_SpecPnt_RR		+= m_SSERenderPipelineList[i]->m_pf_Hit_SpecPnt_RR;

				_pf_nTexRef_PR			+= m_SSERenderPipelineList[i]->m_pf_TexRef_PR;
				_pf_nTexRef_RR			+= m_SSERenderPipelineList[i]->m_pf_TexRef_RR;
				_pf_nHit_ShwPnt_ALL		+= m_SSERenderPipelineList[i]->m_pf_Hit_ShwPnt_ALL;
				_pf_nHit_ShwCnt_PR		+= m_SSERenderPipelineList[i]->m_pf_Hit_ShwCnt_PR;
				_pf_nHit_ShadPnt_PR	+= m_SSERenderPipelineList[i]->m_pf_Hit_ShadPnt_PR;
				_pf_nHit_ShadPnt_RR	+= m_SSERenderPipelineList[i]->m_pf_Hit_ShadPnt_RR;
				_pf_nHit_ShadCnt_PR	+= m_SSERenderPipelineList[i]->m_pf_Hit_ShadCnt_PR;
				_pf_nHit_ShadCnt_RR	+= m_SSERenderPipelineList[i]->m_pf_Hit_ShadCnt_RR;
				_pf_nGen_RaysCnt_PR	+= m_SSERenderPipelineList[i]->G_PR;
				_pf_nGen_RaysCnt_RR	+= m_SSERenderPipelineList[i]->G_RR;
				_pf_nGen_RaysCnt_SR	+= m_SSERenderPipelineList[i]->G_SR;
			}
		}
	#else
		m_pSSERenderPipeline->m_RunStatics = 0;
		_pf_nHit_DiffPnt_PR		= m_pSSERenderPipeline->m_pf_Hit_DiffPnt_PR;
		_pf_nHit_DiffPnt_RR		= m_pSSERenderPipeline->m_pf_Hit_DiffPnt_RR;
		_pf_nHit_SpecPnt_PR		= m_pSSERenderPipeline->m_pf_Hit_SpecPnt_PR;
		_pf_nHit_SpecPnt_RR		= m_pSSERenderPipeline->m_pf_Hit_SpecPnt_RR;
		_pf_nTexRef_PR			= m_pSSERenderPipeline->m_pf_TexRef_PR;
		_pf_nTexRef_RR			= m_pSSERenderPipeline->m_pf_TexRef_RR;
		_pf_nHit_ShwPnt_ALL		= m_pSSERenderPipeline->m_pf_Hit_ShwPnt_ALL;
		_pf_nHit_ShwCnt_PR		= m_pSSERenderPipeline->m_pf_Hit_ShwCnt_PR;
		_pf_nHit_ShadPnt_PR		= m_pSSERenderPipeline->m_pf_Hit_ShadPnt_PR;
		_pf_nHit_ShadPnt_RR		= m_pSSERenderPipeline->m_pf_Hit_ShadPnt_RR;
		_pf_nHit_ShadCnt_PR		= m_pSSERenderPipeline->m_pf_Hit_ShadCnt_PR;
		_pf_nHit_ShadCnt_RR		= m_pSSERenderPipeline->m_pf_Hit_ShadCnt_RR;
		_pf_nGen_RaysCnt_PR		= m_pSSERenderPipeline->G_PR;
		_pf_nGen_RaysCnt_RR		= m_pSSERenderPipeline->G_RR;
		_pf_nGen_RaysCnt_SR		= m_pSSERenderPipeline->G_SR;

		nDupTri = (unsigned int)m_pSSERenderPipeline->vTriID.size();
		if (nDupTri > 0) {
			sort(m_pSSERenderPipeline->vTriID.begin(), m_pSSERenderPipeline->vTriID.end());
			for (unsigned int i = 0; i < nDupTri -1; i++) {
				if (m_pSSERenderPipeline->vTriID[i] == m_pSSERenderPipeline->vTriID[i+1]) continue;
				nUniTri++;
			}
		} else {
			nUniTri = 0;
		}
	#endif

	_pf_nHit_DiffPnt_ALL = _pf_nHit_DiffPnt_PR + _pf_nHit_DiffPnt_RR;
	_pf_nHit_SpecPnt_ALL = _pf_nHit_SpecPnt_PR + _pf_nHit_SpecPnt_RR;
	_pf_nTexRef_ALL      = _pf_nTexRef_PR + _pf_nTexRef_RR;


	GLogManager::logging( LOG_INFO, "PRI-1  , Triangles #           , %10d", m_pSSESceneData->m_TriObjCnt);
	GLogManager::logging( LOG_INFO, "PRI-2  , Visible Tri / Pixels  , %10.3f", 1.0f * nUniTri / (Resolution.x * Resolution.y));

	fRatio = _pf_nHit_SpecPnt_PR==0?0: (100.0f * _pf_nHit_SpecPnt_PR / (_pf_nHit_DiffPnt_PR + _pf_nHit_SpecPnt_PR));
	GLogManager::logging( LOG_INFO, "PRI-6p , Specular Hit     (Pri), %10.2f %%", fRatio);
	fRatio = _pf_nHit_SpecPnt_RR==0?0: (100.0f * _pf_nHit_SpecPnt_RR / (_pf_nHit_DiffPnt_RR + _pf_nHit_SpecPnt_RR));
	GLogManager::logging( LOG_INFO, "PRI-6r , Specular Hit     (Sec), %10.2f %%", fRatio);

	fRatio = 100.0f * (m_pSSESceneData->m_TriAreaSum_Refl + m_pSSESceneData->m_TriAreaSum_Refr) / m_pSSESceneData->m_TriAreaSum_All;
	GLogManager::logging( LOG_INFO, "PRI-7  , Specular Area         , %10.2f %%", fRatio);
	fRatio = 100.0f * m_pSSESceneData->m_TriAreaSum_Tex / m_pSSESceneData->m_TriAreaSum_All;
	GLogManager::logging( LOG_INFO, "PRI-8  , Textured Area         , %10.2f %%", fRatio);

	fRatio = _pf_nHit_ShwPnt_ALL==0?0: (100.0f * _pf_nHit_ShwPnt_ALL / (_pf_nHit_DiffPnt_PR + _pf_nHit_SpecPnt_PR));
	GLogManager::logging( LOG_INFO, "PRI-9  , Shadowed Hit (Pri)    , %10.2f %%", fRatio);

	GLogManager::logging( LOG_INFO, "CMI-1p , Processed Ray    (Pri), %10d", _pf_nGen_RaysCnt_PR);
	GLogManager::logging( LOG_INFO, "CMI-1r , Processed Ray    (Sec), %10d", _pf_nGen_RaysCnt_RR);
	GLogManager::logging( LOG_INFO, "CMI-1s , Processed Ray    (Shw), %10d", _pf_nGen_RaysCnt_SR);
	GLogManager::logging( LOG_INFO, "CMI-6ap, Shading Op Point (Pri), %10d", _pf_nHit_ShadPnt_PR);
	GLogManager::logging( LOG_INFO, "CMI-6ar, Shading Op Point (Sec), %10d", _pf_nHit_ShadPnt_RR);
	GLogManager::logging( LOG_INFO, "CMI-6bp, Shading Op Calc. (Pri), %10d", _pf_nHit_ShadCnt_PR);
	GLogManager::logging( LOG_INFO, "CMI-6br, Shading Op Calc. (Sec), %10d", _pf_nHit_ShadCnt_RR);
	GLogManager::logging( LOG_INFO, "CMI-7p , Texture Access   (Pri), %10d", _pf_nTexRef_PR);
	GLogManager::logging( LOG_INFO, "CMI-7r , Texture Access   (Sec), %10d", _pf_nTexRef_RR);

	//-------------------------------------------------------------------------------------
	fprintf(fp, "PRI-1  , Triangles #           , %10d\n", m_pSSESceneData->m_TriObjCnt);
	fprintf(fp, "PRI-2  , Visible Tri / Pixels  , %10.3f\n", 1.0f * nUniTri / (Resolution.x * Resolution.y));

	fRatio = _pf_nHit_SpecPnt_PR==0?0: (100.0f * _pf_nHit_SpecPnt_PR / (_pf_nHit_DiffPnt_PR + _pf_nHit_SpecPnt_PR));
	fprintf(fp, "PRI-6p , Specular Hit     (Pri), %10.2f %%\n", fRatio);
	fRatio = _pf_nHit_SpecPnt_RR==0?0: (100.0f * _pf_nHit_SpecPnt_RR / (_pf_nHit_DiffPnt_RR + _pf_nHit_SpecPnt_RR));
	fprintf(fp, "PRI-6r , Specular Hit     (Sec), %10.2f %%\n", fRatio);

	fRatio = 100.0f * (m_pSSESceneData->m_TriAreaSum_Refl + m_pSSESceneData->m_TriAreaSum_Refr) / m_pSSESceneData->m_TriAreaSum_All;
	fprintf(fp, "PRI-7  , Specular Area         , %10.2f %%\n", fRatio);
	fRatio = 100.0f * m_pSSESceneData->m_TriAreaSum_Tex / m_pSSESceneData->m_TriAreaSum_All;
	fprintf(fp, "PRI-8  , Textured Area         , %10.2f %%\n", fRatio);

	fRatio = _pf_nHit_ShwPnt_ALL==0?0: (100.0f * _pf_nHit_ShwPnt_ALL / (_pf_nHit_DiffPnt_PR + _pf_nHit_SpecPnt_PR));
	fprintf(fp, "PRI-9  , Shadowed Hit (Pri)    , %10.2f %%\n", fRatio);

	fprintf(fp, "CMI-1p , Processed Ray    (Pri), %10d\n", _pf_nGen_RaysCnt_PR);
	fprintf(fp, "CMI-1r , Processed Ray    (Sec), %10d\n", _pf_nGen_RaysCnt_RR);
	fprintf(fp, "CMI-1s , Processed Ray    (Shw), %10d\n", _pf_nGen_RaysCnt_SR);
	fprintf(fp, "CMI-6ap, Shading Op Point (Pri), %10d\n", _pf_nHit_ShadPnt_PR);
	fprintf(fp, "CMI-6ar, Shading Op Point (Sec), %10d\n", _pf_nHit_ShadPnt_RR);
	fprintf(fp, "CMI-6bp, Shading Op Calc. (Pri), %10d\n", _pf_nHit_ShadCnt_PR);
	fprintf(fp, "CMI-6br, Shading Op Calc. (Sec), %10d\n", _pf_nHit_ShadCnt_RR);
	fprintf(fp, "CMI-7p , Texture Access   (Pri), %10d\n", _pf_nTexRef_PR);
	fprintf(fp, "CMI-7r , Texture Access   (Sec), %10d\n", _pf_nTexRef_RR);

	fclose( fp );

	// Frustum culling 
#if !GOThread
	unsigned int PriIsect_FtnCall_Count		= m_pSSERenderPipeline->PriIsect_FtnCall_Count;
	unsigned int PriIsect_FC_Cull_Count		= m_pSSERenderPipeline->PriIsect_FC_Cull_Count;
	unsigned int PriIsect_TriChk_Count		= m_pSSERenderPipeline->PriIsect_TriChk_Count;
	GLogManager::logging( LOG_INFO, "Pri Isect function call (NonEmpty Node) :  %d", PriIsect_FtnCall_Count);
	GLogManager::logging( LOG_INFO, "Pri Isect Frustum Cull / TriTest        :  %d / %d", PriIsect_FC_Cull_Count, PriIsect_TriChk_Count);


 //min 갯수 : 4
 //-> KDTree Node Count: 163151 (1.244743MB)
 //-> n_leafNode: 81576
 //-> treeLevel: 57
 //-> maxLeafSize: 33
 //-> n_emptyLeaf: 18226 (22.342356 %%%)

	//// time			0.114	0.117
	//// Call			72907
	//// FCull/Test	66,605 / 166,747

 //min 갯수 : 14
 //-> KDTree Node Count: 27867 (0.212608MB)
 //-> n_leafNode: 13934
 //-> treeLevel: 44
 //-> maxLeafSize: 33
 //-> n_emptyLeaf: 3649 (26.187742 %%%)

	//// time			0.131	0.144
	//// Call			62139
	//// FCull/Test	211,082 / 340,249

 //min 갯수 : 24
 //-> KDTree Node Count: 13123 (0.100121MB)
 //-> n_leafNode: 6562
 //-> treeLevel: 36
 //-> maxLeafSize: 33
 //-> n_emptyLeaf: 1729 (26.348674 %%%)

 //	// time			0.175	0.209
	//// Call			64547
	//// FCull/Test	509,584 / 652,093

 //min 갯수 : 40
 //-> KDTree Node Count: 6819 (0.052025MB)
 //-> n_leafNode: 3410
 //-> treeLevel: 31
 //-> maxLeafSize: 40
 //-> n_emptyLeaf: 939 (27.536657 %%%)

	//// time			0.247	0.298
	//// Call			66583
	//// FCull/Test	972,719 / 1,127,276


#endif
}
