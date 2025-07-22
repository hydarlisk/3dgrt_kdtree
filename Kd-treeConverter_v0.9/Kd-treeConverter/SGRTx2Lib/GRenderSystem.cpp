#include "GRenderSystem.h"
#include "GGPURayTracer.h"
#include "GPhotonMapRayTracer.h"
#include "GDistributedPhotonRayTracer.h"
#include "GSSERayTracer.h"
#include "GSSERayTracerForPaper.h"
#include "GSSEBVHRayTracer.h"
#include "GSSEGRIDRayTracer.h"

#include "GSceneManager.h"
#include "GGPUExperimentalRayTracer.h"
#include "GGPURayTracerForPaper.h"
#include "GViewerModeRenderer.h"
#include "GOpenGLPipelineRenderer.h"

#include "GThreadManager.h"
#include "GThreadingOption.h"
#include "FreeImage.h"

#include "mpi.h"

GRenderSystem* GRenderSystem::g_RenderSystem = NULL;

GRenderSystem::GRenderSystem(void)
{
	m_pRenderer = NULL;
	m_pScene = new GScene();
	m_bDebugMode = false;
	m_bEnableShadow = false;
}

GRenderSystem::~GRenderSystem(void)
{
	if ( m_pRenderer ) {
		delete m_pRenderer;
		m_pRenderer = NULL;
	}

	if ( m_pScene ) {
		delete m_pScene;
		m_pScene = NULL;
	}

	MPI_Finalize();
}

/**
 *	System 초기화
 */
GError GRenderSystem::initialize( int argc, char ***argv )
{
	/** image 처리를 위한 freeimage library 초기화 */
	FreeImage_Initialise();

	GError error;
	
	g_RenderSystem = new GRenderSystem();

	int mpiError = MPI_Init( &argc, argv );
	if ( ( error = g_RenderSystem->mpiErrorCheck( mpiError, "initDistributedSystem" ) ) != errorNo )
		return error;
	
	MPI_Comm_rank( MPI_COMM_WORLD, &g_RenderSystem->m_iProcessID );
	MPI_Comm_size( MPI_COMM_WORLD, &g_RenderSystem->m_iProcessCount );

	// logger 에 process id 가 나오게 세팅해둠.
	GLogManager::setProcessID( g_RenderSystem->m_iProcessID );

	if ( g_RenderSystem->m_iProcessID == JEDI_MPI_ROOT ) {
		GLogManager::logging( LOG_INFO, "--------------- Jedi Distributed Render ----------------" );
		GLogManager::logging( LOG_INFO, " -> Current Process Count : %d ", g_RenderSystem->m_iProcessCount );
		GLogManager::logging( LOG_INFO, "--------------------------------------------------------" );
	}

	//------------------------------------------------------------------
	//	쓰레드 기능 초기화
	//------------------------------------------------------------------
	if ( GThreadManager::getMaxThreadCount() == 0) {
		if ( !GThreadManager::initThreadManager( MAX_THREADING ) ) {
			GLogManager::logging( LOG_ERROR, "Thread manager init fail." );
			return errorThreadError;
		}
	}

	return errorNo;
}

void GRenderSystem::uninitialize()
{
	/** image 처리를 위한 freeimage library 해제 */
	FreeImage_DeInitialise();

	//------------------------------------------------------------------
	//	쓰레드 기능 해제
	//------------------------------------------------------------------
	GThreadManager::uninitThreadManager();

	/** 모든 node 에게 종료하라는 메시지를 보낸다. */
	if ( g_RenderSystem->m_iProcessID == JEDI_MPI_ROOT ) {
		g_RenderSystem->sendRootMessageToNode( JEDI_TERMINATION_MESSAGE );
	}
	delete g_RenderSystem;
}

GRenderSystem *GRenderSystem::getInstance()
{
	return g_RenderSystem;
}

void GRenderSystem::setDebugMode( bool flag )
{
	m_bDebugMode = flag;
}

bool GRenderSystem::isDebugMode()
{
	return m_bDebugMode;
}

void GRenderSystem::enableShadow( bool flag )
{
	m_bEnableShadow = flag;
}

bool GRenderSystem::isEnableShadow()
{
	return m_bEnableShadow;
}

int GRenderSystem::getProcessID()
{
	return m_iProcessID;
}

int GRenderSystem::getProcessCount()
{
	return m_iProcessCount;
}
	
bool GRenderSystem::isMasterProcess()
{
	return ( m_iProcessID == JEDI_MPI_ROOT );
}

bool GRenderSystem::isSlaveProcess()
{
	return ( m_iProcessID != JEDI_MPI_ROOT );
}

GError GRenderSystem::createRenderer( rendererType type, const GRenderOption *option )
{
	/** 이전 renderer 가 있다면 제거한다. */
	if ( m_pRenderer )
		delete m_pRenderer;
	m_pRenderer = NULL;

	switch( type ) {
		case rendererOpenGL:
			//m_pRenderer = NULL;
			m_pRenderer = new GOpenGLPipelineRenderer();
			return errorNo;

		case rendererViewerModeRenderer:
			m_pRenderer = new GViewerModeRenderer();
			return errorNo;

		case rendererGPURayTracing:
			m_pRenderer = new GGPURayTracer();
			return errorNo;

		case rendererPhotonRayTracing:
			m_pRenderer = new GPhotonMapRayTracer( (GPhotonMappingOption*) option );
			return errorNo;

		case rendererDistributedPhotonRayTracing:
			m_pRenderer = new GDistributedPhotonRayTracer( this, (GPhotonMappingOption*) option );
			return errorNo;

		case rendererSSERayTracing:
			m_pRenderer = new GSSERayTracer();
			return errorNo;

		case rendererSSERayTracingForPaper:
			m_pRenderer = new GSSERayTracerForPaper();
			return errorNo;

		case rendererGPUExperimentalRayTracing:
			m_pRenderer = new GGPUExperimentalRayTracer();
			return errorNo;

		case rendererGPURayTracingForPaper:
			m_pRenderer = new GGPURayTracerForPaper();
			return errorNo;

		case rendererBVHRayTracing:
			m_pRenderer = new GSSEBVHRayTracer();
			return errorNo;
		
		case rendererGRIDRayTracing:
			m_pRenderer = new GSSEGRIDRayTracer();
			return errorNo;
	}

	return errorUnsupportedRenderer;

}

/**
 *	모든 MPI NODE 에게 Scene 을 로드하게 명령을 내린다. ( master 도 포함 )
 */
GError GRenderSystem::loadScene( const char *fileName )
{
	GError error = errorNo;
	int mpiError;
	char szFileName[ 512 ] = { 0x00, };

	/**
	 *	master 를 제외한 나머지 node 는 messageDispatch() 함수안에서
	 *	메시지가 오기를 기다리고 있으므로 master 가 scene 을 로드하라는
	 *	메시지를 보내서 이 함수 안으로 들어오게 한다.
	 */
	if ( m_iProcessID == JEDI_MPI_ROOT ) {
		sendRootMessageToNode( JEDI_SCENE_LOAD_MESSAGE );
	}

	/** 
	 *	모든 node 가 여기로 들어올때까지 대기 
	 */
	MPI_Barrier( MPI_COMM_WORLD );

	/**
	 *	이전 scene 있다면 제거한다.
	 */
	if ( m_pScene )
		delete m_pScene;
	m_pScene = NULL;

	/**
	 *	현재 root 가 각 node 에게 scene 을 로드하라는 명령을 내린다. file 이름을 전송.
	 *	root 는 loadScene 함수로 들어오는 fileName 인자를 구성해서 각 node 에게 알리고
	 *	각 node 는 fileName 은 무시하고 root 로부터 넘어온 파일이름을 사용해야 한다.
	 */
	if ( m_iProcessID == JEDI_MPI_ROOT ) {
		strncpy( szFileName, fileName, 510 );
		szFileName[ 510 ] = 0x00;
	}

	mpiError = MPI_Bcast( szFileName, 512, MPI_BYTE, JEDI_MPI_ROOT, MPI_COMM_WORLD );
	if ( ( error = mpiErrorCheck( mpiError, "loadScene MPI_Bcast" ) ) != errorNo )
		return error;

	/**
	 *	Scene 을 Load 한다. load 에 실패하면 빈 scene 을 생성.
	 */
	GSceneManager sceneManager;
	error = sceneManager.loadScene( szFileName, &m_pScene );
	if ( error != errorNo ) {
		if ( m_pScene ) {
			delete( m_pScene );
		}
		m_pScene = new GScene();
	}

	/**
	 *	root 가 각 node 의 결과를 취합해서 다 에러 없이 scene 을 로드했는지를 체크한다.
	 *	값에는 error 번호가 기록되어 있을 것이다.
	 */
	int *pErrorList = (int*) malloc( sizeof( int ) * m_iProcessCount );
	int errorData = error;
	mpiError = MPI_Gather( &error, 1, MPI_INT, 
						   pErrorList, 1, MPI_INT, 
						   JEDI_MPI_ROOT, MPI_COMM_WORLD );

	if ( ( error = mpiErrorCheck( mpiError, "loadScene MPI_Gather" ) ) != errorNo ) {
		free( pErrorList );
		return error;
	}

	if ( m_iProcessID == JEDI_MPI_ROOT ) {
		for ( int i = 0; i < m_iProcessCount; ++i ) {
			if ( pErrorList[ i ] != errorNo ) {
				GLogManager::logging( LOG_FATAL, "[ProcessID:%d] failed to scene load.", m_iProcessID );
				error = errorNoScene;
			}
		}
	}

	free( pErrorList );

	return error;
}

GError GRenderSystem::rendering()
{
	return rendering( NULL );
}

GError GRenderSystem::rendering( GCamera *pCamera )
{
	/** 
	 *	현재 생성된 renderer 가 distributed 환경을 지원한다면
	 *	slave node 들과 같이 수행하도록 하고, 아니라면 혼자만 수행하게
	 *	환경을 구성한다.
	 */
	if ( m_pRenderer == NULL )
		return errorNoRenderer;

	if ( m_pRenderer->isDistributed() ) {
		return distributedRendering( pCamera );
	} else {
		return standaloneRendering( pCamera );
	}
}

/**
 *	렌더링을 수행시킨다. master 가 이함수안으로 들어왔을때
 *	slave node 들은 messageDispatch()
 *	함수안에서 돌고 있으므로, master 가 RENDERING 메시지를 보내서
 *	이 함수 안으로 들어오게 한다. 
 */
GError GRenderSystem::distributedRendering( GCamera *pCamera )
{
	GError error;

	if ( m_pScene == NULL )
		return errorNoScene;

	if ( m_pRenderer == NULL )
		return errorNoRenderer;

	if ( pCamera != NULL )
		m_pScene->setRenderCamera( pCamera );

	/**
	 *	master 를 제외한 나머지 node 는 messageDispatch() 함수안에서
	 *	메시지가 오기를 기다리고 있으므로 master 가 scene 을 로드하라는
	 *	메시지를 보내서 이 함수 안으로 들어오게 하고 모든 node 가 들어올때까지
	 *	기다린다.
	 */
	if ( m_iProcessID == JEDI_MPI_ROOT ) {
		sendRootMessageToNode( JEDI_SCENE_RENDERING_MESSAGE );
	}
	MPI_Barrier( MPI_COMM_WORLD );

	/**
	 *	렌더링을 수행하기 위해 Scene 의 초기화 작업을 수행한다.
	 *	Object 의 matrix 를 모두 적용해서 모든 데이터를 World 좌표계로
	 *	변환해 둔다. 내부적으로 geometry 의 변화가 있을때만 수행될 것이다.
	 */
	if ( ( error = m_pScene->convertRenderScene() ) != errorNo ) {
		return error;
	}

	/** rendering option 설정 */
	m_pRenderer->enableShadow( m_pScene->isEnableShadow() );

	return m_pRenderer->rendering( m_pScene, m_bDebugMode );

}

/**
 *	렌더링을 수행시킨다. master 가 이함수안으로 들어왔을때
 *	slave node 들은 messageDispatch()
 *	함수안에서 돌고 있으므로, master 가 RENDERING 메시지를 보내서
 *	이 함수 안으로 들어오게 한다. 
 */
GError GRenderSystem::standaloneRendering( GCamera *pCamera )
{
	GError error;

	if ( m_pScene == NULL )
		return errorNoScene;

	if ( m_pRenderer == NULL )
		return errorNoRenderer;

	if ( pCamera != NULL )
		m_pScene->setRenderCamera( pCamera );

	/**
	 *	렌더링을 수행하기 위해 Scene 의 초기화 작업을 수행한다.
	 *	Object 의 matrix 를 모두 적용해서 모든 데이터를 World 좌표계로
	 *	변환해 둔다. 내부적으로 geometry 의 변화가 있을때만 수행될 것이다.
	 */
	if ( ( error = m_pScene->convertRenderScene() ) != errorNo ) {
		return error;
	}

	/** rendering option 설정 */
	m_pRenderer->enableShadow( m_pScene->isEnableShadow() );

	return m_pRenderer->rendering( m_pScene, m_bDebugMode );

}

/**
 *	slave render node 들의 main loop.
 *	master 로부터 message 받아서 처리.
 */
GError GRenderSystem::slaveRenderNode()
{
	int message;

	//createRenderer( rendererDistributedPhotonRayTracing );

	GLogManager::logging( LOG_INFO, "[ Jedi Distributed process %d ] Running.", m_iProcessID );
	MPI_Status stat;

	while( true ) {

		MPI_Recv( &message, 1, MPI_INT, JEDI_MPI_ROOT, JEDI_MPI_TAG, MPI_COMM_WORLD, &stat );
		GLogManager::logging( LOG_INFO, " << %d recevied", message );

		if ( message == JEDI_TERMINATION_MESSAGE )
			break;

		switch( message ) {
			case JEDI_SCENE_LOAD_MESSAGE:
				loadScene( "" );	// 파일이름은 loadScene 함수안에서 가져올 것이다.
				break;
			case JEDI_SCENE_RENDERING_MESSAGE:
				rendering( NULL );	// 함수안으로 들어가면 rendering 정보가 master 로부터 건너와 세팅될 것이다.
				break;
		}
	}

	GLogManager::logging( LOG_INFO, "[ Jedi Distributed process %d ] Terminated.", m_iProcessID );

	return errorNo;
}

/**
 *	master render node 가 나머지 모든 node 에게 message 를 보낸다.
 */
void GRenderSystem::sendRootMessageToNode( int message )
{
	if ( m_iProcessID != JEDI_MPI_ROOT ) return;

	for ( int i = 1; i < m_iProcessCount; ++i ) {
		MPI_Send( &message, 1, MPI_INT, i, JEDI_MPI_TAG, MPI_COMM_WORLD );
	}
}

inline GError GRenderSystem::mpiErrorCheck( int mpiError, const char* title )
{
	if ( mpiError != MPI_SUCCESS ) {

		char errorString[ 1024 ];
		int errorStringCount;
		MPI_Error_string( mpiError, errorString, &errorStringCount );

		if ( m_iProcessID == JEDI_MPI_ROOT )
			GLogManager::logging( LOG_FATAL, "[MPI ID:%d] %s : %s", title, errorString );

		return errorMPIError;

	}

	return errorNo;
}

GScene* GRenderSystem::getScene()
{
	return m_pScene;
}
