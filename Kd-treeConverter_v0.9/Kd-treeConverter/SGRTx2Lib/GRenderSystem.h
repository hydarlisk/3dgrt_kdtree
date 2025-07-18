#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GScene.h"
#include "GRenderOption.h"

#define JEDI_MPI_ROOT					0			//	루트로 사용할 process id
#define JEDI_MPI_TAG					1

#define JEDI_TERMINATION_MESSAGE		1
#define JEDI_SCENE_LOAD_MESSAGE			2
#define JEDI_SCENE_RENDERING_MESSAGE	3

typedef enum {
	rendererOpenGL,
	rendererViewerModeRenderer,
	rendererGPURayTracing,
	rendererPhotonRayTracing,
	rendererDistributedPhotonRayTracing,
	rendererSSERayTracing,
	rendererGPUExperimentalRayTracing,
	rendererGPURayTracingForPaper,
	rendererSSERayTracingForPaper,
	rendererBVHRayTracing,
	rendererGRIDRayTracing,
} rendererType;

/**
 *	MPICH2 를 사용해서 root 이외의 각 node 들의 message 를
 *	처리할 manager 클래스. 각 node 는 프로그램이 수행되면
 *	이 클래스가 프로그램 동작을 message 를 가지고 핸들링하게
 *	된다. root 의 경우는 이 클래스를 빠져나가서 GUI 안에서
 *	행동이 수행된다.
 *
 *	graphicsian.
 */
class GRenderSystem
{
private:
	// singleton.
	static GRenderSystem *g_RenderSystem;

	GRenderer *m_pRenderer;
	GScene *m_pScene;
	bool m_bDebugMode;
	bool m_bEnableShadow;

	int m_iProcessID;
	int m_iProcessCount;
	
public:
	~GRenderSystem(void);

protected:
	GRenderSystem(void);

public:
	static GError initialize( int argc, char ***argv );
	static GRenderSystem* getInstance();
	static void uninitialize();

	int getProcessID();
	int getProcessCount();
	bool isMasterProcess();
	bool isSlaveProcess();

	void setDebugMode( bool flag );
	bool isDebugMode();
	void enableShadow( bool flag );
	bool isEnableShadow();
	bool isOpenGLPipeline() { return m_pRenderer && m_pRenderer->isOpenGLPipeline(); }

	GRenderer* getRenderer() { return m_pRenderer; }

	GScene* getScene();

	GError createRenderer( rendererType type, const GRenderOption *option );
	GError loadScene( const char *fileName );
	GError rendering( GCamera *pCamera );
	GError rendering();
	GError slaveRenderNode();

	/** 
	 *	root 를 제외한 모든 node 에게 명령을 보냄. 
	 */
	void sendRootMessageToNode( int message );
	GError mpiErrorCheck( int mpiError, const char* title );

protected:
	GError standaloneRendering( GCamera *pCamera );
	GError distributedRendering( GCamera *pCamera );

};
