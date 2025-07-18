#include "GDistributedPhotonRayTracer.h"
#include "GSceneManager.h"
#include "mpi.h"
#include "GRenderSystem.h"
#include "GPhotonMapRayTracer.h"
#include "GGPURayTracer.h"

/**
 *	MPICH2 를 사용한 분산렌더링.
 *	PHOTON MAPPING 과 RayTracing 을 가지고 수행한다.
 *
 *	by graphicsian.
 */
GDistributedPhotonRayTracer::GDistributedPhotonRayTracer( 
										GRenderSystem *pDistManager,
										GPhotonMappingOption *pOption )
{
	m_pDistRenderManager = pDistManager;

	/**
	 *	render node 개수만큼 나누어서 뿌릴 것이므로 emitPhoton 을 
	 *	나눈다. 그리고 다시 몇번의 iteration 을 할지 결정.
	 *	power 도 나누어져야 한다.
	 */
	GPhotonMappingOption newOption = (*pOption);

	/**
	 *	각 node 가 처리할 photon 수치를 결정한다.
	 *	반복할 iteration 을 각 node 가 나우어서 처리. 최소한 한 node 가 한번의
	 *	iteration 은 돌아야 한다( 원하는 photon 개수보다 많아지더라도 )
	 */
	newOption.m_iIteration = max( 1, (int) ( newOption.m_iIteration / (float) pDistManager->getProcessCount() ) );
	newOption.m_fTotalSceneLightPower = pOption->m_fTotalSceneLightPower / (float) pDistManager->getProcessCount();

	m_pPhotonMapRayTracer = new GPhotonMapRayTracer( &newOption );
}

GDistributedPhotonRayTracer::~GDistributedPhotonRayTracer(void)
{
	if ( m_pPhotonMapRayTracer )
		delete m_pPhotonMapRayTracer;
}


/**
 *	모든 Node 가 이 부분은 동시에 진행되게 해야 한다.
 *	root 가 MPI_Bcast 로 rendering 시작을 알리고, 
 *	이때 인자로 Scene 에서 변경된 사항을 가지고 Bcast 를 수행.
 *
 *	서로 network 동기화를 잘 맞추어서 deadlock 이 안생기게 주의할것.
 */
GError GDistributedPhotonRayTracer::rendering( GScene* pScene, bool isDebug )
{
	GError error = errorNo;
	int mpiError;
	GDistributedRenderConfig renderConfig;

	double starttime = MPI_Wtime();
	double endtime = 0.0f;

	float *piPointPower = NULL;
	float *piPointPowerSum = NULL;
	int iPointCount = 0;

	/**
	 *	현재 root 가 rendering config 를 구성해서 node 에 bcast 해서
	 *	렌더링이 수행되게 한다.
	 */
	if ( m_pDistRenderManager->getProcessID() == JEDI_MPI_ROOT ) {
		renderConfig = makeRenderConfig( pScene );
	}

	mpiError = MPI_Bcast( &renderConfig, sizeof( renderConfig ), MPI_BYTE, JEDI_MPI_ROOT, MPI_COMM_WORLD );
	error = m_pDistRenderManager->mpiErrorCheck( mpiError, "rendering MPI_Bcast" );
	if ( error != errorNo )
		goto finalize;

	GLogManager::logging( LOG_DEBUG, "------- Render Config -------------------------------" );
	GLogManager::logging( LOG_DEBUG, " -> Resolution : %d x %d", renderConfig.resolutionX, renderConfig.resolutionY );
	GLogManager::logging( LOG_DEBUG, " -> Sampling   : %d x %d", renderConfig.superSamplingX, renderConfig.superSamplingY );
	GLogManager::logging( LOG_DEBUG, "-----------------------------------------------------" );

	/**----------------------------------------------------------------------------------------------
	 **	error 가 없는 node 들은 root 로 부터 넘겨받은 현재 카메라 정보 및 
	 **	Scene 에서 변한 정보 세팅후 각각 rendering 수행.
	 **--------------------------------------------------------------------------------------------*/

	if ( error == errorNo ) {

		setRenderConfig( renderConfig, pScene );
		error = m_pPhotonMapRayTracer->rendering( pScene, false );

	}
GLogManager::logging( LOG_INFO, "rendering photon");
	/**----------------------------------------------------------------------------------------------
	 **	MPI_AllGather 함수를 통해서 모든 node 들이 전체 rendering 결과를 체크하게 한다.
	 **	error 발생시 종료.
	 **---------------------------------------------------------------------------------------------*/
	GError *pErrorList = (GError*) malloc( sizeof( GError ) * m_pDistRenderManager->getProcessCount() );
	int errorData = error;
	mpiError = MPI_Allgather( &error, 1, MPI_INT, pErrorList, 1, MPI_INT, MPI_COMM_WORLD );

	if ( ( error = m_pDistRenderManager->mpiErrorCheck( mpiError, "rendering result MPI_AllGather" ) ) != errorNo ) {
		free( pErrorList );
		goto finalize;
	}

	for ( int i = 0; i < m_pDistRenderManager->getProcessCount(); ++i ) {
		if ( pErrorList[ i ] != errorNo ) {
			if ( m_pDistRenderManager->isMasterProcess() ) {
				GLogManager::logging( LOG_FATAL, "[ProcessID:%d] failed to rendering. [%s]", i,
					GErrorManager::getGErrorString( pErrorList[ i ] ) );
			}
			error = errorRendering;
		}
	}

	free( pErrorList );
	if ( error != errorNo )
		goto finalize;

	/**--------------------------------------------------------------------------------------------
	 ** node 들은 master 에게 자신이 구한 ipoint 들의 power 를 보내서 합산하게 만든다.
	 **------------------------------------------------------------------------------------------*/

	/**
	 *	ipoint set 정보를 가져온다. 
	 *	power 만 뽑아서 ( r,g,b float 3 ) master 에게 전송한다.
	 */
	piPointPowerSum = NULL;
	GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox = 
										m_pPhotonMapRayTracer->getIPointGridBox();
	iPointCount = pIPointGridBox->m_iTotalCount;
	piPointPower = (float*) malloc( sizeof( float ) * iPointCount * 3 );

	for ( int i = 0; i < iPointCount; ++i ) {
		piPointPower[ i * 3 + 0 ] = pIPointGridBox->m_pData2[ i ].power[0];
		piPointPower[ i * 3 + 1 ] = pIPointGridBox->m_pData2[ i ].power[1];
		piPointPower[ i * 3 + 2 ] = pIPointGridBox->m_pData2[ i ].power[2];
	}
		
	/**
	 *	master 에게 보내서 이 값들을 합산하게 만든다. piPointPowerSum 메모리는 root 만할당하면 됨.
	 */
	if ( m_pDistRenderManager->isMasterProcess() ) {
		piPointPowerSum = (float*) malloc( sizeof( float ) * iPointCount * 3 );
	}

	mpiError = MPI_Reduce( piPointPower, piPointPowerSum, iPointCount * 3, 
						   MPI_FLOAT, MPI_SUM, JEDI_MPI_ROOT, MPI_COMM_WORLD );
	if ( ( error = m_pDistRenderManager->mpiErrorCheck( mpiError, "rendering MPI_Bcast" ) ) != errorNo ) {
		goto finalize;
	}

	/**-------------------------------------------------------------------------------------------
	 **	root 가 자신의 direct illumination 결과와 합산해서, 최종적으로 scene 의 image buffer 에
	 **	기록한다.
	 **------------------------------------------------------------------------------------------*/

	GImageBuffer *pImageBuffer = pScene->getImageBuffer();
	GImageBuffer *pDirectIllumImageBuffer = pScene->getDirectIllumImageBuffer();
	GImageBuffer *pIndirectImageBuffer = pScene->getIndirectIllumImageBuffer();

	/**
	 *	photon 으로 계산된 indirect 와 direct illumination 을 합산한다.
	 */
	if ( m_pDistRenderManager->isMasterProcess() ) {
	
		/** 
		 * indirect 는 모든 node 의 ipoint power 를 합산한 걸로 계산한다.
		 */
		pIndirectImageBuffer->clear();

		int width = pScene->getResolution().x;
		int height = pScene->getResolution().y;
		int samplex = pScene->getSuperSampling().x;
		int sampley = pScene->getSuperSampling().y;
		int x, y;
		GColor color;

		for ( int i = 0; i < pIPointGridBox->m_iTotalCount; ++i ) {

			GGPURayTracer::toImageIndex( pScene, pIPointGridBox->m_pData[ i ].rayIndex, &x, &y );
			cuIntersectionPoint* pIPoint = ( pIPointGridBox->m_pData + i );

			/** power 는 모든 node 의 ipoint power 를 합산한 걸로 계산 */
			color.r = pIPoint->colorWeight.x * piPointPowerSum[ i * 3 + 0 ];
			color.g = pIPoint->colorWeight.y * piPointPowerSum[ i * 3 + 1 ];
			color.b = pIPoint->colorWeight.z * piPointPowerSum[ i * 3 + 2 ];
			color.a = 1.0f;

			pIndirectImageBuffer->addColor( x, y, color );

		}

		pImageBuffer->clear();
		pImageBuffer->add( pDirectIllumImageBuffer );
		pImageBuffer->add( pIndirectImageBuffer );

		endtime = MPI_Wtime();

		GLogManager::logging( LOG_INFO, "-----------------------------------------------------" );
		GLogManager::logging( LOG_INFO, " -> Time : %f sec.", ( endtime - starttime ) );
		GLogManager::logging( LOG_INFO, "-----------------------------------------------------" );
	}

	error = errorNo;

finalize:

	if ( m_pDistRenderManager->isMasterProcess() && error != errorNo ) {

		GLogManager::logging( LOG_INFO, "-----------------------------------------------------" );
		GLogManager::logging( LOG_INFO, " -> Rendering Error ( %s )", GErrorManager::getGErrorString( error ) );
		GLogManager::logging( LOG_INFO, "-----------------------------------------------------" );

	}

	/**-------------------------------------------------------------------------------------------
	 **	사용된 임시 메모리들 해제.
	 **------------------------------------------------------------------------------------------*/

	if ( piPointPower )
		free( piPointPower );
	if ( piPointPowerSum )
		free( piPointPowerSum );

	return error;
}

void GDistributedPhotonRayTracer::setRenderConfig( GDistributedRenderConfig &config,
												   GScene *pScene )
{
	(*pScene->getRenderCamera()) = config.camera;
	pScene->setResolution( config.resolutionX, config.resolutionY );
	pScene->setSuperSampling( config.superSamplingX, config.superSamplingY );

	m_pPhotonMapRayTracer->enableShadow( config.bEnableShadow );
}

GDistributedRenderConfig GDistributedPhotonRayTracer::makeRenderConfig( GScene *pScene )
{
	GDistributedRenderConfig config;

	config.camera = (*pScene->getRenderCamera());
	config.resolutionX = pScene->getResolution().x;
	config.resolutionY = pScene->getResolution().y;
	config.superSamplingX = pScene->getSuperSampling().x;
	config.superSamplingY = pScene->getSuperSampling().y;
	config.bEnableShadow = isEnableShadow();

	return config;
}
