#include "cudaRenderCommon.cuh"
#include "GGPURayTracer.h"
#include "GSpatialStructure.h"
#include "GKDTreeStructure.h"
#include "GPointLight.h"
#include "GRenderCommon.h"
#include "math.h"

GGPURayTracer::GGPURayTracer()
{
	m_iOldSceneNumber = 0;
	m_pCudaRenderPipeline = NULL;
	m_iSceneTimestamp = -1;
	m_pScene = NULL;
}

GGPURayTracer::~GGPURayTracer(void)
{
	uninitialize();
}

void GGPURayTracer::uninitialize()
{
	if ( m_pCudaRenderPipeline ) {
		delete m_pCudaRenderPipeline;
	}
}

/**
 *	scene 정보를 재구성해야할때 초기화한다.
 */
GError GGPURayTracer::initialize( GScene *pScene )
{
	GError error;
	m_pScene = pScene;

	if ( m_iOldSceneNumber != pScene->getSceneNumber() || 
		 m_iSceneTimestamp != pScene->getGeometryChangeTimestamp() ||
		 m_oldResolution != pScene->getResolution() ) {

		if ( m_pCudaRenderPipeline ) {
			delete m_pCudaRenderPipeline;
		}

		cuScene cuSceneInfo;
		cuSceneInfo.globalAmbient = make_float3( pScene->getGlobalAmbient().r, 
												 pScene->getGlobalAmbient().g,
												 pScene->getGlobalAmbient().b );
		cuSceneInfo.iResolutionX = pScene->getResolution().x;
		cuSceneInfo.iResolutionY = pScene->getResolution().y;
		cuSceneInfo.iSuperSamplingX = pScene->getSuperSampling().x;
		cuSceneInfo.iSuperSamplingY = pScene->getSuperSampling().y;
		cuSceneInfo.iBlockSizeX = pScene->getGPUBlockSize().x;
		cuSceneInfo.iBlockSizeY = pScene->getGPUBlockSize().y;
		cuSceneInfo.bEnableShadow = pScene->isEnableShadow();
		cuSceneInfo.bEnableLocalShading = pScene->isEnableLocalShading();
		cuSceneInfo.iShadowRay = 1;

		m_pCudaRenderPipeline = new cudaRenderPipeline();
		int maxRay = cuSceneInfo.iResolutionX * cuSceneInfo.iResolutionY;

		error = m_pCudaRenderPipeline->initialize( cuSceneInfo, maxRay );
		if ( error != errorNo )
			return error; 

		/**
		 *	Light 정보 세팅. 세팅후 필요없으므로 삭제.
		 */
		int lightCount = 0;
		cuLight* pLight = GRenderCommon::makeCudaLightInfo( pScene, &lightCount, m_pCudaRenderPipeline );
		error = m_pCudaRenderPipeline->setLightInfo( pLight, lightCount );
		free( pLight );

		if ( error != errorNo )
			return error;

		/** 
		 *	좀 어색하지만, kdtree 정보를 이렇게 해서 cuda 로 넘김. 
		 */ 
		error = pScene->getKDTreeStructure()->makeCudaRenderStructureInfo( m_pCudaRenderPipeline );
		if ( error != errorNo )
			return error;

		/** blooming 효과를 적용하기를 원한다면 초기화 해둠.*/
		error = m_pCudaRenderPipeline->initBloomingFilter( m_pScene->getBloomingRadius(),
														   m_pScene->getBloomingWeight() );
		if ( error != errorNo )
			return error;

		m_pCudaRenderPipeline->printStatusInfo();
	}

	/**
	 *	현재 Renderer 가 처리한 Scene 을 기억한다.
	 */
	m_iOldSceneNumber = pScene->getSceneNumber();
	m_iSceneTimestamp = pScene->getGeometryChangeTimestamp();
	m_oldResolution = pScene->getResolution();

	return errorNo;
}
//
//GError GGPURayTracer::rendering( GScene *pScene, bool isDebug )
//{
//	GError error;
//
//	/** 
//	 *	Scene 이 이전 geometry 상태에서 변한게 있는지 체크해서 있다면
//	 *	SpatialStructure 를 재구성한다.
//	 */
//	error = initialize( pScene );
//	if ( error != errorNo ) {
//		GLogManager::logging( LOG_ERROR, "can't initialize rendering\n" );
//		return error;
//	}
//
//	/** rendering option 세팅 */
//	m_pCudaRenderPipeline->renderingOption( m_bEnableShadow, pScene->isUseTexture() );
//
//	GImageBuffer *pImageBuffer = pScene->getImageBuffer();
//
//	int depth = 0;
//	int m_iMaxDepth = pScene->getMaxReflectionDepth();
//	int atLeastOneRay = 0;
//
//	GTimer timer1;
//	timer1.start();
//
//	/**
//	 *	cuda 안의 intersection point 를 디폴트 값으로 초기화 한다.
//	 */
//	error = m_pCudaRenderPipeline->clearIntersectionResult();
//	if ( error != errorNo ) {
//		GLogManager::logging( LOG_ERROR, "can't clearIntersectionResult\n" );
//		return errorCudaError;
//	}
//
//	m_pCudaRenderPipeline->clearFrameBuffer();
//
//	GTimer timer2;
//	timer2.start();
//
//	int generatedRayCount = 0;
//	error = makePrimaryRaySet( pScene, &generatedRayCount );
//	if ( error != errorNo ) {
//		GLogManager::logging( LOG_ERROR, "can't makePrimaryRaySet\n" );
//		return error;
//	}
//
//	/**
//	 *	reflection 이나 refraction 이 있으면 최대 max depth 까지
//	 *	추적한다.
//	 */
//	do {
//		error = m_pCudaRenderPipeline->doRayCasting( 0, generatedRayCount, 
//										( pScene->getFrontFace() == faceCCW ), pScene->isBackFaceCulling() );
//		if ( error != errorNo )	break;
//
//		error = m_pCudaRenderPipeline->calDirectIllumination( m_iMaxDepth );
//		if ( error != errorNo )	break;
//
//		if ( depth >= m_iMaxDepth ) break;
//
//		error = m_pCudaRenderPipeline->generateReflectionRay( 0, generatedRayCount, &atLeastOneRay );
//		if ( error != errorNo || atLeastOneRay == 0 ) break;
//
//		depth++;
//	} while( true );
//
//	if ( error != errorNo ) {
//		GLogManager::logging( LOG_FATAL, "Rendering Error, %s\n", GErrorManager::getGErrorString( error ) );
//		return errorNo;
//	}
//
//	/** blooming 효과를 적용하기를 원한다면 */
//	GTimer bloomingtimer;
//	bloomingtimer.start();
//
//	if ( m_pScene->isBloomingFilter() )
//		m_pCudaRenderPipeline->bloomingFiltering();
//
//	bloomingtimer.end();
//	GLogManager::logging( LOG_DEBUG, "Blooming time : %f sec\n", bloomingtimer.getElapsedTime() );
//
//	timer2.end();
//
//	m_pCudaRenderPipeline->getFrameBuffer( pImageBuffer->getBuffer() );
//
//	timer1.end();
//
//	m_pScene->setFPS( 1.0f / timer1.getElapsedTime() );
//	m_pScene->setFPS2( 1.0f / timer2.getElapsedTime() );
//
//	GLogManager::logging( LOG_DEBUG, "traceRayFullBounce time : %f sec. traced depth = %d\n", timer1.getElapsedTime(), depth );
//
//	return errorNo;
//
//}

GError GGPURayTracer::rendering( GScene *pScene, bool isDebug )
{
	GError error;

	/** 
	 *	Scene 이 이전 geometry 상태에서 변한게 있는지 체크해서 있다면
	 *	SpatialStructure 를 재구성한다.
	 */
	error = initialize( pScene );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't initialize rendering\n" );
		return error;
	}

	/** rendering option 세팅 */
	cuScene cuSceneInfo;
	cuSceneInfo.globalAmbient = make_float3( pScene->getGlobalAmbient().r, 
											 pScene->getGlobalAmbient().g,
											 pScene->getGlobalAmbient().b );
	cuSceneInfo.iResolutionX = pScene->getResolution().x;
	cuSceneInfo.iResolutionY = pScene->getResolution().y;
	cuSceneInfo.iSuperSamplingX = pScene->getSuperSampling().x;
	cuSceneInfo.iSuperSamplingY = pScene->getSuperSampling().y;
	cuSceneInfo.iBlockSizeX = pScene->getGPUBlockSize().x;
	cuSceneInfo.iBlockSizeY = pScene->getGPUBlockSize().y;
	cuSceneInfo.bEnableTexture = pScene->isUseTexture();
	cuSceneInfo.bEnableLocalShading = pScene->isEnableLocalShading();
	cuSceneInfo.bEnableShadow = pScene->isEnableShadow();
	cuSceneInfo.iShadowRay = 1;

	m_pCudaRenderPipeline->setSceneInfo( cuSceneInfo );

	GImageBuffer *pImageBuffer = pScene->getImageBuffer();

	int depth = 0;
	int m_iMaxDepth = pScene->getMaxReflectionDepth();
	int atLeastOneRay = 0;

	/**
	 *	cuda 안의 intersection point 를 디폴트 값으로 초기화 한다.
	 */
	error = m_pCudaRenderPipeline->clearIntersectionResult();
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't clearIntersectionResult\n" );
		return errorCudaError;
	}

	m_pCudaRenderPipeline->clearFrameBuffer();

	GTimer timer1;
	timer1.start();

	GTimer timer2;
	timer2.start();

	int generatedRayCount = 0;

	for ( int i = 0; i < pScene->getSuperSampling().x; ++i ) {
		for ( int j = 0; j < pScene->getSuperSampling().y; ++j ) {

			error = makePrimaryRaySet_BlockGrouping( pScene, &generatedRayCount, i, j );
			if ( error != errorNo ) {
				GLogManager::logging( LOG_ERROR, "can't makePrimaryRaySet\n" );
				return error;
			}

			/**
			 *	reflection 이나 refraction 이 있으면 최대 max depth 까지
			 *	추적한다.
			 */
			depth = 0;
			atLeastOneRay = 0;

			do {
				error = m_pCudaRenderPipeline->doRayCasting( 0, generatedRayCount, 
												( pScene->getFrontFace() == faceCCW ), pScene->isBackFaceCulling() );
				if ( error != errorNo )	break;

				error = m_pCudaRenderPipeline->calDirectIllumination( m_iMaxDepth );
				if ( error != errorNo )	break;

				if ( depth >= m_iMaxDepth ) break;

				error = m_pCudaRenderPipeline->generateReflectionRay( 0, generatedRayCount, &atLeastOneRay );
				if ( error != errorNo || atLeastOneRay == 0 ) break;

				depth++;
			} while( true );

		}
	}

	if ( error != errorNo ) {
		GLogManager::logging( LOG_FATAL, "Rendering Error, %s\n", GErrorManager::getGErrorString( error ) );
		return errorNo;
	}

	/** blooming 효과를 적용하기를 원한다면 */
	GTimer bloomingtimer;
	bloomingtimer.start();

	if ( m_pScene->isBloomingFilter() )
		m_pCudaRenderPipeline->bloomingFiltering();

	if ( m_pScene->isUseAntialiasingFilter() )
		m_pCudaRenderPipeline->antialiasingFiltering();

	if ( m_pScene->isUseBlurFilter() )
		m_pCudaRenderPipeline->blurringFiltering();

	if ( m_pScene->isUseEdgeDetectionFilter() )
		m_pCudaRenderPipeline->sobelMethodFiltering();

	if ( m_pScene->isUseGrayScaleFilter() )
		m_pCudaRenderPipeline->grayScaleFiltering();

	timer1.end();

	bloomingtimer.end();
	GLogManager::logging( LOG_DEBUG, "Blooming time : %f sec\n", bloomingtimer.getElapsedTime() );

	timer2.end();

	m_pCudaRenderPipeline->getFrameBuffer( pImageBuffer->getBuffer() );

	m_pScene->setFPS( 1.0f / timer1.getElapsedTime() );
	m_pScene->setFPS2( 1.0f / timer2.getElapsedTime() );

	GLogManager::logging( LOG_DEBUG, "traceRayFullBounce time : %f sec. traced depth = %d\n", timer1.getElapsedTime(), depth );

	return errorNo;

}

cuCamera GGPURayTracer::calCameraInfo( GScene *pScene )
{
	GDimension resolution = pScene->getResolution();
	GDimension samping = pScene->getSuperSampling();
	GCamera* pCamera = pScene->getRenderCamera();
	float aspect = (float) resolution.x / (float) resolution.y;
	float cameraPlaneHeight = 2.0f * pCamera->getNear() * tanf( ( pCamera->getFovy() * 0.5f ) * G_TO_RADIAN );
	float cameraPlaneWidth = aspect * cameraPlaneHeight;

	cuCamera camera;

	camera.eye = make_float3( pCamera->getEye().x, pCamera->getEye().y, pCamera->getEye().z );
	camera.u = make_float3( pCamera->getUVec().x, pCamera->getUVec().y, pCamera->getUVec().z );
	camera.v = make_float3( pCamera->getVVec().x, pCamera->getVVec().y, pCamera->getVVec().z );
	camera.n = make_float3( pCamera->getNVec().x, pCamera->getNVec().y, pCamera->getNVec().z );
	camera.fnear = pCamera->getNear();

	GVector startPoint = pCamera->getEye() - ( pCamera->getNVec() * camera.fnear ) +
						( pCamera->getVVec() * ( cameraPlaneHeight * 0.5f ) ) -
						( pCamera->getUVec() * ( cameraPlaneWidth * 0.5f ) );
	camera.startPoint = make_float3( startPoint.x, startPoint.y, startPoint.z );

	camera.stepX = cameraPlaneWidth / (float)resolution.x;
	camera.stepY = cameraPlaneHeight / (float)resolution.y;

	return camera;
}

/**
 *	CUDA 에서 Primary ray 를 생성시킨다.
 */
GError GGPURayTracer::makePrimaryRaySet( GScene *pScene, int *generatedCount, int currentSampleX, int currentSampleY )
{
	GDimension resolution = pScene->getResolution();
	GDimension SuperSampling = pScene->getSuperSampling();
	GCamera* pCamera = pScene->getRenderCamera();
	float aspect = (float) resolution.x / (float) resolution.y;
	float cameraPlaneHeight = 2.0f * pCamera->getNear() * tanf( ( pCamera->getFovy() * 0.5f ) * G_TO_RADIAN );
	float cameraPlaneWidth = aspect * cameraPlaneHeight;

	cuCamera camera;

	camera.eye = make_float3( pCamera->getEye().x, pCamera->getEye().y, pCamera->getEye().z );
	camera.u = make_float3( pCamera->getUVec().x, pCamera->getUVec().y, pCamera->getUVec().z );
	camera.v = make_float3( pCamera->getVVec().x, pCamera->getVVec().y, pCamera->getVVec().z );
	camera.n = make_float3( pCamera->getNVec().x, pCamera->getNVec().y, pCamera->getNVec().z );
	camera.fnear = pCamera->getNear();

	GVector startPoint = pCamera->getEye() - ( pCamera->getNVec() * camera.fnear ) +
						( pCamera->getVVec() * ( cameraPlaneHeight * 0.5f ) ) -
						( pCamera->getUVec() * ( cameraPlaneWidth * 0.5f ) );
	camera.startPoint = make_float3( startPoint.x, startPoint.y, startPoint.z );

	camera.stepX = cameraPlaneWidth / (float)resolution.x;
	camera.stepY = cameraPlaneHeight / (float)resolution.y;

	return m_pCudaRenderPipeline->generatePrimaryRay( camera, generatedCount, currentSampleX, currentSampleY, pScene->isEnableJittering() );
}


/**
 *	CUDA 에서 Primary ray 를 생성시킨다.
 */
GError GGPURayTracer::makePrimaryRaySet_BlockGrouping( GScene *pScene, int *generatedCount, int currentSampleX, int currentSampleY )
{
	GDimension resolution = pScene->getResolution();
	GDimension SuperSampling = pScene->getSuperSampling();
	GCamera* pCamera = pScene->getRenderCamera();
	float aspect = (float) resolution.x / (float) resolution.y;
	float cameraPlaneHeight = 2.0f * pCamera->getNear() * tanf( ( pCamera->getFovy() * 0.5f ) * G_TO_RADIAN );
	float cameraPlaneWidth = aspect * cameraPlaneHeight;

	cuCamera camera;

	camera.eye = make_float3( pCamera->getEye().x, pCamera->getEye().y, pCamera->getEye().z );
	camera.u = make_float3( pCamera->getUVec().x, pCamera->getUVec().y, pCamera->getUVec().z );
	camera.v = make_float3( pCamera->getVVec().x, pCamera->getVVec().y, pCamera->getVVec().z );
	camera.n = make_float3( pCamera->getNVec().x, pCamera->getNVec().y, pCamera->getNVec().z );
	camera.fnear = pCamera->getNear();

	GVector startPoint = pCamera->getEye() - ( pCamera->getNVec() * camera.fnear ) +
						( pCamera->getVVec() * ( cameraPlaneHeight * 0.5f ) ) -
						( pCamera->getUVec() * ( cameraPlaneWidth * 0.5f ) );
	camera.startPoint = make_float3( startPoint.x, startPoint.y, startPoint.z );

	camera.stepX = cameraPlaneWidth / (float)resolution.x;
	camera.stepY = cameraPlaneHeight / (float)resolution.y;

	return m_pCudaRenderPipeline->generatePrimaryRay( camera, generatedCount, 
		currentSampleX, currentSampleY, pScene->isEnableJittering() );
}

/**
 *	현재 scene 정보를 기반으로 ray index 에 해당하는 ray 가
 *	image 상의 몇 pixel 에 해당할지를 계산한다.
 */
int GGPURayTracer::toImageIndex( GScene *pScene, int rayIndex, int *x, int *y )
{
	/** ray index 를 x,y 좌표로 변환 */
	(*x) = rayIndex % ( pScene->getResolution().x );
	(*y) = rayIndex / ( pScene->getResolution().x );
	
	/** 다시 image index 로 변환 */
	return (*y) * pScene->getResolution().x + (*x);
}


