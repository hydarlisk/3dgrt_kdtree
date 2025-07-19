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
 *	scene ������ �籸���ؾ��Ҷ� �ʱ�ȭ�Ѵ�.
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
		 *	Light ���� ����. ������ �ʿ�����Ƿ� ����.
		 */
		int lightCount = 0;
		cuLight* pLight = GRenderCommon::makeCudaLightInfo( pScene, &lightCount, m_pCudaRenderPipeline );
		error = m_pCudaRenderPipeline->setLightInfo( pLight, lightCount );
		free( pLight );

		if ( error != errorNo )
			return error;

		/** 
		 *	�� ���������, kdtree ������ �̷��� �ؼ� cuda �� �ѱ�. 
		 */ 
		error = pScene->getKDTreeStructure()->makeCudaRenderStructureInfo( m_pCudaRenderPipeline );
		if ( error != errorNo )
			return error;

		/** blooming ȿ���� �����ϱ⸦ ���Ѵٸ� �ʱ�ȭ �ص�.*/
		error = m_pCudaRenderPipeline->initBloomingFilter( m_pScene->getBloomingRadius(),
														   m_pScene->getBloomingWeight() );
		if ( error != errorNo )
			return error;

		m_pCudaRenderPipeline->printStatusInfo();
	}

	/**
	 *	���� Renderer �� ó���� Scene �� ����Ѵ�.
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
//	 *	Scene �� ���� geometry ���¿��� ���Ѱ� �ִ��� üũ�ؼ� �ִٸ�
//	 *	SpatialStructure �� �籸���Ѵ�.
//	 */
//	error = initialize( pScene );
//	if ( error != errorNo ) {
//		GLogManager::logging( LOG_ERROR, "can't initialize rendering\n" );
//		return error;
//	}
//
//	/** rendering option ���� */
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
//	 *	cuda ���� intersection point �� ����Ʈ ������ �ʱ�ȭ �Ѵ�.
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
//	 *	reflection �̳� refraction �� ������ �ִ� max depth ����
//	 *	�����Ѵ�.
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
//	/** blooming ȿ���� �����ϱ⸦ ���Ѵٸ� */
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
	 *	Scene �� ���� geometry ���¿��� ���Ѱ� �ִ��� üũ�ؼ� �ִٸ�
	 *	SpatialStructure �� �籸���Ѵ�.
	 */
	error = initialize( pScene );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't initialize rendering\n" );
		return error;
	}

	/** rendering option ���� */
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
	 *	cuda ���� intersection point �� ����Ʈ ������ �ʱ�ȭ �Ѵ�.
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
			 *	reflection �̳� refraction �� ������ �ִ� max depth ����
			 *	�����Ѵ�.
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

	/** blooming ȿ���� �����ϱ⸦ ���Ѵٸ� */
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
 *	CUDA ���� Primary ray �� ������Ų��.
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
 *	CUDA ���� Primary ray �� ������Ų��.
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
 *	���� scene ������ ������� ray index �� �ش��ϴ� ray ��
 *	image ���� �� pixel �� �ش������� ����Ѵ�.
 */
int GGPURayTracer::toImageIndex( GScene *pScene, int rayIndex, int *x, int *y )
{
	/** ray index �� x,y ��ǥ�� ��ȯ */
	(*x) = rayIndex % ( pScene->getResolution().x );
	(*y) = rayIndex / ( pScene->getResolution().x );
	
	/** �ٽ� image index �� ��ȯ */
	return (*y) * pScene->getResolution().x + (*x);
}


