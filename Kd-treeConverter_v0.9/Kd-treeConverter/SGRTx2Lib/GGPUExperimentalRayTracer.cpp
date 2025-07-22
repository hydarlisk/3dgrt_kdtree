#include "GGPUExperimentalRayTracer.h"
#include "GGPUExperimentalRayTracer.h"
#include "GSpatialStructure.h"
#include "GKDTreeStructure.h"
#include "GPointLight.h"
#include "GRenderCommon.h"
#include "math.h"

GGPUExperimentalRayTracer::GGPUExperimentalRayTracer()
{
	m_iOldSceneNumber = 0;
	m_pCudaRenderPipeline = NULL;
	m_iSceneTimestamp = -1;
	m_pScene = NULL;
}

GGPUExperimentalRayTracer::~GGPUExperimentalRayTracer(void)
{
	uninitialize();
}

void GGPUExperimentalRayTracer::uninitialize()
{
	if ( m_pCudaRenderPipeline ) {
		delete m_pCudaRenderPipeline;
	}
}

/**
 *	scene 정보를 재구성해야할때 초기화한다.
 */
GError GGPUExperimentalRayTracer::initialize( GScene *pScene )
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
		cuSceneInfo.bEnableLocalShading = pScene->isEnableLocalShading();
		cuSceneInfo.bEnableShadow = pScene->isEnableShadow();
		cuSceneInfo.bEnableTexture = pScene->isUseTexture();
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

GError GGPUExperimentalRayTracer::rendering( GScene *pScene, bool isDebug )
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
	cuSceneInfo.bEnableLocalShading = pScene->isEnableLocalShading();
	cuSceneInfo.bEnableTexture = pScene->isUseTexture();
	cuSceneInfo.bEnableShadow = pScene->isEnableShadow();
	cuSceneInfo.iShadowRay = 1;

	m_pCudaRenderPipeline->setSceneInfo( cuSceneInfo );

	cuThreshold cuThresholdInfo;
	cuThresholdInfo.primaryOIDRegionColorThreshold = pScene->getPrimaryOIDRegionColorThreshold();
	cuThresholdInfo.primaryNormalRegionColorThreshold = pScene->getPrimaryNormalRegionColorThreshold();
	cuThresholdInfo.primaryShadowRegionColorThreshold = pScene->getPrimaryShadowRegionColorThreshold();
	cuThresholdInfo.primaryTextureRegionColorThreshold = pScene->getPrimaryTextureRegionColorThreshold();
	cuThresholdInfo.secondaryOIDRegionColorThreshold = pScene->getSecondaryOIDRegionColorThreshold();
	cuThresholdInfo.secondaryNormalRegionColorThreshold = pScene->getSecondaryNormalRegionColorThreshold();
	cuThresholdInfo.secondaryShadowRegionColorThreshold = pScene->getSecondaryShadowRegionColorThreshold();
	cuThresholdInfo.secondaryTextureRegionColorThreshold = pScene->getSecondaryTextureRegionColorThreshold();
	cuThresholdInfo.etcRegionColorThreshold = pScene->getEtcRegionColorThreshold();
	cuThresholdInfo.onlyColorThreshold = pScene->getOnlyColorThreshold();

	m_pCudaRenderPipeline->setThresholdInfo( cuThresholdInfo );

	GImageBuffer *pImageBuffer = pScene->getImageBuffer();

	int depth = 0;
	int m_iMaxDepth = pScene->getMaxReflectionDepth();
	int atLeastOneRay = 0;

	cuCamera camera = calCameraInfo( m_pScene );

	m_pCudaRenderPipeline->clearFrameBuffer();

	GTimer timer1;
	timer1.start();

	//error = m_pCudaRenderPipeline->doSinglePassRayCasting( camera, m_iMaxDepth, 
	//						( pScene->getFrontFace() == faceCCW ), 
	//						pScene->isBackFaceCulling(),
	//						pScene->getSuperSampling().x, pScene->getSuperSampling().y,
	//						pScene->isEnableJittering(),
	//						pScene->isEnableShadow() );

	if ( pScene->getSuperSampling().x == 16 && pScene->getSuperSampling().y == 16 ) {
		error = m_pCudaRenderPipeline->doSinglePassRayCasting( camera, m_iMaxDepth, 
								( pScene->getFrontFace() == faceCCW ), 
								pScene->isBackFaceCulling(),
								pScene->getSuperSampling().x, pScene->getSuperSampling().y,
								pScene->isEnableJittering(),
								pScene->isEnableShadow() );
	} else {
		error = m_pCudaRenderPipeline->fixedOption_doSinglePassRayCasting_Coherent( 
								camera, m_iMaxDepth, 
								( pScene->getFrontFace() == faceCCW ), 
								pScene->isBackFaceCulling(),
								pScene->getSuperSampling().x, pScene->getSuperSampling().y,
								pScene->isEnableJittering(),
								pScene->isEnableShadow() );
	}

	if ( error != errorNo ) {
		GLogManager::logging( LOG_FATAL, "Rendering Error, %s\n", GErrorManager::getGErrorString( error ) );
		return errorNo;
	}

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

	m_pCudaRenderPipeline->getFrameBuffer( pImageBuffer->getBuffer() );

	m_pScene->setFPS( 1.0f / timer1.getElapsedTime() );

	GLogManager::logging( LOG_DEBUG, "traceRayFullBounce time : %f sec. traced depth = %d\n", timer1.getElapsedTime(), depth );

	return errorNo;

}

cuCamera GGPUExperimentalRayTracer::calCameraInfo( GScene *pScene )
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

	return camera;
}

/**
 *	현재 scene 정보를 기반으로 ray index 에 해당하는 ray 가
 *	image 상의 몇 pixel 에 해당할지를 계산한다.
 */
int GGPUExperimentalRayTracer::toImageIndex( GScene *pScene, int rayIndex, int *x, int *y )
{
	/** ray index 를 x,y 좌표로 변환 */
	(*x) = rayIndex % ( pScene->getResolution().x );
	(*y) = rayIndex / ( pScene->getResolution().x );
	
	/** 다시 image index 로 변환 */
	return (*y) * pScene->getResolution().x + (*x);
}


