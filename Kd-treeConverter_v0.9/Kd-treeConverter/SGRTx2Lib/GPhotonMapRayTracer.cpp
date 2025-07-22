#include "GPhotonMapRayTracer.h"
#include "GGPURayTracer.h"
#include "GSpatialStructure.h"
#include "GKDTreeStructure.h"
#include "GPointLight.h"
#include "GRaySetLight.h"
#include "GBoundingBox.h"
#include "math.h"
#include "GAreaDensityEstimate.h"
#include "assert.h"

/**
 */
GPhotonMapRayTracer::GPhotonMapRayTracer( GPhotonMappingOption *pOption )
{
	m_bDebug = false;

	m_iOldSceneNumber = 0;
	m_pIntersectionPointMap = NULL;
	m_pCudaRenderPipeline = NULL;
	m_pCudaPhotonMapping = NULL;

	m_iRandomSeed = 1;

	/** 
	 *	옵션결정 
	 */
	m_Option = (*pOption);
	
	if ( m_Option.m_bDirectIllumByPhotonMap )
		m_Option.m_bSaveDirectPhoton = true;

	m_Option.m_fGridUnitLength = max( m_Option.m_fGridUnitLength, m_Option.m_fSearchRadius );

	m_fSceneLightPowerPerIteration = m_Option.m_fTotalSceneLightPower / (float) m_Option.m_iIteration;
	m_fOnePhotonPower = m_fSceneLightPowerPerIteration / (float) m_Option.m_iEmitPhotonPerIteration;

	/** 
	 *	한번의 iteration 수행시 최대 photon 갯수계산. buffer를 위해서 
	 */
	m_iMaxPhotonSize = m_Option.m_iMaxBound * m_Option.m_iEmitPhotonPerIteration;

	m_pHostPhotonMem = NULL;
	m_pGlobalPhotonGridBox = NULL;
	m_pIPointGridBox = NULL;
}

GPhotonMapRayTracer::~GPhotonMapRayTracer(void)
{
	uninitialize();
}

void GPhotonMapRayTracer::uninitialize()
{
	if ( m_pIntersectionPointMap )
		delete m_pIntersectionPointMap;

	if ( m_pCudaRenderPipeline ) {
		delete m_pCudaRenderPipeline;
	}

	if ( m_pCudaPhotonMapping )
		delete m_pCudaPhotonMapping;

	if ( m_pHostPhotonMem ) {
		free( m_pHostPhotonMem );
		m_pHostPhotonMem = NULL;
	}

	if ( m_pGlobalPhotonGridBox ) {
		delete m_pGlobalPhotonGridBox;
	}

	if ( m_pIPointGridBox )
		delete m_pIPointGridBox;
}

/**
 *	scene 정보를 재구성해야할때 초기화한다.
 */
GError GPhotonMapRayTracer::initialize( GScene *pScene )
{
	GError error;
	m_pScene = pScene;

	/**
	 *	iteration 이 1번이 아니라면 
	 *	이전 rendering 시에 사용한 photon tracing 정보를 재사용할 수 없으므로
	 *  무조건 tracing 을 수행해야 한다.
	 */
	m_bRunTracing = ( m_Option.m_iIteration != 1 );

	if ( m_iOldSceneNumber != pScene->getSceneNumber() || 
		 m_iSceneTimestamp != pScene->getGeometryChangeTimestamp() ||
		 m_oldResolution != pScene->getResolution() ) {

		if ( m_pCudaRenderPipeline ) {
			delete m_pCudaRenderPipeline;
		}
		if ( m_pCudaPhotonMapping ) {
			delete m_pCudaPhotonMapping;
		}

		/**  photon tracing 을 해야할지 여부. */
		m_bRunTracing = true;

		cuScene cuSceneInfo;
		cuSceneInfo.globalAmbient = make_float3( pScene->getGlobalAmbient().r, 
												 pScene->getGlobalAmbient().g,
												 pScene->getGlobalAmbient().b );
		cuSceneInfo.iResolutionX = pScene->getResolution().x;
		cuSceneInfo.iResolutionY = pScene->getResolution().y;
		cuSceneInfo.iSuperSamplingX = pScene->getSuperSampling().x;
		cuSceneInfo.iSuperSamplingY = pScene->getSuperSampling().y;
		cuSceneInfo.bEnableShadow = pScene->isEnableShadow();
		cuSceneInfo.iShadowRay = 1;

		/** ray tracing ray 와 total photon emit 개수중 큰걸로 ray 공간을 할당 */
		int maxRay = cuSceneInfo.iResolutionX * cuSceneInfo.iResolutionY * 
					 cuSceneInfo.iSuperSamplingX * cuSceneInfo.iSuperSamplingY;

		maxRay = max( maxRay, m_iMaxPhotonSize );

		m_pCudaRenderPipeline = new cudaRenderPipeline();
		error = m_pCudaRenderPipeline->initialize( cuSceneInfo, maxRay );
		if ( error != errorNo )
			return error; 

		GLogManager::logging( LOG_DEBUG, "cuPMIntersectionPointcuPMIntersectionPoint %d", sizeof( cuPMIntersectionPoint ) );

		/**
		 *	Light 정보 세팅. light intensity 에 따라서 photon 을 얼마나
		 *	뿌릴지 결정. 세팅후 필요없으므로 삭제.
		 */
		int lightCount = 0;
		cuLight* pLight = GRenderCommon::makeCudaLightInfo( pScene, &lightCount, m_pCudaRenderPipeline );
		if ( pLight == NULL ) {
			GLogManager::logging( LOG_ERROR, "makeCudaLightInfo error" );
			return errorUnknown;
		}

		error = constructPhotonEmitLightInfo( pLight, lightCount );
		if ( error != errorNo ) {
			GLogManager::logging( LOG_ERROR, "constructPhotonEmitLightInfo error" );
			return error;
		}

		/**---------------------------------------------------------------------------------------------
		 *	photon mapping cuda 를 초기화한다.
		----------------------------------------------------------------------------------------------*/
		GBoundingBox bbox = pScene->getKDTreeStructure()->getBoundingBox();
		GVector length = bbox.m_Max - bbox.m_Min;

		/**
		 *	scene 의 크기와 인자로 주어진 grid length 에 따라서
		 *	grid box 의 개수가 엄청많아질수 있으므로
		 *	일단 최대로 Grid 박스를 구성할수 있는 최소한의 gridLength 를 구하고,
		 *	인자로 주어진 gridUnitLength 및 radius 가 이 최대값을 넘는지 체크해서 
		 *	적당한 grid box 를 제안한다.
		 */
		float gridLength = max( length.x / (float) MAX_GRID_COUNT, 
								max( length.y / (float) MAX_GRID_COUNT, length.z / (float) MAX_GRID_COUNT ) );

		m_iRandomSeed = 1;
		m_Option.m_fGridUnitLength = max( m_Option.m_fGridUnitLength, gridLength );
		m_Option.m_fSearchRadius = min( m_Option.m_fSearchRadius, m_Option.m_fGridUnitLength );

		GLogManager::logging( LOG_INFO, " -> Photon Mapping Grid UnitLength : ( %f, %f, %f ), SearchRadius : %f",
										m_Option.m_fGridUnitLength, 
										m_Option.m_fGridUnitLength, 
										m_Option.m_fGridUnitLength,
										m_Option.m_fSearchRadius );

		m_pHostPhotonMem = (cuPhoton*) malloc( sizeof( cuPhoton ) * m_iMaxPhotonSize );
		memset( m_pHostPhotonMem, 0x00, sizeof( cuPhoton ) * m_iMaxPhotonSize );


		int maxIntersection = cuSceneInfo.iResolutionX * cuSceneInfo.iResolutionY * 
					 cuSceneInfo.iSuperSamplingX * cuSceneInfo.iSuperSamplingY * ( m_pScene->getMaxReflectionDepth() + 1 );

		m_pCudaPhotonMapping = new cudaPhotonMapping();
		if ( ( error = m_pCudaPhotonMapping->initialize( m_iMaxPhotonSize, maxIntersection ) ) ) {
			GLogManager::logging( LOG_ERROR, "%s %s", 
								"cudaUploadLightInfo error", 
								GErrorManager::getGErrorString( error ) );
			return errorCudaError;
		}

		/**
		 *	light 정보를 업로드한다.
		 */
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
		
		/**
		 *	intersection point map 초기화.
		 */
		m_pIntersectionPointMap = new GIntersectionPointMap( 
			pScene->getResolution(), pScene->getSuperSampling(), pScene->getMaxReflectionDepth() + 1 );

		/** blooming 효과를 적용하기를 원한다면 초기화 해둠.*/
		if ( m_pScene->isBloomingFilter() ) {
			error = m_pCudaRenderPipeline->initBloomingFilter( m_pScene->getBloomingRadius(),
															   m_pScene->getBloomingWeight() );
			if ( error != errorNo )
				return error;
		}

		m_pCudaRenderPipeline->printStatusInfo();
		m_pCudaPhotonMapping->printStatusInfo();
	}

	/**
	 *	현재 Renderer 가 처리한 Scene 을 기억한다.
	 */
	m_iOldSceneNumber = pScene->getSceneNumber();
	m_iSceneTimestamp = pScene->getGeometryChangeTimestamp();
	m_oldResolution = pScene->getResolution();

	return errorNo;
}

GError GPhotonMapRayTracer::constructPhotonEmitLightInfo( cuLight* plightList, int lightCount )
{
	int photonSum = 0, totalPhotonLightCount = 0;
	int processPhotonLightCount = 0;
	float intensitySum = 0.0f;

	/**
	 *	전체 light 중에서 각 light 의 intensity 가 차지하는 비율만큼 
	 *	각 light 가 뿌릴 photon 의 개수를 배분한다. 
	 */
	for ( int i = 0; i < lightCount; ++i ) {
		if ( plightList[ i ].bUsePhoton == 0 )
			continue;
		intensitySum += plightList[ i ].intensity;
		totalPhotonLightCount++;
	}

	/**
	 *	photon 은 정수형으로 잘리므로 각 light 의 emitPhoton 합이 인자로 주어진 emitPhoton 과 
	 *	1 개정도 차이가 생길수 있다. 따라서 마지막 light 에는 남은 photon 을 다 준다.
	 */
	for ( int i = 0; i < lightCount; ++i ) {

		if ( plightList[ i ].bUsePhoton == 0 )
			continue;

		/** photon 을 emit 시키는 마지막 light 일때 */
		if ( processPhotonLightCount == totalPhotonLightCount - 1 ) {
			plightList[ i ].iPhotonStartIndex = photonSum;
			plightList[ i ].iEmitPhoton = m_Option.m_iEmitPhotonPerIteration - photonSum;
			plightList[ i ].fOnePhotonPower = m_fOnePhotonPower;
			break;
		} else {
			float rate = plightList[ i ].intensity / intensitySum;
			plightList[ i ].iEmitPhoton = (int)( m_Option.m_iEmitPhotonPerIteration * rate );
			plightList[ i ].iPhotonStartIndex = photonSum;
			plightList[ i ].fOnePhotonPower = m_fOnePhotonPower;
			photonSum += plightList[ i ].iEmitPhoton;
			processPhotonLightCount++;
		}

	}
	
	return errorNo;
}

GError GPhotonMapRayTracer::rendering( GScene *pScene, bool isDebug ) 
{
	GError error;

	m_bDebug = isDebug;

	int depth = 0;
	int m_iMaxDepth = pScene->getMaxReflectionDepth();
	int atLeastOneRay = 0;

	/** 
	 *	Scene 이 이전 geometry 상태에서 변한게 있는지 체크해서 있다면
	 *	SpatialStructure 를 재구성한다.
	 */
	error = initialize( pScene );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't initialize rendering" );
		return error;
	}

	/** rendering option 세팅 */
	m_pCudaRenderPipeline->renderingOption( isEnableShadow(), pScene->isUseTexture() );
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

	/**
	 *	cuda 안의 intersection point 를 디폴트 값으로 초기화 한다.
	 */
	error = m_pCudaRenderPipeline->clearIntersectionResult();
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't clearIntersectionResult" );
		return errorCudaError;
	}

	m_pCudaRenderPipeline->clearFrameBuffer();

	GTimer timer;
	timer.start();

	int generatedRayCount = 0;
	error = makePrimaryRaySet_BlockGrouping( pScene, &generatedRayCount, 0, 0 );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "makePrimaryRaySet" );
		return error;
	}

	/**
	 *	reflection 이나 refraction 이 있으면 최대 max depth 까지
	 *	추적해서 intersection point 를 m_pIntersectionPointMap 에 쌓아 넣는다.
	 */
	int count = 0;

	m_pIntersectionPointMap->clear();

	do {
		error = m_pCudaRenderPipeline->doRayCasting( 0, generatedRayCount, 
							( pScene->getFrontFace() == faceCCW ),
							pScene->isBackFaceCulling() );
		if ( error != errorNo )	break;

		/** 
		 *	intersection 정보를 가져와서 intersection map 에 복사해 둔다.
		 *	이거 속도 개선 필요. 이부분때문에 fps 가 반절로 준다.
		 */
		error = backupIntersectionResult( generatedRayCount );
		if ( error != errorNo ) break;

		/**
		 *	photon map 으로 direct illumination 을 처리한다면
		 *	ray tracing 으로 direct illum 처리 안함.
		 */
		if ( !m_Option.m_bDirectIllumByPhotonMap ) {
			error = m_pCudaRenderPipeline->calDirectIllumination( m_iMaxDepth );
			if ( error != errorNo )	break;
		}

		if ( depth >= m_iMaxDepth ) break;

		error = m_pCudaRenderPipeline->generateReflectionRay( 0, generatedRayCount, &atLeastOneRay );
		if ( error != errorNo || atLeastOneRay == 0 ) break;

		depth++;

	} while( true );

	if ( error != errorNo ) {
		GLogManager::logging( LOG_FATAL, "Rendering Error, %s", GErrorManager::getGErrorString( error ) );
		return errorNo;
	}

	GLogManager::logging( LOG_DEBUG, "Intersection Point Count = %d", m_pIntersectionPointMap->getSize() );

	/**-----------------------------------------------------------------------------------------------
	 **	scene 의 direct, indirect, 그리고 합산한 image buffer 를 구성.
	 **----------------------------------------------------------------------------------------------*/

	/**
	 * ray tracing 으로 계산한 direct illum 을 복사해온다.
	 */
	GImageBuffer *pImageBuffer = m_pScene->getImageBuffer();
	GImageBuffer *pDirectIllm = m_pScene->getDirectIllumImageBuffer();
	GImageBuffer *pInDirectIllum = m_pScene->getIndirectIllumImageBuffer();

	/**
	 *	photon mapping 으로 indirect illumination 만 계산해옴.
	 */
	pInDirectIllum->clear();

GLogManager::logging( LOG_INFO, "start photon" );

	error = photonMapIteration( m_bRunTracing, m_Option.m_iIteration, pInDirectIllum );

	if ( error != NULL )
		return errorNo;

	timer.end();

	/** 
	 *	direct 와 indirect illum 을 합산해서 최종 이미지에 저장한다.
	 */
	/**
	 *	photon map 으로 direct illumination 을 처리한다면
	 *	ray tracing 으로 direct illum 누적안함.
	 *  indirect illum 에 photon map 으로 direct 까지 계산한 결과가 있으므로
	 *	그것만 복사.
	 */
	if ( !m_Option.m_bDirectIllumByPhotonMap ) {
		m_pCudaRenderPipeline->getFrameBuffer( pDirectIllm->getBuffer() );
		pImageBuffer->copy( pDirectIllm->getBuffer() );
		pImageBuffer->add( pInDirectIllum );
	} else {
		pImageBuffer->copy( pInDirectIllum->getBuffer() );
	}

	/** blooming 효과를 적용하기를 원한다면 framebuffer 에 올려서 처리하고 다시 가져온다. */
	if ( m_pScene->isBloomingFilter() ) {
		m_pCudaRenderPipeline->setFrameBuffer( pImageBuffer->getBuffer() );
		m_pCudaRenderPipeline->bloomingFiltering();
		m_pCudaRenderPipeline->getFrameBuffer( pImageBuffer->getBuffer() );
	}

	GLogManager::logging( LOG_DEBUG, "traceRayFullBounce time : %f sec. trace depth = %d", timer.getElapsedTime(), depth );


	m_pScene->setFPS( 1.0f / timer.getElapsedTime() );

	return errorNo;
}

/**
 *	계산한 결과를 image buffer 에 누적시킨다.
 *	외부에서도 호출될 수 있으므로, 절대 내부 클래스변수를
 *	함부로 수정하는 연산이 들어가면 안된다.
 */
void GPhotonMapRayTracer::accumulateRadiance ( 
					GImageBuffer *pInDirectIllumImageBuffer,
					GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox )
{
	int width = m_pScene->getResolution().x;
	int height = m_pScene->getResolution().y;
	int samplex = m_pScene->getSuperSampling().x;
	int sampley = m_pScene->getSuperSampling().y;
	int x, y;
	GColor color;

	for ( int i = 0; i < pIPointGridBox->m_iTotalCount; ++i ) {

		GGPURayTracer::toImageIndex( m_pScene, pIPointGridBox->m_pData[ i ].rayIndex, &x, &y );
		cuIntersectionPoint* pIPoint = ( pIPointGridBox->m_pData + i );
		cuPMIntersectionPoint* pPMPoint = ( pIPointGridBox->m_pData2 + i );

		color.r = pIPoint->colorWeight.x * pPMPoint->power[0];
		color.g = pIPoint->colorWeight.y * pPMPoint->power[1];
		color.b = pIPoint->colorWeight.z * pPMPoint->power[2];
		color.a = 1.0f;

		pInDirectIllumImageBuffer->addColor( x, y, color );

	}
	
}

GError GPhotonMapRayTracer::backupIntersectionResult( int count )
{
	GError error;

	cuIntersectionPoint *pIntersectionPoints = m_pCudaRenderPipeline->getIntersectionResult( count );
	if ( pIntersectionPoints == NULL )
		return errorResultError;

	/**
	*	gpu 상에서의 intersection 결과중 hit 된것만 point map 에 insert 한다.
	*/
	for ( int i = 0; i < count; ++i ) {
		if ( pIntersectionPoints[ i ].isHit() ) {
			error = m_pIntersectionPointMap->insertIntersectionPoint( pIntersectionPoints + i, 1 );
			if ( error != errorNo )
				return error;
			}
	}

	return errorNo;
}


/**
 *	CUDA 에서 Primary ray 를 생성시킨다.
 */
GError GPhotonMapRayTracer::makePrimaryRaySet_BlockGrouping( GScene *pScene, int *generatedCount, int currentSampleX, int currentSampleY )
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
 *	CUDA 에서 Primary ray 를 생성시킨다.
 */
GError GPhotonMapRayTracer::makePrimaryRaySet( GScene *pScene, int *generatedRayCount, int currentSampleX, int currentSampleY )
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

	return m_pCudaRenderPipeline->generatePrimaryRay( camera, generatedRayCount, 
		currentSampleX, currentSampleY, pScene->isEnableJittering() );
}

/**
 *	CUDA 를 이용해서 Photon Tracing 과 Gathering Iteration 을 수행한다.
 *	결과를 direct illumination 과 합산해서 imageBuffer 에기록한다.
 */
GError GPhotonMapRayTracer::photonMapIteration( bool bRunTracing, int iteration, 
												GImageBuffer *pInDirectIllumImageBuffer )
{
	GError error;
	GTimer totalTimer, ipointTimer, accumulateTimer, areaTimer;

	totalTimer.start();

	/**-------------------------------------------------------------------------------------------
	 **	모든 iteration 에 관련된 통계자료 초기화.
	 **------------------------------------------------------------------------------------------*/
	m_iLogTotalTracedPhoton = 0;
	m_iLogTotalIsectGridMakingTime = 0.0f;
	m_iLogTotalIsectAreaDensityTime = 0.0f;
	m_iLogTotalTracingTime = 0.0f;
	m_iLogTotalIndexGridTime = 0.0f;
	m_iLogTotalGatheringTime = 0.0f;
	m_iLogTotalPhotonGridMakingTime = 0.0f;
	m_iLogTotalAccumulateTime = 0.0f;
	m_iLogTotalPhotonMappingTime = 0.0f;

	/**-------------------------------------------------------------------------------------------
	 **	intersection point 를 위한 grid 구성을 하고 cuda 에 업로드.
	 **------------------------------------------------------------------------------------------*/
	ipointTimer.start();

	if ( m_pIPointGridBox )
		delete m_pIPointGridBox;
	m_pIPointGridBox = NULL;
	
	m_pIPointGridBox = makeIPointGridBox( m_pIntersectionPointMap );
	error = m_pCudaPhotonMapping->uploadIntersectionPoint( m_pIPointGridBox->m_pData, 
									m_pIPointGridBox->m_iTotalCount );
	if ( error != errorNo )
		return errorNo;
	
	ipointTimer.end();

	/**-------------------------------------------------------------------------------------------
	 **	density 를 추정하기 위해서 각 ipoint 가 사용할 Area 를 구한다.
	 **------------------------------------------------------------------------------------------*/
	areaTimer.start();

	/** projected circle 또는 area photon 사용. */
	error = estimateDensityArea( m_pIPointGridBox, m_Option.m_eDensityMethod );
	if ( error != errorNo )
		return errorNo;
    
	areaTimer.end();

	/**-------------------------------------------------------------------------------------------
	 **	PHOTON TRACING 과 GATHERING 을 수행한다. 
	 **------------------------------------------------------------------------------------------*/
	int randomSeed = m_iRandomSeed;

	for ( int i = 0; i < iteration; ++i ) {

		error = photonMapOneIteration( randomSeed, m_pIPointGridBox, bRunTracing, i, iteration );
		if ( error != errorNo )
			return error;

		/**
		 *	다음번 photon 을 뿌릴때의 random 값을 위한 seed 값
		 */
		randomSeed += m_Option.m_iEmitPhotonPerIteration;
		if ( randomSeed > 1000000000 )
			randomSeed = 1;

	}

	totalTimer.end();

	/**-------------------------------------------------------------------------------------------
	 **	전체 iteration 이 끝나고 계산된 각 ipoint 의 radiance 를 indirect image buffer 에
	 ** 누적시킨다.
	 **------------------------------------------------------------------------------------------*/
	accumulateTimer.start();

	accumulateRadiance( pInDirectIllumImageBuffer, m_pIPointGridBox );

	accumulateTimer.end();

	m_iLogTotalAccumulateTime = accumulateTimer.getElapsedTime();

	/**-------------------------------------------------------------------------------------------
	 **	INFO MATION 출력. 
	 **------------------------------------------------------------------------------------------*/
	m_iLogTotalIsectGridMakingTime = ipointTimer.getElapsedTime();
	m_iLogTotalIsectAreaDensityTime = areaTimer.getElapsedTime();
	m_iLogTotalPhotonMappingTime = totalTimer.getElapsedTime();

	GLogManager::logging( LOG_DEBUG, "-------- photon mapping iteration : %d iteration ----------", iteration );
	GLogManager::logging( LOG_DEBUG, " -> Resolution          : %d x %d", 
						m_pScene->getResolution().x, m_pScene->getResolution().y );
	GLogManager::logging( LOG_DEBUG, " -> Sampling            : %d x %d", 
						m_pScene->getSuperSampling().x, m_pScene->getSuperSampling().y );
	GLogManager::logging( LOG_DEBUG, " -> isect points        : %d", m_pIPointGridBox->m_iTotalCount );
	GLogManager::logging( LOG_DEBUG, " -> total traced photon : %d", m_iLogTotalTracedPhoton );
	GLogManager::logging( LOG_DEBUG, " -> grid box            : %d x %d x %d", 
				m_pIPointGridBox->m_iCellXCount, m_pIPointGridBox->m_iCellYCount, m_pIPointGridBox->m_iCellZCount );
	GLogManager::logging( LOG_DEBUG, " -> grid unit length    : %f x %f x %f",
				m_pIPointGridBox->m_fXUnitLength, m_pIPointGridBox->m_fYUnitLength, m_pIPointGridBox->m_fZUnitLength );
	GLogManager::logging( LOG_DEBUG, " -> gahtering radius    : %f", m_Option.m_fSearchRadius );
	GLogManager::logging( LOG_DEBUG, " -> Scene Light power   : %f", m_Option.m_fTotalSceneLightPower );
	GLogManager::logging( LOG_DEBUG, " -> One Photon  power   : %f", m_fOnePhotonPower );

	GLogManager::logging( LOG_DEBUG, " -> total isect grid making   : %f sec.", m_iLogTotalIsectGridMakingTime );
	GLogManager::logging( LOG_DEBUG, " -> total area density        : %f sec.", m_iLogTotalIsectAreaDensityTime );
	GLogManager::logging( LOG_DEBUG, " -> total photon tracing      : %f sec.", m_iLogTotalTracingTime );
	GLogManager::logging( LOG_DEBUG, " -> total photon gathering    : %f sec.", m_iLogTotalGatheringTime );
	GLogManager::logging( LOG_DEBUG, " -> total photon grid making  : %f sec.", m_iLogTotalPhotonGridMakingTime );
	GLogManager::logging( LOG_DEBUG, " -> total index grid making   : %f sec.", m_iLogTotalIndexGridTime );
	GLogManager::logging( LOG_DEBUG, " -> total accumulate radiance : %f sec.", m_iLogTotalAccumulateTime );

	GLogManager::logging( LOG_DEBUG, " -> total time                : %f sec.", m_iLogTotalPhotonMappingTime );
	GLogManager::logging( LOG_DEBUG, "------------------------------------------------------------" );

	return errorNo;

}

/**
 *	CUDA 를 이용해서 Photon Tracing 과 Gathering 한 Iteration 을 수행한다.
 *	Scene structure 가 변하지 않았다면 tracing 을 다시 수행할 필요는 없기
 *	때문에 bRunTracing 인자로 조절한다.
 */
GError GPhotonMapRayTracer::photonMapOneIteration( int randomSeed,
					GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox,				
					bool bRunTracing, int iterationId, int maxIteration )
{
	GError error;
	GTimer timer, tracingTimer, gatheringTimer, shadingTimer, photonGridTimer;
	GTimer indexGridTimer;

	timer.start();

	GLogManager::logging( LOG_DEBUG, " -> Photon Mapping Iteration( %d / %d ), random seed = ( %d )", 
		iterationId, maxIteration, randomSeed );

	//-----------------------------------------------------------------------------------------//
	/** 
	 *	debug option 일때 이전 rendering 에서 추가한 debug photon option 을 삭제한다. 
	 */
	if ( m_bDebug && bRunTracing ) {
		char name[1024] = { 0x00, };
		sprintf( name, "DEBUG_PHOTON_MAP_ITERATION_%d", iterationId );
		m_pScene->removeObject( name );
	}

	/**
	 *	Tracing 을 수행할 필요가 있을때만. iteration 이 1번이고 
	 *	scene 구조가 변하지 않았다면 이전에 tracing 한 photon 정보를
	 *	사용한다. iteration 이 여러번이라면 매번 map 이 바뀌므로 재사용할 수 없다.
	 */
	if ( bRunTracing ) {

		tracingTimer.start();

		/**
		*	photon tracing 을 수행한다.
		*/
		int iTracedPhotonSize = 0, iTracedBound = 0;
		error = m_pCudaPhotonMapping->photonTracing( 
						m_Option.m_iEmitPhotonPerIteration, 
						m_Option.m_iMaxBound, 
						randomSeed,
						m_Option.m_bSaveDirectPhoton,
						m_pCudaRenderPipeline, m_pHostPhotonMem, 
						&iTracedPhotonSize, &iTracedBound,
						( m_pScene->getFrontFace() == faceCCW ) );
		if ( error != errorNo )
			return error;

		tracingTimer.end();

		//-----------------------------------------------------------------------------------------//

		photonGridTimer.start();

		/**
		*	미리 업로드해놓은 intersection point 와 tracing 된 photon 을 가지고
		*	intersectionPoint 주변의 photon 을 gathering 한다.
		*	먼저 tracing 된 photon 을 grid box 에서 정렬한다.
		*/
		if ( m_pGlobalPhotonGridBox ) {
			delete m_pGlobalPhotonGridBox;
			m_pGlobalPhotonGridBox = NULL;
		}

		m_pGlobalPhotonGridBox = makePhotonGridBox( m_pHostPhotonMem, iTracedPhotonSize );
		GLogManager::logging( LOG_INFO, " -> Traced Photon Count = %d", m_pGlobalPhotonGridBox->m_iTotalCount );

		photonGridTimer.end();

	}

	/** ipoint 와 photon 의 연관관계만듬. */
	indexGridTimer.start();
	int photonIndexCount = 0;
	cuPhotonIndex *pPhotonIndex = makeIPointVsPhotonIndexData( 
						pIPointGridBox, m_pGlobalPhotonGridBox, &photonIndexCount );
	indexGridTimer.end();


	gatheringTimer.start();

	/**
	 *	ipoint 주변에 photon 이 있을때만
	 */
	if ( pPhotonIndex != NULL ) {
		error = m_pCudaPhotonMapping->photonGathering( pIPointGridBox->m_pData2, pIPointGridBox->m_iTotalCount,
								 m_pGlobalPhotonGridBox->m_pData, m_pGlobalPhotonGridBox->m_iTotalCount,
								 pPhotonIndex, photonIndexCount, m_Option.m_fSearchRadius );
		delete pPhotonIndex;
	}

	gatheringTimer.end();

	/** 
	 *	debug option 일때 rendering 에서 생성한 photon 정보를 object 만들어서 scene 에 추가한다. 
	 *	data 가 너무 많으면 안되므로 iterationid 가 0 인것만 만듬.
	 */
	if ( m_bDebug && bRunTracing && iterationId == 0 ) {
		
		char name[1024] = { 0x00, };
		GPolygonObject *pObject = new GPolygonObject();
		pObject->setDebugObject( true );

		cuPhoton *pPhoton = NULL;
		sprintf( name, "DEBUG_PHOTON_MAP_ITERATION_%d", iterationId );

		int *indexArray = (int*) malloc( m_pGlobalPhotonGridBox->m_iTotalCount * sizeof( int ) );
		float *vertexArray = (float*) malloc( m_pGlobalPhotonGridBox->m_iTotalCount * sizeof( float ) * 3 );
		float *colorArray = (float*) malloc( m_pGlobalPhotonGridBox->m_iTotalCount * sizeof( float ) * 3 );
		double maxpower = 0.0f, scale = 0.0f;

		for ( int i = 0; i < m_pGlobalPhotonGridBox->m_iTotalCount; ++i ) {

			pPhoton = ( m_pGlobalPhotonGridBox->m_pData + i );
			vertexArray[ i * 3 + 0 ] = pPhoton->pos.x;
			vertexArray[ i * 3 + 1 ] = pPhoton->pos.y;
			vertexArray[ i * 3 + 2 ] = pPhoton->pos.z;

			// debug 화면세서 photon 정보를 보여줄때
			// photon 의 power 가 너무 작기때문에 파워를 키운다.
			maxpower = max( pPhoton->power.x, max( pPhoton->power.y, pPhoton->power.z ) );
			if ( maxpower > 0.0f )
				scale = 1.0 / maxpower;

			colorArray[ i * 3 + 0 ] = (float)( (double) pPhoton->power.x * scale );
			colorArray[ i * 3 + 1 ] = (float)( (double) pPhoton->power.y * scale );
			colorArray[ i * 3 + 2 ] = (float)( (double) pPhoton->power.z * scale );
			indexArray[ i ] = i;

		}

		pObject->setName( name );
		pObject->setVertexCount( m_pGlobalPhotonGridBox->m_iTotalCount ); 
		pObject->setVertexArray( vertexArray );

		pObject->setTriangleCount( m_pGlobalPhotonGridBox->m_iTotalCount );
		pObject->setIndexArray( indexArray );
		pObject->setColorArray( colorArray );

		pObject->setPolygonType( typePolygonPoint );

		m_pScene->addObject( pObject );

	}
	//-----------------------------------------------------------------------------------------//

	timer.end();

	GLogManager::logging( LOG_DEBUG, "-------- photon mapping  ( seed : %d )----------------------", randomSeed );
	GLogManager::logging( LOG_DEBUG, " -> photon tracing run : %d", m_bRunTracing );
	GLogManager::logging( LOG_DEBUG, " -> isect point count  : %d", pIPointGridBox->m_iTotalCount );
	GLogManager::logging( LOG_DEBUG, " -> traced photons     : %d", m_pGlobalPhotonGridBox->m_iTotalCount );
	GLogManager::logging( LOG_DEBUG, " -> grid box           : %d x %d x %d", 
				pIPointGridBox->m_iCellXCount, pIPointGridBox->m_iCellYCount, pIPointGridBox->m_iCellZCount );
	GLogManager::logging( LOG_DEBUG, " -> grid unit length   : %f x %f x %f",
				pIPointGridBox->m_fXUnitLength, pIPointGridBox->m_fYUnitLength, pIPointGridBox->m_fZUnitLength );

	GLogManager::logging( LOG_DEBUG, " -> traced photons     : %d", m_pGlobalPhotonGridBox->m_iTotalCount );
	GLogManager::logging( LOG_DEBUG, " -> gahtering radius   : %f", m_Option.m_fSearchRadius );
	GLogManager::logging( LOG_DEBUG, " -> power Per Iteration : %f", m_fSceneLightPowerPerIteration );
	GLogManager::logging( LOG_DEBUG, " -> One Photon power    : %f", m_fOnePhotonPower );

	GLogManager::logging( LOG_DEBUG, " -> photon tracing     : %f sec.", tracingTimer.getElapsedTime() );
	GLogManager::logging( LOG_DEBUG, " -> photon grid making : %f sec.", photonGridTimer.getElapsedTime() );
	GLogManager::logging( LOG_DEBUG, " -> index grid making  : %f sec.", indexGridTimer.getElapsedTime() );
	GLogManager::logging( LOG_DEBUG, " -> photon gathering   : %f sec.", gatheringTimer.getElapsedTime() );
	GLogManager::logging( LOG_DEBUG, " -> total time         : %f sec.", timer.getElapsedTime() );
	GLogManager::logging( LOG_DEBUG, "------------------------------------------------------------" );

	m_iLogTotalTracedPhoton += m_pGlobalPhotonGridBox->m_iTotalCount;

	m_iLogTotalTracingTime += tracingTimer.getElapsedTime();
	m_iLogTotalPhotonGridMakingTime += photonGridTimer.getElapsedTime();
	m_iLogTotalIndexGridTime += indexGridTimer.getElapsedTime();
	m_iLogTotalGatheringTime += gatheringTimer.getElapsedTime();

	return errorNo;
}

/**
 *	ray 데이터를 이용해서 Ray Grid Box 를 만든다.
 */
GGridBox<cuIntersectionPoint, cuPMIntersectionPoint>*
		GPhotonMapRayTracer::makeIPointGridBox( GIntersectionPointMap *pIntersectionPointMap )
{
	GBoundingBox sceneBBox = m_pScene->getKDTreeStructure()->getBoundingBox();

	GGridBox<cuIntersectionPoint, cuPMIntersectionPoint>* pIGridBox = 
				new GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> 
							( sceneBBox, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength );

	/**
	 *	각 Cell 에 들어갈 ray counting.
	 */
	const cuIntersectionPoint* pPoint = NULL;
	int pointSize = pIntersectionPointMap->getSize();

	for ( int i = 0; i < pointSize; ++i ) {
		pPoint = pIntersectionPointMap->getIntersectionPoint( i );
		pIGridBox->counting( pPoint->pos.x, pPoint->pos.y, pPoint->pos.z );
	}

	pIGridBox->allocate();

	/**
	 *	각 Cell 에 ray 정보를 insert.
	 */
	int rayNumber = 0;
	cuIntersectionPoint iPoint;
	cuPMIntersectionPoint pmiPoint;

	for ( int i = 0; i < pointSize; ++i ) {

		pPoint = pIntersectionPointMap->getIntersectionPoint( i );
		memcpy( &iPoint, pPoint, sizeof( cuIntersectionPoint ) );

		// power 는 0.0f 으로 다 초기화 해야함.
		pmiPoint.power[0] = 0.0f; pmiPoint.power[1] = 0.0f; pmiPoint.power[2] = 0.0f;
		pmiPoint.photonIndexOffset = 0;
		pmiPoint.photonIndexCount = 0;
		pmiPoint.area = 0.0f;
		
		pIGridBox->insertData( pPoint->pos.x, pPoint->pos.y, pPoint->pos.z, &iPoint, &pmiPoint );

	}

	pIGridBox->printInfo( "RayGridBox" );

	return pIGridBox;

}


/**
 *	area photon 데이터를 이용해서 Area Photon Grid Box 를 만든다.
 */
GGridBox<cuPhoton, char> *GPhotonMapRayTracer::makeAreaPhotonGridBox( const vector<cuPhoton*> &list )
{
	int photonCount = (int) list.size();
	cuPhoton photonInfo;

	GGridBox<cuPhoton, char> *pAreaPhotonGridBox = new GGridBox<cuPhoton, char> 
			( m_pScene->getKDTreeStructure()->getBoundingBox(), m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength );

	/**
	 *	각 Cell 에 들어갈 photon counting.
	 */
	for ( int i = 0; i < photonCount; ++i ) {
		pAreaPhotonGridBox->counting( list[ i ]->pos.x, list[ i ]->pos.y,	list[ i ]->pos.z );
	}
	pAreaPhotonGridBox->allocate();

	/**
	 *	각 Cell 에 photon insert.
	 */
	for ( int i = 0; i < photonCount; ++i ) {

		photonInfo.pos = list[ i ]->pos;
		photonInfo.normal = list[ i ]->normal;

		// area 정보가 들어있음.
		photonInfo.power = list[ i ]->power;

		pAreaPhotonGridBox->insertData( 
			photonInfo.pos.x, photonInfo.pos.y, photonInfo.pos.z, &photonInfo, NULL );
	}

	pAreaPhotonGridBox->printInfo( "Area PhotonGridBox" );

	return pAreaPhotonGridBox;
}

GError GPhotonMapRayTracer::estimateAreaByAreaPhoton( 
				GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox )
{
	GError error = errorNo;
	float circleArea = 3.14159f * m_Option.m_fSearchRadius * m_Option.m_fSearchRadius;
	int zeroAreaCount = 0, loop = 10, offset = 0, blocks = 0;
	int triangleIndex = 0, orderIndex = 0;
	int overAreaCount = 0;
	int totalSize = 0, totalAreaPhoton = 0;
	int indexCount = 0;

	GTimer timer, timer1;
	GLogManager::logging( LOG_DEBUG, "--------------------------------- Area Estimate Start --------------------------" );

	timer.start();

	GTriangleWrapperList *triangleList = m_pScene->getKDTreeStructure()->getTriangleWrapperList();

	/** 
	 *	샘플링갯수가 많으면 iteration 한다. 
	 */
	GAreaDensityEstimate estimate( 500000 );
	totalSize = triangleList->size();

	for ( int i = 0; i < pIPointGridBox->m_iTotalCount; ++i ) {
		pIPointGridBox->m_pData2[ i ].area = 0.0f;
	}

	while( triangleIndex < totalSize ) {

		timer1.start();

		estimate.generateAreaPhoton( triangleList, &triangleIndex, &orderIndex, 
			circleArea / 1000.0f );

		const vector<cuPhoton*> *photons = estimate.getAreaPhoton();
		totalAreaPhoton += (int) photons->size();

		timer1.end();

		GLogManager::logging( LOG_DEBUG,
				"AreaPhoton processed size : %d, current triangle Index = %d, time = %f", 
				photons->size(), triangleIndex, timer1.getElapsedTime() );

		/**
		 *	intersection point 와 연관된 grid 박스를 구성해서 cuda 로 수행.
		 */
		GGridBox<cuPhoton, char> *pPhotonGridBox = makeAreaPhotonGridBox( *photons );
		cuPhotonIndex *pPhotonIndex = makeIPointVsPhotonIndexData( pIPointGridBox, pPhotonGridBox, &indexCount );

		/**
		 *	ipoint 주변에 photon 이 있을때만
		 */
		if ( pPhotonIndex != NULL ) {
			error = m_pCudaPhotonMapping->calDensityArea( 
								pIPointGridBox->m_pData2, pIPointGridBox->m_iTotalCount,
								pPhotonGridBox->m_pData, 
								pPhotonGridBox->m_iTotalCount, 
								pPhotonIndex, indexCount, m_Option.m_fSearchRadius );
			delete pPhotonIndex;
		}

		delete pPhotonGridBox;

		if ( error != errorNo ) {
			GLogManager::logging( LOG_FATAL, "fail to cal density area" );
			return error;
		}
	}

	timer.end();

	/** 
	 *	최종적인 area 를 봐서 area 가 0 이거나 projected circle area 보다 크면 projected circle area 로 만든다. 
	 */
	for ( int i = 0; i < pIPointGridBox->m_iTotalCount; ++i ) {
		if ( pIPointGridBox->m_pData2[ i ].area <= 0.0f ) {
			pIPointGridBox->m_pData2[ i ].area = circleArea;
			zeroAreaCount++;
		} else if ( pIPointGridBox->m_pData2[ i ].area > circleArea ) {
			pIPointGridBox->m_pData2[ i ].area = circleArea;
			overAreaCount++;
		}
	}

	GLogManager::logging( LOG_DEBUG, "totalTriangle = %d, totalProcessedTriangles = %d, total area photon = %d", 
		totalSize, triangleIndex, totalAreaPhoton );
	GLogManager::logging( LOG_DEBUG, "estimateAreaByAreaPhoton time : %f", timer.getElapsedTime() );
	GLogManager::logging( LOG_DEBUG, "totalRay = %d, zeroAreaRayCount = %d, overAreaRayCount = %d", 
		pIPointGridBox->m_iTotalCount, zeroAreaCount, overAreaCount );
	GLogManager::logging( LOG_DEBUG, "--------------------------------- Area Estimate End --------------------------" );

	return errorNo;

}

/**
 *	현재 ray texture 와 area photon 을 이용해서 
 *	Photon Gathering 시 사용할 면적 계산을 수행한 다음에
 *	ray texture 에 area 정보를 갱신하기 위해서 texture 를 다시 올린다.
 */
GError GPhotonMapRayTracer::estimateDensityArea( 
							GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox,
							enumDensityMethod method )
{
	/** 
	 *	area photon 을 사용해서 측정한다. 
	 *	측정한다음에 area 를 갱신하고, cuda 의 ray geometry 를 재로딩시켜야한다.
	 */
	/** 
	 *	만약 search 반경이 0.1 보다 작다면 그냥 circle 로 한다. 
	 */

	if ( m_Option.m_fSearchRadius > 0.1f && method == densityAreaPhoton ) {

		return estimateAreaByAreaPhoton( pIPointGridBox );

	} else {

		if ( m_Option.m_fSearchRadius <= 0.1f ) {
			GLogManager::logging( LOG_WARNING, "search radius is too small. so use projected circle." );
		}

		/**
		 *	area 를 projected circle 로 만듬.
		 */
		for ( int i = 0; i < pIPointGridBox->m_iTotalCount; ++i ) {
			pIPointGridBox->m_pData2[ i ].area =
				3.14159f * m_Option.m_fSearchRadius * m_Option.m_fSearchRadius;
		}

		return errorNo;

	}
}

cuPhotonIndex* GPhotonMapRayTracer::makeIPointVsPhotonIndexData( 
						GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox,
						GGridBox<cuPhoton, char> *pPhotonGridBox, int *pIndexCount )
{
	/**
	 *	각 Ray Cell 주변의 Photon Cell 안에 들어있는 Photon Data
	 *	개수를 계산해온다.
	 */
	CellInfo *rayCellInfo = NULL;
	CellInfo *photonCellInfo = NULL;

	int validRayCellCount = 0;
	int totalNeighborPhotonCellCount = 0;
	int neighborPhotonCellCount = 0;
	int index = 0;

	for ( int z = 0; z < pIPointGridBox->m_iCellZCount; ++z ) {
		for ( int y = 0; y < pIPointGridBox->m_iCellYCount; ++y ) {
			for ( int x = 0; x < pIPointGridBox->m_iCellXCount; ++x ) {

				rayCellInfo = pIPointGridBox->getCellInfo( x, y, z );
				
				/**
				 *	ray 가 존재하는 Cell만.
				 */
				if ( rayCellInfo->dataCount > 0 ) {
					/** 
					 *	photon grid 에서 ray cell 주변의 27 neighbor cell 에 데이터가 있는지를
					 *	체크해서 현재 ray cell 이 찾아야할 주변 cell 이 몇개인지를 계산한다.
					 */
					neighborPhotonCellCount = 0;
					for ( int k = z - 1; k <= z + 1; ++k ) {
						for ( int j = y - 1; j <= y + 1; ++j ) {
							for ( int i = x - 1; i <= x + 1; ++i ) {
								photonCellInfo = pPhotonGridBox->getCellInfo( i, j, k );
								if ( photonCellInfo != NULL && photonCellInfo->dataCount > 0 ) {
									neighborPhotonCellCount++;
								}
							}
						}
					}

					/** 
					 *	현재 ray cell 안에 존재하는 모든 ray 데이터에 이웃 photon cell 의 index 를
					 *	저장할 memory 의 offset 과 index 개수를 기록한다.
					 */
					cuPMIntersectionPoint *pPMPoint = pIPointGridBox->getData2( rayCellInfo );
					for ( int k = 0; k < rayCellInfo->dataCount; ++k ) {
						pPMPoint[ k ].photonIndexOffset = totalNeighborPhotonCellCount;
						pPMPoint[ k ].photonIndexCount = neighborPhotonCellCount;
					}

					totalNeighborPhotonCellCount += neighborPhotonCellCount;
					validRayCellCount++;
				}

			}
		}
	}

	GLogManager::logging( LOG_DEBUG, "RayPhotonIndexCount = %d, ValidRayCellCount = %d", 
		totalNeighborPhotonCellCount, validRayCellCount );

	/**
	 *	Memory 할당.
	 */
	(*pIndexCount) = totalNeighborPhotonCellCount;
	if ( (*pIndexCount) == 0 ) {
		GLogManager::logging( LOG_DEBUG, "there is no photons" );
		return NULL;
	}
	cuPhotonIndex* pPhotonIndex = (cuPhotonIndex*) malloc ( sizeof( cuPhotonIndex ) * (*pIndexCount) );
	memset( pPhotonIndex, 0x00, sizeof( cuPhotonIndex ) * (*pIndexCount) );

	/**
	 *	다시 돌면서 실제로 현재 ray cell 과 연관된 주변 photon cell 정보를 기록한다.
	 */
	for ( int z = 0; z < pIPointGridBox->m_iCellZCount; ++z ) {
		for ( int y = 0; y < pIPointGridBox->m_iCellYCount; ++y ) {
			for ( int x = 0; x < pIPointGridBox->m_iCellXCount; ++x ) {

				rayCellInfo = pIPointGridBox->getCellInfo( x, y, z );
				
				/**
				 *	ray 가 존재하는 Cell만. cell 안의 첫번째 ray 정보를 가져와서
				 *	이 cell 과 연관된 neighbor photon 을 위한 index 를 어디에 저장할지를 결정한다.
				 */
				if ( rayCellInfo->dataCount > 0 ) {
					
					cuPMIntersectionPoint *pPMPoint = pIPointGridBox->getData2( rayCellInfo );

					/** 
					 *	photon grid 에서 ray cell 주변의 27 neighbor cell 에 데이터가 있는지를
					 *	체크해서 현재 ray cell 이 찾아야할 주변 cell 의 photon 에 대한 offset 과 count 를
					 *	index 정보로 저장한다.
					 */
					index = 0;

					for ( int k = z - 1; k <= z + 1; ++k ) {
						for ( int j = y - 1; j <= y + 1; ++j ) {
							for ( int i = x - 1; i <= x + 1; ++i ) {
								photonCellInfo = pPhotonGridBox->getCellInfo( i, j, k );
								if ( photonCellInfo != NULL && photonCellInfo->dataCount > 0 ) {
									pPhotonIndex[ pPMPoint->photonIndexOffset + index ].offset = 
										photonCellInfo->offset;
									pPhotonIndex[ pPMPoint->photonIndexOffset + index ].count = 
										photonCellInfo->dataCount;
									index++;
								}
							}
						}
					}
				}

			}
		}
	}
	
	return pPhotonIndex;
}

/**
 *	photon 데이터를 이용해서 Photon Grid Box 를 만든다.
 *	pIntersectionPoints 정보에서 isHit() 가 아닌것은 제외해야 한다.
 */
GGridBox<cuPhoton, char> *GPhotonMapRayTracer::makePhotonGridBox( cuPhoton *pPhotons, int size )
{
	cuPhoton photonInfo;

	GGridBox<cuPhoton, char> *pPhotonGridBox = new GGridBox<cuPhoton, char> 
		( m_pScene->getKDTreeStructure()->getBoundingBox(), m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength );

	/**
	 *	각 Cell 에 들어갈 photon counting.
	 *	cuPhoton 정보에서 dir 이 모두 0.0f 인것은 쓰레기 photon 이므로 제외.
	 */
	for ( int i = 0; i < size; ++i ) {
		if ( pPhotons[ i ].dir.x == 0.0f && 
			 pPhotons[ i ].dir.y == 0.0f && 
			 pPhotons[ i ].dir.z == 0.0f  ) continue;
		pPhotonGridBox->counting( pPhotons[ i ].pos.x, pPhotons[ i ].pos.y, pPhotons[ i ].pos.z );
	}

	/**
	 *	Grid 정보를 구성하기 위해서 메모리 allocation.
	 */
	pPhotonGridBox->allocate();

	/**
	 *	각 Cell 에 photon insert.
	 */
	for ( int i = 0; i < size; ++i ) {

		if ( pPhotons[ i ].dir.x == 0.0f && 
			 pPhotons[ i ].dir.y == 0.0f && 
			 pPhotons[ i ].dir.z == 0.0f  ) continue;

		photonInfo.pos = pPhotons[ i ].pos;
		photonInfo.dir = pPhotons[ i ].dir;
		photonInfo.power = pPhotons[ i ].power;
		photonInfo.normal = pPhotons[ i ].normal;

		pPhotonGridBox->insertData( photonInfo.pos.x, photonInfo.pos.y, photonInfo.pos.z, &photonInfo, NULL );
	}
	
	return pPhotonGridBox;
}

GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *GPhotonMapRayTracer::getIPointGridBox()
{
	return m_pIPointGridBox;
}

