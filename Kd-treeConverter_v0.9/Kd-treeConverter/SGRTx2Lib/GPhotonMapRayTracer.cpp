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
	 *	�ɼǰ��� 
	 */
	m_Option = (*pOption);
	
	if ( m_Option.m_bDirectIllumByPhotonMap )
		m_Option.m_bSaveDirectPhoton = true;

	m_Option.m_fGridUnitLength = max( m_Option.m_fGridUnitLength, m_Option.m_fSearchRadius );

	m_fSceneLightPowerPerIteration = m_Option.m_fTotalSceneLightPower / (float) m_Option.m_iIteration;
	m_fOnePhotonPower = m_fSceneLightPowerPerIteration / (float) m_Option.m_iEmitPhotonPerIteration;

	/** 
	 *	�ѹ��� iteration ����� �ִ� photon �������. buffer�� ���ؼ� 
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
 *	scene ������ �籸���ؾ��Ҷ� �ʱ�ȭ�Ѵ�.
 */
GError GPhotonMapRayTracer::initialize( GScene *pScene )
{
	GError error;
	m_pScene = pScene;

	/**
	 *	iteration �� 1���� �ƴ϶�� 
	 *	���� rendering �ÿ� ����� photon tracing ������ ������ �� �����Ƿ�
	 *  ������ tracing �� �����ؾ� �Ѵ�.
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

		/**  photon tracing �� �ؾ����� ����. */
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

		/** ray tracing ray �� total photon emit ������ ū�ɷ� ray ������ �Ҵ� */
		int maxRay = cuSceneInfo.iResolutionX * cuSceneInfo.iResolutionY * 
					 cuSceneInfo.iSuperSamplingX * cuSceneInfo.iSuperSamplingY;

		maxRay = max( maxRay, m_iMaxPhotonSize );

		m_pCudaRenderPipeline = new cudaRenderPipeline();
		error = m_pCudaRenderPipeline->initialize( cuSceneInfo, maxRay );
		if ( error != errorNo )
			return error; 

		GLogManager::logging( LOG_DEBUG, "cuPMIntersectionPointcuPMIntersectionPoint %d", sizeof( cuPMIntersectionPoint ) );

		/**
		 *	Light ���� ����. light intensity �� ���� photon �� �󸶳�
		 *	�Ѹ��� ����. ������ �ʿ�����Ƿ� ����.
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
		 *	photon mapping cuda �� �ʱ�ȭ�Ѵ�.
		----------------------------------------------------------------------------------------------*/
		GBoundingBox bbox = pScene->getKDTreeStructure()->getBoundingBox();
		GVector length = bbox.m_Max - bbox.m_Min;

		/**
		 *	scene �� ũ��� ���ڷ� �־��� grid length �� ����
		 *	grid box �� ������ ��û�������� �����Ƿ�
		 *	�ϴ� �ִ�� Grid �ڽ��� �����Ҽ� �ִ� �ּ����� gridLength �� ���ϰ�,
		 *	���ڷ� �־��� gridUnitLength �� radius �� �� �ִ밪�� �Ѵ��� üũ�ؼ� 
		 *	������ grid box �� �����Ѵ�.
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
		 *	light ������ ���ε��Ѵ�.
		 */
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
		
		/**
		 *	intersection point map �ʱ�ȭ.
		 */
		m_pIntersectionPointMap = new GIntersectionPointMap( 
			pScene->getResolution(), pScene->getSuperSampling(), pScene->getMaxReflectionDepth() + 1 );

		/** blooming ȿ���� �����ϱ⸦ ���Ѵٸ� �ʱ�ȭ �ص�.*/
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
	 *	���� Renderer �� ó���� Scene �� ����Ѵ�.
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
	 *	��ü light �߿��� �� light �� intensity �� �����ϴ� ������ŭ 
	 *	�� light �� �Ѹ� photon �� ������ ����Ѵ�. 
	 */
	for ( int i = 0; i < lightCount; ++i ) {
		if ( plightList[ i ].bUsePhoton == 0 )
			continue;
		intensitySum += plightList[ i ].intensity;
		totalPhotonLightCount++;
	}

	/**
	 *	photon �� ���������� �߸��Ƿ� �� light �� emitPhoton ���� ���ڷ� �־��� emitPhoton �� 
	 *	1 ������ ���̰� ����� �ִ�. ���� ������ light ���� ���� photon �� �� �ش�.
	 */
	for ( int i = 0; i < lightCount; ++i ) {

		if ( plightList[ i ].bUsePhoton == 0 )
			continue;

		/** photon �� emit ��Ű�� ������ light �϶� */
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
	 *	Scene �� ���� geometry ���¿��� ���Ѱ� �ִ��� üũ�ؼ� �ִٸ�
	 *	SpatialStructure �� �籸���Ѵ�.
	 */
	error = initialize( pScene );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't initialize rendering" );
		return error;
	}

	/** rendering option ���� */
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
	 *	cuda ���� intersection point �� ����Ʈ ������ �ʱ�ȭ �Ѵ�.
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
	 *	reflection �̳� refraction �� ������ �ִ� max depth ����
	 *	�����ؼ� intersection point �� m_pIntersectionPointMap �� �׾� �ִ´�.
	 */
	int count = 0;

	m_pIntersectionPointMap->clear();

	do {
		error = m_pCudaRenderPipeline->doRayCasting( 0, generatedRayCount, 
							( pScene->getFrontFace() == faceCCW ),
							pScene->isBackFaceCulling() );
		if ( error != errorNo )	break;

		/** 
		 *	intersection ������ �����ͼ� intersection map �� ������ �д�.
		 *	�̰� �ӵ� ���� �ʿ�. �̺κж����� fps �� ������ �ش�.
		 */
		error = backupIntersectionResult( generatedRayCount );
		if ( error != errorNo ) break;

		/**
		 *	photon map ���� direct illumination �� ó���Ѵٸ�
		 *	ray tracing ���� direct illum ó�� ����.
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
	 **	scene �� direct, indirect, �׸��� �ջ��� image buffer �� ����.
	 **----------------------------------------------------------------------------------------------*/

	/**
	 * ray tracing ���� ����� direct illum �� �����ؿ´�.
	 */
	GImageBuffer *pImageBuffer = m_pScene->getImageBuffer();
	GImageBuffer *pDirectIllm = m_pScene->getDirectIllumImageBuffer();
	GImageBuffer *pInDirectIllum = m_pScene->getIndirectIllumImageBuffer();

	/**
	 *	photon mapping ���� indirect illumination �� ����ؿ�.
	 */
	pInDirectIllum->clear();

GLogManager::logging( LOG_INFO, "start photon" );

	error = photonMapIteration( m_bRunTracing, m_Option.m_iIteration, pInDirectIllum );

	if ( error != NULL )
		return errorNo;

	timer.end();

	/** 
	 *	direct �� indirect illum �� �ջ��ؼ� ���� �̹����� �����Ѵ�.
	 */
	/**
	 *	photon map ���� direct illumination �� ó���Ѵٸ�
	 *	ray tracing ���� direct illum ��������.
	 *  indirect illum �� photon map ���� direct ���� ����� ����� �����Ƿ�
	 *	�װ͸� ����.
	 */
	if ( !m_Option.m_bDirectIllumByPhotonMap ) {
		m_pCudaRenderPipeline->getFrameBuffer( pDirectIllm->getBuffer() );
		pImageBuffer->copy( pDirectIllm->getBuffer() );
		pImageBuffer->add( pInDirectIllum );
	} else {
		pImageBuffer->copy( pInDirectIllum->getBuffer() );
	}

	/** blooming ȿ���� �����ϱ⸦ ���Ѵٸ� framebuffer �� �÷��� ó���ϰ� �ٽ� �����´�. */
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
 *	����� ����� image buffer �� ������Ų��.
 *	�ܺο����� ȣ��� �� �����Ƿ�, ���� ���� Ŭ����������
 *	�Ժη� �����ϴ� ������ ���� �ȵȴ�.
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
	*	gpu �󿡼��� intersection ����� hit �Ȱ͸� point map �� insert �Ѵ�.
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
 *	CUDA ���� Primary ray �� ������Ų��.
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
 *	CUDA ���� Primary ray �� ������Ų��.
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
 *	CUDA �� �̿��ؼ� Photon Tracing �� Gathering Iteration �� �����Ѵ�.
 *	����� direct illumination �� �ջ��ؼ� imageBuffer ������Ѵ�.
 */
GError GPhotonMapRayTracer::photonMapIteration( bool bRunTracing, int iteration, 
												GImageBuffer *pInDirectIllumImageBuffer )
{
	GError error;
	GTimer totalTimer, ipointTimer, accumulateTimer, areaTimer;

	totalTimer.start();

	/**-------------------------------------------------------------------------------------------
	 **	��� iteration �� ���õ� ����ڷ� �ʱ�ȭ.
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
	 **	intersection point �� ���� grid ������ �ϰ� cuda �� ���ε�.
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
	 **	density �� �����ϱ� ���ؼ� �� ipoint �� ����� Area �� ���Ѵ�.
	 **------------------------------------------------------------------------------------------*/
	areaTimer.start();

	/** projected circle �Ǵ� area photon ���. */
	error = estimateDensityArea( m_pIPointGridBox, m_Option.m_eDensityMethod );
	if ( error != errorNo )
		return errorNo;
    
	areaTimer.end();

	/**-------------------------------------------------------------------------------------------
	 **	PHOTON TRACING �� GATHERING �� �����Ѵ�. 
	 **------------------------------------------------------------------------------------------*/
	int randomSeed = m_iRandomSeed;

	for ( int i = 0; i < iteration; ++i ) {

		error = photonMapOneIteration( randomSeed, m_pIPointGridBox, bRunTracing, i, iteration );
		if ( error != errorNo )
			return error;

		/**
		 *	������ photon �� �Ѹ����� random ���� ���� seed ��
		 */
		randomSeed += m_Option.m_iEmitPhotonPerIteration;
		if ( randomSeed > 1000000000 )
			randomSeed = 1;

	}

	totalTimer.end();

	/**-------------------------------------------------------------------------------------------
	 **	��ü iteration �� ������ ���� �� ipoint �� radiance �� indirect image buffer ��
	 ** ������Ų��.
	 **------------------------------------------------------------------------------------------*/
	accumulateTimer.start();

	accumulateRadiance( pInDirectIllumImageBuffer, m_pIPointGridBox );

	accumulateTimer.end();

	m_iLogTotalAccumulateTime = accumulateTimer.getElapsedTime();

	/**-------------------------------------------------------------------------------------------
	 **	INFO MATION ���. 
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
 *	CUDA �� �̿��ؼ� Photon Tracing �� Gathering �� Iteration �� �����Ѵ�.
 *	Scene structure �� ������ �ʾҴٸ� tracing �� �ٽ� ������ �ʿ�� ����
 *	������ bRunTracing ���ڷ� �����Ѵ�.
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
	 *	debug option �϶� ���� rendering ���� �߰��� debug photon option �� �����Ѵ�. 
	 */
	if ( m_bDebug && bRunTracing ) {
		char name[1024] = { 0x00, };
		sprintf( name, "DEBUG_PHOTON_MAP_ITERATION_%d", iterationId );
		m_pScene->removeObject( name );
	}

	/**
	 *	Tracing �� ������ �ʿ䰡 ��������. iteration �� 1���̰� 
	 *	scene ������ ������ �ʾҴٸ� ������ tracing �� photon ������
	 *	����Ѵ�. iteration �� �������̶�� �Ź� map �� �ٲ�Ƿ� ������ �� ����.
	 */
	if ( bRunTracing ) {

		tracingTimer.start();

		/**
		*	photon tracing �� �����Ѵ�.
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
		*	�̸� ���ε��س��� intersection point �� tracing �� photon �� ������
		*	intersectionPoint �ֺ��� photon �� gathering �Ѵ�.
		*	���� tracing �� photon �� grid box ���� �����Ѵ�.
		*/
		if ( m_pGlobalPhotonGridBox ) {
			delete m_pGlobalPhotonGridBox;
			m_pGlobalPhotonGridBox = NULL;
		}

		m_pGlobalPhotonGridBox = makePhotonGridBox( m_pHostPhotonMem, iTracedPhotonSize );
		GLogManager::logging( LOG_INFO, " -> Traced Photon Count = %d", m_pGlobalPhotonGridBox->m_iTotalCount );

		photonGridTimer.end();

	}

	/** ipoint �� photon �� �������踸��. */
	indexGridTimer.start();
	int photonIndexCount = 0;
	cuPhotonIndex *pPhotonIndex = makeIPointVsPhotonIndexData( 
						pIPointGridBox, m_pGlobalPhotonGridBox, &photonIndexCount );
	indexGridTimer.end();


	gatheringTimer.start();

	/**
	 *	ipoint �ֺ��� photon �� ��������
	 */
	if ( pPhotonIndex != NULL ) {
		error = m_pCudaPhotonMapping->photonGathering( pIPointGridBox->m_pData2, pIPointGridBox->m_iTotalCount,
								 m_pGlobalPhotonGridBox->m_pData, m_pGlobalPhotonGridBox->m_iTotalCount,
								 pPhotonIndex, photonIndexCount, m_Option.m_fSearchRadius );
		delete pPhotonIndex;
	}

	gatheringTimer.end();

	/** 
	 *	debug option �϶� rendering ���� ������ photon ������ object ���� scene �� �߰��Ѵ�. 
	 *	data �� �ʹ� ������ �ȵǹǷ� iterationid �� 0 �ΰ͸� ����.
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

			// debug ȭ�鼼�� photon ������ �����ٶ�
			// photon �� power �� �ʹ� �۱⶧���� �Ŀ��� Ű���.
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
 *	ray �����͸� �̿��ؼ� Ray Grid Box �� �����.
 */
GGridBox<cuIntersectionPoint, cuPMIntersectionPoint>*
		GPhotonMapRayTracer::makeIPointGridBox( GIntersectionPointMap *pIntersectionPointMap )
{
	GBoundingBox sceneBBox = m_pScene->getKDTreeStructure()->getBoundingBox();

	GGridBox<cuIntersectionPoint, cuPMIntersectionPoint>* pIGridBox = 
				new GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> 
							( sceneBBox, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength );

	/**
	 *	�� Cell �� �� ray counting.
	 */
	const cuIntersectionPoint* pPoint = NULL;
	int pointSize = pIntersectionPointMap->getSize();

	for ( int i = 0; i < pointSize; ++i ) {
		pPoint = pIntersectionPointMap->getIntersectionPoint( i );
		pIGridBox->counting( pPoint->pos.x, pPoint->pos.y, pPoint->pos.z );
	}

	pIGridBox->allocate();

	/**
	 *	�� Cell �� ray ������ insert.
	 */
	int rayNumber = 0;
	cuIntersectionPoint iPoint;
	cuPMIntersectionPoint pmiPoint;

	for ( int i = 0; i < pointSize; ++i ) {

		pPoint = pIntersectionPointMap->getIntersectionPoint( i );
		memcpy( &iPoint, pPoint, sizeof( cuIntersectionPoint ) );

		// power �� 0.0f ���� �� �ʱ�ȭ �ؾ���.
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
 *	area photon �����͸� �̿��ؼ� Area Photon Grid Box �� �����.
 */
GGridBox<cuPhoton, char> *GPhotonMapRayTracer::makeAreaPhotonGridBox( const vector<cuPhoton*> &list )
{
	int photonCount = (int) list.size();
	cuPhoton photonInfo;

	GGridBox<cuPhoton, char> *pAreaPhotonGridBox = new GGridBox<cuPhoton, char> 
			( m_pScene->getKDTreeStructure()->getBoundingBox(), m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength );

	/**
	 *	�� Cell �� �� photon counting.
	 */
	for ( int i = 0; i < photonCount; ++i ) {
		pAreaPhotonGridBox->counting( list[ i ]->pos.x, list[ i ]->pos.y,	list[ i ]->pos.z );
	}
	pAreaPhotonGridBox->allocate();

	/**
	 *	�� Cell �� photon insert.
	 */
	for ( int i = 0; i < photonCount; ++i ) {

		photonInfo.pos = list[ i ]->pos;
		photonInfo.normal = list[ i ]->normal;

		// area ������ �������.
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
	 *	���ø������� ������ iteration �Ѵ�. 
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
		 *	intersection point �� ������ grid �ڽ��� �����ؼ� cuda �� ����.
		 */
		GGridBox<cuPhoton, char> *pPhotonGridBox = makeAreaPhotonGridBox( *photons );
		cuPhotonIndex *pPhotonIndex = makeIPointVsPhotonIndexData( pIPointGridBox, pPhotonGridBox, &indexCount );

		/**
		 *	ipoint �ֺ��� photon �� ��������
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
	 *	�������� area �� ���� area �� 0 �̰ų� projected circle area ���� ũ�� projected circle area �� �����. 
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
 *	���� ray texture �� area photon �� �̿��ؼ� 
 *	Photon Gathering �� ����� ���� ����� ������ ������
 *	ray texture �� area ������ �����ϱ� ���ؼ� texture �� �ٽ� �ø���.
 */
GError GPhotonMapRayTracer::estimateDensityArea( 
							GGridBox<cuIntersectionPoint, cuPMIntersectionPoint> *pIPointGridBox,
							enumDensityMethod method )
{
	/** 
	 *	area photon �� ����ؼ� �����Ѵ�. 
	 *	�����Ѵ����� area �� �����ϰ�, cuda �� ray geometry �� ��ε����Ѿ��Ѵ�.
	 */
	/** 
	 *	���� search �ݰ��� 0.1 ���� �۴ٸ� �׳� circle �� �Ѵ�. 
	 */

	if ( m_Option.m_fSearchRadius > 0.1f && method == densityAreaPhoton ) {

		return estimateAreaByAreaPhoton( pIPointGridBox );

	} else {

		if ( m_Option.m_fSearchRadius <= 0.1f ) {
			GLogManager::logging( LOG_WARNING, "search radius is too small. so use projected circle." );
		}

		/**
		 *	area �� projected circle �� ����.
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
	 *	�� Ray Cell �ֺ��� Photon Cell �ȿ� ����ִ� Photon Data
	 *	������ ����ؿ´�.
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
				 *	ray �� �����ϴ� Cell��.
				 */
				if ( rayCellInfo->dataCount > 0 ) {
					/** 
					 *	photon grid ���� ray cell �ֺ��� 27 neighbor cell �� �����Ͱ� �ִ�����
					 *	üũ�ؼ� ���� ray cell �� ã�ƾ��� �ֺ� cell �� ������� ����Ѵ�.
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
					 *	���� ray cell �ȿ� �����ϴ� ��� ray �����Ϳ� �̿� photon cell �� index ��
					 *	������ memory �� offset �� index ������ ����Ѵ�.
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
	 *	Memory �Ҵ�.
	 */
	(*pIndexCount) = totalNeighborPhotonCellCount;
	if ( (*pIndexCount) == 0 ) {
		GLogManager::logging( LOG_DEBUG, "there is no photons" );
		return NULL;
	}
	cuPhotonIndex* pPhotonIndex = (cuPhotonIndex*) malloc ( sizeof( cuPhotonIndex ) * (*pIndexCount) );
	memset( pPhotonIndex, 0x00, sizeof( cuPhotonIndex ) * (*pIndexCount) );

	/**
	 *	�ٽ� ���鼭 ������ ���� ray cell �� ������ �ֺ� photon cell ������ ����Ѵ�.
	 */
	for ( int z = 0; z < pIPointGridBox->m_iCellZCount; ++z ) {
		for ( int y = 0; y < pIPointGridBox->m_iCellYCount; ++y ) {
			for ( int x = 0; x < pIPointGridBox->m_iCellXCount; ++x ) {

				rayCellInfo = pIPointGridBox->getCellInfo( x, y, z );
				
				/**
				 *	ray �� �����ϴ� Cell��. cell ���� ù��° ray ������ �����ͼ�
				 *	�� cell �� ������ neighbor photon �� ���� index �� ��� ���������� �����Ѵ�.
				 */
				if ( rayCellInfo->dataCount > 0 ) {
					
					cuPMIntersectionPoint *pPMPoint = pIPointGridBox->getData2( rayCellInfo );

					/** 
					 *	photon grid ���� ray cell �ֺ��� 27 neighbor cell �� �����Ͱ� �ִ�����
					 *	üũ�ؼ� ���� ray cell �� ã�ƾ��� �ֺ� cell �� photon �� ���� offset �� count ��
					 *	index ������ �����Ѵ�.
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
 *	photon �����͸� �̿��ؼ� Photon Grid Box �� �����.
 *	pIntersectionPoints �������� isHit() �� �ƴѰ��� �����ؾ� �Ѵ�.
 */
GGridBox<cuPhoton, char> *GPhotonMapRayTracer::makePhotonGridBox( cuPhoton *pPhotons, int size )
{
	cuPhoton photonInfo;

	GGridBox<cuPhoton, char> *pPhotonGridBox = new GGridBox<cuPhoton, char> 
		( m_pScene->getKDTreeStructure()->getBoundingBox(), m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength, m_Option.m_fGridUnitLength );

	/**
	 *	�� Cell �� �� photon counting.
	 *	cuPhoton �������� dir �� ��� 0.0f �ΰ��� ������ photon �̹Ƿ� ����.
	 */
	for ( int i = 0; i < size; ++i ) {
		if ( pPhotons[ i ].dir.x == 0.0f && 
			 pPhotons[ i ].dir.y == 0.0f && 
			 pPhotons[ i ].dir.z == 0.0f  ) continue;
		pPhotonGridBox->counting( pPhotons[ i ].pos.x, pPhotons[ i ].pos.y, pPhotons[ i ].pos.z );
	}

	/**
	 *	Grid ������ �����ϱ� ���ؼ� �޸� allocation.
	 */
	pPhotonGridBox->allocate();

	/**
	 *	�� Cell �� photon insert.
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

