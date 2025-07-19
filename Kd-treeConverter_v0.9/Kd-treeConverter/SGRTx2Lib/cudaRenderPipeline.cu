/**
 *	Cuda �� Rendering �� �����ϱ� ���ؼ�
 *	���������� �����ϴ� class.
 *
 *	light, texture, shading, ray tracing, photon mapping
 *	���.
 *	
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
//#include <cutil.h>
#include <cuda_runtime.h>

#include "cudaRenderPipeline.cuh"
#include "cudaRenderPipelineKernel.cu"
#include "cudaPhotonMapping.cu"

#pragma comment(lib, "cudart.lib")
//#pragma comment(lib, "cutil32.lib")
#include <GL/freeglut.h> 

#define GENERAL_THREAD_COUNT	128

cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

/**
 *	Adaptive Sampling �� 4������ ���ø����θ� üũ�Ҷ� ����� �ֺ� �ȼ��� ���������� ���� index.
 *	(x,y) ������ ��3���� �� 24��
 */
static int g_staticPatternData[] = { 
		-1, 0, -1, -1, 0, -1,			// left-top corner.
		0, -1, 1, -1, 1, 0,				// right-top corner.
		-1, 0, -1, 1, 0, 1,				// left-bottom corner.
		1, 0, 1, 1, 0, 1,				// right-bottom corner.
};

/**
 *	�� ������ pixel index �� �ش�Ǵ� pixel weight. �ϳ��� sub-pixel �� ĥ�� ����
 *	interpolation �Ҷ� ����� ��. �� g_staticPatternData �� ����Ű�� index �������
 *	weight �� �����Ǿ� �־�� �Ѵ�.
 */
static float g_staticPixelColorWeight[] = {
		0.1875f, 0.0625f, 0.1875f,
		0.1875f, 0.0625f, 0.1875f,
		0.1875f, 0.0625f, 0.1875f,
		0.1875f, 0.0625f, 0.1875f
};


cudaRenderPipeline::cudaRenderPipeline()
{
	m_pHostIntersectionPoint = NULL;
	m_pDeviceRays = NULL;
	m_pDeviceIntersectionPoint = NULL;
	m_pDeviceKDTreeNodes = NULL;
	m_pDeviceTriangleOffsetList = NULL;
	m_pDeviceWaldTriangleInfo = NULL;
	m_pDevicePlueckerTriangleInfo = NULL;
	m_pDeviceTriangleGeometry = NULL;
	m_pDeviceObjectMaterial = NULL;
	m_pDeviceSamplingMap = NULL;
	m_pDeviceIntResult = NULL;
	m_pDeviceFrameBuffer = NULL;
	m_pDeviceFrameBuffer2 = NULL;
	m_pDeviceASBuffer = NULL;
	m_pDeviceTextureData = NULL;
	m_pDeviceLightRaySetData = NULL;
	m_pDeviceBloomingFilter = NULL;

	m_iTotalTextureWidth = 0;
	m_iTotalTextureHeight = 0;
	m_iBloomingWidth = 0;
	m_fBloomingWeight = 0.0f;
}

cudaRenderPipeline::~cudaRenderPipeline()
{
	uninitialize();
}

/**
 *	�ʱ�ȭ. GScene ���� ���� sceneInfo �� �����ϰ�
 *	KDTree �� ���� data �� �����Ѵ�.
 */
GError cudaRenderPipeline::initialize( cuScene pScene, int maxray )
{
	GError error;
	
	int argc = 1;	char *argv[] ={"init"};
	//CUT_DEVICE_INIT(argc, argv);
	glutInit(&argc, argv);
	//glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	//glutInitWindowSize(800, 600);         // �ʿ信 ���� ����
	//glutCreateWindow("SGRTx2 Viewer");
	
	if ( ( error = setSceneInfo( pScene ) ) != errorNo )
		return error;

	/**
	 *	ray �� intersection point �� ���� ���� �Ҵ�.
	 */	
	error = initRayIntersection( maxray );
	if ( error != errorNo )
		return error;

	/**
	 *	Intersection check �� ���� Stack size ����.
	 */
	unsigned int depth = SHORT_STACK_DEPTH;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( shortStackDepth, &depth, sizeof( unsigned int ) ) );
	if ( checkError( "shortStackDepth" ) != cudaSuccess )
		return errorCudaError;
	
	return errorNo;
}

GError cudaRenderPipeline::setSceneInfo( cuScene pScene )
{
	m_SceneInfo = pScene;

	/**
	 *	Scene ������ constant �� �ø�.
	 */
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( g_SceneInfo, &m_SceneInfo, sizeof( cuScene ) ) );
	if ( checkError( "SceneInfo Upload" ) != cudaSuccess )
		return errorCudaError;

	return errorNo;
}

GError cudaRenderPipeline::setThresholdInfo( cuThreshold threshold )
{
	m_ThresholdInfo = threshold;

	/**
	 *	Scene ������ constant �� �ø�.
	 */
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( g_ThresholdInfo, &m_ThresholdInfo, sizeof( cuThreshold ) ) );
	if ( checkError( "ThresholdInfo Upload" ) != cudaSuccess )
		return errorCudaError;

	return errorNo;
}


void cudaRenderPipeline::swapFrameBuffer()
{
	float *temp = m_pDeviceFrameBuffer;
	m_pDeviceFrameBuffer = m_pDeviceFrameBuffer2;
	m_pDeviceFrameBuffer2 = temp;

	/** texture �� �޸��ּҵ� �ٲپ�� �Ѵ�. */

	CUDA_SAFE_CALL( cudaUnbindTexture( inFrameBufferTexture ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, inFrameBufferTexture, m_pDeviceFrameBuffer ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( inFrameBuffer2Texture ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, inFrameBuffer2Texture, m_pDeviceFrameBuffer2 ) );

}

GError cudaRenderPipeline::renderingOption( bool shadow, bool texture )
{
	m_SceneInfo.bEnableShadow = shadow;
	m_SceneInfo.bEnableTexture = texture;
	
	/**
	 *	Scene ������ constant �� �ø�.
	 */
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( g_SceneInfo, &m_SceneInfo, sizeof( cuScene ) ) );
	if ( checkError( "SceneInfo Upload" ) != cudaSuccess )
		return errorCudaError;
	
	return errorNo;
}

/**
 *	Light RaySet Data �� Device �� �ø���.
 */
GError cudaRenderPipeline::setLightRaySetData( cuLightRaySet *pRaySet, int count )
{
	if ( m_pDeviceLightRaySetData ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inLightRaySetTexture ) );
		CUDA_SAFE_CALL( cudaFree( m_pDeviceLightRaySetData ) );
	}

	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceLightRaySetData, 
								sizeof( cuLightRaySet ) * count ) );
	CUDA_SAFE_CALL( cudaMemcpy( m_pDeviceLightRaySetData, pRaySet,
								sizeof( cuLightRaySet ) * count,
								cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, inLightRaySetTexture, m_pDeviceLightRaySetData ) );

	if ( checkError( "setLightRaySetData" ) != cudaSuccess )
		return errorCudaError;
	
	GLogManager::logging( LOG_INFO, " -> Light Ray Set Data Uploaded." );

	return errorNo;
}

GError cudaRenderPipeline::initRayIntersection( int maxray )
{
	m_iMaxRay = maxray;
	m_iMaxIntersectionPoint = maxray;
	m_iImagePixelCount = m_SceneInfo.iResolutionX *  m_SceneInfo.iResolutionY;
	
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceRays, sizeof( cuRay ) *  m_iMaxRay ) );
	if ( checkError( "cudaMalloc::m_pDeviceRays" ) != cudaSuccess )
		return errorCudaError;

	size_t rayDataSize = sizeof(float4) * maxray * 2;
	CUDA_SAFE_CALL( cudaBindTexture( 0, inRayListTex,  m_pDeviceRays, rayDataSize) );
	if ( checkError( "cudaBindTexture::inRayListTex" ) != cudaSuccess )
		return errorCudaError;
	
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceIntersectionPoint,
					sizeof( cuIntersectionPoint ) *  m_iMaxIntersectionPoint ) );

	/** 
	 *	host, device intersection buffer
	 */
	m_pHostIntersectionPoint = (cuIntersectionPoint*) 
					malloc( sizeof( cuIntersectionPoint ) * m_iMaxIntersectionPoint );

	/**
	 *	int result 4���� ���� ����.
	 */									
	CUDA_SAFE_CALL( cudaMalloc( (void**) & m_pDeviceIntResult, sizeof( int ) * 4 ) );

	/**
	 *	frame buffer. rgb �̹Ƿ� float * 3
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceFrameBuffer, 
								sizeof( float ) * 3 * m_iImagePixelCount ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, inFrameBufferTexture, m_pDeviceFrameBuffer ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceFrameBuffer2, 
								sizeof( float ) * 3 * m_iImagePixelCount ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, inFrameBuffer2Texture, m_pDeviceFrameBuffer2 ) );

	/**
	 *	Adaptive Sampling �� ���� AS-Buffer. �� pixel �� �ִ� 4���� sub-pixel �� ����� �ְ�
	 *	�ش� ������ float �ϳ��� ����Ѵ�. �׸��� padding subpixel ���� ���� �� �����Ƿ�
	 *	�ϹǷ� �̹����ػ��� * 5 ��size �� ��´�.
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceASBuffer, sizeof( int ) * 5 * m_iImagePixelCount ) );
	if ( checkError( "cudaRenderPipeline::initRayIntersection" ) != cudaSuccess )
		return errorCudaError;
	CUDA_SAFE_CALL( cudaBindTexture( 0, inASBufferTexture, m_pDeviceASBuffer ) );

	/**
	 *	Pattern Data�� constant �� ���ε�.
	 */
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( constantIndexTablePattern, g_staticPatternData, sizeof( int ) * 24 ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( constantPixelWeight, g_staticPixelColorWeight, sizeof( float ) * 12 ) );
	
	/**
	 * sampling map
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceSamplingMap, sizeof( cuSamplingMap ) * m_iImagePixelCount ) );
	if ( checkError( "cudaRenderPipeline::initRayIntersection" ) != cudaSuccess )
		return errorCudaError;

	CUDA_SAFE_CALL( cudaBindTexture( 0, inSamplingMapTexture, m_pDeviceSamplingMap ) );

		
	return errorNo;
}

/**
 *	blooming �� ���� ����� �ʱ�ȭ �Ѵ�.
 */
GError cudaRenderPipeline::initBloomingFilter( float radius, float weight )
{
	if ( m_pDeviceBloomingFilter != NULL ) {
		CUDA_SAFE_CALL( cudaFree( m_pDeviceBloomingFilter ) );
	}

	float *bloomFilter = NULL;
	int bloomSupport = (int)( radius * max( m_SceneInfo.iResolutionX , m_SceneInfo.iResolutionY ) );
	float dist;

	m_iBloomingWidth = bloomSupport / 2;
	m_fBloomingWeight = weight;

	if ( m_iBloomingWidth <= 1 ) {
		GLogManager::logging( LOG_FATAL, "m_iBloomingWidth is error." );
		return errorCudaError;
	}
	
	if ( m_iBloomingWidth >= 50 ) {
		m_iBloomingWidth = 50;
	}

	bloomFilter = (float *) malloc( m_iBloomingWidth * m_iBloomingWidth * sizeof( float ) );
	for ( int i = 0; i < m_iBloomingWidth * m_iBloomingWidth; ++i ) {
		dist = (float) sqrt( (float)( i ) ) / (float)( m_iBloomingWidth );
		bloomFilter[i] = (float) pow( max( 0.f, 1.f - dist ), 4.f );
	}

	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceBloomingFilter, 
								m_iBloomingWidth * m_iBloomingWidth * sizeof( float ) ) );
	CUDA_SAFE_CALL( cudaMemcpy( m_pDeviceBloomingFilter, bloomFilter, 
								m_iBloomingWidth * m_iBloomingWidth * sizeof( float ),
								cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, inBloomingFilterTexture, m_pDeviceBloomingFilter ) );

	free( bloomFilter );

	if ( checkError( "m_pDeviceBloomingFilter malloc" ) != cudaSuccess )
		return errorCudaError;

	return errorNo;
	
}


void cudaRenderPipeline::uninitialize()
{
	if ( m_pHostIntersectionPoint )
		free( m_pHostIntersectionPoint );
		
	if ( m_pDeviceIntersectionPoint ) {
		CUDA_SAFE_CALL( cudaFree(  m_pDeviceIntersectionPoint ) );
	}

	if ( m_pDeviceRays ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inRayListTex ) );
		CUDA_SAFE_CALL( cudaFree( m_pDeviceRays ) );
	}

	if ( m_pDeviceKDTreeNodes ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inKdTreeNodeTex ) );
		CUDA_SAFE_CALL( cudaFree(  m_pDeviceKDTreeNodes ) );
	}

	if ( m_pDeviceTriangleOffsetList ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inObjectOffsetListTex ) );
		CUDA_SAFE_CALL( cudaFree(  m_pDeviceTriangleOffsetList ) );
	}

	if ( m_pDeviceWaldTriangleInfo ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inWaldTriangleTex ) );
		CUDA_SAFE_CALL( cudaFree(  m_pDeviceWaldTriangleInfo ) );
	}

	if ( m_pDevicePlueckerTriangleInfo ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inPlueckerTriangleTex ) );
		CUDA_SAFE_CALL( cudaFree(  m_pDevicePlueckerTriangleInfo ) );
	}

	if ( m_pDeviceTriangleGeometry ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inTriangleGeometryTex ) );
		CUDA_SAFE_CALL( cudaFree(  m_pDeviceTriangleGeometry ) );
	}

	if ( m_pDeviceObjectMaterial ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inObjectMaterialTex ) );
		CUDA_SAFE_CALL( cudaFree(  m_pDeviceObjectMaterial ) );
	}

	if ( m_pDeviceSamplingMap ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inSamplingMapTexture ) );
		CUDA_SAFE_CALL( cudaFree(  m_pDeviceSamplingMap ) );
	}

	if ( m_pDeviceFrameBuffer ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inFrameBufferTexture ) );
		CUDA_SAFE_CALL( cudaFree( m_pDeviceFrameBuffer ) );
	}
	if ( m_pDeviceFrameBuffer2 ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inFrameBuffer2Texture ) );
		CUDA_SAFE_CALL( cudaFree( m_pDeviceFrameBuffer2 ) );
	}
	if ( m_pDeviceASBuffer ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inASBufferTexture ) );
		CUDA_SAFE_CALL( cudaFree( m_pDeviceASBuffer ) );
	}

	if ( m_pDeviceIntResult )
		CUDA_SAFE_CALL( cudaFree( m_pDeviceIntResult ) );
	
	if ( m_pDeviceTextureData ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inObjectTexture ) );
		CUDA_SAFE_CALL( cudaFreeArray( m_pDeviceTextureData ) );
	}
	if ( m_pDeviceLightRaySetData ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inLightRaySetTexture ) );
		CUDA_SAFE_CALL( cudaFree( m_pDeviceLightRaySetData ) );
	}
	if ( m_pDeviceBloomingFilter ) {
		CUDA_SAFE_CALL( cudaUnbindTexture( inBloomingFilterTexture ) );
		CUDA_SAFE_CALL( cudaFree( m_pDeviceBloomingFilter ) );
	}
	
	checkError( "cudaRenderPipeline::uninitialize" );
}

GError cudaRenderPipeline::setLightInfo( cuLight *pLight, int count )
{
	if ( count > CUDA_MAX_LIGHT ) {
		return errorCudaError;
	}
	
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( constantLightInfo, pLight, sizeof( cuLight ) * count ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( constantLightCount, &count, sizeof( int ) ) );
	if ( checkError( "setLightInfo" ) != cudaSuccess )
		return errorCudaError;
			
	return errorNo;	
}

int cudaRenderPipeline::getDeviceRayBufferSize()
{
	return m_iMaxRay;
}

cuRay *cudaRenderPipeline::getDeviceRayBuffer()
{
	return m_pDeviceRays;
}

cuIntersectionPoint* cudaRenderPipeline::getDeviceIntersectionBuffer()
{
	return m_pDeviceIntersectionPoint;
}

/**
 *	���� Texture �� cuda ���� �������� ������ �Ұ��� �ϹǷ�.
 *	texture �ϳ��� ��� texture �� �� ��� �ø��� ���������� ó���Ѵ�.
 *	���ڷ� �־� texture ���� width ���� padding �� ���ٴ� ����.
 */
GError cudaRenderPipeline::setTextureData( cuTexture* pTextureData, int count )
{
	GError error = errorNo;
	
	int totalWidth = -100, totalHeight = 0;
	int widthOffset = 0, heightOffset = 0;
	if ( count > CUDA_MAX_TEXTURE ) {
		GLogManager::logging( LOG_ERROR, "too many texture. max texture=%d", CUDA_MAX_TEXTURE );
		return errorCudaError;
	}

	cuTextureRef *pTextureRef = new cuTextureRef[ count ];
	unsigned char *pTempTexture = NULL;
	
	/** TODO: ȿ�������� texture size �����. �ϴ��� �����ϰ� */
	/** ��ü texture �߿��� width �� ���� ū���� ã�´�. height �� ��ü��. */
	for ( int i = 0; i < count; ++i ) {
		totalWidth = max( totalWidth, pTextureData[ i ].width );
		totalHeight += pTextureData[ i ].height;
	}
	
	m_iTotalTextureWidth = totalWidth;
	m_iTotalTextureHeight = totalHeight;
	
	pTempTexture = (unsigned char*) malloc( sizeof( unsigned char ) * 4 * totalWidth * totalHeight );
	
	/** pTempTexture �ȿ� texture ���� ��ġ�Ѵ�. */
	/** 
	 *	�ϴ��� �����ϰ� row ������. 
	 *	���� totalTexture �� �� �ȿ� ¤��������� texture �� �ػ󵵰� Ʋ���Ƿ�
	 *	�� texture �� �����Ҷ��� �������� �����ؾ� �Ѵ�. 
	 */
	heightOffset = 0;
	for ( int i = 0; i < count; ++i ) {
		for ( int j = 0; j < pTextureData[ i ].height; ++j ) {
			memcpy( pTempTexture + ( heightOffset + j ) * totalWidth * 4 + widthOffset * 4, 
					pTextureData[ i ].pData + ( j * pTextureData[ i ].width * 4 ), 
					sizeof( unsigned char ) * 4 * pTextureData[ i ].width );
		}

		/** texture ref info ���� ���� */
		pTextureRef[ i ].x = widthOffset;
		pTextureRef[ i ].y = heightOffset;
		pTextureRef[ i ].width = pTextureData[ i ].width;
		pTextureRef[ i ].height = pTextureData[ i ].height;

		heightOffset += pTextureData[ i ].height;
	}
	
	GLogManager::logging( LOG_INFO, 
		"Texture Collection Size : %d x %d, texture count=%d", totalWidth, totalHeight, count );
	
	/**
	 *	CUDA Texture �� �ø���.
	 */
	CUDA_SAFE_CALL( cudaMallocArray( &m_pDeviceTextureData, &uchar4tex, totalWidth, totalHeight ) );
						 
	if ( checkError( "texture cudaMallocArray" ) != cudaSuccess ) {
		return errorCudaError;
	}

	CUDA_SAFE_CALL( cudaMemcpyToArray( m_pDeviceTextureData, 0, 0,
									   pTempTexture,
									   sizeof( unsigned char ) * 4 * totalWidth * totalHeight,
									   cudaMemcpyHostToDevice ) );

	//inObjectTexture.addressMode[ 0 ] = cudaAddressModeWrap;
	//inObjectTexture.addressMode[ 1 ] = cudaAddressModeWrap;
	inObjectTexture.filterMode = cudaFilterModeLinear;
	
	/** ������ǥ�� ���ٽ��Ѿ� �ϱ� ������ normalized = false */
	inObjectTexture.normalized = false;

	CUDA_SAFE_CALL( cudaBindTextureToArray( inObjectTexture, m_pDeviceTextureData ) );
	if ( checkError( "setTextureData" ) != cudaSuccess ) {
		error = errorCudaError;
		goto end;
	}
	
	/**
	 *	Texture info �� constant ������ �ѱ��.
	 */
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( g_TextureRefInfo, pTextureRef, sizeof( cuTextureRef ) * count ) );
	if ( checkError( "g_TextureRefInfo g_TextureRefCount" ) != cudaSuccess ) {
		error = errorCudaError;
		goto end;
	}

	CUDA_SAFE_CALL( cudaMemcpyToSymbol( g_TextureRefCount, &count, sizeof( int ) ) );
	if ( checkError( "g_TextureRefInfo g_TextureRefCount" ) != cudaSuccess ) {
		error = errorCudaError;
		goto end;
	}
		
	error = errorNo;
	
end:
	
	delete[] pTextureRef;
	free( pTempTexture );

	return error;	
}

GError cudaRenderPipeline::setCameraInfo( cuCamera *pCamera )
{
	/**
	 *	Camera ������ constant �� �ѱ��.
	 */
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( g_CameraInfo, pCamera, sizeof( cuCamera ) ) );
	cudaError_t error = checkError( "g_CameraInfo" );	
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}
	return errorNo;
}

GError cudaRenderPipeline::setObjectMaterial( cuObjectMaterial *pObjectMaterial, int count )
{
	 m_iObjectMaterialCount = count;

	/**
	 *	�ﰢ���� ���Ե� object �� material ����.
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) & m_pDeviceObjectMaterial, 
						sizeof( cuObjectMaterial ) * count ) );
	CUDA_SAFE_CALL( cudaMemcpy(  m_pDeviceObjectMaterial, pObjectMaterial, 
								sizeof( cuObjectMaterial ) * count, 
								cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, inObjectMaterialTex,  m_pDeviceObjectMaterial ) );

	if ( checkError( "setGeometryInfo pObjectMaterial" ) != cudaSuccess )
		return errorCudaError;
		
	return errorNo;
}

GError cudaRenderPipeline::setKDTreeNodeData( kdtreeNode *pKDTreeNodes, int nodeCount, cuBoundingBox sceneBox )
{
	 m_iNodeCount = nodeCount;

	/**
	 *	KD Tree Node Data.
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceKDTreeNodes, sizeof( kdtreeNode ) * nodeCount ) );
	if ( checkError( "setKDTreeData" ) != cudaSuccess )
		return errorCudaError;

	CUDA_SAFE_CALL( cudaMemcpy(  m_pDeviceKDTreeNodes, pKDTreeNodes, 
					sizeof( kdtreeNode ) * nodeCount, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture(0, inKdTreeNodeTex,  m_pDeviceKDTreeNodes ) );
	if ( checkError( "setKDTreeData" ) != cudaSuccess )
		return errorCudaError;

	/**
	 *	BBox ������ constant �� �ѱ��.
	 */
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( g_SceneBBox, &sceneBox, sizeof( cuBoundingBox ) ) );
	cudaError_t error = checkError( "g_SceneBBox" );	
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}


	return errorNo;
}

GError cudaRenderPipeline::setTriangleOffsetList( unsigned int *pTriangleOffsetList, int offsetCount )
{
	 m_iTriangleOffsetCount = offsetCount;

	/**
	 *	Triangle Offset List
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceTriangleOffsetList, sizeof( unsigned int ) * offsetCount ) );
	CUDA_SAFE_CALL( cudaMemcpy(  m_pDeviceTriangleOffsetList, pTriangleOffsetList, 
						sizeof( unsigned int ) * offsetCount, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture(0, inObjectOffsetListTex,  m_pDeviceTriangleOffsetList ) );
	if ( checkError( "m_pDeviceTriangleOffsetList" ) != cudaSuccess )
		return errorCudaError;

	return errorNo;
}

GError cudaRenderPipeline::setTriangleGeometry( cuTriangleGeometry *pTriangleGeometry, int triangleCount )
{
	m_iTriangleCount = triangleCount;

	/**
	 *	�ﰢ�� geometry ����.
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceTriangleGeometry, 
								sizeof( cuTriangleGeometry ) * triangleCount ) );
	if ( checkError( "pDeviceRayTracingContext.kdtreeNodes1" ) != cudaSuccess )
		return errorCudaError;
		
	CUDA_SAFE_CALL( cudaMemcpy( m_pDeviceTriangleGeometry, pTriangleGeometry, 
								sizeof( cuTriangleGeometry ) * triangleCount, 
								cudaMemcpyHostToDevice ) );
	
	if ( checkError( "setTriangleGeometry" ) != cudaSuccess )
		return errorCudaError;
		
	CUDA_SAFE_CALL( cudaBindTexture( 0, inTriangleGeometryTex,  m_pDeviceTriangleGeometry ) );
	if ( checkError( "setTriangleGeometry" ) != cudaSuccess )
		return errorCudaError;

	return errorNo;
}

GError cudaRenderPipeline::setPlueckerTriangleInfo( cuPlueckerTriangleInfo *pTriangleInfo, int triangleCount )
{
	/**
	 *	�ﰢ�� intersection üũ�� ���� ����.
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) & m_pDevicePlueckerTriangleInfo, 
								sizeof( cuPlueckerTriangleInfo ) * triangleCount ) );
	CUDA_SAFE_CALL( cudaMemcpy(  m_pDevicePlueckerTriangleInfo, pTriangleInfo, 
								sizeof( cuPlueckerTriangleInfo ) * triangleCount, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture(0, inPlueckerTriangleTex,  m_pDevicePlueckerTriangleInfo ) );
	if ( checkError( "setPlueckerTriangleInfo" ) != cudaSuccess )
		return errorCudaError;

	return errorNo;
}

GError cudaRenderPipeline::setWaldTriangleInfo( cuWaldTriangleInfo *pTriangleInfo, int triangleCount )
{
	/**
	 *	�ﰢ�� intersection üũ�� ���� ����.
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) & m_pDeviceWaldTriangleInfo, 
								sizeof( cuWaldTriangleInfo ) * triangleCount ) );
	CUDA_SAFE_CALL( cudaMemcpy(  m_pDeviceWaldTriangleInfo, pTriangleInfo, 
								sizeof( cuWaldTriangleInfo ) * triangleCount, cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture(0, inWaldTriangleTex,  m_pDeviceWaldTriangleInfo ) );
	if ( checkError( "setWaldTriangleInfo" ) != cudaSuccess )
		return errorCudaError;

	return errorNo;
}

/**
 *	CPU ���� ray ������ �����Ҷ� ���.
 */
GError cudaRenderPipeline::setRayInfo( cuRay *pRays, int destOffset, int count )
{
	if ( m_pDeviceRays == NULL ) {
		GLogManager::logging( LOG_ERROR, "ray memory not allocated. " );
		return errorCudaError;
	}
	
	if ( count >  m_iMaxRay ) {
		GLogManager::logging( LOG_ERROR, "Too many ray data. %d", count );
		return errorCudaError;
	}
	
	CUDA_SAFE_CALL( cudaMemcpy(  m_pDeviceRays + destOffset, pRays, 
						sizeof( cuRay ) * count, cudaMemcpyHostToDevice ) );
	if ( checkError( "setRayInfo" ) != cudaSuccess )
		return errorCudaError;
	
	return errorNo;
}

/**
 *	Intersection ����� �����ϴ� device buffer �� clear �Ѵ�.
 */
GError cudaRenderPipeline::clearIntersectionResult()
{
	if ( m_pDeviceIntersectionPoint == NULL ) {
		GLogManager::logging( LOG_ERROR, "intersection point memory not allocated. " );
		return errorCudaError;
	}
		
	/** 
	 *	hit �������� �˸��� objIndex �� ���� -1 �̹Ƿ� -1 �� �ʱ�ȭ �Ѵ�. 
	 */
	CUDA_SAFE_CALL( cudaMemset(  m_pDeviceIntersectionPoint, -1, 
					sizeof( cuIntersectionPoint ) *  m_iMaxIntersectionPoint ) );
	if ( checkError( "clearIntersectionResult" ) != cudaSuccess )
		return errorCudaError;
		
	return errorNo;
}

/**
 *	���� �޸𸮻� �ִ� intersection point ���� ����ؼ� Shading �� �����Ų��.
 */
GError cudaRenderPipeline::calDirectIllumination( int maxReflectionDepth )
{
	int imagePixelNum =  m_iImagePixelCount;
	int startImageIndex = 0;

	if ( m_pDeviceIntersectionPoint == NULL || m_pDeviceFrameBuffer == NULL ) {
		GLogManager::logging( LOG_ERROR, "intersection point memory not allocated. " );
		return errorCudaError;
	}

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	blocks.x = m_SceneInfo.iResolutionX / threads.x;
	blocks.y = m_SceneInfo.iResolutionY / threads.y;

	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	/**
	 *	shading ����.
	 */
	shadingKernel<<< blocks, threads, 
				sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2 >>>
		( startImageIndex,  imagePixelNum, m_pDeviceIntersectionPoint,  m_pDeviceFrameBuffer, maxReflectionDepth );
			
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	cudaError_t error = checkError( "cudaShading" );
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
	
}


void cudaRenderPipeline::clearFrameBuffer( )
{
	int imagePixelNum =  m_iImagePixelCount;
	CUDA_SAFE_CALL( cudaMemset(  m_pDeviceFrameBuffer, 0x00, sizeof( float ) * 3 * imagePixelNum ) );	
}

void cudaRenderPipeline::getFrameBuffer( float* pBuffer )
{	
	int imagePixelNum =  m_iImagePixelCount;
	CUDA_SAFE_CALL( cudaMemcpy( pBuffer,  m_pDeviceFrameBuffer, 
								sizeof( float ) * 3 * imagePixelNum, 
								cudaMemcpyDeviceToHost ) );	
}
void cudaRenderPipeline::setFrameBuffer( float* pBuffer )
{	
	int imagePixelNum =  m_iImagePixelCount;
	CUDA_SAFE_CALL( cudaMemcpy( m_pDeviceFrameBuffer, pBuffer, 
								sizeof( float ) * 3 * imagePixelNum, 
								cudaMemcpyHostToDevice ) );	
}

/**
 *	primary ray �� ������Ų��. ������ ray ������ device �޸𸮿�
 *	����ȴ�. ī�޶� ������ constant �� �ø���.
 */
GError cudaRenderPipeline::generatePrimaryRay( cuCamera camera, int *pGeneratedCount, 
											   int currentSampleX, int currentSampleY,
											   bool bJittering )
{
	int rayNum =  m_iImagePixelCount;
				  
	int startRayIndex = 0;
	
	if ( m_pDeviceRays == NULL || m_iMaxRay < rayNum ) {
		GLogManager::logging( LOG_ERROR, "cudaGeneratePrimaryRay overflow max ray ( %d > %d )",
										 rayNum,  m_iMaxRay );
		return errorCudaGeneratePrimaryRay;
	}
	
	setCameraInfo( &camera );
	
	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	blocks.x = m_SceneInfo.iResolutionX / threads.x;
	blocks.y = m_SceneInfo.iResolutionY / threads.y;

	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	/**
	 *	primary ����.
	 */
	generatePrimaryRayKernel<<< blocks, threads >>> ( 
		startRayIndex, rayNum, m_pDeviceRays,  m_pDeviceIntersectionPoint, 
		currentSampleX, currentSampleY, bJittering );	
	
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

	if ( checkError( "cudaGeneratePrimaryRay" ) != cudaSuccess ) {
		return errorCudaGeneratePrimaryRay;
	}
	
	/** 
	 *	cuda ���� ������ ray ���� ���� 
	 */
	 (*pGeneratedCount) = rayNum;
	
	return errorNo;

}

/**
 *	intersection point �� üũ�ؼ� secondary reflection ray �� ������Ų��. 
 *	intersection point �� ��� ���� �������� üũ�ؼ� second ray �� ����������
 *	startOffset �� count.
 *
 *	������ ray ������ device �޸𸮿� ����ȴ�. 
 *	
 */
GError cudaRenderPipeline::generateReflectionRay(  int startOffset, int count, int *atLeastOneRay )
{
	int checkCount = count;
	int startIntersectionIndex = startOffset;
	
	CUDA_SAFE_CALL( cudaMemset( m_pDeviceIntResult, 0x00, sizeof( int ) ) );	
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	
	if (  m_iMaxRay < startOffset + checkCount ) {
		GLogManager::logging( LOG_ERROR, "cudaGenerateReflectionRay overflow max ray ( %d > %d )",
										( startOffset + checkCount ),  m_iMaxRay );
		return errorCudaGeneratePrimaryRay;
	}

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	blocks.x = m_SceneInfo.iResolutionX / threads.x;
	blocks.y = m_SceneInfo.iResolutionY / threads.y;

	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	generateReflectionRayKernel<<< blocks, threads >>> ( 
					startIntersectionIndex,  checkCount, m_pDeviceIntersectionPoint,
					m_pDeviceRays,  m_pDeviceIntResult );	
	
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

	cudaError_t error = checkError( "cudaGeneratePrimaryRay" );
	if ( error != cudaSuccess ) {
		return errorCudaGeneratePrimaryRay;
	}

	CUDA_SAFE_CALL( cudaMemcpy( atLeastOneRay, 
								m_pDeviceIntResult, 
								sizeof( int ), cudaMemcpyDeviceToHost ) );	
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	error = checkError( "copy atLeastOneRay" );
	if ( error != cudaSuccess ) {
		return errorCudaGeneratePrimaryRay;
	}
	
	return errorNo;

}

/**
 *	Anti-aliasing �� ���� filter. pixel �� 3x3 filter �� ���Ƿ� ������
 *	block size �� 12 x .. ���·� ������.
 */
GError cudaRenderPipeline::antialiasingFiltering()
{
	dim3 threads( 16, 8 );
	dim3 blocks( m_SceneInfo.iResolutionX / threads.x, m_SceneInfo.iResolutionY / threads.y );
	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	antialiasingFilteringKernel<<< blocks, threads >>> ( m_pDeviceFrameBuffer2 );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
	swapFrameBuffer();

	cudaError_t error = checkError( "antialiasingFiltering" );
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
}

/**
 *	Anti-aliasing �� ���� filter. pixel �� 3x3 filter �� ���Ƿ� ������
 *	block size �� 12 x .. ���·� ������.
 */
GError cudaRenderPipeline::grayScaleFiltering()
{
	dim3 threads( 16, 8 );
	dim3 blocks( m_SceneInfo.iResolutionX / threads.x, m_SceneInfo.iResolutionY / threads.y );
	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	grayScaleFilteringKernel<<< blocks, threads >>> ( m_pDeviceFrameBuffer2 );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
	swapFrameBuffer();

	cudaError_t error = checkError( "grayScaleFilteringKernel" );
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
}

/**
 *	Anti-aliasing �� ���� filter. pixel �� 3x3 filter �� ���Ƿ� ������
 *	block size �� 12 x .. ���·� ������.
 */
GError cudaRenderPipeline::sobelMethodFiltering()
{
	dim3 threads( 16, 8 );
	dim3 blocks( m_SceneInfo.iResolutionX / threads.x, m_SceneInfo.iResolutionY / threads.y );
	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	sobelMethodFilteringKernel<<< blocks, threads >>> ( m_pDeviceFrameBuffer2 );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
	swapFrameBuffer();

	cudaError_t error = checkError( "sobelMethodFilteringKernel" );
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
}

/**
 *	Anti-aliasing �� ���� filter. pixel �� 3x3 filter �� ���Ƿ� ������
 *	block size �� 12 x .. ���·� ������.
 */
GError cudaRenderPipeline::blurringFiltering()
{
	dim3 threads( 16, 8 );
	dim3 blocks( m_SceneInfo.iResolutionX / threads.x, m_SceneInfo.iResolutionY / threads.y );
	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	blurringFilteringKernel<<< blocks, threads >>> ( m_pDeviceFrameBuffer2 );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
	swapFrameBuffer();

	cudaError_t error = checkError( "blurringFilteringKernel" );
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
}

/**
 *	���� frame buffer �� blooming ȿ���� �ش�.
 *	�̹��� ũ�⸸ŭ cuda thread �� �����ؼ� ������.
 */
GError cudaRenderPipeline::bloomingFiltering()
{
	int imagePixelCount = m_iImagePixelCount;
	int startImageIndex = 0;

	if (  m_pDeviceBloomingFilter == NULL || m_pDeviceFrameBuffer == NULL || m_pDeviceFrameBuffer2 == NULL ) {
		GLogManager::logging( LOG_ERROR, "blooming is not initialized." );
		return errorCudaError;
	}

	dim3 threads( 32, 4 );
	dim3 blocks( m_SceneInfo.iResolutionX / threads.x, m_SceneInfo.iResolutionY / threads.y );
	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	/**
	 *	����.
	 */
	bloomingFilteringKernel<<< blocks, threads >>>( 
							startImageIndex,
							imagePixelCount,
							m_pDeviceFrameBuffer,
							m_pDeviceFrameBuffer2,
							m_pDeviceBloomingFilter,
							m_iBloomingWidth,
							m_fBloomingWeight );
	
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

	swapFrameBuffer();

	cudaError_t error = checkError( "bloomingFrameBufferKernel" );
	if ( error != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;


}

/**
 *	Intersection Result �� Debugging ���� ���.
 */
void cudaRenderPipeline::printIntersectionResultDebugInfo( int offset, int count )
{
	cuIntersectionPoint *pPoint = new cuIntersectionPoint[ count ];
	
	CUDA_SAFE_CALL( cudaMemcpy( pPoint, m_pDeviceIntersectionPoint + offset, 
								sizeof( cuIntersectionPoint ) * count,
								cudaMemcpyDeviceToHost ) );
	
	int loop = 0;
	for ( int i = 0; i < count && loop < 100; ++i ) {
		if ( (int)pPoint[ i ].triIndex == -1 ) continue;
		GLogManager::logging( LOG_DEBUG, "Intersection Point : (u=%6.3f, v=%6.3f)",
			pPoint[i].u, pPoint[i].v );
		loop++;
	}
	
	delete[] pPoint;
}

GError cudaRenderPipeline::doSinglePassRayCasting( cuCamera camera, int maxReflectionDepth, 
												   bool faceCCW, bool backFaceCulling,
												   int samplingX, int samplingY, bool jittering,
												   bool bShadowEnable )
{
	if ( m_pDeviceKDTreeNodes == NULL || m_pDeviceTriangleOffsetList == NULL || m_pDeviceTriangleGeometry == NULL ||
		 m_pDeviceObjectMaterial == NULL || m_pDeviceIntResult == NULL ) {
		 GLogManager::logging( LOG_ERROR, "There is no spatial structure info" );
		 return errorCudaError;
	}

	setCameraInfo( &camera );

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	blocks.x = m_SceneInfo.iResolutionX / threads.x;
	blocks.y = m_SceneInfo.iResolutionY / threads.y;

	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	/**
	 *	��ü�� ����, reflection depth �� ���� ��� kernel �ȿ��� sampling ������ŭ �ݺ��ϸ�
	 *	Ŀ�ο��귮 �ʰ��� GPU �� �״°�찡 �ֱ� ������ kernel �� sampling ������ŭ ȣ���Ѵ�.
	 */
	for ( int i = 0; i < samplingX; ++i ) {
		for ( int j = 0; j < samplingY; ++j ) {
			if ( bShadowEnable ) {
				singlePassRayTracingKernel_ShadowOn<<<blocks, threads, sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
					( m_pDeviceFrameBuffer, maxReflectionDepth, i, j, jittering );		
			} else {
				singlePassRayTracingKernel_ShadowOff<<<blocks, threads, sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
					( m_pDeviceFrameBuffer, maxReflectionDepth, i, j, jittering );		
			}
			CUDA_SAFE_CALL( cudaThreadSynchronize() );
		}
	}

	cudaError_t error_t = checkError( "doSinglePassRayCasting" );
	if ( error_t != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;

}


GError cudaRenderPipeline::doSinglePassRayCasting_Coherent( cuCamera camera, int maxReflectionDepth, 
														   bool faceCCW, bool backFaceCulling,
														   int samplingX, int samplingY, bool jittering,
														   bool bShadowEnable )
{
	if ( m_pDeviceKDTreeNodes == NULL || m_pDeviceTriangleOffsetList == NULL || m_pDeviceTriangleGeometry == NULL ||
		 m_pDeviceObjectMaterial == NULL || m_pDeviceIntResult == NULL ) {
		 GLogManager::logging( LOG_ERROR, "There is no spatial structure info" );
		 return errorCudaError;
	}

	setCameraInfo( &camera );

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	if ( samplingX == 3 && samplingY == 3 ) {
		threads.x = 3;
		threads.y = 42;
	}
	if ( samplingX == 4 && samplingY == 3 ) {
		threads.x = 4;
		threads.y = 30;
	}

	blocks.x = ( m_SceneInfo.iResolutionX * samplingX ) / threads.x;
	blocks.y = ( m_SceneInfo.iResolutionY * samplingY ) / threads.y;

	if ( ( samplingX * m_SceneInfo.iResolutionX ) % threads.x != 0 )
		blocks.x++;
	if ( ( samplingY * m_SceneInfo.iResolutionY ) % threads.y != 0 )
		blocks.y++;

	if ( bShadowEnable ) {
		singlePassRayTracingKernel_Coherent_ShadowOn<<<blocks, threads, sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
			( m_pDeviceFrameBuffer, maxReflectionDepth, jittering );		
	} else {
		singlePassRayTracingKernel_Coherent_ShadowOff<<<blocks, threads, sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
			( m_pDeviceFrameBuffer, maxReflectionDepth, jittering );		
	}

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	cudaError_t error_t = checkError( "doSinglePassRayCasting" );
	if ( error_t != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;

}


GError cudaRenderPipeline::fixedOption_doSinglePassRayCasting_Coherent( cuCamera camera, int maxReflectionDepth, 
														   bool faceCCW, bool backFaceCulling,
														   int samplingX, int samplingY, bool jittering,
														   bool bShadowEnable )
{
	if ( m_pDeviceKDTreeNodes == NULL || m_pDeviceTriangleOffsetList == NULL || m_pDeviceTriangleGeometry == NULL ||
		 m_pDeviceObjectMaterial == NULL || m_pDeviceIntResult == NULL ) {
		 GLogManager::logging( LOG_ERROR, "There is no spatial structure info" );
		 return errorCudaError;
	}

	setCameraInfo( &camera );

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	if ( samplingX == 3 && samplingY == 3 ) {
		threads.x = 3;
		threads.y = 42;
	}
	if ( samplingX == 4 && samplingY == 3 ) {
		threads.x = 4;
		threads.y = 30;
	}

	blocks.x = ( m_SceneInfo.iResolutionX * samplingX ) / threads.x;
	blocks.y = ( m_SceneInfo.iResolutionY * samplingY ) / threads.y;

	if ( ( samplingX * m_SceneInfo.iResolutionX ) % threads.x != 0 )
		blocks.x++;
	if ( ( samplingY * m_SceneInfo.iResolutionY ) % threads.y != 0 )
		blocks.y++;

	if ( bShadowEnable ) {
		fixedOption_singlePassRayTracingKernel_Coherent_ShadowOn<<<blocks, threads, sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
			( m_pDeviceFrameBuffer, maxReflectionDepth, jittering );		
	} else {
		fixedOption_singlePassRayTracingKernel_Coherent_ShadowOff<<<blocks, threads, sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
			( m_pDeviceFrameBuffer, maxReflectionDepth, jittering );		
	}

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	cudaError_t error_t = checkError( "doSinglePassRayCasting" );
	if ( error_t != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;

}

GError cudaRenderPipeline::doSelectiveAndAdaptiveSamplingRayTracing( 
												   int maxReflectionDepth, 
												   int samplingX, int samplingY, bool jittering,
												   bool bAdaptiveInfo,
												   int compareType,
												   bool bEnableShadow,
												   float *pResultTime,
												   float *pRayRate )
{
	int samplingCount[4] = { 0x00, };

	if ( m_pDeviceKDTreeNodes == NULL || m_pDeviceTriangleOffsetList == NULL || m_pDeviceTriangleGeometry == NULL ||
		 m_pDeviceObjectMaterial == NULL || m_pDeviceIntResult == NULL ) {
		 GLogManager::logging( LOG_ERROR, "There is no spatial structure info" );
		 return errorCudaError;
	}

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	blocks.x = m_SceneInfo.iResolutionX / threads.x;
	blocks.y = m_SceneInfo.iResolutionY / threads.y;

	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	
	/**
	 *	��ü�� ����, reflection depth �� ���� ��� kernel �ȿ��� sampling ������ŭ �ݺ��ϸ�
	 *	Ŀ�ο��귮 �ʰ��� GPU �� �״°�찡 �ֱ� ������ kernel �� sampling ������ŭ ȣ���Ѵ�.
	 */
	if ( samplingX * samplingY == 1 ) {

		if ( bEnableShadow ) {
			singlePassRayTracingKernel_1_SamplingKernel_ShadowOn<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		} else {
			singlePassRayTracingKernel_1_SamplingKernel_ShadowOff<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		}
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

	} else {

		if ( bEnableShadow ) {
			singlePassRayTracingKernel_1_SamplingKernel_ShadowOn<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		} else {
			singlePassRayTracingKernel_1_SamplingKernel_ShadowOff<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		}
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		/** 2���� ������ ����. */
		CUDA_SAFE_CALL( cudaMemset( m_pDeviceIntResult, 0x00, sizeof( int ) * 4 ) );	
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		GTimer detectionTime;
		detectionTime.start();

		///** 
		// *	color difference map �� �����. �� kernel �� ���ϰ� ���Ǿ��� ������ thread �� ���̽ᵵ �� 
		// */
		//threads.x = 16;
		//threads.y = 16;
		//blocks.x = ( m_SceneInfo.iResolutionX ) / threads.x;
		//blocks.y = ( m_SceneInfo.iResolutionY ) / threads.y;

		//if ( (  m_SceneInfo.iResolutionX ) % threads.x != 0 )
		//	blocks.x++;
		//if ( ( m_SceneInfo.iResolutionY ) % threads.y != 0 )
		//	blocks.y++;

		//colorDifferenceMapGenerationKernel_SobelMethod<<<blocks, threads>>>( m_pDeviceFrameBuffer2 );
		//CUDA_SAFE_CALL( cudaThreadSynchronize() );

		/** 
		 *	���ȼ��� 4sub-pixel �� ������ �����尡 ó���ϰ� �ϱ� ���ؼ�.
		 * x, y �� 2�� ����̾�� �Ѵ�. �׸��� �� ����� width*2, height*2 ũ���� �Ѵ�. 
		 */
		threads.x = 8;
		threads.y = 16;
		blocks.x = ( 2 * m_SceneInfo.iResolutionX ) / threads.x;
		blocks.y = ( 2 * m_SceneInfo.iResolutionY ) / threads.y;

		if ( ( 2 * m_SceneInfo.iResolutionX ) % threads.x != 0 )
			blocks.x++;
		if ( ( 2 * m_SceneInfo.iResolutionY ) % threads.y != 0 )
			blocks.y++;

		/** ���� shared memory ������� stack ������� ������� �Ʒ�ó�� ��ƾ��� */
		singlePassRayTracingKernel_DetectionStage<<<blocks, threads, 6 * sizeof( float ) * threads.x * threads.y>>>
			( m_pDeviceFrameBuffer, m_pDeviceASBuffer,
			  m_pDeviceIntResult, bAdaptiveInfo, compareType );

		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		CUDA_SAFE_CALL( cudaMemcpy( samplingCount, m_pDeviceIntResult, 4 * sizeof( int ), cudaMemcpyDeviceToHost ) );	

		detectionTime.end();

		/**
		 *	����� ������ ���� ��.
		 *	�߰������� ��� ray �� sampling �ؾ��ϴ����� ����Ѵ�. padding �� subpixel �� �����ϰ� ����ؾ� �ϹǷ�
		 *	�޸𸮸� �m���.
		 */
		if ( bAdaptiveInfo ) {
			
			int *tempBuffer = (int*) malloc( sizeof( int ) * samplingCount[0] );

			CUDA_SAFE_CALL( cudaMemcpy( tempBuffer, m_pDeviceASBuffer, sizeof( int ) * samplingCount[0], cudaMemcpyDeviceToHost ) );
			CUDA_SAFE_CALL( cudaThreadSynchronize() );

			int activeSubPixels = 0;
			int value = 0, prev = 0;
			int error = 0, samepixel = 1;

			for ( int i = 0; i < samplingCount[0]; ++i ) {

				value = tempBuffer[ i ];

				if ( value != -1 ) {
					if ( prev == value && ( i % ADAPTIVE_THREADS ) == 0 )
						error++;

					if ( prev != value )
						samepixel = 1;
					else
						samepixel++;

					activeSubPixels++;
					prev = value;

					if ( GET_SUBPIXEL_INDEX( value ) >= m_iImagePixelCount ) error++;
					if ( GET_SUBPIXEL_INDEX( value ) < 0 ) error++;
					if ( ! (GET_SUBPIXEL_CORNER( value ) >= 0 && GET_SUBPIXEL_CORNER( value ) <= 3) &&
						 ! (GET_SUBPIXEL_CORNER( value ) >= 10 && GET_SUBPIXEL_CORNER( value ) <= 13) ) error++;
					if ( samepixel > 4 ) error++;
				} else {
					samepixel = 1;
				}
			}
			
			float rate = ((float)( m_iImagePixelCount + activeSubPixels * 4.0f )  / (float)( 16.0f * m_iImagePixelCount ) ) * 100.0f;
			GLogManager::logging( LOG_ERROR, 
				"Total ASBuffer = %d, ASP = %d, Padding = %d ->Adaptive ray count = %d vs fixed 4x4 ray count = %d, rate = %f", 
				samplingCount[0], 
				activeSubPixels, samplingCount[0] - activeSubPixels, 
				m_iImagePixelCount + activeSubPixels * 4, 16 * m_iImagePixelCount,
				rate );
			
			if ( error > 0 ) {
				GLogManager::logging( LOG_ERROR, "Error !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ======== %d", error );
			}

			free( tempBuffer );

			*pResultTime = detectionTime.getElapsedTime();
			*pRayRate = rate;
		}

		GLogManager::logging( LOG_ERROR, "Detection Stage Time Elapsed = %6.3f", detectionTime.getElapsedTime() );

		if ( !bAdaptiveInfo ) {

			/**
			 *	Kernel �ȿ��� shared �޸𸮸� �̿��ؼ� �� pixel �� ���ؼ� sampling ������ճ��� �����Ѵ�.
			 *	block ���� thread ������ �ݵ�� �� pixel �� sampling ������ ����̾�� �Ѵ�.
			 *	3x3 �� adaptive sampling ���� �����Ѵ�.
			 */
			threads.x = ADAPTIVE_THREADS;
			threads.y = 1;
			blocks.x = ( 4 * samplingCount[ 0 ] ) / threads.x;
			blocks.y = 1;

			if ( ( 4 * samplingCount[ 0 ] ) % threads.x != 0 )
				blocks.x++;

			if ( bEnableShadow ) {
				singlePassRayTracingKernel_SuperSamplingStage_ShadowOn<<<blocks, threads, 
						 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
								( m_pDeviceFrameBuffer, samplingCount[0], maxReflectionDepth );		
			} else {
				singlePassRayTracingKernel_SuperSamplingStage_ShadowOff<<<blocks, threads, 
						 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
								( m_pDeviceFrameBuffer, samplingCount[0], maxReflectionDepth );		
			}

			CUDA_SAFE_CALL( cudaThreadSynchronize() );

		}

	}

	cudaError_t error_t = checkError( "doAdaptiveSamplingRayTracing" );
	if ( error_t != cudaSuccess ) {
		clearFrameBuffer();
		return errorCudaError;
	}
	
	return errorNo;

}


/**
 *	reflection depth=1, ���� 1�� ����.
 */
GError cudaRenderPipeline::fixedOption_doSelectiveAndAdaptiveSamplingRayTracing( 
													int maxReflectionDepth, 
												   int samplingX, int samplingY, bool jittering,
												   bool bAdaptiveInfo,
												   int compareType,
												   bool bEnableShadow,
												   float *pResultTime,
												   float *pRayRate )
{
	int samplingCount[4] = { 0x00, };

	if ( m_pDeviceKDTreeNodes == NULL || m_pDeviceTriangleOffsetList == NULL || m_pDeviceTriangleGeometry == NULL ||
		 m_pDeviceObjectMaterial == NULL || m_pDeviceIntResult == NULL ) {
		 GLogManager::logging( LOG_ERROR, "There is no spatial structure info" );
		 return errorCudaError;
	}

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	blocks.x = m_SceneInfo.iResolutionX / threads.x;
	blocks.y = m_SceneInfo.iResolutionY / threads.y;

	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	
	/**
	 *	��ü�� ����, reflection depth �� ���� ��� kernel �ȿ��� sampling ������ŭ �ݺ��ϸ�
	 *	Ŀ�ο��귮 �ʰ��� GPU �� �״°�찡 �ֱ� ������ kernel �� sampling ������ŭ ȣ���Ѵ�.
	 */
	if ( samplingX * samplingY == 1 ) {

		if ( bEnableShadow ) {
			fixedOption_singlePassRayTracingKernel_1_SamplingKernel_ShadowOn<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		} else {
			fixedOption_singlePassRayTracingKernel_1_SamplingKernel_ShadowOff<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		}
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

	} else {

		if ( bEnableShadow ) {
			fixedOption_singlePassRayTracingKernel_1_SamplingKernel_ShadowOn<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		} else {
			fixedOption_singlePassRayTracingKernel_1_SamplingKernel_ShadowOff<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
				( m_pDeviceSamplingMap, m_pDeviceFrameBuffer, m_pDeviceFrameBuffer2, maxReflectionDepth );		
		}
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		/** 2���� ������ ����. */
		CUDA_SAFE_CALL( cudaMemset( m_pDeviceIntResult, 0x00, sizeof( int ) * 4 ) );	
		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		GTimer detectionTime;
		detectionTime.start();

		/** 
		 *	���ȼ��� 4sub-pixel �� ������ �����尡 ó���ϰ� �ϱ� ���ؼ�.
		 * x, y �� 2�� ����̾�� �Ѵ�. �׸��� �� ����� width*2, height*2 ũ���� �Ѵ�. 
		 */
		threads.x = 8;
		threads.y = 16;
		blocks.x = ( 2 * m_SceneInfo.iResolutionX ) / threads.x;
		blocks.y = ( 2 * m_SceneInfo.iResolutionY ) / threads.y;

		if ( ( 2 * m_SceneInfo.iResolutionX ) % threads.x != 0 )
			blocks.x++;
		if ( ( 2 * m_SceneInfo.iResolutionY ) % threads.y != 0 )
			blocks.y++;

		/** ���� shared memory ������� stack ������� ������� �Ʒ�ó�� ��ƾ��� */
		singlePassRayTracingKernel_DetectionStage<<<blocks, threads, 6 * sizeof( float ) * threads.x * threads.y>>>
			( m_pDeviceFrameBuffer, m_pDeviceASBuffer,
			  m_pDeviceIntResult, bAdaptiveInfo, compareType );

		CUDA_SAFE_CALL( cudaThreadSynchronize() );

		CUDA_SAFE_CALL( cudaMemcpy( samplingCount, m_pDeviceIntResult, 4 * sizeof( int ), cudaMemcpyDeviceToHost ) );	

		detectionTime.end();

		/**
		 *	����� ������ ���� ��.
		 *	�߰������� ��� ray �� sampling �ؾ��ϴ����� ����Ѵ�. padding �� subpixel �� �����ϰ� ����ؾ� �ϹǷ�
		 *	�޸𸮸� �m���.
		 */
		if ( bAdaptiveInfo ) {
			
			int *tempBuffer = (int*) malloc( sizeof( int ) * samplingCount[0] );

			CUDA_SAFE_CALL( cudaMemcpy( tempBuffer, m_pDeviceASBuffer, sizeof( int ) * samplingCount[0], cudaMemcpyDeviceToHost ) );
			CUDA_SAFE_CALL( cudaThreadSynchronize() );

			int activeSubPixels = 0;
			int value = 0, prev = 0;
			int error = 0, samepixel = 1;

			for ( int i = 0; i < samplingCount[0]; ++i ) {

				value = tempBuffer[ i ];

				if ( value != -1 ) {
					if ( prev == value && ( i % ADAPTIVE_THREADS ) == 0 )
						error++;

					if ( prev != value )
						samepixel = 1;
					else
						samepixel++;

					activeSubPixels++;
					prev = value;

					if ( GET_SUBPIXEL_INDEX( value ) >= m_iImagePixelCount ) error++;
					if ( GET_SUBPIXEL_INDEX( value ) < 0 ) error++;
					if ( ! (GET_SUBPIXEL_CORNER( value ) >= 0 && GET_SUBPIXEL_CORNER( value ) <= 3) &&
						 ! (GET_SUBPIXEL_CORNER( value ) >= 10 && GET_SUBPIXEL_CORNER( value ) <= 13) ) error++;
					if ( samepixel > 4 ) error++;
				} else {
					samepixel = 1;
				}
			}
			
			float rate = ((float)( m_iImagePixelCount + activeSubPixels * 4.0f )  / (float)( 9.0f * m_iImagePixelCount ) ) * 100.0f;
			GLogManager::logging( LOG_ERROR, 
				"Total ASBuffer = %d, ASP = %d, Padding = %d ->Adaptive ray count = %d vs fixed 3x3 ray count = %d, rate = %f", 
				samplingCount[0], 
				activeSubPixels, samplingCount[0] - activeSubPixels, 
				m_iImagePixelCount + activeSubPixels * 4, 9 * m_iImagePixelCount,
				rate );
			
			if ( error > 0 ) {
				GLogManager::logging( LOG_ERROR, "Error !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ======== %d", error );
			}

			free( tempBuffer );

			*pResultTime = detectionTime.getElapsedTime();
			*pRayRate = rate;
		}

		GLogManager::logging( LOG_ERROR, "Detection Stage Time Elapsed = %6.3f", detectionTime.getElapsedTime() );

		if ( !bAdaptiveInfo ) {

			/**
			 *	Kernel �ȿ��� shared �޸𸮸� �̿��ؼ� �� pixel �� ���ؼ� sampling ������ճ��� �����Ѵ�.
			 *	block ���� thread ������ �ݵ�� �� pixel �� sampling ������ ����̾�� �Ѵ�.
			 *	3x3 �� adaptive sampling ���� �����Ѵ�.
			 */

			/** �ʹ����� block �� ����ٸ� ������ ����� �Ѵ�. */
			int activeSubPixelCount = samplingCount[ 0 ];
			int oneIterationSubPixel = 1000000;
			int mincount = 0;

			for ( int startSubPixels = 0; startSubPixels < activeSubPixelCount; startSubPixels += oneIterationSubPixel ) {

				threads.x = ADAPTIVE_THREADS;
				threads.y = 1;

				mincount = min( activeSubPixelCount - startSubPixels, oneIterationSubPixel );
				blocks.x = ( 4 * mincount ) / threads.x;
				blocks.y = 1;

				if ( ( 4 * mincount ) % threads.x != 0 )
					blocks.x++;

				if ( bEnableShadow ) {
					fixedOption_singlePassRayTracingKernel_SuperSamplingStage_ShadowOn<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
									( m_pDeviceFrameBuffer, startSubPixels, samplingCount[0], maxReflectionDepth );		
				} else {
					fixedOption_singlePassRayTracingKernel_SuperSamplingStage_ShadowOff<<<blocks, threads, 
							 sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
									( m_pDeviceFrameBuffer, startSubPixels, samplingCount[0], maxReflectionDepth );		
				}

				CUDA_SAFE_CALL( cudaThreadSynchronize() );

			}

		}

	}

	cudaError_t error_t = checkError( "doAdaptiveSamplingRayTracing" );
	if ( error_t != cudaSuccess ) {
		clearFrameBuffer();
		return errorCudaError;
	}
	
	return errorNo;

}

/**
 *	Ray Casting �� �����Ѵ�.
 *	�̹��� ��ü�� ���ؼ� �����ϴ°�.
 */
GError cudaRenderPipeline::doRayCasting( int rayOffset, int rayCount, bool faceCCW, bool backFaceCulling )
{
	int startOffset = rayOffset;

	if ( m_pDeviceKDTreeNodes == NULL || m_pDeviceTriangleOffsetList == NULL || m_pDeviceTriangleGeometry == NULL ||
		 m_pDeviceObjectMaterial == NULL || m_pDeviceIntResult == NULL ) {
		 GLogManager::logging( LOG_ERROR, "There is no spatial structure info" );
		 return errorCudaError;
	}

	dim3 blocks;
	dim3 threads( m_SceneInfo.iBlockSizeX, m_SceneInfo.iBlockSizeY );

	blocks.x = m_SceneInfo.iResolutionX / threads.x;
	blocks.y = m_SceneInfo.iResolutionY / threads.y;

	if ( m_SceneInfo.iResolutionX % threads.x != 0 )
		blocks.x++;
	if ( m_SceneInfo.iResolutionY % threads.y != 0 )
		blocks.y++;

	rayCastingKernel<<<blocks, threads, sizeof( float ) * ( SHORT_STACK_DEPTH * threads.x * threads.y ) * 2>>>
		( startOffset, rayCount, m_pDeviceIntersectionPoint, faceCCW, backFaceCulling );			
	
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

	cudaError_t error_t = checkError( "cudaRayCasting" );
	if ( error_t != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
}

/**
 *	Ray Casting �� �����Ѵ�.
 *	� ray �� ���������� �̹� device ���� ray memory �� �ö� �־�� �Ѵ�.
 *	generatePrimaryRay �� setRay ���� �̿��ؼ� �̸� �����ؾ� ��. intersection �����
 *	device �޸� �� ������� �д�.
 *	���ڴ� global memory �� �ö� �ִ� ray �������� ��� ��������
 *	intersection check ������ backface �� ��� culling �ɼ��� �ִ� �ﰢ���� ���ؼ�
 *	culling ���� ����.
 */
GError cudaRenderPipeline::doRayCastingSequentialData( int rayOffset, int rayCount, bool faceCCW, bool backFaceCulling )
{
	int startOffset = rayOffset;
	
	if ( rayCount == 0 || rayOffset + rayCount > m_iMaxRay ) {
		GLogManager::logging( LOG_ERROR, "max ray overflow." );
		return errorCudaError;
	}

	if ( m_pDeviceKDTreeNodes == NULL || m_pDeviceTriangleOffsetList == NULL || m_pDeviceTriangleGeometry == NULL ||
		 m_pDeviceObjectMaterial == NULL || m_pDeviceIntResult == NULL ) {
		 GLogManager::logging( LOG_ERROR, "There is no spatial structure info" );
		 return errorCudaError;
	}

	GLogManager::logging( LOG_DEBUG, "DO RAY CASTING START ( %d ~ %d ) ray check ", rayOffset, rayOffset + rayCount );
	
	int blockNum = rayCount / INTERSECTION_THREAD_DIM + 1;
	rayCastingKernelSequentialData<<<blockNum, INTERSECTION_THREAD_DIM, sizeof( float ) * ( SHORT_STACK_DEPTH * INTERSECTION_THREAD_DIM ) * 2>>>
		(  startOffset, rayCount, m_pDeviceIntersectionPoint, faceCCW, backFaceCulling );			
	
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

	cudaError_t error_t = checkError( "cudaRayCasting" );
	if ( error_t != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
}

/**
 *	���� device �� �����ϴ� Intersection ����� �޾ƿ´�. 
 *	count �� ����� �޾ƿ��� �����ϴ� ��.
 */
cuIntersectionPoint *cudaRenderPipeline::getIntersectionResult( int count )
{
	/** 
	 *	���� Device ���� ����� �����ؼ� �����´�. 
	 */
	CUDA_SAFE_CALL( cudaMemcpy( m_pHostIntersectionPoint, 
								m_pDeviceIntersectionPoint, 
								sizeof( cuIntersectionPoint ) * count, 
								cudaMemcpyDeviceToHost ) );
	
	return m_pHostIntersectionPoint;
}

/**
 *	Memory ���¸� ����Ѵ�.
 */
void cudaRenderPipeline::printStatusInfo()
{
	float fRayBufferSize = ( sizeof( cuRay ) * m_iMaxRay ) / 1048576.0f;
	float fIntersectionBufferSize = ( sizeof( cuIntersectionPoint ) * m_iMaxIntersectionPoint ) /  1048576.0f;
	// framebuffer �� 2��.
	float fFrameBufferSize = ( ( sizeof( float ) * 3 * m_iImagePixelCount ) / 1048576.0f ) * 2;
	
	float fKDTreeSize = ( sizeof( kdtreeNode ) * m_iNodeCount ) / 1048576.0f;
	float fObjectMaterial = ( sizeof( cuObjectMaterial ) * m_iObjectMaterialCount ) / 1048576.0f;
	float fTriangleOffsetList = ( sizeof( unsigned int ) * m_iTriangleOffsetCount ) / 1048576.0f;
	float fTriangleIntersectionInfo = ( sizeof( cuWaldTriangleInfo ) * m_iTriangleCount ) / 1048576.0f;
	float fTriangleGeometry = ( sizeof( cuTriangleGeometry ) * m_iTriangleCount ) / 1048576.0f;
	float fTexture = ( sizeof( uint4 ) * m_iTotalTextureWidth * m_iTotalTextureHeight ) / 1048576.0f;
	
	float fTotal = fRayBufferSize + fIntersectionBufferSize + fFrameBufferSize +
				   fKDTreeSize + fObjectMaterial + fTriangleOffsetList + fTriangleIntersectionInfo +
				   fTriangleGeometry + fTexture;
	
	GLogManager::logging( LOG_INFO, "------------------ SGRTx2 GPU RENDER PIPELINE INFO ----------------------------" );
	
	GLogManager::logging( LOG_INFO, " -> Screen Resolution ( %d, %d ), SuperSampling ( %d, %d )", 
									m_SceneInfo.iResolutionX, m_SceneInfo.iResolutionY,
									m_SceneInfo.iSuperSamplingX, m_SceneInfo.iSuperSamplingY );
	GLogManager::logging( LOG_INFO, " -> RayBuffer : %d ( %f MB )", m_iMaxRay, fRayBufferSize );
	GLogManager::logging( LOG_INFO, " -> IntersectionBuffer = %d ( %f MB )", m_iMaxIntersectionPoint, fIntersectionBufferSize );
	GLogManager::logging( LOG_INFO, " -> FrameBuffer x 2 : %d pixels x 2. ( %f MB )", m_iImagePixelCount,	fFrameBufferSize );
	GLogManager::logging( LOG_INFO, " -> KDTree Buffer : %d node. ( %f MB )", m_iNodeCount, fKDTreeSize );
	GLogManager::logging( LOG_INFO, " -> Triangle Offset : %d ( %f MB )", m_iTriangleOffsetCount, fTriangleOffsetList );
	GLogManager::logging( LOG_INFO, " -> Triangle Intersection Info : %d ( %f MB )", m_iTriangleCount, fTriangleIntersectionInfo );
	GLogManager::logging( LOG_INFO, " -> Triangle Geometry Info : %d ( %f MB )", m_iTriangleCount, fTriangleGeometry );
	GLogManager::logging( LOG_INFO, " -> Object Material : %d ( %f MB )", m_iObjectMaterialCount, fObjectMaterial );
	GLogManager::logging( LOG_INFO, " -> Texture Data : %d x %d ( %f MB )", m_iTotalTextureWidth, m_iTotalTextureHeight, fTexture );
	GLogManager::logging( LOG_INFO, " -> Total Allocated GPU Memory : ( %f MB )", fTotal );
									
	GLogManager::logging( LOG_INFO, "------------------------------------------------------------------------------" );
	
	
}



