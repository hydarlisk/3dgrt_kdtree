/**
 *	Cuda 로 Rendering 을 수행하기 위해서
 *	여러가지를 관리하는 class.
 *
 *	light, texture, shading, ray tracing, photon mapping
 *	등등.
 *	
 *	by graphicsian.
 */
#ifndef _CUDA_RENDER_PIPELINE_CUH_
#define _CUDA_RENDER_PIPELINE_CUH_

#include "GBase.h"
#include "cudaRenderCommon.cuh"

class cudaRenderPipeline {

private:
	cuScene m_SceneInfo;
	cuThreshold m_ThresholdInfo;

	cuRay *m_pDeviceRays;
	cuIntersectionPoint *m_pDeviceIntersectionPoint;
	cuIntersectionPoint *m_pHostIntersectionPoint;
	
	int m_iMaxRay, m_iMaxIntersectionPoint;
	
	kdtreeNode *m_pDeviceKDTreeNodes;
	unsigned int* m_pDeviceTriangleOffsetList;
	cuWaldTriangleInfo *m_pDeviceWaldTriangleInfo;
	cuPlueckerTriangleInfo *m_pDevicePlueckerTriangleInfo;

	cuTriangleGeometry *m_pDeviceTriangleGeometry;
	cuObjectMaterial *m_pDeviceObjectMaterial;
	cuSamplingMap *m_pDeviceSamplingMap;

	int m_iNodeCount;
	int m_iObjectMaterialCount;
	int m_iTriangleOffsetCount;
	int m_iTriangleCount;
	
	cuLightRaySet *m_pDeviceLightRaySetData;
	
	float *m_pDeviceFrameBuffer;
	float *m_pDeviceFrameBuffer2;
	int *m_pDeviceASBuffer;

	float *m_pDeviceBloomingFilter;
	
	int m_iBloomingWidth;
	float m_fBloomingWeight;
	
	int m_iImagePixelCount;
	int *m_pDeviceIntResult;

	int m_iTotalTextureWidth;
	int m_iTotalTextureHeight;
	
	cudaArray *m_pDeviceTextureData;

public:
	cudaRenderPipeline();
	~cudaRenderPipeline();
	
	/**
	*	CUDA Render Context 초기화.
	*/
	GError initialize( cuScene pScene, int maxRay );
	GError setLightRaySetData( cuLightRaySet *pRaySet, int count );
	GError setLightInfo( cuLight *pLight, int count );
	GError setCameraInfo( cuCamera *pCamera );
	GError setSceneInfo( cuScene pScene );
	GError setThresholdInfo( cuThreshold threshold );

	GError initRayIntersection( int maxRay );

	GError initBloomingFilter( float radius, float weight );
	GError bloomingFiltering();
	
	GError antialiasingFiltering();
	GError sobelMethodFiltering();
	GError grayScaleFiltering();
	GError blurringFiltering();


	/**
	 *	RAY 와 삼각형 Intersection Check 를 위한 데이터 세팅.
	 */
	GError setKDTreeNodeData( kdtreeNode *pKDTreeNodes, int nodeCount, cuBoundingBox sceneBox );
	GError setTriangleOffsetList( unsigned int *pTriangleOffsetList, int offsetCount );

	GError setTriangleGeometry( cuTriangleGeometry *pTriangleGeometry, int triangleCount );
	GError setWaldTriangleInfo( cuWaldTriangleInfo *pTriangleInfo, int triangleCount );
	GError setPlueckerTriangleInfo( cuPlueckerTriangleInfo *pTriangleInfo, int triangleCount );

	GError setObjectMaterial( cuObjectMaterial *pObjectMaterial, int count );
	
	GError setTextureData( cuTexture* pTextureData, int count );
	
	int getDeviceRayBufferSize();
	cuRay *getDeviceRayBuffer();
	cuIntersectionPoint* getDeviceIntersectionBuffer();
	
	/**
	 *	추적할 ray 정보를 cpu 로 부터 세팅.
	 */
	GError setRayInfo( cuRay *pRay, int destOffset, int count );
	
	/**
	 *	현재 frame buffer 의 값을 복사해온다. pBuffer 는 
	 *	frame buffer 와 크기가 같아야 한다.
	 */
	void getFrameBuffer( float* pBuffer );
	
	/**
	 *	현재 frame buffer 을 pBuffer 로 덮어쓴다.
	 *	데이터는 크기는 frame buffer size 와 같아야 한다.
	 */
	void setFrameBuffer( float* pBuffer );
	
	/**
	 *	FrameBuffer 를 clear.
	 */
	void clearFrameBuffer();
	
	/**
	 *	현재 Intersection point 정보를 clear.
	 */
	GError clearIntersectionResult();

	/**
	 *	GPU 상에 있는 intersection 정보를 기반으로 해당 point 의
	 *	direct illumination 을 수행해서 Frame Buffer 에 누적.
	 */
	GError calDirectIllumination( int maxReflectionDepth );

	/** 
	 *	GPU 상에서 primary ray 생성.
	 */
	GError generatePrimaryRay( cuCamera camera, int *pGeneratedCount, int currentSampleX, int currentSampleY, bool Jittering );

	/** 
	 *	GPU 상에서 primary ray 생성.
	 */
	GError generateReflectionRay( int offset, int count, int *atLeastOneRay );

	void printIntersectionResultDebugInfo( int offset, int count );
	
	GError fixedOption_doSinglePassRayCasting_Coherent( cuCamera camera, int maxReflectionDepth, 
											   bool faceCCW, bool backFaceCulling,
											   int samplingX, int samplingY, bool jittering,
											   bool bShadowEnable );
	GError doSinglePassRayCasting_Coherent( cuCamera camera, int maxReflectionDepth, 
											   bool faceCCW, bool backFaceCulling,
											   int samplingX, int samplingY, bool jittering,
											   bool bShadowEnable );
	GError doSinglePassRayCasting( cuCamera camera, int maxReflectionDepth, 
								   bool faceCCW, bool backFaceCulling,
								   int samplingX, int samplingY, bool jittering, bool bShadowEnable );

	GError doSelectiveAndAdaptiveSamplingRayTracing( int maxReflectionDepth, 
												   int samplingX, int samplingY, bool jittering,
												   bool bAdaptiveDebugInfo,
												   int compareType, 
												   bool bEnableShadow,
												   float *pResultTime,
												   float *pRayRate );

	/**
	 *	광원1개 고정, reflection 고정, reflectionDepth=1 적용한 ray tracer.
	 */
	GError fixedOption_doSelectiveAndAdaptiveSamplingRayTracing( 
													int maxReflectionDepth,
												   int samplingX, int samplingY, bool jittering,
												   bool bAdaptiveDebugInfo,
												   int compareType, 
												   bool bEnableShadow,
												   float *pResultTime,
												   float *pRayRate );

	/** 
	 *	GPU Device 상에 올라가 있는 cuRay 정보를 기반으로 Ray Casting 수행.
	 *	checkRayCount 는 offset 0 부터 몇 개 까지의 ray 를 체크할지 여부.
	 */
	GError doRayCastingSequentialData( int offset, int count, bool faceCCW, bool backFaceCulling );
	GError doRayCasting( int offset, int count, bool faceCCW, bool backFaceCulling );

	/**
	 *	현재 device 상에 존재하는 Intersection 결과를 cpu 로 복사한뒤 리턴한다. 
	 *	pCount 는 몇개의 ray 결과가 있는지 결과.
	 */
	cuIntersectionPoint *getIntersectionResult( int count );
	
	GError renderingOption( bool shadow, bool texture );
		
	void printStatusInfo();
	
private:
	void swapFrameBuffer();
	void uninitialize();
	
};

#endif

