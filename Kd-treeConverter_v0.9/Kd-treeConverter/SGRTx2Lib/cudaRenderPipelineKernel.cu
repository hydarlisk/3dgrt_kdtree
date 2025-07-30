/**
 *	CUDA 로 Rendering 을 수행하기 위한 기능들.
 *
 *	by graphicsian.
 */
#ifndef __RENDER_PIPELINE_KERNEL_CU_
#define __RENDER_PIPELINE_KERNEL_CU_

#include "GKDTreeNode.h"
#include "cuda_math.h"
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cutil.h>
#include "cudaRenderPipelineCommonKernel.cu"
#include "cudaRenderPipeline.h"


/**------------------------------------------------------------------------------------------
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **	MULTI-PASS RAY TRACING
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **--------------------------------------------------------------------------------------------/


/**
 *	Scene 안의 광원으로부터 Direct Illumination 을 계산한다.
 *	만약 현재 물체 자체가 광원이라면, 현재 물체의 색을 그대로 사용한다.
 *
 *	Phong Shading.
 */
__device__ float3 calDirectIllumination( cuIntersectionPoint &intersectResult, 
									     cuObjectMaterial &material,
									     int maxReflectionDepth )
{
	cuLight *pLight = NULL;
	float3 L, N, R;
	float3 texColor, color;
	int texture = float_as_int( material.textureNumber );
	int visible = 0;
	int objectID = float_as_int( material.iObjectID );
	float3 pos = intersectResult.pos;
	float3 dir = intersectResult.dir;
	float roughness = material.roughness;

	N = intersectResult.normal;

	/** 
	 *	투명한 물체일때, Normal 과 dir 의 dot 이 < 0 이라면
	 *	normal 을 뒤짚는다.
	 */
	if ( material.transparency > 0.0f && dot( dir, N ) < 0.0f ) {
		N = -1.0f * N;
	}

	R = reflection( dir, N );
	
	/** 
	 *	texture number 이제 필요없으므로 texture 변수를
	 *	texture 유무값으로 사용한다. texture 를 1로 세팅. 없다면 0  
	 */
	texture = fetchTexture( texture, intersectResult.u, intersectResult.v, texColor );
	
	color = material.ambient_emission;

	for ( int i = 0; i < constantLightCount; ++i ) {
	
		pLight = ( constantLightInfo + i );

		///**
		// *	현재 물체가 광원이고, 지금 처리하려는 광원과 동일한지를 체크한다.
		// *	Direct Illumination 에 이용하지 않는 광원이라도 이건 처리해야 한다.
		// *	같은 광원이라면 shading 없이 자신의 색을 그대로 사용한다. light 가 0.0 이 아니면 광원이다.
		// */
		if ( material.light != 0.0f ) {
			if ( pLight->iObjectID == objectID ) {
				color += pLight->color * pLight->intensity;
			}
			continue;
		}

		if ( pLight->bUseDirect == 0 )
			continue;

		/** 
		 *	visibility 체크. shadow 가 enable 아니면 무조건 보이는걸로 처리. 아니면 ray 를 쏴서 체크.
		 */
		visible = ( !g_SceneInfo.bEnableShadow || checkVisibility( pos, pLight->pos ) );
		
		if ( visible ) {
		
			L = normalize( pLight->pos - pos );
			
			/** 
			 *	texture 존재 여부에 따라서 material 색깔 선택 
			 */
			if ( texture == 1 ) {
				color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
						material.specular * pLight->color * powf( max( 0.0f, dot( R, L ) ), 1.0 );
			} else {
				color += material.diffuse * pLight->color * max( 0.0f, dot( L, N ) ) +
						 material.specular * pLight->color * powf( max( 0.0f, dot( R, L ) ), 1.0 );
			}
		}
	}

	/**
	 *	최종적으로	kr, kt 를 체크해서 현재 hit 의 color 를 최종 이미지에
	 *	얼마만큼 누적해야할지를 계산. 만약 현재 intersection point 가
	 *	마지막 depth 라면 더이상 반사나 굴절을 통한 데이터를 가져올 수 없으므로
	 *	현재 local shading 전체를 color 에 누적시키기 위해서 kr 이나 kt 비율을 적용하지
	 *	않는다.
	 */
	if ( intersectResult.boundDepth < maxReflectionDepth && material.light == 0.0f ) {
		color = color * ( 1 - material.reflection - material.transparency );
	}

	return color;
	
}

/**
 *	Intersection Point 의 shading 을 계산한다.
 *	 이 안에서도 intersection 체크를 해야하므로 stack 을 위한 shared memory 를
 *	지정하고 호출해야 한다.
 */
__global__ void shadingKernel( int startImageIndex, int totalCount,
							   cuIntersectionPoint *intersectResult, 
							   float *pFrameBuffer,
							   int maxReflectionDepth )
{
	/** 
	 *	현재 thread 에서 생성할 image pixel index. 화면 left, top 에서부터의 순서를 의미 
	 */
	int imageIndex = samplingImageIndex_BlockGrouping( startImageIndex );
	float3 color = make_float3( 0.0f, 0.0f, 0.0f ), tempcolor = make_float3( 0.0f, 0.0f, 0.0f );
	
	/**
	 *	몇개의 쓰레드가 여분으로 더 실행될지 모르므로 index 범위를 초과한
	 *	쓰레드는 그냥 종료.
	 */
	if ( imageIndex >= totalCount )
		return;

	/** 
	 *	imageIndex 를 image 좌표( x, y ) 로 변환한다. 
	 *	각 pixel 당 sampling ray 를 하나씩 처리해한다.
	 */
	int x = imageIndex % ( g_SceneInfo.iResolutionX );
	int y = imageIndex / ( g_SceneInfo.iResolutionX );
		
	/**
	*	현재 이미지상의 좌표의 sub sampling 좌표를 계산해서 
	*	intersectResult 배열안에서의 rayIndex 를 구한다.
	*/
	int rayIndex = y * g_SceneInfo.iResolutionX + x;
			
	if ( intersectResult[ rayIndex ].isHit() ) {
	
		cuObjectMaterial material;
		getObjectMaterial( intersectResult[ rayIndex ].objectIndex, material );
				
		tempcolor = calDirectIllumination( intersectResult[ rayIndex ], material, maxReflectionDepth );
		color += tempcolor * intersectResult[ rayIndex ].colorWeight;
				
	}

	/** 
	 *	해당 좌표의intersectResult 가 hit 이면 object 를
	 *	가져와서 light 과의 관계를 적용해서 shading 을 계산해서 framebuffer 에기록한다.
	*/
	int idx = ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX * 3 + 3 * x;
	
	pFrameBuffer[ idx + 0 ] += color.x;
	pFrameBuffer[ idx + 1 ] += color.y;
	pFrameBuffer[ idx + 2 ] += color.z;
	
}


/**
 *	ray casting kernel. 추적해야할 data 가 sequential 하게 구성되어 있는
 *	메모리를 참조해서 추적.
 *	intersection point 를 리턴한다.
 *	backFaceCulling 이 true 이면 물체중 culling 옵션이 켜져 있는것에
 *	대해서는 뒷면에 맞았을경우 intersection 체크를 하지 않는다.
 */
__global__ void rayCastingKernelSequentialData( int startOffset, int maxIndex, 
												cuIntersectionPoint *intersecResult, 
												bool faceCCW, bool backFaceCulling )
{
	const int tid = samplingRayID( startOffset );

	/**
	 *	몇개의 쓰레드가 여분으로 더 실행될지 모르므로 index 범위를 초과한
	 *	쓰레드는 그냥 종료.
	 */
	if ( tid >= maxIndex )
		return;

	cuRay currRay;

//	currRay.dir = tex1Dfetch( inRayListTex, 2 * tid + 0 );
//	currRay.pos = tex1Dfetch( inRayListTex, 2 * tid + 1 );
	float4 temp;
	temp = tex1Dfetch( inRayListTex, 2 * tid + 0 );
	currRay.dir = make_float3(temp.x, temp.y, temp.z);
	temp = tex1Dfetch( inRayListTex, 2 * tid + 1 );
	currRay.pos = make_float3(temp.x, temp.y, temp.z);

	//currRay.info = tex1Dfetch( inRayListTex, 3 * tid + 2 );

	cuIntersectionCheck currIsectCheck;
	currIsectCheck.init();

	MultipassIntersect( currRay, currIsectCheck, faceCCW, backFaceCulling );
	
	/**
	 *	intersection 했다면, intersection point 정보를 구성한다.
	 */
	if ( currIsectCheck.isHit() ) {
	
		makeIntersectionPoint( &currRay, &currIsectCheck, &intersecResult[ tid ] );
		
	} else {
	
		intersecResult[ tid ].init();
	
	}
	
}

/**
 *	ray casting kernel. 
 *	intersection point 를 리턴한다.
 *	backFaceCulling 이 true 이면 물체중 culling 옵션이 켜져 있는것에
 *	대해서는 뒷면에 맞았을경우 intersection 체크를 하지 않는다.
 */
__global__ void rayCastingKernel( int startOffset, int maxIndex, 
								  cuIntersectionPoint *intersecResult, 
								  bool faceCCW, bool backFaceCulling )
{
	const int tid = samplingRayID_BlockGrouping( startOffset );

	/**
	 *	몇개의 쓰레드가 여분으로 더 실행될지 모르므로 index 범위를 초과한
	 *	쓰레드는 그냥 종료.
	 */
	if ( tid >= maxIndex )
		return;

	cuRay currRay;

//	currRay.dir = tex1Dfetch( inRayListTex, 2 * tid + 0 );
//	currRay.pos = tex1Dfetch( inRayListTex, 2 * tid + 1 );
	float4 temp;
	temp = tex1Dfetch( inRayListTex, 2 * tid + 0 );
	currRay.dir = make_float3(temp.x, temp.y, temp.z);
	temp = tex1Dfetch( inRayListTex, 2 * tid + 1 );
	currRay.pos = make_float3(temp.x, temp.y, temp.z);

	//currRay.info = tex1Dfetch( inRayListTex, 3 * tid + 2 );

	cuIntersectionCheck currIsectCheck;
	currIsectCheck.init();

	MultipassIntersect( currRay, currIsectCheck, faceCCW, backFaceCulling );
	
	/**
	 *	intersection 했다면, intersection point 정보를 구성한다.
	 */
	if ( currIsectCheck.isHit() ) {
	
		makeIntersectionPoint( &currRay, &currIsectCheck, &intersecResult[ tid ] );
		
	} else {
	
		intersecResult[ tid ].init();
	
	}
	
}

/**
 *	primary ray 를 생성해서, device 의 inRay 에 저장해 둔다.
 */
__global__ void generatePrimaryRayKernel( int startRayIndex, int rayCount, 
										  cuRay* inRays, cuIntersectionPoint *pIntersectResult,
										  int currentSampleX, int currentSampleY, bool bJittering )
{
	float sx = 0.0f, sy = 0.0f;

	/** 
	 *	현재 thread 에서 생성할 ray index. 화면 left, top 에서부터의 순서를 의미 
	 */
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	
	int rayIndex = y * g_SceneInfo.iResolutionX + x;

	/**
	 *	몇개의 쓰레드가 여분으로 더 실행될지 모르므로 index 범위를 초과한
	 *	쓰레드는 그냥 종료.
	 */
	if ( rayIndex >= rayCount ) 
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */
	float3 dir, pos;

	pos = g_CameraInfo.eye;

	if ( g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering ) {
		sx = (float) x + ( (float) currentSampleX + radicalInverse( x << 8 + currentSampleX, 3 ) ) / g_SceneInfo.iSuperSamplingX;
		sy = (float) y + ( (float) currentSampleY + radicalInverse( y << 8 + currentSampleY, 5 ) ) / g_SceneInfo.iSuperSamplingY;
	} else {
		sx = (float) x + ( (float)currentSampleX + 0.5f ) / g_SceneInfo.iSuperSamplingX;
		sy = (float) y + ( (float)currentSampleY + 0.5f ) / g_SceneInfo.iSuperSamplingY;
	}

	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * sx * g_CameraInfo.stepX - 
		  g_CameraInfo.v * sy * g_CameraInfo.stepY;
	dir = normalize( dir - pos );

	//inRays[ rayIndex ].init();
	inRays[ rayIndex ].dir.x = dir.x;
	inRays[ rayIndex ].dir.y = dir.y;
	inRays[ rayIndex ].dir.z = dir.z;
//	inRays[ rayIndex ].dir.w = 0.0f;
	inRays[ rayIndex ].pos.x = pos.x;
	inRays[ rayIndex ].pos.y = pos.y;
	inRays[ rayIndex ].pos.z = pos.z;
//	inRays[ rayIndex ].pos.w = FLT_MAX;
	
	/**
	 *	super sampling 경우, 한 pixel 의 하나의 sample ray 가 pixel 에 미칠 영향.
	 */
	float samplingWeight = __fdividef( 1.0f, ( g_SceneInfo.iSuperSamplingX * g_SceneInfo.iSuperSamplingY ) );

	/**
	 *	intersection 결과를 저장할 데이터에 colorWeight 는 1.0 으로 한다.
	 *	primary ray 가 이미지에 영향을 줄 weight.
	 */
	pIntersectResult[ rayIndex ].init();
	pIntersectResult[ rayIndex ].boundDepth = 0;
	pIntersectResult[ rayIndex ].rayIndex = rayIndex;
	pIntersectResult[ rayIndex ].colorWeight = make_float3( samplingWeight, samplingWeight, samplingWeight );
}

/**
 *	intersection point 를 체크해서, reflection ray 를 생성한다. 
 *	그리고 다시 intersection point 에는 해당 ray 를 추적한 결과를 위해서
 *	초기화 한다.
 *	최소한 하나라도 intersection point 가 있을때 atLeast 가 1 로 세팅됨.
 *	멀티쓰레드에 의해서 동시에 atLeast 가 접근되더라도 상관없다. 하나라도
 *	1이라면 1이 될테니.
 *
 *	만약 ray 의 colorWeight 가 지정된 threashold 보다 밑으로 내려가는 경우는
 *	더이상 추적하지 않는다. color 값에 거의 영향을 안주는 weight 가 되었을때.
 *	TODO: 
 */
__global__ void generateReflectionRayKernel( int startIntersectionIndex, int maxIndex,
											 cuIntersectionPoint *intersectResult,
											 cuRay* inRays,
											 int *atLeast )
{
	int generate = 0, rayIndex = 0;
	
	/** 
	 *	현재 thread 에서 생성할 ray index. 화면 left, top 에서부터의 순서를 의미 
	 */

	int index = samplingRayID_BlockGrouping( startIntersectionIndex );

	/**
	 *	몇개의 쓰레드가 여분으로 더 실행될지 모르므로 index 범위를 초과한
	 *	쓰레드는 그냥 종료.
	 */
	if ( index >= maxIndex )
		return;

	float3 R;

	rayIndex = intersectResult[ index ].rayIndex;
	
	if ( intersectResult[ index ].isHit() ) {
		
		cuObjectMaterial material;
		getObjectMaterial( intersectResult[ index ].objectIndex, material );

		if ( material.transparency > 0.0f ) {
		
			generate = 1;
		
			R = refraction( intersectResult[ index ].dir, 
							intersectResult[ index ].normal, 
							material.refractionIndex );
			
			//inRays[ index ].setPrevTriIndex( intersectResult[ index ].triIndex );
			inRays[ index ].dir.x = R.x;
			inRays[ index ].dir.y = R.y;
			inRays[ index ].dir.z = R.z;
//			inRays[ index ].dir.w = 0.0f;

			/** 
			 *	self intersection 을 막기위해서 위치 조금 증가.
			 */
			inRays[ index ].pos.x = intersectResult[ index ].pos.x + R.x * RAY_START_EPSILON;
			inRays[ index ].pos.y = intersectResult[ index ].pos.y + R.y * RAY_START_EPSILON;
			inRays[ index ].pos.z = intersectResult[ index ].pos.z + R.z * RAY_START_EPSILON;
//			inRays[ index ].pos.w = FLT_MAX;
			
			/**
			 *	ray 의 intersection 결과를 저장할 장소를 초기화 하고,
			 *	intersection 결과가 이미지에 줄 영향을 colorWeight 에 세팅한다.
			 *	rayIndex 는 기존 ray 와 동일한 index 로 세팅한다.
			 */
			intersectResult[ index ].init();
			intersectResult[ index ].colorWeight = 
				intersectResult[ index ].colorWeight * material.transparency * ( material.diffuse );

			intersectResult[ index ].boundDepth++;
			intersectResult[ index ].rayIndex = rayIndex;

			(*atLeast) = 1;

		} else if ( material.reflection > 0.0f ) {
		
			generate = 1;

			R = reflection( intersectResult[ index ].dir, intersectResult[ index ].normal );
				
			//inRays[ index ].setPrevTriIndex( intersectResult[ index ].triIndex );
			inRays[ index ].dir.x = R.x;
			inRays[ index ].dir.y = R.y;
			inRays[ index ].dir.z = R.z;
//			inRays[ index ].dir.w = 0.0f;

			/** 
			 *	self intersection 을 막기위해서 위치 조금 증가.
			 */
			inRays[ index ].pos.x = intersectResult[ index ].pos.x + R.x * RAY_START_EPSILON;
			inRays[ index ].pos.y = intersectResult[ index ].pos.y + R.y * RAY_START_EPSILON;
			inRays[ index ].pos.z = intersectResult[ index ].pos.z + R.z * RAY_START_EPSILON;
//			inRays[ index ].pos.w = FLT_MAX;
				
			/**
			 *	ray 의 intersection 결과를 저장할 장소를 초기화 하고,
			 *	intersection 결과가 이미지에 줄 영향을 colorWeight 에 세팅하고
			 *	rayIndex 는 기존 ray 와 동일한 index 로 세팅한다.
			 */
			intersectResult[ index ].init();
			intersectResult[ index ].colorWeight = 
				intersectResult[ index ].colorWeight * material.reflection * ( material.diffuse );
			intersectResult[ index ].boundDepth++;
			intersectResult[ index ].rayIndex = rayIndex;

			(*atLeast) = 1;

		}
		
	}
	
	/**
	 *	secondary ray 가 없거나
	 *	color 의 weight 를 계산해서, 지정된 threshold 값 밑이면 추적하지 않는다.
	 *  mint 를 FLT_MAX 로 세팅.
	 */
	if ( generate == 0 || (
			intersectResult[ index ].colorWeight.x <= COLOR_WEIGHT_THREADHOLD &&
			intersectResult[ index ].colorWeight.y <= COLOR_WEIGHT_THREADHOLD &&
			intersectResult[ index ].colorWeight.z <= COLOR_WEIGHT_THREADHOLD ) )
	{
		intersectResult[ index ].init();
		intersectResult[ index ].rayIndex = rayIndex;

		inRays[ index ].pos = make_float3( FLT_MAX, FLT_MAX, FLT_MAX );
		inRays[ index ].dir = make_float3( 1.0f, 0.0f, 0.0f );

		//inRays[ index ].init();
//		inRays[ index ].dir.w = FLT_MAX;
//		inRays[ index ].pos.w = FLT_MAX;
	}
	
}


/**------------------------------------------------------------------------------------------
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **	SINGLE PASS RAY TRACING
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **------------------------------------------------------------------------------------------*/

/**
 *	anti-aliasing 을 위한 filter. frame buffer 는 texture 로도 올려져
 *	있으므로 inFrameBufferTexture 로 접근하면 된다.
 */
__global__ void antialiasingFilteringKernel( float* m_pFrameBuffer2 )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float3 avgColor = make_float3( 0.0f, 0.0f, 0.0f );
	int count = 0, index = 0;
	float grayScale = 0.0f;
	float xvalue = 0.0f, yvalue = 0.0f;
	float hx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	float hy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;
	
	/** sobel method 적용 */
	for ( int j = -1; j <= 1; ++j ) {
		for ( int i = -1; i <= 1; ++i ) {
			if ( x + i >= 0 && x + i < g_SceneInfo.iResolutionX && 
				 y + j >= 0 && y + j < g_SceneInfo.iResolutionY ) {
				index = 3 * ( ( y + j ) * g_SceneInfo.iResolutionX + ( x + i ) );
				grayScale = 0.3f * tex1Dfetch( inFrameBufferTexture, index + 0 ) +
							0.59f * tex1Dfetch( inFrameBufferTexture, index + 1 ) +
							0.11f * tex1Dfetch( inFrameBufferTexture, index + 2 );
				xvalue += hx[ j + 1 ][ i + 1 ] * grayScale;
				yvalue += hy[ j + 1 ][ i + 1 ] * grayScale;
			}
		}
	}

	if ( fabs( xvalue ) + fabs( yvalue ) >= EDGE_THRESHOLD ) {

		for ( int j = -2; j <= 2; ++j ) {
			for ( int i = -2; i <= 2; ++i ) {
				if ( x + i >= 0 && x + i < g_SceneInfo.iResolutionX && 
					 y + j >= 0 && y + j < g_SceneInfo.iResolutionY ) {
					index = 3 * ( ( y + j ) * g_SceneInfo.iResolutionX + ( x + i ) );
					avgColor.x += tex1Dfetch( inFrameBufferTexture, index + 0 );
					avgColor.y += tex1Dfetch( inFrameBufferTexture, index + 1 );
					avgColor.z += tex1Dfetch( inFrameBufferTexture, index + 2 );

					count++;
				}
			}
		}

	} else {
		index = 3 * ( y * g_SceneInfo.iResolutionX + x );
		avgColor.x += tex1Dfetch( inFrameBufferTexture, index + 0 );
		avgColor.y += tex1Dfetch( inFrameBufferTexture, index + 1 );
		avgColor.z += tex1Dfetch( inFrameBufferTexture, index + 2 );
		count = 1;
	}

	index = 3 * ( y * g_SceneInfo.iResolutionX + x );

	m_pFrameBuffer2[ index + 0 ] = avgColor.x / count;
	m_pFrameBuffer2[ index + 1 ] = avgColor.y / count;
	m_pFrameBuffer2[ index + 2 ] = avgColor.z / count;
}


/**
 *	anti-aliasing 을 위한 filter. frame buffer 는 texture 로도 올려져
 *	있으므로 inFrameBufferTexture 로 접근하면 된다.
 */
__global__ void blurringFilteringKernel( float* m_pFrameBuffer2 )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float3 avgColor = make_float3( 0.0f, 0.0f, 0.0f );
	int count = 0, index = 0;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;
	
	for ( int j = -1; j <= 1; ++j ) {
		for ( int i = -1; i <= 1; ++i ) {
			if ( x + i >= 0 && x + i < g_SceneInfo.iResolutionX && 
				 y + j >= 0 && y + j < g_SceneInfo.iResolutionY ) {
				index = 3 * ( ( y + j ) * g_SceneInfo.iResolutionX + ( x + i ) );
				avgColor.x += tex1Dfetch( inFrameBufferTexture, index + 0 );
				avgColor.y += tex1Dfetch( inFrameBufferTexture, index + 1 );
				avgColor.z += tex1Dfetch( inFrameBufferTexture, index + 2 );
				count++;
			}
		}
	}

	index = 3 * ( y * g_SceneInfo.iResolutionX + x );

	m_pFrameBuffer2[ index + 0 ] = avgColor.x / count;
	m_pFrameBuffer2[ index + 1 ] = avgColor.y / count;
	m_pFrameBuffer2[ index + 2 ] = avgColor.z / count;
}

/**
 *	anti-aliasing 을 위한 filter. frame buffer 는 texture 로도 올려져
 *	있으므로 inFrameBufferTexture 로 접근하면 된다.
 */
__global__ void sobelMethodFilteringKernel( float* m_pFrameBuffer2 )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = 0;
	float grayScale = 0.0f;
	float xvalue = 0.0f, yvalue = 0.0f;
	float hx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	float hy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;
	
	/** sobel method 적용 */
	for ( int j = -1; j <= 1; ++j ) {
		for ( int i = -1; i <= 1; ++i ) {
			if ( x + i >= 0 && x + i < g_SceneInfo.iResolutionX && 
				 y + j >= 0 && y + j < g_SceneInfo.iResolutionY ) {
				index = 3 * ( ( y + j ) * g_SceneInfo.iResolutionX + ( x + i ) );
				grayScale = 0.3f * tex1Dfetch( inFrameBufferTexture, index + 0 ) +
							0.59f * tex1Dfetch( inFrameBufferTexture, index + 1 ) +
							0.11f * tex1Dfetch( inFrameBufferTexture, index + 2 );
				xvalue += hx[ j + 1 ][ i + 1 ] * grayScale;
				yvalue += hy[ j + 1 ][ i + 1 ] * grayScale;
			}
		}
	}

	if ( fabs( xvalue ) + fabs( yvalue ) > EDGE_THRESHOLD ) {
		index = 3 * ( y * g_SceneInfo.iResolutionX + x );
		m_pFrameBuffer2[ index + 0 ] = 1.0;
		m_pFrameBuffer2[ index + 1 ] = 1.0;
		m_pFrameBuffer2[ index + 2 ] = 1.0;
	} else {
		index = 3 * ( y * g_SceneInfo.iResolutionX + x );
		m_pFrameBuffer2[ index + 0 ] = 0.0;
		m_pFrameBuffer2[ index + 1 ] = 0.0;
		m_pFrameBuffer2[ index + 2 ] = 0.0;
	}
}


/**
 *	gray scale filter.
 */
__global__ void grayScaleFilteringKernel( float* m_pFrameBuffer2 )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = 0;
	float grayScale = 0.0f;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;
	
	index = 3 * ( y * g_SceneInfo.iResolutionX + x );
	grayScale = 0.3f * tex1Dfetch( inFrameBufferTexture, index + 0 ) +
				0.59f * tex1Dfetch( inFrameBufferTexture, index + 1 ) +
				0.11f * tex1Dfetch( inFrameBufferTexture, index + 2 );

	m_pFrameBuffer2[ index + 0 ] = grayScale;
	m_pFrameBuffer2[ index + 1 ] = grayScale;
	m_pFrameBuffer2[ index + 2 ] = grayScale;
}

/**
 *	현재 frame buffer 에 blooming 효과를 적용한다.
 *	blooming 코드는 Dr. 차득현군의 headlight 소스에서 가져왔음을 밝히는 바입니다.
 */
__global__ void bloomingFilteringKernel( int startImageIndex,
										 int maxIndex,
										 float* m_pFrameBuffer,
										 float* m_pFrameBuffer2,
										 float* m_pBloomingFilter, 
										 int m_iBloomingWidth,
										 float m_fBloomingWeight )
{
	/** 
	 *	현재 thread 에서 처리할 image pixel index. 화면 left, top 에서부터의 순서를 의미 
	 */
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int imageIndex = y * g_SceneInfo.iResolutionX + x;
	
	/**
	 *	몇개의 쓰레드가 여분으로 더 실행될지 모르므로 index 범위를 초과한
	 *	쓰레드는 그냥 종료.
	 */
	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	int imageWidth = g_SceneInfo.iResolutionX;
	int imageHeight = g_SceneInfo.iResolutionY;

	int x0 = max( 0, x - m_iBloomingWidth );
	int x1 = min( x + m_iBloomingWidth, imageWidth - 1 );
	int y0 = max( 0, y - m_iBloomingWidth );
	int y1 = min( y + m_iBloomingWidth, imageHeight - 1 );
	int dx, dy, dist2;
	float sumWt = 0.0f, wt = 0.0f;
	float3 blooming = make_float3( 0.0f, 0.0f, 0.0f );
	int bloomOffset;

	for ( int by = y0; by <= y1; ++by ) {

		for ( int bx = x0; bx <= x1; ++bx ) {

			dx = x - bx; dy = y - by;
			if ( dx == 0 && dy == 0 ) continue;

			dist2 = dx * dx + dy * dy;

			if ( dist2 < m_iBloomingWidth * m_iBloomingWidth ) {
			
				bloomOffset = bx + by * imageWidth;
				wt = tex1Dfetch( inBloomingFilterTexture, dist2 );
				sumWt += wt;

				/** loop 안에서 많이 접근하므로 texture 로 접근하는게 더 빠를것이다. */
				blooming.x += wt * tex1Dfetch( inFrameBufferTexture, 3 * bloomOffset + 0 );
				blooming.y += wt * tex1Dfetch( inFrameBufferTexture, 3 * bloomOffset + 1 );
				blooming.z += wt * tex1Dfetch( inFrameBufferTexture, 3 * bloomOffset + 2 );

			}

		}
	}

	blooming /= sumWt;

	/** 
	 *	m_pFrameBuffer2 에 써놓는다. 
	 *	이 kernel 이 끝나고 나면 m_pFrameBuffer 과 m_pFrameBuffer2 를 swap 할 것이다. 
	 */
	m_pFrameBuffer2[ imageIndex * 3 + 0 ] = 
		( 1.0f- m_fBloomingWeight ) * m_pFrameBuffer[ imageIndex * 3 + 0 ] + m_fBloomingWeight * blooming.x;	
	m_pFrameBuffer2[ imageIndex * 3 + 1 ] = 
		( 1.0f- m_fBloomingWeight ) * m_pFrameBuffer[ imageIndex * 3 + 1 ] + m_fBloomingWeight * blooming.y;	
	m_pFrameBuffer2[ imageIndex * 3 + 2 ] = 
		( 1.0f- m_fBloomingWeight ) * m_pFrameBuffer[ imageIndex * 3 + 2 ] + m_fBloomingWeight * blooming.z;	
}


/**----------------------------------------------------------------------------------------------------
 **
 **
 **
 **
 **
 **
 **
 **
 **	SINGLE PASS RAYTRACING KERNEL 관련
 **	다른 논문들과의 속도비교를 위해서 한 kernel 안에서 primary 생성. shadow. shading 까지 최적으로
 **	수행하는 버전. 속도를 위한 것이므로 flexibility 가 떨어져 실제 응용에는 적합하지 않다.
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **
 **---------------------------------------------------------------------------------------------------*/

/**
 *	Pluecker Triangle Intersection.
 */
__device__ inline void PlueckerIntersection_ForPaper( const cuRay &ray, const int id,
													  cuIntersectionCheck &hit,
													  const float t_near, const float t_far )
{
	cuPlueckerTriangleInfo tri;
	float4 temp;
	float3 dir;
	float t;

	temp = tex1Dfetch( inPlueckerTriangleTex, 4 * id );
	tri.p0.x = temp.x; tri.p0.y = temp.y; tri.p0.z = temp.z; tri.p1.x = temp.w;
	temp = tex1Dfetch( inPlueckerTriangleTex, 4 * id + 1 );
	tri.p1.y = temp.x; tri.p1.z = temp.y; tri.p2.x = temp.z; tri.p2.y = temp.w;
	temp = tex1Dfetch( inPlueckerTriangleTex, 4 * id + 2 );
	tri.p2.z = temp.x; tri.normal.x = temp.y; tri.normal.y = temp.z; tri.normal.z = temp.w;
	temp = tex1Dfetch( inPlueckerTriangleTex, 4 * id + 3 );
	tri.attrib.x = temp.x; tri.attrib.y = temp.y;

	tri.p0.x = tri.p0.x - ray.pos.x; tri.p0.y = tri.p0.y - ray.pos.y; tri.p0.z = tri.p0.z - ray.pos.z;
	tri.p1.x = tri.p1.x - ray.pos.x; tri.p1.y = tri.p1.y - ray.pos.y; tri.p1.z = tri.p1.z - ray.pos.z;
	tri.p2.x = tri.p2.x - ray.pos.x; tri.p2.y = tri.p2.y - ray.pos.y; tri.p2.z = tri.p2.z - ray.pos.z;

	dir.x = ray.dir.x; dir.y = ray.dir.y; dir.z = ray.dir.z;

	temp.x = dot( dir, cross( tri.p1, tri.p0 ) );
	temp.y = dot( dir, cross( tri.p0, tri.p2 ) );
	temp.z = dot( dir, cross( tri.p2, tri.p1 ) );
	
	if ( ( temp.x >= 0.0f && temp.y >= 0.0f && temp.z >= 0.0f ) ||
		( temp.x <= 0.0f && temp.y <= 0.0f && temp.z <= 0.0f ) ) {

		t = __fdividef( dot( tri.normal, tri.p0 ), dot( dir, tri.normal ) );
		if ( ( hit.tHit <= t ) | ( t < t_near - EPSILON4 ) | ( t > t_far + EPSILON4 ) ) return;
	
		float in = 1.0f / ( temp.x + temp.y + temp.z );

		hit.tHit = t;
		hit.beta = temp.y * in;
		hit.gamma = temp.x * in;
		hit.triIndex = id;
		hit.objectIndex = tri.getObjectIndex();

	}
}

/**
 *	WALD Intersection method.
 */
__device__ inline void singlePassIntersectRoutine( const cuRay &ray, const int id, cuIntersectionCheck &hit, 
												   const float t_near, const float t_far  )
{
	cuWaldTriangleInfo tri;
	tri.internal0 = tex1Dfetch( inWaldTriangleTex, 3 * id );
	tri.internal1 = tex1Dfetch( inWaldTriangleTex, 3 * id + 1 );
	tri.internal2 = tex1Dfetch( inWaldTriangleTex, 3 * id + 2 );

	cuWaldTriangleInfo::perm_t p = tri.get_perm( ray );
	//const float dot = ( tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z );
	p.pos.x = ( tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z );
	const float denum = ( p.dir.x + tri.n_u() * p.dir.y + tri.n_v() * p.dir.z );
	const float t = __fdividef( p.pos.x, denum );
	
	if ( isnan( t ) ) return;
	if ( ( hit.tHit <= t ) | ( t < t_near - EPSILON4 ) | ( t > t_far + EPSILON4 ) ) return;
	
	/**
	 *	culling 옵션이 있고, object 가 transparent 하지 않다면
	 *	앞면인지 뒷면인지 체크. 뒷면에 맞은거면 hit 처리 안함.
	 */
	const float hu = p.pos.y + t * p.dir.y - tri.vert_ku();
	const float hv = p.pos.z + t * p.dir.z - tri.vert_kv();
	const float beta = hv * tri.b_nu() + hu * tri.b_nv();
	const float gamma = hu * tri.c_nu() + hv * tri.c_nv();
	
	/** 삼각형의 edge 와 부딪힐때, 수치오차가 있으므로 epsilon 을 좀 준다. */
	//if ( isnan( beta * gamma ) ) return;
	if ( ( beta < 0.f - BARYCENTRY_EPSILON ) | ( gamma < 0.f - BARYCENTRY_EPSILON ) | ( ( 1.0f - beta - gamma ) < 0.0f - BARYCENTRY_EPSILON ) ) return;

	hit.tHit = t;
	hit.beta = beta;
	hit.gamma = gamma;
	hit.triIndex = id;
	hit.objectIndex = tri.getObjectIndex();
	hit.bSelected = tri.isSelection();
}

__device__ inline void singlePassIntersect( cuRay &currRay, cuIntersectionCheck &intersectionCheck  )
{
	float t_scene_near = 0.0f, t_scene_far = FLT_MAX;	
	//float t_scene_near = currRay.mint, t_scene_far = currRay.maxt;
	if ( BoundsRayIntersect( g_SceneBBox, currRay, t_scene_near, t_scene_far ) )
	{
		float t_near = t_scene_near, t_far = t_scene_far;		
		const unsigned smem_baseOffset =  umul24(threadIdx.y , blockDim.x) + threadIdx.x;
		shortStack stack;
		stack.init(smem_baseOffset);

		kdtreeNode node = tex1Dfetch(inKdTreeNodeTex, 0);
		while(true)
		{
			while(!IS_LEAF(node))
			{
				//const int axis = SPLIT_AXIS(node);
				//const float splitPos = SPLIT_POS(node);
				const float2 pos_dir = currRay.get_dir_pos(SPLIT_AXIS(node)); 
				//const float dir = pos_dir.y;			
				const float t_split = __fdividef(SPLIT_POS(node) - pos_dir.x, pos_dir.y);				
				const unsigned sign = signbit(pos_dir.y);
				const unsigned childOffset = FIRST_CHILD_OFFSET(node);
				unsigned idx = childOffset + (sign^(t_split <= t_near));
				//if(t_split <= t_near) 
				//	idx = childOffset + (sign^1);
				if(t_near < t_split && t_split < t_far){
					stack.push(childOffset + (sign^1), t_far);
					t_far = t_split;
				}
				node = tex1Dfetch(inKdTreeNodeTex, idx);
			}

			
			unsigned baseOffset = OBJECTLIST_OFFSET(node);
			int objectSize = OBJECT_SIZE(node) + baseOffset;

			for(; baseOffset<objectSize ; baseOffset++) {
				const unsigned objListOffset = tex1Dfetch(inObjectOffsetListTex, baseOffset);
				#if INTERSECTION_METHOD == 0
					singlePassIntersectRoutine( currRay, objListOffset, intersectionCheck, t_near, t_far );
				#elif INTERSECTION_METHOD == 1
					PlueckerIntersection( currRay, objListOffset, intersectionCheck, t_near, t_far, faceCCW, bCulling );
				#endif
			}
			if( intersectionCheck.tHit <= t_far | t_far >= t_scene_far)
				break;
			if(stack.empty())
			{
				node = tex1Dfetch(inKdTreeNodeTex, 0);
				t_near = t_far;		t_far = t_scene_far;
			}
			else
			{	
				const cu_traceState &trace = stack.top(); stack.pop();
				node = tex1Dfetch(inKdTreeNodeTex, trace.nodeID);
				t_near = t_far;									
				t_far = trace.tMax;				
			}
		}
	}
}

/** 
 *	Texture 가 존재하는 경우 현재 diffuse color 를 texture 내의 u, v 상의
 *	color 로 대체.
 */
__device__ inline void calTextureColor( cuIntersectionPoint &intersectResult,
								   cuObjectMaterial &material, float3 &diffuse )
{
	int texture = float_as_int( material.textureNumber );
	intersectResult.bTexture = fetchTexture( texture, intersectResult.u, intersectResult.v, diffuse );
}

/**
 *	Scene 안의 광원으로부터 Direct Illumination 을 계산한다.
 *	만약 현재 물체 자체가 광원이라면, 현재 물체의 색을 그대로 사용한다.
 *
 *	Phong Shading.
 */
__device__ inline float3 calSinglePassDirectIllumination_ShadowOn( 
													cuIntersectionPoint &intersectResult, 
													cuObjectMaterial &material,
													const float3 &diffuse )
{
	cuLight *pLight = NULL;
	float3 L, N, R;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	N = intersectResult.normal;

	if ( material.transparency > 0.0f && dot( intersectResult.dir, N ) < 0.0f ) {
		N = -1.0f * N;
	}

	R = reflection( intersectResult.dir, N );
	
	color = material.ambient_emission;

	intersectResult.shadowCount = 0.0f;

	for ( int i = 0; i < constantLightCount; ++i ) {
	
		pLight = ( constantLightInfo + i );

		if ( checkVisibility( intersectResult.pos, pLight->pos ) ) {
		
			L = normalize( pLight->pos - intersectResult.pos );
			color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
						material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );

		} else {

			intersectResult.shadowCount++;

		}

	}

	return color;
	
}


__device__ inline float3 calSinglePassDirectIllumination_ShadowOff( 
										cuIntersectionPoint &intersectResult, 
										cuObjectMaterial &material,
										const float3 &diffuse )
{
	cuLight *pLight = NULL;
	float3 L, N, R;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	N = intersectResult.normal;

	if ( material.transparency > 0.0f && dot( intersectResult.dir, N ) < 0.0f ) {
		N = -1.0f * N;
	}

	R = reflection( intersectResult.dir, N );

	color = material.ambient_emission;

	for ( int i = 0; i < constantLightCount; ++i ) {
	
		pLight = ( constantLightInfo + i );

		L = normalize( pLight->pos - intersectResult.pos );
		color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
					material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );

	}

	return color;
	
}


/**
 *	광원개수 고정. 최대2개.
 *
 *	Phong Shading.
 */
__device__ inline float3 fixedOption_calSinglePassDirectIllumination_ShadowOn( 
													cuIntersectionPoint &intersectResult, 
													cuObjectMaterial &material,
													const float3 &diffuse )
{
	cuLight *pLight = NULL;
	float3 L, N, R;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	N = intersectResult.normal;

	if ( material.transparency > 0.0f && dot( intersectResult.dir, N ) < 0.0f ) {
		N = -1.0f * N;
	}

	R = reflection( intersectResult.dir, N );
	
	color = material.ambient_emission;

	intersectResult.shadowCount = 0.0f;

	if ( constantLightCount >= 1 ) {
	
		pLight = ( constantLightInfo + 0 );
		if ( checkVisibility( intersectResult.pos, pLight->pos ) ) {
		
			L = normalize( pLight->pos - intersectResult.pos );
			color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
					material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );

		} else {

			intersectResult.shadowCount++;

		}

	}

	if ( constantLightCount >= 2 ) {
	
		pLight = ( constantLightInfo + 1 );
		if ( checkVisibility( intersectResult.pos, pLight->pos ) ) {
		
			L = normalize( pLight->pos - intersectResult.pos );
			color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
					material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );

		} else {

			intersectResult.shadowCount++;

		}

	}

	return color;
	
}


/**
 *	광원개수 고정. 최대2개.
 *
 *	Phong Shading.
 */
__device__ inline float3 fixedOption_calSinglePassDirectIllumination_ShadowOn_ForSelective( 
													cuIntersectionPoint &intersectResult, 
													cuObjectMaterial &material,
													const float3 &diffuse )
{
	cuLight *pLight = NULL;
	float3 L, N, R;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	bool bSelected;

	N = intersectResult.normal;

	if ( material.transparency > 0.0f && dot( intersectResult.dir, N ) < 0.0f ) {
		N = -1.0f * N;
	}

	R = reflection( intersectResult.dir, N );
	
	color = material.ambient_emission;

	intersectResult.shadowCount = 0.0f;

	if ( constantLightCount >= 1 ) {
	
		pLight = ( constantLightInfo + 0 );
		if ( checkVisibility_ForSelective( intersectResult.pos, pLight->pos, bSelected ) ) {
		
			L = normalize( pLight->pos - intersectResult.pos );
			color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
					material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );
			intersectResult.bSelected += bSelected;

		} else {

			intersectResult.shadowCount++;

		}

	}

	if ( constantLightCount >= 2 ) {
	
		pLight = ( constantLightInfo + 1 );
		if ( checkVisibility_ForSelective( intersectResult.pos, pLight->pos, bSelected ) ) {
		
			L = normalize( pLight->pos - intersectResult.pos );
			color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
					material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );
			intersectResult.bSelected += bSelected;

		} else {

			intersectResult.shadowCount++;

		}

	}

	return color;
	
}

/**
 *	광원개수 고정. 최대 2개
 */
__device__ inline float3 fixedOption_calSinglePassDirectIllumination_ShadowOff( 
										cuIntersectionPoint &intersectResult, 
										cuObjectMaterial &material,
										const float3 &diffuse )
{
	cuLight *pLight = NULL;
	float3 L, N, R;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	N = intersectResult.normal;

	if ( material.transparency > 0.0f && dot( intersectResult.dir, N ) < 0.0f ) {
		N = -1.0f * N;
	}

	R = reflection( intersectResult.dir, N );

	color = material.ambient_emission;

	pLight = ( constantLightInfo + 0 );

	if ( constantLightCount >= 1 ) {
		L = normalize( pLight->pos - intersectResult.pos );
		color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
					material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );
	}

	if ( constantLightCount >= 2 ) {
		pLight = ( constantLightInfo + 1 );

		L = normalize( pLight->pos - intersectResult.pos );
		color += pLight->color * ( diffuse * max( 0.0f, dot( L, N ) ) + 
					material.specular * powf( max( 0.0f, dot( R, L ) ), material.roughness ) );
	}

	return color;
	
}

/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_ShadowOff( float* pFrameBuffer, 
													  int maxReflectionDepth, 
													  int sampleX, int sampleY, bool bJittering )
{
	float3 dir;	
	float sx, sy;
	float invSamplingX = __fdividef( 1.0f , g_SceneInfo.iSuperSamplingX );
	float invSamplingY = __fdividef( 1.0f , g_SceneInfo.iSuperSamplingY );
	int depth = 0;

	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int x = umul24(blockIdx.x , blockDim.x) + threadIdx.x;
	int y = umul24(blockIdx.y , blockDim.y) + threadIdx.y;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */

	if ( g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering ) {
		sx = (float) x + ( (float) sampleX + radicalInverse( x * 256 + sampleX, 3 ) ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + radicalInverse( y * 256 + sampleY, 5 ) ) * invSamplingY;
	} else {
		sx = (float) x + ( (float) sampleX + 0.5f ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + 0.5f ) * invSamplingY;
	}

	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * sx * g_CameraInfo.stepX - 
		  g_CameraInfo.v * sy * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	cuRay currRay;
	currRay.dir = dir;
	currRay.pos = g_CameraInfo.eye;

	cuIntersectionPoint point;
	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	bool secondary;

	do {
		secondary = false;
		cuIntersectionCheck currIsectCheck;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck );

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {
			
			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* calSinglePassDirectIllumination_ShadowOff( point, material, diffuse );				

			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight = point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			}


		}

		depth++;

	} while( secondary );
			
	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
	pFrameBuffer[ idx + 0 ] += color.x * invSamplingX * invSamplingY;
	pFrameBuffer[ idx + 1 ] += color.y * invSamplingX * invSamplingY;
	pFrameBuffer[ idx + 2 ] += color.z * invSamplingX * invSamplingY;
}

/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_ShadowOn( float* pFrameBuffer, 
													 int maxReflectionDepth, 
													 int sampleX, int sampleY, bool bJittering )
{
	float3 dir;	
	float sx, sy;
	float invSamplingX = __fdividef(1.0f , g_SceneInfo.iSuperSamplingX);
	float invSamplingY = __fdividef(1.0f , g_SceneInfo.iSuperSamplingY);
	int depth = 0;

	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int x = umul24(blockIdx.x , blockDim.x) + threadIdx.x;
	int y = umul24(blockIdx.y , blockDim.y) + threadIdx.y;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */

	if ( g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering ) {
		sx = (float) x + ( (float) sampleX + radicalInverse( x * 256 + sampleX, 3 ) ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + radicalInverse( y * 256 + sampleY, 5 ) ) * invSamplingY;
	} else {
		sx = (float) x + ( (float) sampleX + 0.5f ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + 0.5f ) * invSamplingY;
	}

	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * sx * g_CameraInfo.stepX - 
		  g_CameraInfo.v * sy * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	cuRay currRay;
	currRay.dir = dir;
	currRay.pos = g_CameraInfo.eye;

	cuIntersectionPoint point;
	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	bool secondary;

	do {
		secondary = false;
		cuIntersectionCheck currIsectCheck;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck );

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {
			
			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* calSinglePassDirectIllumination_ShadowOn( point, material, diffuse );				

			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight = point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			}


		}

		depth++;

	} while( secondary );
			
	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
	pFrameBuffer[ idx + 0 ] += color.x * invSamplingX * invSamplingY;
	pFrameBuffer[ idx + 1 ] += color.y * invSamplingX * invSamplingY;
	pFrameBuffer[ idx + 2 ] += color.z * invSamplingX * invSamplingY;

}

/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_Coherent_ShadowOff( float* pFrameBuffer, 
																int maxReflectionDepth, bool bJittering )
{
	float3 dir;	
	float sx, sy;
	float invSamplingX = __fdividef(1.0f , g_SceneInfo.iSuperSamplingX);
	float invSamplingY = __fdividef(1.0f , g_SceneInfo.iSuperSamplingY);
	int depth = 0;

	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int x = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) / g_SceneInfo.iSuperSamplingX;
	int y = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) / g_SceneInfo.iSuperSamplingY;
	int sampleX = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) % g_SceneInfo.iSuperSamplingX;
	int sampleY = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) % g_SceneInfo.iSuperSamplingY;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */

	if ( g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering ) {
		sx = (float) x + ( (float) sampleX + radicalInverse( x * 256 + sampleX, 3 ) ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + radicalInverse( y * 256 + sampleY, 5 ) ) * invSamplingY;
	} else {
		sx = (float) x + ( (float) sampleX + 0.5f ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + 0.5f ) * invSamplingY;
	}

	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * sx * g_CameraInfo.stepX - 
		  g_CameraInfo.v * sy * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	cuRay currRay;
	currRay.dir = dir;
	currRay.pos = g_CameraInfo.eye;

	cuIntersectionPoint point;
	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	bool secondary;

	do {
		secondary = false;
		cuIntersectionCheck currIsectCheck;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck );

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {
			
			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* calSinglePassDirectIllumination_ShadowOff( point, material, diffuse );				

			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight = point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			}


		}

		depth++;

	} while( secondary );
			
	__syncthreads();

	sharedMemory[ threadIndex * 3 + 0 ] = color.x * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 1 ] = color.y * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 2 ] = color.z * invSamplingX * invSamplingY;

	__syncthreads();

	if ( sampleX == 0 && sampleY == 0 ) {
		for ( int i = 0; i < g_SceneInfo.iSuperSamplingX; ++i ) {
			for ( int j = 0; j < g_SceneInfo.iSuperSamplingY; ++j ) {
				if ( i == 0 && j == 0 ) continue;
				int index2 = ( threadIdx.y + j ) * blockDim.x + ( threadIdx.x + i );
				sharedMemory[ threadIndex * 3 + 0 ] += sharedMemory[ index2 * 3 + 0 ];
				sharedMemory[ threadIndex * 3 + 1 ] += sharedMemory[ index2 * 3 + 1 ];
				sharedMemory[ threadIndex * 3 + 2 ] += sharedMemory[ index2 * 3 + 2 ];
			}
		}

		int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
		pFrameBuffer[ idx + 0 ] = sharedMemory[ threadIndex * 3 + 0 ];
		pFrameBuffer[ idx + 1 ] = sharedMemory[ threadIndex * 3 + 1 ];
		pFrameBuffer[ idx + 2 ] = sharedMemory[ threadIndex * 3 + 2 ];

	}

}


/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_Coherent_ShadowOn( float* pFrameBuffer, 
													 int maxReflectionDepth, bool bJittering )
{
	float3 dir;	
	float sx, sy;
	float invSamplingX = __fdividef(1.0f , g_SceneInfo.iSuperSamplingX);
	float invSamplingY = __fdividef(1.0f , g_SceneInfo.iSuperSamplingY);
	int depth = 0;

	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int x = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) / g_SceneInfo.iSuperSamplingX;
	int y = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) / g_SceneInfo.iSuperSamplingY;
	int sampleX = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) % g_SceneInfo.iSuperSamplingX;
	int sampleY = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) % g_SceneInfo.iSuperSamplingY;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */

	if ( g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering ) {
		sx = (float) x + ( (float) sampleX + radicalInverse( x * 256 + sampleX, 3 ) ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + radicalInverse( y * 256 + sampleY, 5 ) ) * invSamplingY;
	} else {
		sx = (float) x + ( (float) sampleX + 0.5f ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + 0.5f ) * invSamplingY;
	}

	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * sx * g_CameraInfo.stepX - 
		  g_CameraInfo.v * sy * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	cuRay currRay;
	currRay.dir = dir;
	currRay.pos = g_CameraInfo.eye;

	cuIntersectionPoint point;
	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	bool secondary;

	do {
		secondary = false;
		cuIntersectionCheck currIsectCheck;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck );

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {
			
			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* calSinglePassDirectIllumination_ShadowOn( point, material, diffuse );				

			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight = point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			}


		}

		depth++;

	} while( secondary );
			
	__syncthreads();

	sharedMemory[ threadIndex * 3 + 0 ] = color.x * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 1 ] = color.y * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 2 ] = color.z * invSamplingX * invSamplingY;

	__syncthreads();

	if ( sampleX == 0 && sampleY == 0 ) {
		for ( int i = 0; i < g_SceneInfo.iSuperSamplingX; ++i ) {
			for ( int j = 0; j < g_SceneInfo.iSuperSamplingY; ++j ) {
				if ( i == 0 && j == 0 ) continue;
				int index2 = ( threadIdx.y + j ) * blockDim.x + ( threadIdx.x + i );
				sharedMemory[ threadIndex * 3 + 0 ] += sharedMemory[ index2 * 3 + 0 ];
				sharedMemory[ threadIndex * 3 + 1 ] += sharedMemory[ index2 * 3 + 1 ];
				sharedMemory[ threadIndex * 3 + 2 ] += sharedMemory[ index2 * 3 + 2 ];
			}
		}

		int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
		pFrameBuffer[ idx + 0 ] = sharedMemory[ threadIndex * 3 + 0 ];
		pFrameBuffer[ idx + 1 ] = sharedMemory[ threadIndex * 3 + 1 ];
		pFrameBuffer[ idx + 2 ] = sharedMemory[ threadIndex * 3 + 2 ];

	}

}


/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void fixedOption_singlePassRayTracingKernel_Coherent_ShadowOff( float* pFrameBuffer, 
																int maxReflectionDepth, bool bJittering )
{
	float3 dir;	
	float sx, sy;
	float invSamplingX = __fdividef(1.0f , g_SceneInfo.iSuperSamplingX);
	float invSamplingY = __fdividef(1.0f , g_SceneInfo.iSuperSamplingY);
	int depth = 0;

	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int x = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) / g_SceneInfo.iSuperSamplingX;
	int y = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) / g_SceneInfo.iSuperSamplingY;
	int sampleX = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) % g_SceneInfo.iSuperSamplingX;
	int sampleY = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) % g_SceneInfo.iSuperSamplingY;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */

	if ( g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering ) {
		sx = (float) x + ( (float) sampleX + radicalInverse( x * 256 + sampleX, 3 ) ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + radicalInverse( y * 256 + sampleY, 5 ) ) * invSamplingY;
	} else {
		sx = (float) x + ( (float) sampleX + 0.5f ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + 0.5f ) * invSamplingY;
	}

	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * sx * g_CameraInfo.stepX - 
		  g_CameraInfo.v * sy * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	cuRay currRay;
	currRay.dir = dir;
	currRay.pos = g_CameraInfo.eye;

	cuIntersectionPoint point;
	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	bool secondary;

	do {
		secondary = false;
		cuIntersectionCheck currIsectCheck;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck );

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {
			
			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* fixedOption_calSinglePassDirectIllumination_ShadowOff( point, material, diffuse );				

			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight = point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			}


		}

		depth++;

	} while( secondary );
			
	__syncthreads();

	sharedMemory[ threadIndex * 3 + 0 ] = color.x * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 1 ] = color.y * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 2 ] = color.z * invSamplingX * invSamplingY;

	__syncthreads();

	if ( sampleX == 0 && sampleY == 0 ) {
		for ( int i = 0; i < g_SceneInfo.iSuperSamplingX; ++i ) {
			for ( int j = 0; j < g_SceneInfo.iSuperSamplingY; ++j ) {
				if ( i == 0 && j == 0 ) continue;
				int index2 = ( threadIdx.y + j ) * blockDim.x + ( threadIdx.x + i );
				sharedMemory[ threadIndex * 3 + 0 ] += sharedMemory[ index2 * 3 + 0 ];
				sharedMemory[ threadIndex * 3 + 1 ] += sharedMemory[ index2 * 3 + 1 ];
				sharedMemory[ threadIndex * 3 + 2 ] += sharedMemory[ index2 * 3 + 2 ];
			}
		}

		int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
		pFrameBuffer[ idx + 0 ] = sharedMemory[ threadIndex * 3 + 0 ];
		pFrameBuffer[ idx + 1 ] = sharedMemory[ threadIndex * 3 + 1 ];
		pFrameBuffer[ idx + 2 ] = sharedMemory[ threadIndex * 3 + 2 ];

	}

}


/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void fixedOption_singlePassRayTracingKernel_Coherent_ShadowOn( float* pFrameBuffer, 
													 int maxReflectionDepth, bool bJittering )
{
	float3 dir;	
	float sx, sy;
	float invSamplingX = __fdividef(1.0f , g_SceneInfo.iSuperSamplingX);
	float invSamplingY = __fdividef(1.0f , g_SceneInfo.iSuperSamplingY);
	int depth = 0;

	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int x = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) / g_SceneInfo.iSuperSamplingX;
	int y = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) / g_SceneInfo.iSuperSamplingY;
	int sampleX = ( umul24(blockIdx.x , blockDim.x) + threadIdx.x ) % g_SceneInfo.iSuperSamplingX;
	int sampleY = ( umul24(blockIdx.y , blockDim.y) + threadIdx.y ) % g_SceneInfo.iSuperSamplingY;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */

	if ( g_SceneInfo.iSuperSamplingX > 1 && g_SceneInfo.iSuperSamplingY > 1 && bJittering ) {
		sx = (float) x + ( (float) sampleX + radicalInverse( x * 256 + sampleX, 3 ) ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + radicalInverse( y * 256 + sampleY, 5 ) ) * invSamplingY;
	} else {
		sx = (float) x + ( (float) sampleX + 0.5f ) * invSamplingX;
		sy = (float) y + ( (float) sampleY + 0.5f ) * invSamplingY;
	}

	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * sx * g_CameraInfo.stepX - 
		  g_CameraInfo.v * sy * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	cuRay currRay;
	currRay.dir = dir;
	currRay.pos = g_CameraInfo.eye;

	cuIntersectionPoint point;
	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	bool secondary;

	do {
		secondary = false;
		cuIntersectionCheck currIsectCheck;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck );

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {
			
			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* fixedOption_calSinglePassDirectIllumination_ShadowOn( point, material, diffuse );				

			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight = point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;


				secondary = true;

			}


		}

		depth++;

	} while( secondary );
			
	__syncthreads();

	sharedMemory[ threadIndex * 3 + 0 ] = color.x * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 1 ] = color.y * invSamplingX * invSamplingY;
	sharedMemory[ threadIndex * 3 + 2 ] = color.z * invSamplingX * invSamplingY;

	__syncthreads();

	if ( sampleX == 0 && sampleY == 0 ) {
		for ( int i = 0; i < g_SceneInfo.iSuperSamplingX; ++i ) {
			for ( int j = 0; j < g_SceneInfo.iSuperSamplingY; ++j ) {
				if ( i == 0 && j == 0 ) continue;
				int index2 = ( threadIdx.y + j ) * blockDim.x + ( threadIdx.x + i );
				sharedMemory[ threadIndex * 3 + 0 ] += sharedMemory[ index2 * 3 + 0 ];
				sharedMemory[ threadIndex * 3 + 1 ] += sharedMemory[ index2 * 3 + 1 ];
				sharedMemory[ threadIndex * 3 + 2 ] += sharedMemory[ index2 * 3 + 2 ];
			}
		}

		int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
		pFrameBuffer[ idx + 0 ] = sharedMemory[ threadIndex * 3 + 0 ];
		pFrameBuffer[ idx + 1 ] = sharedMemory[ threadIndex * 3 + 1 ];
		pFrameBuffer[ idx + 2 ] = sharedMemory[ threadIndex * 3 + 2 ];

	}

}

/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_1_SamplingKernel_ShadowOff( 
											cuSamplingMap *pSamplingMap,
											float* pFrameBuffer, float* pFrameBuffer2, 
											int maxReflectionDepth )
{
	float3 dir;
	bool secondary = false;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int rayIndex = y * g_SceneInfo.iResolutionX + x;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */
	dir = g_CameraInfo.startPoint + 
		  g_CameraInfo.u * ((float)x + 0.5f ) * g_CameraInfo.stepX - 
		  g_CameraInfo.v * ((float)y + 0.5f ) * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	currRay.dir = make_float3( dir.x, dir.y, dir.z );
	currRay.pos = make_float3( g_CameraInfo.eye.x, g_CameraInfo.eye.y, g_CameraInfo.eye.z );

	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	/** 
	 *	현재지점의 sampling info 초기화.
	 *	integer 를 float 로 int_as_float_H 식으로 저장.
	 *
	 *	normal 은 다 1.0 으로 세팅해야 한다.
	 */
	pSamplingMap[ rayIndex ].primaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].primaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );
	pSamplingMap[ rayIndex ].secondaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].secondaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );

	do {

		secondary = false;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck  );

		point.normal = make_float3( 1.0f, 1.0f, 1.0f );
		point.objectIndex = OBJECT_MAX_ID;
		point.shadowCount = 0;
		point.bTexture = 0;

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {

			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* calSinglePassDirectIllumination_ShadowOff( point, material, diffuse );				

			/**
			 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
			 */
			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight =  point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z; 
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			}

			if ( depth == 0 ) {
				pSamplingMap[ rayIndex ].primaryNormal = point.normal;
				pSamplingMap[ rayIndex ].primaryAttr =  
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

			/** 2차 ray 가 intersection 한경우. */
			if ( depth == 1 ) {
				pSamplingMap[ rayIndex ].secondaryNormal = point.normal;
				pSamplingMap[ rayIndex ].secondaryAttr = 
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

		}

		depth++;

	} while( secondary );
			
	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer2[ idx + 0 ] = color.x;
	pFrameBuffer2[ idx + 1 ] = color.y;
	pFrameBuffer2[ idx + 2 ] = color.z;

	pFrameBuffer[ idx + 0 ] = color.x;
	pFrameBuffer[ idx + 1 ] = color.y;
	pFrameBuffer[ idx + 2 ] = color.z;

}


/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_1_SamplingKernel_ShadowOn( 
											cuSamplingMap *pSamplingMap,
											float* pFrameBuffer, float* pFrameBuffer2, 
											int maxReflectionDepth )
{
	float3 dir;
	bool secondary = false;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int rayIndex = y * g_SceneInfo.iResolutionX + x;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */
	dir = g_CameraInfo.startPoint + g_CameraInfo.u * ((float)x + 0.5f ) * g_CameraInfo.stepX - 
		  g_CameraInfo.v * ((float)y + 0.5f ) * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	currRay.dir = make_float3( dir.x, dir.y, dir.z );
	currRay.pos = make_float3( g_CameraInfo.eye.x, g_CameraInfo.eye.y, g_CameraInfo.eye.z );

	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	/** 
	 *	현재지점의 sampling info 초기화.
	 *	integer 를 float 로 int_as_float_H 식으로 저장.
	 *
	 *	normal 은 다 1.0 으로 세팅해야 한다.
	 */
	pSamplingMap[ rayIndex ].primaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].primaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );
	pSamplingMap[ rayIndex ].secondaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].secondaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );

	do {

		secondary = false;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck  );

		point.normal = make_float3( 1.0f, 1.0f, 1.0f );
		point.objectIndex = OBJECT_MAX_ID;
		point.shadowCount = 0;
		point.bTexture = 0;

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {

			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* calSinglePassDirectIllumination_ShadowOn( point, material, diffuse );				

			/**
			 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
			 */
			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight =  point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z; 
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			}

			if ( depth == 0 ) {
				pSamplingMap[ rayIndex ].primaryNormal = point.normal;
				pSamplingMap[ rayIndex ].primaryAttr =  
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

			/** 2차 ray 가 intersection 한경우. */
			if ( depth == 1 ) {
				pSamplingMap[ rayIndex ].secondaryNormal = point.normal;
				pSamplingMap[ rayIndex ].secondaryAttr = 
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

		}

		depth++;

	} while( secondary );
			
	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer2[ idx + 0 ] = color.x;
	pFrameBuffer2[ idx + 1 ] = color.y;
	pFrameBuffer2[ idx + 2 ] = color.z;

	pFrameBuffer[ idx + 0 ] = color.x;
	pFrameBuffer[ idx + 1 ] = color.y;
	pFrameBuffer[ idx + 2 ] = color.z;

}

/**
 *	Sobel Method 로 현재 Pixel 의 difference value 를 생성.
 */
__global__ void colorDifferenceMapGenerationKernel_SobelMethod( float* pFrameBuffer2 )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = 0;
	float grayScale = 0.0f;
	float xvalue = 0.0f, yvalue = 0.0f;
	float hx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	float hy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;
	
	/** sobel method 적용 */
	for ( int j = -1; j <= 1; ++j ) {
		for ( int i = -1; i <= 1; ++i ) {
			if ( x + i >= 0 && x + i < g_SceneInfo.iResolutionX && 
				 y + j >= 0 && y + j < g_SceneInfo.iResolutionY ) {
				index = 3 * ( ( y + j ) * g_SceneInfo.iResolutionX + ( x + i ) );
				grayScale = 0.3f * tex1Dfetch( inFrameBufferTexture, index + 0 ) +
							0.59f * tex1Dfetch( inFrameBufferTexture, index + 1 ) +
							0.11f * tex1Dfetch( inFrameBufferTexture, index + 2 );
				xvalue += hx[ j + 1 ][ i + 1 ] * grayScale;
				yvalue += hy[ j + 1 ][ i + 1 ] * grayScale;
			}
		}
	}
	
	/** 상하가 바뀌어 있으므로 framebuffer2 에는 뒤집어서 저장 */
	index = ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
	pFrameBuffer2[ index ] = fmin( 1.0f, fabs( xvalue ) + fabs( yvalue ) );

}

__global__ void laplacianOfGaussianKernel9x9( float* pFrameBuffer2 )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float edgevalue = 0.0f;
	float grayScale = 0.0f;
	int index;

	float log[9][9] = { { 0, 1, 1, 2, 2, 2, 1, 1, 0 }, 
						{ 1, 2, 4, 5, 5, 5, 4, 2, 1 }, 
						{ 1, 4, 5, 3, 0, 3, 5, 4, 1 }, 
						{ 2, 5, 3, -12, -24, -12, 3, 5, 2 },
						{ 2, 5, 0, -24, -40, -24, 0, 5, 2 }, 
						{ 2, 5, 3, -12, -24, -12, 3, 5, 2 },
						{ 1, 4, 5, 3, 0, 3, 5, 4, 1 }, 
						{ 1, 2, 4, 5, 5, 5, 4, 2, 1 }, 
						{ 0, 1, 1, 2, 2, 2, 1, 1, 0 } };

	for ( int j = -4; j <= 4; ++j ) {
		for ( int i = -4; i <= 4; ++i ) {
			if ( x + i >= 0 && x + i < g_SceneInfo.iResolutionX && y + j >= 0 && y + j < g_SceneInfo.iResolutionY ) {
				index = 3 * ( ( g_SceneInfo.iResolutionY - ( y + j ) - 1 ) * g_SceneInfo.iResolutionX + ( x + i ) );
				grayScale = 0.3f * tex1Dfetch( inFrameBufferTexture, index + 0 ) +
							0.59f * tex1Dfetch( inFrameBufferTexture, index + 1 ) +
							0.11f * tex1Dfetch( inFrameBufferTexture, index + 2 );
				edgevalue += log[ j + 4 ][ i + 4 ] * grayScale;
			}
		}
	}

	/** 상하가 바뀌어 있으므로 framebuffer2 에는 뒤집어서 저장 */
	index = y * g_SceneInfo.iResolutionX + x;
	pFrameBuffer2[ index ] = edgevalue;
}


/**
 *	어느지점에 대해서 SuperSampling 을 수행할지를 결정한다.
 *	각 thread 는 한픽셀의 4x4 를 쪼갠 하나씩에 대응된다. 즉 총4개의 쓰레드가 하나의
 *	픽셀에서 각 모퉁이를 샘플링할지 여부를 결정.
 */
__global__ void singlePassRayTracingKernel_DetectionStage(
											float* pFrameBuffer,
											int* pASBuffer,
											int* atomicVariable, 
											bool bDebug, 
											int compareType )
{
	int x = ( blockIdx.x * blockDim.x + threadIdx.x ) / 2;
	int y = ( blockIdx.y * blockDim.y + threadIdx.y ) / 2;
	int index = 0, index2 = 0;

	/** block 내의 쓰레드 번호 */
	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int primaryAttr, secondaryAttr;
	float3 primaryNormal, secondaryNormal;

	// pixel corner number 는 left-top 부터 0, 1, 2, 3
	int pixelCornerIndex = threadIdx.x % 2 + ( threadIdx.y % 2 ) * 2;	
	int2 pattern_x_y;

	bool flag = false;
	bool checkRegion = false;

	float4 temp, temp2;
	float3 contrast = make_float3( 0.0f, 0.0f, 0.0f );

	/** active sub-pixel 인지와 active sub-pixel 이 몇 개인지를 위한 변수 초기화 */
	sharedMemory[ threadIndex * 6 + 0 ] = -1.0f;
	sharedMemory[ threadIndex * 6 + 1 ] = 0.0f;
	sharedMemory[ threadIndex * 6 + 2 ] = 0.0f;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;
	
	index = ( y * g_SceneInfo.iResolutionX + x );

	temp = tex1Dfetch( inSamplingMapTexture, index * 2 + 0 );
	primaryAttr = float_as_int( temp.x ); primaryNormal.x = temp.y; 
	primaryNormal.y = temp.z; primaryNormal.z = temp.w;

	temp = tex1Dfetch( inSamplingMapTexture, index * 2 + 1 );
	secondaryAttr = float_as_int( temp.x ); secondaryNormal.x = temp.y; 
	secondaryNormal.y = temp.z; secondaryNormal.z = temp.w;

	/**
	 *  현재 sub-pixel 과 관계된 4pixel 의
	 *	contrast 를 구함.
	 *	temp, temp2 를 각각 colormax, colormin 으로 사용.
	 */
	index2 = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	temp.x = tex1Dfetch( inFrameBuffer2Texture, index2 + 0 );	//r
	temp.y = tex1Dfetch( inFrameBuffer2Texture, index2 + 1 );	//g
	temp.z = tex1Dfetch( inFrameBuffer2Texture, index2 + 2 );	//b
	temp2 = temp;

	/** 
	 *	sub-pixel 이 active 가 아닐때 color 를 계산하기 위한 shared memory 
	 *	중심 pixel 은 9/16 의 weight 를 곱해야 한다.
	 */
	sharedMemory[ threadIndex * 6 + 3 ] = temp.x * 0.5625f;
	sharedMemory[ threadIndex * 6 + 4 ] = temp.y * 0.5625f;
	sharedMemory[ threadIndex * 6 + 5 ] = temp.z * 0.5625f;

	for ( int i = 0; i < 3; ++i ) {

		pattern_x_y.x = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 0 ] + x;
		pattern_x_y.y = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 1 ] + y;

		if ( pattern_x_y.x < 0 || pattern_x_y.y < 0  || 
			 pattern_x_y.x >= g_SceneInfo.iResolutionX || pattern_x_y.y >= g_SceneInfo.iResolutionY )
			continue;

		index2 = 3 * ( ( g_SceneInfo.iResolutionY - pattern_x_y.y - 1 ) * g_SceneInfo.iResolutionX + pattern_x_y.x );

		/**
		 *	현재 sub-pixel 과 관계된 4-pixel 의 presampling 정보를 이용해서 interpolation 해서 sub-pixel color 를 
		 *	결정한다.
		 */
		sharedMemory[ threadIndex * 6 + 3 ] += constantPixelWeight[ 3 * pixelCornerIndex + i ] * tex1Dfetch( inFrameBuffer2Texture, index2 + 0 );
		sharedMemory[ threadIndex * 6 + 4 ] += constantPixelWeight[ 3 * pixelCornerIndex + i ] * tex1Dfetch( inFrameBuffer2Texture, index2 + 1 );
		sharedMemory[ threadIndex * 6 + 5 ] += constantPixelWeight[ 3 * pixelCornerIndex + i ] * tex1Dfetch( inFrameBuffer2Texture, index2 + 2 );


		
		temp.x = fmin( temp.x, tex1Dfetch( inFrameBuffer2Texture, index2 + 0 ) );
		temp.y = fmin( temp.y, tex1Dfetch( inFrameBuffer2Texture, index2 + 1 ) );
		temp.z = fmin( temp.z, tex1Dfetch( inFrameBuffer2Texture, index2 + 2 ) );

		temp2.x = fmax( temp2.x, tex1Dfetch( inFrameBuffer2Texture, index2 + 0 ) );
		temp2.y = fmax( temp2.y, tex1Dfetch( inFrameBuffer2Texture, index2 + 1 ) );
		temp2.z = fmax( temp2.z, tex1Dfetch( inFrameBuffer2Texture, index2 + 2 ) );

	}

	// min, max 를 가지고 contrast 를 구한다.
	if ( temp.x + temp2.x > 0.0f ) contrast.x = ( temp2.x - temp.x ) / ( temp2.x + temp.x );
	if ( temp.y + temp2.y > 0.0f ) contrast.y = ( temp2.y - temp.y ) / ( temp2.y + temp.y );
	if ( temp.z + temp2.z > 0.0f ) contrast.z = ( temp2.z - temp.z ) / ( temp2.z + temp.z );

	///** 각 Thread 별로 해당 sub-pixel 을 추출하기 위한 계산 */
	for ( int i = 0; i < 3; ++i ) {

		pattern_x_y.x = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 0 ] + x;
		pattern_x_y.y = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 1 ] + y;

		/**
		 *	체크할 픽셀이 이미지범위를 넘어서면 continue 
		 */
		if ( pattern_x_y.x < 0 || pattern_x_y.y < 0  || 
			 pattern_x_y.x >= g_SceneInfo.iResolutionX || pattern_x_y.y >= g_SceneInfo.iResolutionY )
			continue;

		index = ( pattern_x_y.y * g_SceneInfo.iResolutionX + pattern_x_y.x );

		temp = tex1Dfetch( inSamplingMapTexture, index * 2 + 0 );
		temp2 = tex1Dfetch( inSamplingMapTexture, index * 2 + 1 );

		/** primary ray 가 맞은 물체가 selection 영역의 경계라면 무조건 supersampling 을 수행한다. */
		if ( GET_SELECTED_ATTR( primaryAttr ) != GET_SELECTED_ATTR( float_as_int( temp.x ) ) ) {
			flag = true;
			break;;
		}
		

		/** primary ray 가 맞은 물체가 selection 영역이 아니라면 supersampling 하지 않는다. */
		if ( GET_SELECTED_ATTR( primaryAttr ) == 0 && GET_SELECTED_ATTR( float_as_int( temp.x ) ) == 0 ) //&& GET_SELECTED_ATTR( float_as_int( temp2.x ) ) == 0 )
			continue;

		/** 오직 하나의 threshold 로 전체이미지를 color 비교하는경우는 이것만 하고 break */
		if ( ( compareType & 512 ) == 512 ) {
			if ( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.onlyColorThreshold ||
				 contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.onlyColorThreshold ||
				 contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.onlyColorThreshold ) {
				flag = true;
				break;
			}
			break;
		}

		/** primary oid 영역. */
		if ( GET_OID_ATTR( primaryAttr ) != GET_OID_ATTR( float_as_int( temp.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 1 ) == 1 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryOIDRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryOIDRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryOIDRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** primary normal */
		if ( primaryNormal.x * temp.y + primaryNormal.y * temp.z + primaryNormal.z * temp.w <= DETECTOR_NORMAL_THREADHOLD ) {
			checkRegion = true;
			if ( ( compareType & 2 ) == 2 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryNormalRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryNormalRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryNormalRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}
		
		/** primary shadow */
		if ( GET_SHADOW_ATTR( primaryAttr ) != GET_SHADOW_ATTR( float_as_int( temp.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 4 ) == 4 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryShadowRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryShadowRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryShadowRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** primary texture */
		if ( GET_TEXTURE_ATTR( primaryAttr ) == 1 ) {
			checkRegion = true;
			if ( ( compareType & 8 ) == 8 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryTextureRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryTextureRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryTextureRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary oid */
		if ( GET_OID_ATTR( secondaryAttr ) != GET_OID_ATTR( float_as_int( temp2.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 16 ) == 16 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryOIDRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryOIDRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryOIDRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary normal */
		if ( secondaryNormal.x * temp2.y + secondaryNormal.y * temp2.z + secondaryNormal.z * temp2.w <= DETECTOR_NORMAL_THREADHOLD ) {
			checkRegion = true;
			if ( ( compareType & 32 ) == 32 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryNormalRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryNormalRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryNormalRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary shadow */
		if ( GET_SHADOW_ATTR( secondaryAttr ) != GET_SHADOW_ATTR( float_as_int( temp2.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 64 ) == 64 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryShadowRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryShadowRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryShadowRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary texture */
		if ( GET_TEXTURE_ATTR( secondaryAttr ) == 1 ) {
			checkRegion = true;
			if ( ( compareType & 128 ) == 128 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryTextureRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryTextureRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryTextureRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** target 영역이외에 대해서는 pixel color 로 비교 */
		if ( !checkRegion && ( compareType & 256 ) == 256 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.etcRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.etcRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.etcRegionColorThreshold ) ) {
			flag = true;
			break;
		}

	}

	if ( flag ) {
		/** x,y 픽셀의 어느모퉁이 인지 기록 */
		sharedMemory[ threadIndex * 6 + 0 ] = pixelCornerIndex;
		sharedMemory[ threadIndex * 6 + 1 ] = y * g_SceneInfo.iResolutionX + x;
	}

	__syncthreads();


	/** 
	 *	한픽셀을 처리하는 thread 4개중 첫번째 놈이 현재 pixel 4귀퉁이중 몇개를 샘플링해야하는지를
	 *	계산해서 super sampling 하지 않는 영역은 1x1 sampling 한 결과를 weight 를 줘서 다시 저장한다.
	 */
	int count = 0;
	
	if ( pixelCornerIndex == 0 ) {

		//count += ( ( sharedMemory[ ( ( threadIdx.y + 0 ) * blockDim.x + ( threadIdx.x + 0 ) ) * 6 ] ) >= 0.0f );
		//count += ( ( sharedMemory[ ( ( threadIdx.y + 0 ) * blockDim.x + ( threadIdx.x + 1 ) ) * 6 ] ) >= 0.0f );
		//count += ( ( sharedMemory[ ( ( threadIdx.y + 1 ) * blockDim.x + ( threadIdx.x + 0 ) ) * 6 ] ) >= 0.0f );
		//count += ( ( sharedMemory[ ( ( threadIdx.y + 1 ) * blockDim.x + ( threadIdx.x + 1 ) ) * 6 ] ) >= 0.0f );

		if ( ( sharedMemory[ threadIndex * 6 ] ) >= 0.0f ) {
			sharedMemory[ threadIndex * 6 + 3 ] = 0.0f;
			sharedMemory[ threadIndex * 6 + 4 ] = 0.0f;
			sharedMemory[ threadIndex * 6 + 5 ] = 0.0f;
		} else {
			sharedMemory[ threadIndex * 6 + 3 ] = sharedMemory[ threadIndex * 6 + 3 ] * 0.25f;
			sharedMemory[ threadIndex * 6 + 4 ] = sharedMemory[ threadIndex * 6 + 4 ] * 0.25f;
			sharedMemory[ threadIndex * 6 + 5 ] = sharedMemory[ threadIndex * 6 + 5 ] * 0.25f;
		}

		for ( int i = 0; i <= 1; ++i ) {
			for ( int j = 0; j <= 1; ++j ) {
				count += ( ( sharedMemory[ ( ( threadIdx.y + i ) * blockDim.x + ( threadIdx.x + j ) ) * 6 ] ) >= 0.0f );
				if ( i == 0 && j == 0 ) continue;
				
				if ( ( sharedMemory[ ( ( threadIdx.y + i ) * blockDim.x + ( threadIdx.x + j ) ) * 6 ] ) < 0.0f ) {
					sharedMemory[ threadIndex * 6 + 3 ] += sharedMemory[ ( ( threadIdx.y + i ) * blockDim.x + ( threadIdx.x + j ) ) * 6 + 3 ] * 0.25f;
					sharedMemory[ threadIndex * 6 + 4 ] += sharedMemory[ ( ( threadIdx.y + i ) * blockDim.x + ( threadIdx.x + j ) ) * 6 + 4 ] * 0.25f;
					sharedMemory[ threadIndex * 6 + 5 ] += sharedMemory[ ( ( threadIdx.y + i ) * blockDim.x + ( threadIdx.x + j ) ) * 6 + 5 ] * 0.25f;
				}
			}
		}

		index = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

		pFrameBuffer[ index + 0 ] = sharedMemory[ threadIndex * 6 + 3 ];
		pFrameBuffer[ index + 1 ] = sharedMemory[ threadIndex * 6 + 4 ];
		pFrameBuffer[ index + 2 ] = sharedMemory[ threadIndex * 6 + 5 ];

		if ( !bDebug && count > 0 ) {

			pFrameBuffer[ index + 0 ] = sharedMemory[ threadIndex * 6 + 3 ];
			pFrameBuffer[ index + 1 ] = sharedMemory[ threadIndex * 6 + 4 ];
			pFrameBuffer[ index + 2 ] = sharedMemory[ threadIndex * 6 + 5 ];

			//pFrameBuffer[ index + 0 ] = pFrameBuffer[ index + 0 ] * ( (float)( 4 - count ) * 0.25f );
			//pFrameBuffer[ index + 1 ] = pFrameBuffer[ index + 1 ] * ( (float)( 4 - count ) * 0.25f );
			//pFrameBuffer[ index + 2 ] = pFrameBuffer[ index + 2 ] * ( (float)( 4 - count ) * 0.25f );

		} else if ( bDebug && count > 0 ) {

			if ( count == 1 ) {	pFrameBuffer[ index + 0 ] = 1.0f; pFrameBuffer[ index + 1 ] = 0.0f;	pFrameBuffer[ index + 2 ] = 0.0f; }
			if ( count == 2 ) {	pFrameBuffer[ index + 0 ] = 0.0f; pFrameBuffer[ index + 1 ] = 1.0f;	pFrameBuffer[ index + 2 ] = 0.0f; }
			if ( count == 3 ) {	pFrameBuffer[ index + 0 ] = 0.0f; pFrameBuffer[ index + 1 ] = 0.0f;	pFrameBuffer[ index + 2 ] = 1.0f; }
			if ( count == 4 ) {	pFrameBuffer[ index + 0 ] = 1.0f; pFrameBuffer[ index + 1 ] = 1.0f;	pFrameBuffer[ index + 2 ] = 1.0f; }

		}

		/** 현재 pixel 에 몇개의 active sub-pixel 이 있는지 저장해둔다. */
		sharedMemory[ threadIndex * 6 + 2 ] = count;

	}

	__syncthreads();

	/** 
	 *	한줄의 데이터를 한번에 처리 atomicAdd 를 매번부르면 비효율적이고,
	 *	image 상의 한 block 의 locallity 를 최대한 살리기 위해서.
	 */
	count = 0;
	if ( threadIdx.x == 0 && threadIdx.y == 0 ) {

		for ( int i = 0; i < blockDim.x * blockDim.y; ++i ) {
			count += sharedMemory[ i * 6 + 2 ];
		}

		/** 
		 *	다음 커널에서 돌릴때 한 pixel 에 더해질 subpixel 들의 color 를 sum 할때, global memory 를
		 *	이용하면 속도가 느리므로, shared memory 를 사용하기 위한 구조가 될수 있게 해야한다.
		 *	 따라서 다음커널에서 한 pixel 의 subpixel 들은 반드시 같은 block 안에 들어가게 해야 한다.
		 *	 그러기 위해서 다음커널의 block 안의 thread 개수를 고려해서 아래와 같은 구조를 만든다.
		 *  한 sub-pixel 은 다시 4개의 ray 로 구성되고 이 각각의 ray 에는 하나씩 thread 가 할당되므로
		 *	다음커널의 블락안에는 threads / 4 개의 sub-pixel 이 들어갈수 있다. 따라서 경우에 따라 
		 *	한 pixel 을 구성하는 sub-pixel 들이 서로 다른 block 으로 나뉘어질수 있다. 따라서 경계부분에서
		 *	sub-pixel 이 나뉘어질것 같으면 첫번째 block 에 들어갈 자리를 무효화 표시를 하고 다음 block 에
		 *  한 pixel 을 구성하는 sub-pixel 들을 다 넣어야 한다. 동기화 함수를 써야 하므로 쉽지는 않다.
		 *	자세한 설명은 앞으로쓸 논문을 참조하라.
		 *
		 *	각 active sub-pixel 중 가장 첫번째 것을 master 로 표시하기 위해서 corner 넘버에 10 을 더한다.
		 */
		if ( count > 0 ) {
			
			int moreRequireIndex = 0;
			int masterCheck = 1;
			int subpixels = 0;

			index = atomicAdd( atomicVariable, count );

			/** 각 pixel 순서로 sub-pixel 들을 체크하기 위해서 */
			for ( int j = 0; j < blockDim.y; j += 2 ) {
			for ( int i = 0; i < blockDim.x; i += 2 ) {

				/** 
				 *	다음 pixel 의 sub-pixel 들을 buffer 에 넣기 전에, 쓰레드 block 의 경계를 넘어서는지를
				 *	체크해서 넘어선다면, 비워두고 경계이후부터 채운다. 한 쓰레드 block 이 256 개로 구성될때
				 *	이 안에 64개의 sub-pixel 이 들어갈수 있으므로 64을 체크. 128 개라면 32 개
				 */
				subpixels = sharedMemory[ ( j * blockDim.x + i ) * 6 + 2 ];

				while( subpixels > 0 ) {

					if ( count >= subpixels && ( index % SUBPIXEL_CAPABILITY ) + subpixels > SUBPIXEL_CAPABILITY ) {
						masterCheck = SUBPIXEL_CAPABILITY - ( index % SUBPIXEL_CAPABILITY );		// 임시로 masterCheck 변수 사용.
						for ( int loop = 0; loop < masterCheck; ++loop ) {
							pASBuffer[ index++ ] = -1;							// buffer 의 공간에 무효하다는것을 세팅.
							count--;
							moreRequireIndex++;									// 할당된 buffer 보다 나중에 더 잡아야할 개수.
						}
					}

					/** 
					 *	남아있는 buffer 가 모자란다면, 더 필요한 개수를 할당는데, 남아있는 buffer 는
					 *	다시 무효화를 체크하고 새로 필요한 만큼 buffer 를 할당받은다음에 다시 align 을 체크한다.
					 */
					if ( count < subpixels ) {

						for ( int loop = 0; loop < count; ++loop ) {
							pASBuffer[ index++ ] = -1;						// buffer 의 공간에 무효하다는것을 세팅.
							moreRequireIndex++;								// 할당된 buffer 보다 나중에 더 잡아야할 개수.
						}

						index = atomicAdd( atomicVariable, moreRequireIndex );
						count = moreRequireIndex;
						moreRequireIndex = 0;

					} else {

						masterCheck = 1;

						for ( int k = 0; k < 4; ++k ) {

							x = i + ( k % 2 ); y = j + ( k / 2 );	// 픽셀 x, y 가 아니라 2차원상에서의 쓰레드 좌표.
							threadIndex = y * blockDim.x + x;

							if ( sharedMemory[ threadIndex * 6 ] >= 0.0f ) {
								pASBuffer[ index++ ] = MAKE_ACTIVE_SUBPIXEL( (int) sharedMemory[ threadIndex * 6 + 1 ], 
																			 (int) ( masterCheck * 10 + sharedMemory[ threadIndex * 6 + 0 ] ) );
								count--;
								masterCheck = 0;
							}
						}

						break;

					}

				};

			}
			}
		}

	}

}



/**
 *	어느지점에 대해서 SuperSampling 을 수행할지를 결정한다.
 *	각 thread 는 한픽셀의 4x4 를 쪼갠 하나씩에 대응된다. 즉 총4개의 쓰레드가 하나의
 *	픽셀에서 각 모퉁이를 샘플링할지 여부를 결정.
 */
__global__ void singlePassRayTracingKernel_DetectionStage_ForSelective(
											float* pFrameBuffer,
											int* pASBuffer,
											int* atomicVariable, 
											bool bDebug, 
											int compareType )
{
	int x = ( blockIdx.x * blockDim.x + threadIdx.x ) / 2;
	int y = ( blockIdx.y * blockDim.y + threadIdx.y ) / 2;
	int index = 0, index2 = 0;

	/** block 내의 쓰레드 번호 */
	int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	int primaryAttr, secondaryAttr;
	float3 primaryNormal, secondaryNormal;

	// pixel corner number 는 left-top 부터 0, 1, 2, 3
	int pixelCornerIndex = threadIdx.x % 2 + ( threadIdx.y % 2 ) * 2;	
	int2 pattern_x_y;

	bool flag = false;
	bool checkRegion = false;

	float4 temp, temp2;
	float3 contrast = make_float3( 0.0f, 0.0f, 0.0f );

	/** active sub-pixel 인지와 active sub-pixel 이 몇 개인지를 위한 변수 초기화 */
	sharedMemory[ threadIndex * 6 + 0 ] = -1.0f;
	sharedMemory[ threadIndex * 6 + 1 ] = 0.0f;
	sharedMemory[ threadIndex * 6 + 2 ] = 0.0f;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;
	
	index = ( y * g_SceneInfo.iResolutionX + x );

	temp = tex1Dfetch( inSamplingMapTexture, index * 2 + 0 );
	primaryAttr = float_as_int( temp.x ); primaryNormal.x = temp.y; 
	primaryNormal.y = temp.z; primaryNormal.z = temp.w;

	temp = tex1Dfetch( inSamplingMapTexture, index * 2 + 1 );
	secondaryAttr = float_as_int( temp.x ); secondaryNormal.x = temp.y; 
	secondaryNormal.y = temp.z; secondaryNormal.z = temp.w;

	/**
	 *  현재 sub-pixel 과 관계된 4pixel 의
	 *	contrast 를 구함.
	 *	temp, temp2 를 각각 colormax, colormin 으로 사용.
	 */
	index2 = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	temp.x = tex1Dfetch( inFrameBuffer2Texture, index2 + 0 );	//r
	temp.y = tex1Dfetch( inFrameBuffer2Texture, index2 + 1 );	//g
	temp.z = tex1Dfetch( inFrameBuffer2Texture, index2 + 2 );	//b
	temp2 = temp;

	for ( int i = 0; i < 3; ++i ) {

		pattern_x_y.x = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 0 ] + x;
		pattern_x_y.y = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 1 ] + y;

		if ( pattern_x_y.x < 0 || pattern_x_y.y < 0  || 
			 pattern_x_y.x >= g_SceneInfo.iResolutionX || pattern_x_y.y >= g_SceneInfo.iResolutionY )
			continue;

		index2 = 3 * ( ( g_SceneInfo.iResolutionY - pattern_x_y.y - 1 ) * g_SceneInfo.iResolutionX + pattern_x_y.x );
		
		temp.x = fmin( temp.x, tex1Dfetch( inFrameBuffer2Texture, index2 + 0 ) );
		temp.y = fmin( temp.y, tex1Dfetch( inFrameBuffer2Texture, index2 + 1 ) );
		temp.z = fmin( temp.z, tex1Dfetch( inFrameBuffer2Texture, index2 + 2 ) );

		temp2.x = fmax( temp2.x, tex1Dfetch( inFrameBuffer2Texture, index2 + 0 ) );
		temp2.y = fmax( temp2.y, tex1Dfetch( inFrameBuffer2Texture, index2 + 1 ) );
		temp2.z = fmax( temp2.z, tex1Dfetch( inFrameBuffer2Texture, index2 + 2 ) );

	}

	// min, max 를 가지고 contrast 를 구한다.
	if ( temp.x + temp2.x > 0.0f ) contrast.x = ( temp2.x - temp.x ) / ( temp2.x + temp.x );
	if ( temp.y + temp2.y > 0.0f ) contrast.y = ( temp2.y - temp.y ) / ( temp2.y + temp.y );
	if ( temp.z + temp2.z > 0.0f ) contrast.z = ( temp2.z - temp.z ) / ( temp2.z + temp.z );

	/**
	 *	selective 한 물체에 대해서는 9pixel 경계를 다 비교한다.
	 */
	bool boundary = false;

	for ( int i = -1; i <= 1; ++i ) {
		for ( int j = -1; j <= 1; ++j ) {

			pattern_x_y.x = i + x;
			pattern_x_y.y = j + y;

			/**
			 *	체크할 픽셀이 이미지범위를 넘어서면 continue 
			 */
			if ( pattern_x_y.x < 0 || pattern_x_y.y < 0  || 
				 pattern_x_y.x >= g_SceneInfo.iResolutionX || pattern_x_y.y >= g_SceneInfo.iResolutionY )
				continue;

			index = ( pattern_x_y.y * g_SceneInfo.iResolutionX + pattern_x_y.x );

			temp = tex1Dfetch( inSamplingMapTexture, index * 2 + 0 );

			/** primary ray 가 맞은 물체가 selection 영역의 경계라면 무조건 supersampling 을 수행한다. */
			if ( GET_SELECTED_ATTR( primaryAttr ) != GET_SELECTED_ATTR( float_as_int( temp.x ) ) ) {
				boundary = true;
				break;
			}
		}
	}

	///** 각 Thread 별로 해당 sub-pixel 을 추출하기 위한 계산 */
	for ( int i = 0; i < 3; ++i ) {

		pattern_x_y.x = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 0 ] + x;
		pattern_x_y.y = constantIndexTablePattern[ 6 * pixelCornerIndex + i * 2 + 1 ] + y;

		/**
		 *	체크할 픽셀이 이미지범위를 넘어서면 continue 
		 */
		if ( pattern_x_y.x < 0 || pattern_x_y.y < 0  || 
			 pattern_x_y.x >= g_SceneInfo.iResolutionX || pattern_x_y.y >= g_SceneInfo.iResolutionY )
			continue;

		index = ( pattern_x_y.y * g_SceneInfo.iResolutionX + pattern_x_y.x );

		temp = tex1Dfetch( inSamplingMapTexture, index * 2 + 0 );
		temp2 = tex1Dfetch( inSamplingMapTexture, index * 2 + 1 );


		/** primary ray 가 맞은 물체가 selection 영역이 아니라면 supersampling 하지 않는다. */
		if ( boundary ) {
			flag = true;
			break;
		}

		/** primary ray 가 맞은 물체가 selection 영역이 아니라면 supersampling 하지 않는다. */
		if ( GET_SELECTED_ATTR( primaryAttr ) == 0 && GET_SELECTED_ATTR( float_as_int( temp.x ) ) == 0 ) //&& GET_SELECTED_ATTR( float_as_int( temp2.x ) ) == 0 )
			continue;

		/** 오직 하나의 threshold 로 전체이미지를 color 비교하는경우는 이것만 하고 break */
		if ( ( compareType & 512 ) == 512 ) {
			if ( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.onlyColorThreshold ||
				 contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.onlyColorThreshold ||
				 contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.onlyColorThreshold ) {
				flag = true;
				break;
			}
			break;
		}

		/** primary oid 영역. */
		if ( GET_OID_ATTR( primaryAttr ) != GET_OID_ATTR( float_as_int( temp.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 1 ) == 1 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryOIDRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryOIDRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryOIDRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** primary normal */
		if ( primaryNormal.x * temp.y + primaryNormal.y * temp.z + primaryNormal.z * temp.w <= DETECTOR_NORMAL_THREADHOLD ) {
			checkRegion = true;
			if ( ( compareType & 2 ) == 2 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryNormalRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryNormalRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryNormalRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}
		
		/** primary shadow */
		if ( GET_SHADOW_ATTR( primaryAttr ) != GET_SHADOW_ATTR( float_as_int( temp.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 4 ) == 4 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryShadowRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryShadowRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryShadowRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** primary texture */
		if ( GET_TEXTURE_ATTR( primaryAttr ) == 1 ) {
			checkRegion = true;
			if ( ( compareType & 8 ) == 8 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryTextureRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryTextureRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.primaryTextureRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary oid */
		if ( GET_OID_ATTR( secondaryAttr ) != GET_OID_ATTR( float_as_int( temp2.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 16 ) == 16 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryOIDRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryOIDRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryOIDRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary normal */
		if ( secondaryNormal.x * temp2.y + secondaryNormal.y * temp2.z + secondaryNormal.z * temp2.w <= DETECTOR_NORMAL_THREADHOLD ) {
			checkRegion = true;
			if ( ( compareType & 32 ) == 32 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryNormalRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryNormalRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryNormalRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary shadow */
		if ( GET_SHADOW_ATTR( secondaryAttr ) != GET_SHADOW_ATTR( float_as_int( temp2.x ) ) ) {
			checkRegion = true;
			if ( ( compareType & 64 ) == 64 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryShadowRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryShadowRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryShadowRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** secondary texture */
		if ( GET_TEXTURE_ATTR( secondaryAttr ) == 1 ) {
			checkRegion = true;
			if ( ( compareType & 128 ) == 128 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryTextureRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryTextureRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.secondaryTextureRegionColorThreshold ) ) {
				flag = true;
				break;
			}
		}

		/** target 영역이외에 대해서는 pixel color 로 비교 */
		if ( !checkRegion && ( compareType & 256 ) == 256 && 
					( contrast.x > CONTRAST_RED_DEFAULT_THRESHOLD * g_ThresholdInfo.etcRegionColorThreshold ||
					  contrast.y > CONTRAST_GREEN_DEFAULT_THRESHOLD * g_ThresholdInfo.etcRegionColorThreshold ||
					  contrast.z > CONTRAST_BLUE_DEFAULT_THRESHOLD * g_ThresholdInfo.etcRegionColorThreshold ) ) {
			flag = true;
			break;
		}

	}

	if ( flag ) {

		/** x,y 픽셀의 어느모퉁이 인지 기록 */
		sharedMemory[ threadIndex * 6 + 0 ] = pixelCornerIndex;
		sharedMemory[ threadIndex * 6 + 1 ] = y * g_SceneInfo.iResolutionX + x;

		if ( bDebug ) {
			index = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
			pFrameBuffer[ index + 0 ] = 1.0f;
			pFrameBuffer[ index + 1 ] = 1.0f;
			pFrameBuffer[ index + 2 ] = 1.0f;
		}

	} else {

		if ( bDebug ) {
			index = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
			pFrameBuffer[ index + 0 ] = 0.0f;
			pFrameBuffer[ index + 1 ] = 0.0f;
			pFrameBuffer[ index + 2 ] = 0.0f;
		}

	}

	__syncthreads();

	/** 
	 *	한픽셀을 처리하는 thread 4개중 첫번째 놈이 현재 pixel 4귀퉁이중 몇개를 샘플링해야하는지를
	 *	계산해서 super sampling 하지 않는 영역은 1x1 sampling 한 결과를 weight 를 줘서 다시 저장한다.
	 */
	int count = 0;
	
	if ( pixelCornerIndex == 0 ) {

		count += ( ( sharedMemory[ ( ( threadIdx.y + 0 ) * blockDim.x + ( threadIdx.x + 0 ) ) * 6 ] ) >= 0.0f );
		count += ( ( sharedMemory[ ( ( threadIdx.y + 0 ) * blockDim.x + ( threadIdx.x + 1 ) ) * 6 ] ) >= 0.0f );
		count += ( ( sharedMemory[ ( ( threadIdx.y + 1 ) * blockDim.x + ( threadIdx.x + 0 ) ) * 6 ] ) >= 0.0f );
		count += ( ( sharedMemory[ ( ( threadIdx.y + 1 ) * blockDim.x + ( threadIdx.x + 1 ) ) * 6 ] ) >= 0.0f );

		if ( !bDebug && count > 0 ) {

			index = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
			pFrameBuffer[ index + 0 ] = pFrameBuffer[ index + 0 ] * ( (float)( 4 - count ) / 4.0f );
			pFrameBuffer[ index + 1 ] = pFrameBuffer[ index + 1 ] * ( (float)( 4 - count ) / 4.0f );
			pFrameBuffer[ index + 2 ] = pFrameBuffer[ index + 2 ] * ( (float)( 4 - count ) / 4.0f );

		} else if ( bDebug && count > 0 ) {

			index = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );
			if ( count == 1 ) {	pFrameBuffer[ index + 0 ] = 1.0f; pFrameBuffer[ index + 1 ] = 0.0f;	pFrameBuffer[ index + 2 ] = 0.0f; }
			if ( count == 2 ) {	pFrameBuffer[ index + 0 ] = 0.0f; pFrameBuffer[ index + 1 ] = 1.0f;	pFrameBuffer[ index + 2 ] = 0.0f; }
			if ( count == 3 ) {	pFrameBuffer[ index + 0 ] = 0.0f; pFrameBuffer[ index + 1 ] = 0.0f;	pFrameBuffer[ index + 2 ] = 1.0f; }
			if ( count == 4 ) {	pFrameBuffer[ index + 0 ] = 1.0f; pFrameBuffer[ index + 1 ] = 1.0f;	pFrameBuffer[ index + 2 ] = 1.0f; }
		}

		/** 현재 pixel 에 몇개의 유효한 sub-pixel 이 있는지 저장해둔다. */
		sharedMemory[ threadIndex * 6 + 2 ] = count;

	}

	__syncthreads();

	/** 
	 *	한줄의 데이터를 한번에 처리 atomicAdd 를 매번부르면 비효율적이고,
	 *	image 상의 한 block 의 locallity 를 최대한 살리기 위해서.
	 */
	count = 0;
	if ( threadIdx.x == 0 && threadIdx.y == 0 ) {

		for ( int i = 0; i < blockDim.x * blockDim.y; ++i ) {
			count += sharedMemory[ i * 6 + 2 ];
		}

		/** 
		 *	다음 커널에서 돌릴때 한 pixel 에 더해질 subpixel 들의 color 를 sum 할때, global memory 를
		 *	이용하면 속도가 느리므로, shared memory 를 사용하기 위한 구조가 될수 있게 해야한다.
		 *	 따라서 다음커널에서 한 pixel 의 subpixel 들은 반드시 같은 block 안에 들어가게 해야 한다.
		 *	 그러기 위해서 다음커널의 block 안의 thread 개수를 고려해서 아래와 같은 구조를 만든다.
		 *  한 sub-pixel 은 다시 4개의 ray 로 구성되고 이 각각의 ray 에는 하나씩 thread 가 할당되므로
		 *	다음커널의 블락안에는 threads / 4 개의 sub-pixel 이 들어갈수 있다. 따라서 경우에 따라 
		 *	한 pixel 을 구성하는 sub-pixel 들이 서로 다른 block 으로 나뉘어질수 있다. 따라서 경계부분에서
		 *	sub-pixel 이 나뉘어질것 같으면 첫번째 block 에 들어갈 자리를 무효화 표시를 하고 다음 block 에
		 *  한 pixel 을 구성하는 sub-pixel 들을 다 넣어야 한다. 동기화 함수를 써야 하므로 쉽지는 않다.
		 *	자세한 설명은 앞으로쓸 논문을 참조하라.
		 *
		 *	각 active sub-pixel 중 가장 첫번째 것을 master 로 표시하기 위해서 corner 넘버에 10 을 더한다.
		 */
		if ( count > 0 ) {
			
			int moreRequireIndex = 0;
			int masterCheck = 1;
			int subpixels = 0;

			index = atomicAdd( atomicVariable, count );

			/** 각 pixel 순서로 sub-pixel 들을 체크하기 위해서 */
			for ( int j = 0; j < blockDim.y; j += 2 ) {
			for ( int i = 0; i < blockDim.x; i += 2 ) {

				/** 
				 *	다음 pixel 의 sub-pixel 들을 buffer 에 넣기 전에, 쓰레드 block 의 경계를 넘어서는지를
				 *	체크해서 넘어선다면, 비워두고 경계이후부터 채운다. 한 쓰레드 block 이 256 개로 구성될때
				 *	이 안에 64개의 sub-pixel 이 들어갈수 있으므로 64을 체크. 128 개라면 32 개
				 */
				subpixels = sharedMemory[ ( j * blockDim.x + i ) * 6 + 2 ];

				while( subpixels > 0 ) {

					if ( count >= subpixels && ( index % SUBPIXEL_CAPABILITY ) + subpixels > SUBPIXEL_CAPABILITY ) {
						masterCheck = SUBPIXEL_CAPABILITY - ( index % SUBPIXEL_CAPABILITY );		// 임시로 masterCheck 변수 사용.
						for ( int loop = 0; loop < masterCheck; ++loop ) {
							pASBuffer[ index++ ] = -1;							// buffer 의 공간에 무효하다는것을 세팅.
							count--;
							moreRequireIndex++;									// 할당된 buffer 보다 나중에 더 잡아야할 개수.
						}
					}

					/** 
					 *	남아있는 buffer 가 모자란다면, 더 필요한 개수를 할당는데, 남아있는 buffer 는
					 *	다시 무효화를 체크하고 새로 필요한 만큼 buffer 를 할당받은다음에 다시 align 을 체크한다.
					 */
					if ( count < subpixels ) {

						for ( int loop = 0; loop < count; ++loop ) {
							pASBuffer[ index++ ] = -1;						// buffer 의 공간에 무효하다는것을 세팅.
							moreRequireIndex++;								// 할당된 buffer 보다 나중에 더 잡아야할 개수.
						}

						index = atomicAdd( atomicVariable, moreRequireIndex );
						count = moreRequireIndex;
						moreRequireIndex = 0;

					} else {

						masterCheck = 1;

						for ( int k = 0; k < 4; ++k ) {

							x = i + ( k % 2 ); y = j + ( k / 2 );	// 픽셀 x, y 가 아니라 2차원상에서의 쓰레드 좌표.
							threadIndex = y * blockDim.x + x;

							if ( sharedMemory[ threadIndex * 6 ] >= 0.0f ) {
								pASBuffer[ index++ ] = MAKE_ACTIVE_SUBPIXEL( (int) sharedMemory[ threadIndex * 6 + 1 ], 
																			 (int) ( masterCheck * 10 + sharedMemory[ threadIndex * 6 + 0 ] ) );
								count--;
								masterCheck = 0;
							}
						}

						break;

					}

				};

			}
			}
		}

	}

}

/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_SuperSamplingStage_ShadowOn( 
											float* pFrameBuffer,
											int count,
											int maxReflectionDepth )
{
	float3 pos, dir;
	bool secondary = false;
	float sx, sy;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	bool master = false;
	int index = ( blockIdx.x * blockDim.x + threadIdx.x ) / 4;

	int data = -1;
	int pixelIndex = 0, cornerNumber = 0, x = 0, y = 0;
	int samplingIndex = ( blockIdx.x * blockDim.x + threadIdx.x ) % 4;

	/**
	 *	index 범위안에 포함되지 않은 thread 라도 return 시키면 안된다. 
	 *	이 thread 에 할당될 shared memory 를 다른 thread 가 사용하므로,
	 *	이 thread 도 sharedmemory 를 초기화 해야하므로.
	 */
	if ( index < count ) {
		data = tex1Dfetch( inASBufferTexture, index );
	}

	/** 
	 *	무효한 sub-pixel 이 아닐때만 처리. 
	 *	이런 방식말고 if ( cornerNumber >= 0 ) return; 처럼 여기서 return 시키면 안되고!! 반드시 이렇게 조건문을 사용 
	 *	아래에서 합산할때 cornerNumber < 0 인 경우의 sharedMemory 값을 참조해야 하므로. sharedMemory 초기화 부분이 수행되어야 한다.
	 */
	if ( data >= 0 ) {

		pixelIndex = GET_SUBPIXEL_INDEX( data );
		cornerNumber = GET_SUBPIXEL_CORNER( data );
		x = pixelIndex % g_SceneInfo.iResolutionX;
		y = pixelIndex / g_SceneInfo.iResolutionX;

		if ( cornerNumber >= 10 ) {
			master = true;
			cornerNumber = cornerNumber % 10;
		}

		/** 
		 *	어떤 corner 인지에 따라서 샘플링 구간의위치 결정
		 *	( cornerNumber / 2 ) * 2 는 2 로 나눌때 int 로 round-off 되고 * 2를 하는것.
		 */
		int samplingX = ( cornerNumber % 2 ) * 2 + ( samplingIndex % 2 );
		int samplingY = ( cornerNumber / 2 ) * 2 + ( samplingIndex / 2 );

		/**
		 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
		 */
		pos = g_CameraInfo.eye;

		/** 가상으로 4 by 4 를 했을때의 각 샘플링지점을 구한다. */
		sx = (float) x + ( (float) samplingX + radicalInverse( x << 8 + samplingX, 3 ) ) * 0.25f;
		sy = (float) y + ( (float) samplingY + radicalInverse( y << 8 + samplingY, 5 ) ) * 0.25f;

		dir = g_CameraInfo.startPoint + 
			  g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
		dir = normalize( dir - pos );

		currRay.dir = make_float3( dir.x, dir.y, dir.z );
		currRay.pos = make_float3( pos.x, pos.y, pos.z );

		point.init();
		point.colorWeight.x = 1.0f;
		point.colorWeight.y = 1.0f;
		point.colorWeight.z = 1.0f;

		do {

			secondary = false;
			currIsectCheck.init();
			singlePassIntersect( currRay, currIsectCheck  );

			/**
			 *	intersection check
			 */
			if ( currIsectCheck.isHit() ) {

				makeIntersectionPoint( &currRay, &currIsectCheck, &point );

				cuObjectMaterial material;
				getObjectMaterial( currIsectCheck.objectIndex, material );
				
				/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
				float3 diffuse = material.diffuse;
				calTextureColor( point, material, diffuse );

				color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection)*( maxReflectionDepth > depth ) ) * 
						calSinglePassDirectIllumination_ShadowOn( point, material, diffuse );

				/**
				 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
				 */
				if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

					dir = refraction( point.dir, point.normal, material.refractionIndex );
					point.colorWeight = point.colorWeight * material.transparency * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

					dir = reflection( point.dir, point.normal );
					point.colorWeight =  point.colorWeight * material.reflection * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				}

			}

			depth++;

		} while( secondary );

	}

	/**
	 *	위에서 ray 추적을할때도 shared memory 를 사용하므로 반드시 여기까지 동기화를 한번한다음에
	 *	위에서 사용한 shared memory 를 color 누적하는 buffer 로 사용해야 한다.
	 */
	__syncthreads();

	/** 
	 *	반드시 이 sharedMemory 는 ray 를 추적할때 사용되는 stack 을 위한 shared memory 
	 *	사이즈이내 에서 사용해야 한다. 
	 *	 같은 pixel 에 대해서 샘플링한 결과를 평균내서 framebuffer 에 칠하기 위해서 일단 각자의
	 *	데이터를 shared memory 에 저장하고나서, 대표thread 들이 합산해서 frame buffer 에 저장한다.
	 */
	/** 
	 *	각 ray 는 천체픽셀에 1/16 만큼만 weighted 된다. 
	 *	shared memory 는 ray 를 추적할때 stack 으로도 사용되므로 ray 추적이 끝난후에 반드시 여기서 초기화후 사용해야 한다. 
	 */
	sharedMemory[ threadIdx.x * 4 + 0 ] = color.x * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 1 ] = color.y * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 2 ] = color.z * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 3 ] = pixelIndex;

	__syncthreads();

	/** 각 sub-pixel 의 첫번째 thread 가 해당 sub-pixel 의 color 를 누적 계산한다. */
	if ( samplingIndex != 0 ) 
		return;

	for ( int i = 1; i < 4; ++i ) {

		sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
		sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
		sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];

	}

	__syncthreads();

	/** 
	 *	각 pixel 의 master sub-pixel 의 첫번째 thread 가 해당 pixel 에 칠해질 color 를 누적한다. 
	 *	한 pixel 을 구성하는 sub pixel 들은 thread index 가 증가하는순으로 붙어있고 최대 4개만 존재하므로
	 *  현재 master thread 로 부터 최대 4 sub-pixel 의 첫번째 thread 만 체크하면 된다.
	 */
	if ( !master ) return;

	for ( int i = 4; i <= 12; i += 4 ) {
		/** 같은 pixel 을 구성하는 sub-pixel 일때만 누적 */
		if ( threadIdx.x + i < blockDim.x && 
			sharedMemory[ ( threadIdx.x + i ) * 4 + 3 ] == pixelIndex ) {
			sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
			sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
			sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];
		}
	}

	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer[ idx + 0 ] += ( sharedMemory[ threadIdx.x * 4 + 0 ] );
	pFrameBuffer[ idx + 1 ] += ( sharedMemory[ threadIdx.x * 4 + 1 ] );
	pFrameBuffer[ idx + 2 ] += ( sharedMemory[ threadIdx.x * 4 + 2 ] );

}


/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void singlePassRayTracingKernel_SuperSamplingStage_ShadowOff( 
											float* pFrameBuffer,
											int count,
											int maxReflectionDepth )
{
	float3 pos, dir;
	bool secondary = false;
	float sx, sy;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	bool master = false;
	int index = ( blockIdx.x * blockDim.x + threadIdx.x ) / 4;

	int data = -1;
	int pixelIndex = 0, cornerNumber = 0, x = 0, y = 0;
	int samplingIndex = ( blockIdx.x * blockDim.x + threadIdx.x ) % 4;

	/**
	 *	index 범위안에 포함되지 않은 thread 라도 return 시키면 안된다. 
	 *	이 thread 에 할당될 shared memory 를 다른 thread 가 사용하므로,
	 *	이 thread 도 sharedmemory 를 초기화 해야하므로.
	 */
	if ( index < count ) {
		data = tex1Dfetch( inASBufferTexture, index );
	}

	/** 
	 *	무효한 sub-pixel 이 아닐때만 처리. 
	 *	이런 방식말고 if ( cornerNumber >= 0 ) return; 처럼 여기서 return 시키면 안되고!! 반드시 이렇게 조건문을 사용 
	 *	아래에서 합산할때 cornerNumber < 0 인 경우의 sharedMemory 값을 참조해야 하므로. sharedMemory 초기화 부분이 수행되어야 한다.
	 */
	if ( data >= 0 ) {

		pixelIndex = GET_SUBPIXEL_INDEX( data );
		cornerNumber = GET_SUBPIXEL_CORNER( data );
		x = pixelIndex % g_SceneInfo.iResolutionX;
		y = pixelIndex / g_SceneInfo.iResolutionX;

		if ( cornerNumber >= 10 ) {
			master = true;
			cornerNumber = cornerNumber % 10;
		}

		/** 
		 *	어떤 corner 인지에 따라서 샘플링 구간의위치 결정
		 *	( cornerNumber / 2 ) * 2 는 2 로 나눌때 int 로 round-off 되고 * 2를 하는것.
		 */
		int samplingX = ( cornerNumber % 2 ) * 2 + ( samplingIndex % 2 );
		int samplingY = ( cornerNumber / 2 ) * 2 + ( samplingIndex / 2 );

		/**
		 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
		 */
		pos = g_CameraInfo.eye;

		/** 가상으로 4 by 4 를 했을때의 각 샘플링지점을 구한다. */
		sx = (float) x + ( (float) samplingX + radicalInverse( x << 8 + samplingX, 3 ) ) * 0.25f;
		sy = (float) y + ( (float) samplingY + radicalInverse( y << 8 + samplingY, 5 ) ) * 0.25f;

		dir = g_CameraInfo.startPoint + 
			  g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
		dir = normalize( dir - pos );

		currRay.dir = make_float3( dir.x, dir.y, dir.z );
		currRay.pos = make_float3( pos.x, pos.y, pos.z );

		point.init();
		point.colorWeight.x = 1.0f;
		point.colorWeight.y = 1.0f;
		point.colorWeight.z = 1.0f;

		do {

			secondary = false;
			currIsectCheck.init();
			singlePassIntersect( currRay, currIsectCheck  );

			/**
			 *	intersection check
			 */
			if ( currIsectCheck.isHit() ) {

				makeIntersectionPoint( &currRay, &currIsectCheck, &point );

				cuObjectMaterial material;
				getObjectMaterial( currIsectCheck.objectIndex, material );
				
				/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
				float3 diffuse = material.diffuse;
				calTextureColor( point, material, diffuse );

				color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection)*( maxReflectionDepth > depth ) ) * 
						calSinglePassDirectIllumination_ShadowOff( point, material, diffuse );

				/**
				 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
				 */
				if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

					dir = refraction( point.dir, point.normal, material.refractionIndex );
					point.colorWeight = point.colorWeight * material.transparency * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

					dir = reflection( point.dir, point.normal );
					point.colorWeight =  point.colorWeight * material.reflection * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				}

			}

			depth++;

		} while( secondary );

	}

	/**
	 *	위에서 ray 추적을할때도 shared memory 를 사용하므로 반드시 여기까지 동기화를 한번한다음에
	 *	위에서 사용한 shared memory 를 color 누적하는 buffer 로 사용해야 한다.
	 */
	__syncthreads();

	/** 
	 *	반드시 이 sharedMemory 는 ray 를 추적할때 사용되는 stack 을 위한 shared memory 
	 *	사이즈이내 에서 사용해야 한다. 
	 *	 같은 pixel 에 대해서 샘플링한 결과를 평균내서 framebuffer 에 칠하기 위해서 일단 각자의
	 *	데이터를 shared memory 에 저장하고나서, 대표thread 들이 합산해서 frame buffer 에 저장한다.
	 */
	/** 
	 *	각 ray 는 천체픽셀에 1/16 만큼만 weighted 된다. 
	 *	shared memory 는 ray 를 추적할때 stack 으로도 사용되므로 ray 추적이 끝난후에 반드시 여기서 초기화후 사용해야 한다. 
	 */
	sharedMemory[ threadIdx.x * 4 + 0 ] = color.x * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 1 ] = color.y * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 2 ] = color.z * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 3 ] = pixelIndex;

	__syncthreads();

	/** 각 sub-pixel 의 첫번째 thread 가 해당 sub-pixel 의 color 를 누적 계산한다. */
	if ( samplingIndex != 0 ) 
		return;

	for ( int i = 1; i < 4; ++i ) {

		sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
		sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
		sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];

	}

	__syncthreads();

	/** 
	 *	각 pixel 의 master sub-pixel 의 첫번째 thread 가 해당 pixel 에 칠해질 color 를 누적한다. 
	 *	한 pixel 을 구성하는 sub pixel 들은 thread index 가 증가하는순으로 붙어있고 최대 4개만 존재하므로
	 *  현재 master thread 로 부터 최대 4 sub-pixel 의 첫번째 thread 만 체크하면 된다.
	 */
	if ( !master ) return;

	for ( int i = 4; i <= 12; i += 4 ) {
		/** 같은 pixel 을 구성하는 sub-pixel 일때만 누적 */
		if ( threadIdx.x + i < blockDim.x && 
			sharedMemory[ ( threadIdx.x + i ) * 4 + 3 ] == pixelIndex ) {
			sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
			sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
			sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];
		}
	}

	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer[ idx + 0 ] += ( sharedMemory[ threadIdx.x * 4 + 0 ] );
	pFrameBuffer[ idx + 1 ] += ( sharedMemory[ threadIdx.x * 4 + 1 ] );
	pFrameBuffer[ idx + 2 ] += ( sharedMemory[ threadIdx.x * 4 + 2 ] );
}


/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void fixedOption_singlePassRayTracingKernel_SuperSamplingStage_ShadowOn( 
											float* pFrameBuffer,
											int startSubPixelIndex,
											int count,
											int maxReflectionDepth )
{
	float3 pos, dir;
	bool secondary = false;
	float sx, sy;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	bool master = false;
	int index = startSubPixelIndex + ( blockIdx.x * blockDim.x + threadIdx.x ) / 4;

	int data = -1;
	int pixelIndex = 0, cornerNumber = 0, x = 0, y = 0;
	int samplingIndex = ( blockIdx.x * blockDim.x + threadIdx.x ) % 4;

	/**
	 *	index 범위안에 포함되지 않은 thread 라도 return 시키면 안된다. 
	 *	이 thread 에 할당될 shared memory 를 다른 thread 가 사용하므로,
	 *	이 thread 도 sharedmemory 를 초기화 해야하므로.
	 */
	if ( index < count ) {
		data = tex1Dfetch( inASBufferTexture, index );
	}

	/** 
	 *	무효한 sub-pixel 이 아닐때만 처리. 
	 *	이런 방식말고 if ( cornerNumber >= 0 ) return; 처럼 여기서 return 시키면 안되고!! 반드시 이렇게 조건문을 사용 
	 *	아래에서 합산할때 cornerNumber < 0 인 경우의 sharedMemory 값을 참조해야 하므로. sharedMemory 초기화 부분이 수행되어야 한다.
	 */
	if ( data >= 0 ) {

		pixelIndex = GET_SUBPIXEL_INDEX( data );
		cornerNumber = GET_SUBPIXEL_CORNER( data );
		x = pixelIndex % g_SceneInfo.iResolutionX;
		y = pixelIndex / g_SceneInfo.iResolutionX;

		if ( cornerNumber >= 10 ) {
			master = true;
			cornerNumber = cornerNumber % 10;
		}

		/** 
		 *	어떤 corner 인지에 따라서 샘플링 구간의위치 결정
		 *	( cornerNumber / 2 ) * 2 는 2 로 나눌때 int 로 round-off 되고 * 2를 하는것.
		 */
		int samplingX = ( cornerNumber % 2 ) * 2 + ( samplingIndex % 2 );
		int samplingY = ( cornerNumber / 2 ) * 2 + ( samplingIndex / 2 );

		/**
		 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
		 */
		pos = g_CameraInfo.eye;

		/** 가상으로 4 by 4 를 했을때의 각 샘플링지점을 구한다. */
		sx = (float) x + ( (float) samplingX + radicalInverse( x << 8 + samplingX, 3 ) ) * 0.25f;
		sy = (float) y + ( (float) samplingY + radicalInverse( y << 8 + samplingY, 5 ) ) * 0.25f;

		dir = g_CameraInfo.startPoint + 
			  g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
		dir = normalize( dir - pos );

		currRay.dir = make_float3( dir.x, dir.y, dir.z );
		currRay.pos = make_float3( pos.x, pos.y, pos.z );

		point.init();
		point.colorWeight.x = 1.0f;
		point.colorWeight.y = 1.0f;
		point.colorWeight.z = 1.0f;

		do {

			secondary = false;
			currIsectCheck.init();
			singlePassIntersect( currRay, currIsectCheck  );

			/**
			 *	intersection check
			 */
			if ( currIsectCheck.isHit() ) {

				makeIntersectionPoint( &currRay, &currIsectCheck, &point );

				cuObjectMaterial material;
				getObjectMaterial( currIsectCheck.objectIndex, material );
				
				/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
				float3 diffuse = material.diffuse;
				calTextureColor( point, material, diffuse );

				color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection)*( maxReflectionDepth > depth ) ) * 
						fixedOption_calSinglePassDirectIllumination_ShadowOn( point, material, diffuse );

				/**
				 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
				 */
				if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

					dir = refraction( point.dir, point.normal, material.refractionIndex );
					point.colorWeight = point.colorWeight * material.transparency * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

					dir = reflection( point.dir, point.normal );
					point.colorWeight =  point.colorWeight * material.reflection * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				}

			}

			depth++;

		} while( secondary );

	}

	/**
	 *	위에서 ray 추적을할때도 shared memory 를 사용하므로 반드시 여기까지 동기화를 한번한다음에
	 *	위에서 사용한 shared memory 를 color 누적하는 buffer 로 사용해야 한다.
	 */
	__syncthreads();

	/** 
	 *	반드시 이 sharedMemory 는 ray 를 추적할때 사용되는 stack 을 위한 shared memory 
	 *	사이즈이내 에서 사용해야 한다. 
	 *	 같은 pixel 에 대해서 샘플링한 결과를 평균내서 framebuffer 에 칠하기 위해서 일단 각자의
	 *	데이터를 shared memory 에 저장하고나서, 대표thread 들이 합산해서 frame buffer 에 저장한다.
	 */
	/** 
	 *	각 ray 는 천체픽셀에 1/16 만큼만 weighted 된다. 
	 *	shared memory 는 ray 를 추적할때 stack 으로도 사용되므로 ray 추적이 끝난후에 반드시 여기서 초기화후 사용해야 한다. 
	 */
	sharedMemory[ threadIdx.x * 4 + 0 ] = color.x * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 1 ] = color.y * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 2 ] = color.z * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 3 ] = pixelIndex;

	__syncthreads();

	/** 각 sub-pixel 의 첫번째 thread 가 해당 sub-pixel 의 color 를 누적 계산한다. */
	if ( samplingIndex != 0 ) 
		return;

	for ( int i = 1; i < 4; ++i ) {

		sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
		sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
		sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];

	}

	__syncthreads();

	/** 
	 *	각 pixel 의 master sub-pixel 의 첫번째 thread 가 해당 pixel 에 칠해질 color 를 누적한다. 
	 *	한 pixel 을 구성하는 sub pixel 들은 thread index 가 증가하는순으로 붙어있고 최대 4개만 존재하므로
	 *  현재 master thread 로 부터 최대 4 sub-pixel 의 첫번째 thread 만 체크하면 된다.
	 */
	if ( !master ) return;

	for ( int i = 4; i <= 12; i += 4 ) {
		/** 같은 pixel 을 구성하는 sub-pixel 일때만 누적 */
		if ( threadIdx.x + i < blockDim.x && 
			sharedMemory[ ( threadIdx.x + i ) * 4 + 3 ] == pixelIndex ) {
			sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
			sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
			sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];
		}
	}

	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer[ idx + 0 ] += ( sharedMemory[ threadIdx.x * 4 + 0 ] );
	pFrameBuffer[ idx + 1 ] += ( sharedMemory[ threadIdx.x * 4 + 1 ] );
	pFrameBuffer[ idx + 2 ] += ( sharedMemory[ threadIdx.x * 4 + 2 ] );

}


/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void fixedOption_singlePassRayTracingKernel_SuperSamplingStage_ShadowOff( 
											float* pFrameBuffer,
											int startSubPixelIndex,
											int count,
											int maxReflectionDepth )
{
	float3 pos, dir;
	bool secondary = false;
	float sx, sy;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );
	bool master = false;
	int index = startSubPixelIndex + ( blockIdx.x * blockDim.x + threadIdx.x ) / 4;

	int data = -1;
	int pixelIndex = 0, cornerNumber = 0, x = 0, y = 0;
	int samplingIndex = ( blockIdx.x * blockDim.x + threadIdx.x ) % 4;

	/**
	 *	index 범위안에 포함되지 않은 thread 라도 return 시키면 안된다. 
	 *	이 thread 에 할당될 shared memory 를 다른 thread 가 사용하므로,
	 *	이 thread 도 sharedmemory 를 초기화 해야하므로.
	 */
	if ( index < count ) {
		data = tex1Dfetch( inASBufferTexture, index );
	}

	/** 
	 *	무효한 sub-pixel 이 아닐때만 처리. 
	 *	이런 방식말고 if ( cornerNumber >= 0 ) return; 처럼 여기서 return 시키면 안되고!! 반드시 이렇게 조건문을 사용 
	 *	아래에서 합산할때 cornerNumber < 0 인 경우의 sharedMemory 값을 참조해야 하므로. sharedMemory 초기화 부분이 수행되어야 한다.
	 */
	if ( data >= 0 ) {

		pixelIndex = GET_SUBPIXEL_INDEX( data );
		cornerNumber = GET_SUBPIXEL_CORNER( data );
		x = pixelIndex % g_SceneInfo.iResolutionX;
		y = pixelIndex / g_SceneInfo.iResolutionX;

		if ( cornerNumber >= 10 ) {
			master = true;
			cornerNumber = cornerNumber % 10;
		}

		/** 
		 *	어떤 corner 인지에 따라서 샘플링 구간의위치 결정
		 *	( cornerNumber / 2 ) * 2 는 2 로 나눌때 int 로 round-off 되고 * 2를 하는것.
		 */
		int samplingX = ( cornerNumber % 2 ) * 2 + ( samplingIndex % 2 );
		int samplingY = ( cornerNumber / 2 ) * 2 + ( samplingIndex / 2 );

		/**
		 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
		 */
		pos = g_CameraInfo.eye;

		/** 가상으로 4 by 4 를 했을때의 각 샘플링지점을 구한다. */
		sx = (float) x + ( (float) samplingX + radicalInverse( x << 8 + samplingX, 3 ) ) * 0.25f;
		sy = (float) y + ( (float) samplingY + radicalInverse( y << 8 + samplingY, 5 ) ) * 0.25f;

		dir = g_CameraInfo.startPoint + 
			  g_CameraInfo.u * sx * g_CameraInfo.stepX - g_CameraInfo.v * sy * g_CameraInfo.stepY;
		dir = normalize( dir - pos );

		currRay.dir = make_float3( dir.x, dir.y, dir.z );
		currRay.pos = make_float3( pos.x, pos.y, pos.z );

		point.init();
		point.colorWeight.x = 1.0f;
		point.colorWeight.y = 1.0f;
		point.colorWeight.z = 1.0f;

		do {

			secondary = false;
			currIsectCheck.init();
			singlePassIntersect( currRay, currIsectCheck  );

			/**
			 *	intersection check
			 */
			if ( currIsectCheck.isHit() ) {

				makeIntersectionPoint( &currRay, &currIsectCheck, &point );

				cuObjectMaterial material;
				getObjectMaterial( currIsectCheck.objectIndex, material );
				
				/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
				float3 diffuse = material.diffuse;
				calTextureColor( point, material, diffuse );

				color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection)*( maxReflectionDepth > depth ) ) * 
						fixedOption_calSinglePassDirectIllumination_ShadowOff( point, material, diffuse );

				/**
				 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
				 */
				if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

					dir = refraction( point.dir, point.normal, material.refractionIndex );
					point.colorWeight = point.colorWeight * material.transparency * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

					dir = reflection( point.dir, point.normal );
					point.colorWeight =  point.colorWeight * material.reflection * diffuse;
					currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
					currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
					currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
					currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

					secondary = true;

				}

			}

			depth++;

		} while( secondary );

	}

	/**
	 *	위에서 ray 추적을할때도 shared memory 를 사용하므로 반드시 여기까지 동기화를 한번한다음에
	 *	위에서 사용한 shared memory 를 color 누적하는 buffer 로 사용해야 한다.
	 */
	__syncthreads();

	/** 
	 *	반드시 이 sharedMemory 는 ray 를 추적할때 사용되는 stack 을 위한 shared memory 
	 *	사이즈이내 에서 사용해야 한다. 
	 *	 같은 pixel 에 대해서 샘플링한 결과를 평균내서 framebuffer 에 칠하기 위해서 일단 각자의
	 *	데이터를 shared memory 에 저장하고나서, 대표thread 들이 합산해서 frame buffer 에 저장한다.
	 */
	/** 
	 *	각 ray 는 천체픽셀에 1/16 만큼만 weighted 된다. 
	 *	shared memory 는 ray 를 추적할때 stack 으로도 사용되므로 ray 추적이 끝난후에 반드시 여기서 초기화후 사용해야 한다. 
	 */
	sharedMemory[ threadIdx.x * 4 + 0 ] = color.x * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 1 ] = color.y * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 2 ] = color.z * 0.0625f;
	sharedMemory[ threadIdx.x * 4 + 3 ] = pixelIndex;

	__syncthreads();

	/** 각 sub-pixel 의 첫번째 thread 가 해당 sub-pixel 의 color 를 누적 계산한다. */
	if ( samplingIndex != 0 ) 
		return;

	for ( int i = 1; i < 4; ++i ) {

		sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
		sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
		sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];

	}

	__syncthreads();

	/** 
	 *	각 pixel 의 master sub-pixel 의 첫번째 thread 가 해당 pixel 에 칠해질 color 를 누적한다. 
	 *	한 pixel 을 구성하는 sub pixel 들은 thread index 가 증가하는순으로 붙어있고 최대 4개만 존재하므로
	 *  현재 master thread 로 부터 최대 4 sub-pixel 의 첫번째 thread 만 체크하면 된다.
	 */
	if ( !master ) return;

	for ( int i = 4; i <= 12; i += 4 ) {
		/** 같은 pixel 을 구성하는 sub-pixel 일때만 누적 */
		if ( threadIdx.x + i < blockDim.x && 
			sharedMemory[ ( threadIdx.x + i ) * 4 + 3 ] == pixelIndex ) {
			sharedMemory[ threadIdx.x * 4 + 0 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 0 ];
			sharedMemory[ threadIdx.x * 4 + 1 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 1 ];
			sharedMemory[ threadIdx.x * 4 + 2 ] += sharedMemory[ ( threadIdx.x + i ) * 4 + 2 ];
		}
	}

	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer[ idx + 0 ] += ( sharedMemory[ threadIdx.x * 4 + 0 ] );
	pFrameBuffer[ idx + 1 ] += ( sharedMemory[ threadIdx.x * 4 + 1 ] );
	pFrameBuffer[ idx + 2 ] += ( sharedMemory[ threadIdx.x * 4 + 2 ] );
}

//-------------------------------------------------------------------------------------------------------------
//
//
//
//
//	Option 을 Fix 한 버전.
//
//	Max Reflection Bounce = 1
//	광원 최대 2개.
//
//
//
//-------------------------------------------------------------------------------------------------------------


/**
 *	fixed option 버전. 
 */
__global__ void fixedOption_singlePassRayTracingKernel_1_SamplingKernel_ShadowOff( 
												cuSamplingMap *pSamplingMap,
												float* pFrameBuffer, 
												float* pFrameBuffer2,
												int maxReflectionDepth )
{
	float3 dir;
	bool secondary = false;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int rayIndex = y * g_SceneInfo.iResolutionX + x;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */
	dir = g_CameraInfo.startPoint + g_CameraInfo.u * ((float)x + 0.5f ) * g_CameraInfo.stepX - 
		  g_CameraInfo.v * ((float)y + 0.5f ) * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	currRay.dir = make_float3( dir.x, dir.y, dir.z );
	currRay.pos = make_float3( g_CameraInfo.eye.x, g_CameraInfo.eye.y, g_CameraInfo.eye.z );

	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	/** 
	 *	현재지점의 sampling info 초기화.
	 *	integer 를 float 로 int_as_float_H 식으로 저장.
	 *
	 *	normal 은 다 1.0 으로 세팅해야 한다.
	 */
	pSamplingMap[ rayIndex ].primaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].primaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );
	pSamplingMap[ rayIndex ].secondaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].secondaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );

	do {

		secondary = false;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck  );

		point.normal = make_float3( 1.0f, 1.0f, 1.0f );
		point.objectIndex = OBJECT_MAX_ID;
		point.shadowCount = 0;
		point.bTexture = 0;

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {

			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* fixedOption_calSinglePassDirectIllumination_ShadowOff( point, material, diffuse );				

			/**
			 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
			 */
			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight =  point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z; 
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			}

			if ( depth == 0 ) {
				pSamplingMap[ rayIndex ].primaryNormal = point.normal;
				pSamplingMap[ rayIndex ].primaryAttr =  
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

			/** 2차 ray 가 intersection 한경우. */
			if ( depth == 1 ) {
				pSamplingMap[ rayIndex ].secondaryNormal = point.normal;
				pSamplingMap[ rayIndex ].secondaryAttr = 
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

		}

		depth++;

	} while( secondary );
			
	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer2[ idx + 0 ] = color.x;
	pFrameBuffer2[ idx + 1 ] = color.y;
	pFrameBuffer2[ idx + 2 ] = color.z;

	//pFrameBuffer2[ idx ] = 0.3 * color.x + 0.59 * color.y + 0.11 * color.z;

	pFrameBuffer[ idx + 0 ] = color.x;
	pFrameBuffer[ idx + 1 ] = color.y;
	pFrameBuffer[ idx + 2 ] = color.z;


}




/**
 *	Single Pass 로 RayTracing 을 수행.
 */
__global__ void fixedOption_singlePassRayTracingKernel_1_SamplingKernel_ShadowOn( 
											cuSamplingMap *pSamplingMap,
											float* pFrameBuffer, float* pFrameBuffer2, 
											int maxReflectionDepth )
{
	float3 dir;
	bool secondary = false;
	int depth = 0;
	cuRay currRay;
	cuIntersectionCheck currIsectCheck;
	cuIntersectionPoint point;
	float3 color = make_float3( 0.0f, 0.0f, 0.0f );

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int rayIndex = y * g_SceneInfo.iResolutionX + x;

	if ( x >= g_SceneInfo.iResolutionX || y >= g_SceneInfo.iResolutionY )
		return;

	/**
	 *	왼쪽,상단 포인트의 카메라 plane 상에서의 좌표.
	 */
	dir = g_CameraInfo.startPoint + g_CameraInfo.u * ((float)x + 0.5f ) * g_CameraInfo.stepX - 
		  g_CameraInfo.v * ((float)y + 0.5f ) * g_CameraInfo.stepY;
	dir = normalize( dir - g_CameraInfo.eye );

	currRay.dir = make_float3( dir.x, dir.y, dir.z );
	currRay.pos = make_float3( g_CameraInfo.eye.x, g_CameraInfo.eye.y, g_CameraInfo.eye.z );

	point.init();
	point.colorWeight.x = 1.0f;
	point.colorWeight.y = 1.0f;
	point.colorWeight.z = 1.0f;

	/** 
	 *	현재지점의 sampling info 초기화.
	 *	integer 를 float 로 int_as_float_H 식으로 저장.
	 *
	 *	normal 은 다 1.0 으로 세팅해야 한다.
	 */
	pSamplingMap[ rayIndex ].primaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].primaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );
	pSamplingMap[ rayIndex ].secondaryNormal = make_float3( 1.0f, 1.0f, 1.0f );
	pSamplingMap[ rayIndex ].secondaryAttr = MAKE_PIXEL_ATTR_ASFLOAT( OBJECT_MAX_ID, 0, 0, 0 );

	do {

		secondary = false;
		currIsectCheck.init();
		singlePassIntersect( currRay, currIsectCheck  );

		point.normal = make_float3( 1.0f, 1.0f, 1.0f );
		point.objectIndex = OBJECT_MAX_ID;
		point.shadowCount = 0;
		point.bTexture = 0;

		/**
		 *	intersection check
		 */
		if ( currIsectCheck.isHit() ) {

			makeIntersectionPoint( &currRay, &currIsectCheck, &point );

			cuObjectMaterial material;
			getObjectMaterial( currIsectCheck.objectIndex, material );

			/** texture 가 존재하면 diffuse color 를 texture 내의 u, v 에서 계산된 color 로 대체 */
			float3 diffuse = material.diffuse;
			calTextureColor( point, material, diffuse );

			color += point.colorWeight * 
						( 1.0f - (material.transparency + material.reflection) * ( maxReflectionDepth > depth ) )
						* fixedOption_calSinglePassDirectIllumination_ShadowOn_ForSelective( point, material, diffuse );				

			/**
			 *	reflection depth 가 켜져 있다면, 1-bounce 만큼 처리.
			 */
			if ( maxReflectionDepth > depth && material.transparency > 0.0f ) {

				dir = refraction( point.dir, point.normal, material.refractionIndex );
				point.colorWeight = point.colorWeight * material.transparency * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z;
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			} else if ( maxReflectionDepth > depth && material.reflection > 0.0f ) {

				dir = reflection( point.dir, point.normal );
				point.colorWeight =  point.colorWeight * material.reflection * diffuse;
				currRay.dir.x = dir.x; currRay.dir.y = dir.y; currRay.dir.z = dir.z; 
				currRay.pos.x = point.pos.x + dir.x * RAY_START_EPSILON;
				currRay.pos.y = point.pos.y + dir.y * RAY_START_EPSILON;
				currRay.pos.z = point.pos.z + dir.z * RAY_START_EPSILON;

				secondary = true;

			}

			if ( depth == 0 ) {
				pSamplingMap[ rayIndex ].primaryNormal = point.normal;
				pSamplingMap[ rayIndex ].primaryAttr =  
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

			/** 2차 ray 가 intersection 한경우. */
			if ( depth == 1 ) {
				pSamplingMap[ rayIndex ].secondaryNormal = point.normal;
				pSamplingMap[ rayIndex ].secondaryAttr = 
						MAKE_PIXEL_ATTR_ASFLOAT( (int)point.objectIndex, (int)point.shadowCount, (int)currIsectCheck.bSelected, (int)( point.bTexture ) );
			}

		}

		depth++;

	} while( secondary );
			
	int idx = 3 * ( ( g_SceneInfo.iResolutionY - y - 1 ) * g_SceneInfo.iResolutionX + x );

	pFrameBuffer2[ idx + 0 ] = color.x;
	pFrameBuffer2[ idx + 1 ] = color.y;
	pFrameBuffer2[ idx + 2 ] = color.z;

	//pFrameBuffer2[ idx ] = 0.3 * color.x + 0.59 * color.y + 0.11 * color.z;

	pFrameBuffer[ idx + 0 ] = color.x;
	pFrameBuffer[ idx + 1 ] = color.y;
	pFrameBuffer[ idx + 2 ] = color.z;

}

#endif

