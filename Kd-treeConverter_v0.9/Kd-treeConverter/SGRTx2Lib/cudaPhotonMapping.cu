#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
//#include <cutil.h>

#include "GLogManager.h"

#include "cuda_math.h"
#include "cudaPhotonMapping.cuh"
#include "cudaPhotonMappingKernel.cu"

#define PHOTONMAPPING_THREAD			256
#define PHOTONMAPPING_BOUNDING_THREAD	128

#define GATHERING_MAX_BLOCK_COUNT		1024		// gathering 시 한번에 돌릴 Block 갯수.
#define GATHERING_THREAD				64

/**
 *	Cuda 로 Photon Mapping 을 수행.
 *
 *	by graphicsian
 */

cudaPhotonMapping::cudaPhotonMapping()
{
}

cudaPhotonMapping::~cudaPhotonMapping()
{
	uninitialize();
}

/**
 *	CUDA Global Memory 에 Light 데이터를 업로드하고, photon
 *	저장공간을 할당한다. 
 *
 *	@param pContext photon mapping context.
 *	@return GError result
 */ 
GError cudaPhotonMapping::initialize( int maxPhotonSize, int maxIntersectionPoint )
{
	m_iMaxPhotonSize = maxPhotonSize;
	m_iMaxIntersectionPoint = maxIntersectionPoint;
	
	/** 
	 *	device 공간에 photon 저장공간 할당 
	 */	
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDevicePhotonMem,
								sizeof( cuPhoton ) * m_iMaxPhotonSize ) );
	CUDA_SAFE_CALL( cudaMemset( m_pDevicePhotonMem, 0x00, 
								sizeof( cuPhoton ) * m_iMaxPhotonSize ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceIntResult, sizeof( int ) ) );
	
	if ( checkError( "cudaAllocPhotonStorage" ) != cudaSuccess )
		return errorCudaPhotonAllocError;
	
	/**
	 *	intersection point 과 결과를 위한 공간 할당.
	 */
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDevicePMIsectPointMem, 
								sizeof( cuPMIntersectionPoint ) * m_iMaxIntersectionPoint ) );
	if ( checkError( "pDeviceIPointMem malloc" ) != cudaSuccess )
		return errorCudaPhotonAllocError;
		
	CUDA_SAFE_CALL( cudaMalloc( (void**) &m_pDeviceIsectPointMem, 
								sizeof( cuIntersectionPoint ) * m_iMaxIntersectionPoint ) );
	if ( checkError( "pDeviceIPointMem malloc" ) != cudaSuccess )
		return errorCudaPhotonAllocError;
	
	return errorNo;
}

/**
 *	photon map 을 clear 하고 싶을때.
 */
GError cudaPhotonMapping::photonMapclear()
{
	CUDA_SAFE_CALL( cudaMemset( m_pDevicePhotonMem, 
								0x00, sizeof( cuPhoton ) * m_iMaxPhotonSize ) );

	if ( checkError( "photonMapclear" ) != cudaSuccess )
		return errorCudaPhotonAllocError;
		
	return errorNo;
}

/**
 *	자원해제.
 */
GError cudaPhotonMapping::uninitialize()
{
	if ( m_pDevicePhotonMem ) {
		CUDA_SAFE_CALL( cudaFree( m_pDevicePhotonMem ) );
		m_pDevicePhotonMem = NULL;
	}
	if ( m_pDeviceIntResult ) {
		CUDA_SAFE_CALL( cudaFree( m_pDeviceIntResult ) );
		m_pDeviceIntResult = NULL;
	}
	
	if ( m_pDeviceIsectPointMem ) {
		CUDA_SAFE_CALL( cudaFree( m_pDeviceIsectPointMem ) );
		m_pDeviceIsectPointMem = NULL;
	}
	if ( m_pDevicePMIsectPointMem ) {
		CUDA_SAFE_CALL( cudaFree( m_pDevicePMIsectPointMem ) );
		m_pDevicePMIsectPointMem = NULL;
	}
	
	if ( checkError( "uninitialize" ) != cudaSuccess )
		return errorCudaFreeError;
	
	return errorNo;
}

/**
 *	각 광원으로부터 photon 을 출발시켜서 bound 되는 photon 까지
 *	모두 추적한다음에 결과를 global photon memory 저장한다.
 *	photon global memory 는 각 bound 당 고정 메모리 emitPhoton 만큼씩 사용될 것이다. 
 *	따라서 필요한 총 메모리는 emitPhoton * m_iMaxBound 만큼 잡혀 있어야 한다.
 *	m_iMaxPhotonSize 가 그것. 
 *	결과는 pOutHostMem, pOutPhotonSize, pOutTracedBound 에 담아서 보내준다.
 *	pOutHostMem 은 최소한 emitPhoton * maxBound 만큼 잡혀있어야 한다.
 *
 *	direct photon 을 저장하지 않는 옵션이라면 bound 가 0 인것은 저장하지 않는다.
 *
 */
GError cudaPhotonMapping::photonTracing( int emitPhoton, int maxBound, int randomSeed,
										 bool bSaveDirectPhoton,
										 cudaRenderPipeline *pRenderPipeline,
										 cuPhoton *pOutHostMem, int *pOutTracedPhotonSize, 
										 int *pOutTracedBound,
										 bool faceCCW )
{
	GError error = errorNo;
	GTimer timer;
	timer.start();

	/**
	 *	CUDA RENDER Pipeline class 에서 ray 저장소를 가져와서 emit photon 을 추적할
	 *	ray 정보를 기록하고 ray casting 을 수행시킨다. ray 공간은 한번에 뿌릴
	 *	photon 의 개수보다 커야한다.
	 */
	if ( emitPhoton * maxBound < m_iMaxPhotonSize ) {
		GLogManager::logging( LOG_ERROR, "overflow max photon size : %d", m_iMaxPhotonSize );
		return errorNotEnoughRayMem;
	}
	
	if ( pRenderPipeline->getDeviceRayBufferSize() < emitPhoton ) {
		GLogManager::logging( LOG_ERROR, "ray memory is smaller than photon emit count." );
		return errorNotEnoughRayMem;
	}

	/**
	 *	한번에 수행시킬 thread block 개수를 구한다.
	 */
	int threads = PHOTONMAPPING_THREAD;
	int block = ( emitPhoton / threads ) + 1;

	GLogManager::logging( LOG_DEBUG, "----------------- Jedi Photon Tracing ---------------------------", 
						block, PHOTONMAPPING_THREAD, block * PHOTONMAPPING_THREAD );
						
	GLogManager::logging( LOG_DEBUG, " -> Block = %d, Thread = %d, Total Threas = %d", 
						block, PHOTONMAPPING_THREAD, block * PHOTONMAPPING_THREAD );
	
	/**-----------------------------------------------------------------------------------------------
	 *	photon 을 각 광원으로부터 emit 시켜서 
	 *	pContex 는 host 메모리상에 있는것이므로 포인터를 넘기면 안되고
	 *	반드시 복사되게끔 call by value 로 넘겨야 한다.!
	 **-----------------------------------------------------------------------------------------------*/
	 
	cuPhotonEmitKernel<<< block, threads >>>( 
			emitPhoton, randomSeed, m_pDevicePhotonMem, pRenderPipeline->getDeviceRayBuffer() );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	if ( checkError( "cuPhotonEmitKernel" ) != cudaSuccess )
		return errorphotonTracingError;

	/**-----------------------------------------------------------------------------------------------
	 *	max bound 에 도달하거나 bound 되는 photon 이 하나도 없을때까지
	 *	반복하면서 photon 을 global photon memory 에 저장한다. 한번 photon 을 뿌린후에는
	 *	photon 정보가 photon memory 에 block 단위로 기록된다. block 크기는 emitPhoton 개수만큼
	 *	
	 *	ray 체크 cudaRenderPipeline 을 사용하는데, 항상 새로 추적할 ray 는
	 *	ray 공간의 offset 0 부터 emitPhoton 개만큼저장되어 block 단위로 추적된다.
	 *	 이중 bound 되지 않아서 추적하지 않아도	되는 ray 는 mint 가 FLT_MAX로 세팅되어 있을것이다. 
	 **-----------------------------------------------------------------------------------------------*/

	int bound = 0;
	int atLeastOne = 0;
	
	do {
		/**
		 *	각 bound 별 photon ray 를 추적후 결과는 device 상의 intersection point memory 공간에
		 *	offset 0 부터 emitPhoton 사이에 매치되어 기록되게 된다.
		 *  pRenderPipeline->getDeviceRayBuffer() 와 pRenderPipeline->getDeviceIntersectionBuffer()
		 *	이 각각 device 상의 ray 공간 intersection point 공간을 의미.
		 *
		 *	photon 을 뿌릴때는 back face culling 을 하지 않는다.
		 */
		error = pRenderPipeline->doRayCastingSequentialData( 0, emitPhoton, faceCCW, false );
		if ( error != errorNo ) {
			return error;
		}

		CUDA_SAFE_CALL( cudaMemset( m_pDeviceIntResult, 0x00, sizeof( int ) ) );
	
		/** 
		 *	block 과 thread 개수 곱한게 emitPhoton 개수가 되는지 확인할것 
		 */
		threads = PHOTONMAPPING_BOUNDING_THREAD;
		block = ( emitPhoton / threads ) + 1;
		cuMakePhotonMapAndBoundingKernel<<< block, threads >>>( 
											emitPhoton, randomSeed, bound, maxBound,
											bSaveDirectPhoton,
											m_pDevicePhotonMem, 
											pRenderPipeline->getDeviceRayBuffer(),
											pRenderPipeline->getDeviceIntersectionBuffer(),
											m_pDeviceIntResult );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		if ( checkError( "cuMakePhotonMapAndBoundingKernel" ) != cudaSuccess ) {
			error = errorPhotonBoundingError;
			break;
		}

		CUDA_SAFE_CALL( cudaMemcpy( &atLeastOne, 
									m_pDeviceIntResult, 
									sizeof( int ),
									cudaMemcpyDeviceToHost ) );
									
		++bound;
											
	} while( atLeastOne > 0 );

	/**
	 *	최종적으로 photon map 을 device->host 로 복사해온다.
	 */
	CUDA_SAFE_CALL( cudaMemcpy( pOutHostMem, m_pDevicePhotonMem,
								sizeof( cuPhoton ) * bound * emitPhoton, cudaMemcpyDeviceToHost ) );
	if ( checkError( "pHostPhotonMem cudaMemcpy" ) != cudaSuccess ) {
		(*pOutTracedBound) = 0;
		(*pOutTracedPhotonSize) = 0;
		return errorPhotonBoundingError;
	}

	(*pOutTracedBound) = bound;
	(*pOutTracedPhotonSize) = bound * emitPhoton;
	
	timer.end();
	GLogManager::logging( LOG_DEBUG, " -> Tracing end. time=%f sec. bound = %d", 
								timer.getElapsedTime(), bound );

	GLogManager::logging( LOG_DEBUG, "-----------------------------------------------------------------" );

	return error;
}

/**
 *	Intersection point 정보를 DEVICE 에 올린다.
 */	
GError cudaPhotonMapping::uploadIntersectionPoint( cuIntersectionPoint *isectPointData, int iCount )
{
	if ( iCount > m_iMaxIntersectionPoint ) {
		GLogManager::logging( LOG_FATAL, "intersection point (%d) too many.", iCount );
		return errorCudaError;
	}
	
	/**
	 *	intersection point 정보 복사.
	 */	
	CUDA_SAFE_CALL( cudaMemcpy( m_pDeviceIsectPointMem, isectPointData, 
					sizeof( cuIntersectionPoint ) * iCount, cudaMemcpyHostToDevice ) );
	
	if ( checkError( "pDeviceIPointMem copy" ) != cudaSuccess ) {
		return errorCudaError;
	}
	
	return errorNo;
}

/**
 *	area photon 으로 density area 를 계산한다.
 *	intersection point 는 이미 이전에 올라가 있어야 한다.
 *	area 를 계산해서 intersection point device memory 의 area 를 업데이트 시킨다.
 */
GError cudaPhotonMapping::calDensityArea( cuPMIntersectionPoint *pPMIsectPoint, int iCount,
										  cuPhoton *pPhotonInfo, int photonCount,
										  cuPhotonIndex *pPhotonIndex, 
										  int photonIndexCount, float radius )
{
	int startOffset = 0;
	int loop = 0;
	bool bSuccess = false;
	
	if ( iCount > m_iMaxIntersectionPoint ) {
		GLogManager::logging( LOG_FATAL, "intersection point too many." );
		return errorCudaError;
	}
	
	/**
	 *	photon info를 texture 로 올리기
	 */
	float *pDeviceAreaPhotonData = NULL;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&pDeviceAreaPhotonData, sizeof( cuPhoton ) * photonCount ) );
	CUDA_SAFE_CALL( cudaMemcpy( pDeviceAreaPhotonData, pPhotonInfo, sizeof( cuPhoton ) * photonCount,
						cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, areaPhotonTexture, pDeviceAreaPhotonData ) );

	/**
	 *	photon index 를 texture 로 올리기
	 */
	int *pDevicePhotonIndexData = NULL;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&pDevicePhotonIndexData, 
						sizeof( cuPhotonIndex ) * photonIndexCount ) );
	CUDA_SAFE_CALL( cudaMemcpy( pDevicePhotonIndexData, pPhotonIndex, 
						sizeof( cuPhotonIndex ) * photonIndexCount,
						cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, photonIndexTexture, pDevicePhotonIndexData ) );

	/**
	 *	intersection point 와 photon 연관관계 정보 복사.
	 *	pPMIsectPoint 크기는 context 안의 iCurrentIntersectionPoint 와 같다.
	 */	
	CUDA_SAFE_CALL( cudaMemcpy( m_pDevicePMIsectPointMem, pPMIsectPoint, 
						sizeof( cuPMIntersectionPoint ) * iCount, cudaMemcpyHostToDevice ) );
	
	if ( checkError( "pDeviceIPointMem copy" ) != cudaSuccess ) {
		return errorCudaError;
	}

	/**
	 *	ray 데이터를 cuda 에서 처리할 수 있는 일정 개수만큼
	 *	잘라서 처리시킨다.
	 */
	while( true ) {
	
		loop++;
		
		int blockCount = min( GATHERING_MAX_BLOCK_COUNT, 
				( ( iCount - startOffset ) / GATHERING_THREAD ) + 1 );

		cuCalDensityAreaKernel<<< blockCount, GATHERING_THREAD >>> ( 
								m_pDeviceIsectPointMem, 
								m_pDevicePMIsectPointMem,
								iCount,
								startOffset, 
								radius * radius );
								
		startOffset += blockCount * GATHERING_THREAD;

		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		cudaError_t result = cudaGetLastError();

		if ( result != cudaSuccess ) {
			GLogManager::logging( LOG_FATAL, "Result : failed. ( %s ). ray count = %d, photon count = %d, index count = %d", 
				cudaGetErrorString( result ), iCount, photonCount, photonIndexCount  );
			bSuccess = false;
			break;
		}

		if ( startOffset >= iCount ) {
			bSuccess = true;
			break;
		}
	}

	/**
	 *	결과를 받는다.
	 */
	if ( bSuccess ) {
	
		CUDA_SAFE_CALL( cudaMemcpy( pPMIsectPoint, m_pDevicePMIsectPointMem, 
						sizeof( cuPMIntersectionPoint ) * iCount, cudaMemcpyDeviceToHost ) );
	
			int validRay = 0;
			for ( int i = 0; i < iCount; ++i ) {
				if ( pPMIsectPoint[i].usedPhotonCount > 0 )
					validRay++;
			}

			GLogManager::logging( LOG_DEBUG, "valid ray count = %d", validRay );

			// test
			for ( int i = 0; i < 10 && i < iCount; ++i ) {
				GLogManager::logging( LOG_DEBUG, "RayResult : total photon = %d, used photon = %d, area = %f", 
					pPMIsectPoint[i].totalPhotonCount, 
					pPMIsectPoint[i].usedPhotonCount, 
					pPMIsectPoint[i].area );
			}

			
	}

	/**
	 *	Data Free
	 */
	CUDA_SAFE_CALL( cudaUnbindTexture( areaPhotonTexture ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( photonIndexTexture ) );

	CUDA_SAFE_CALL( cudaFree( pDeviceAreaPhotonData ) );
	CUDA_SAFE_CALL( cudaFree( pDevicePhotonIndexData ) );

	if ( bSuccess )
		return errorNo;
	
	return errorCudaError;	
}

GError cudaPhotonMapping::photonGathering( cuPMIntersectionPoint *pPMIsectPoint, int iCount,
										   cuPhoton *pPhotonInfo, int photonCount,
										   cuPhotonIndex *pPhotonIndex, int photonIndexCount, float radius )
{
	unsigned int timer;
	GError error;
	int startOffset = 0;
	int loop = 0;
	bool bSuccess = false;

	if ( iCount > m_iMaxIntersectionPoint ) {
		GLogManager::logging( LOG_FATAL, "intersection point too many." );
		return errorCudaError;
	}

	//CUDA_SAFE_CALL( cutCreateTimer( &timer ) );
	//CUDA_SAFE_CALL( cutStartTimer( timer ) );

	/**
	 *	photon info를 texture 로 올리기
	 */
	float *pDevicePhotonData = NULL;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&pDevicePhotonData, sizeof( cuPhoton ) * photonCount ) );
	CUDA_SAFE_CALL( cudaMemcpy( pDevicePhotonData, pPhotonInfo, 
						sizeof( cuPhoton ) * photonCount,
						cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, photonTexture, pDevicePhotonData ) );

	/**
	 *	photon index 를 texture 로 올리기
	 */
	int *pDevicePhotonIndexData = NULL;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&pDevicePhotonIndexData, 
						sizeof( cuPhotonIndex ) * photonIndexCount ) );
	CUDA_SAFE_CALL( cudaMemcpy( pDevicePhotonIndexData, pPhotonIndex, 
						sizeof( cuPhotonIndex ) * photonIndexCount,
						cudaMemcpyHostToDevice ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, photonIndexTexture, pDevicePhotonIndexData ) );
	
	/**
	 *	intersection point 와 photon 연관관계 정보 복사.
	 *	pPMIsectPoint 크기는 context 안의 iCurrentIntersectionPoint 와 같다.
	 */	
	CUDA_SAFE_CALL( cudaMemcpy( m_pDevicePMIsectPointMem, pPMIsectPoint, 
						sizeof( cuPMIntersectionPoint ) * iCount, cudaMemcpyHostToDevice ) );

	/**
	 *	ray 데이터를 cuda 에서 처리할 수 있는 일정 개수만큼
	 *	잘라서 처리시킨다.
	 */
	while( true ) {
	
		loop++;
		int blockCount = min( GATHERING_MAX_BLOCK_COUNT, 
							( ( iCount - startOffset ) / GATHERING_THREAD ) + 1 );

		GLogManager::logging( LOG_DEBUG, "%d) %d ray will be processed block=%d, threads=%d", 
			loop, startOffset, blockCount, GATHERING_THREAD );

		caIPointRadianceKernel<<< blockCount, GATHERING_THREAD >>> ( 
								m_pDeviceIsectPointMem, 
								m_pDevicePMIsectPointMem, 
								iCount, startOffset, radius * radius );

		startOffset += blockCount * GATHERING_THREAD;

		CUDA_SAFE_CALL( cudaThreadSynchronize() );
		cudaError_t result = cudaGetLastError();
		
		if ( result != cudaSuccess ) {
			GLogManager::logging( LOG_DEBUG, "Result : failed. ( %s )", cudaGetErrorString( result ) );
			bSuccess = false;
			break;
		}

		if ( startOffset >= iCount ) {
			bSuccess = true;
			break;
		}
	}
	
	/**
	 *	결과를 받는다.
	 */
	if ( bSuccess ) {
	
		CUDA_SAFE_CALL( cudaMemcpy( pPMIsectPoint, m_pDevicePMIsectPointMem, 
						sizeof( cuPMIntersectionPoint ) * iCount, cudaMemcpyDeviceToHost ) );


		//-----------------------------------------------------------
		int validRay = 0, invalidPowerRay = 0;
		for ( int i = 0; i < iCount; ++i ) {
			if ( pPMIsectPoint[i].usedPhotonCount > 0 )
				validRay++;
			if ( pPMIsectPoint[i].power[0] < 0.0f || 
				pPMIsectPoint[i].power[0] < 0.0f ||
				pPMIsectPoint[i].power[0] < 0.0f ) {
				invalidPowerRay++;
			}
		}

			GLogManager::logging( LOG_DEBUG, "*********** total ray = %d, valid ray count = %d, invalid power ray = %d", 
				iCount, validRay, invalidPowerRay );

			// test
			for ( int i = 0; i < 10 && i < iCount; ++i ) {
				GLogManager::logging( LOG_DEBUG, "RayResult : total photon = %d, used photon = %d, power( %f, %f, %f ), area( %10.8f )", 
					pPMIsectPoint[i].totalPhotonCount, 
					pPMIsectPoint[i].usedPhotonCount, 
					pPMIsectPoint[i].power[0],
					pPMIsectPoint[i].power[1],
					pPMIsectPoint[i].power[2],
					pPMIsectPoint[i].area );
			}
			
	}

	//GLogManager::logging( 
	//	LOG_DEBUG, "ray-photon processing time : %f sec.", cutGetTimerValue( timer ) / 1000.0f );

	//CUT_SAFE_CALL( cutStopTimer( timer ) );
	//CUT_SAFE_CALL( cutDeleteTimer( timer ) );

	/**
	 *	Data Free
	 */
	CUDA_SAFE_CALL( cudaUnbindTexture( photonTexture ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( photonIndexTexture ) );

	CUDA_SAFE_CALL( cudaFree( pDevicePhotonData ) );
	CUDA_SAFE_CALL( cudaFree( pDevicePhotonIndexData ) );

	if ( bSuccess )
		error = errorNo;
	else
		error = errorCudaError;
		
	return error;
}

void cudaPhotonMapping::printStatusInfo()
{
	float fPhotonMem = ( sizeof( cuPhoton ) * m_iMaxPhotonSize ) / 1048576.0f;
	float fIsectPointMem = ( sizeof( cuIntersectionPoint ) * m_iMaxIntersectionPoint ) /  1048576.0f;
	float fPMIsectPointMem = ( sizeof( cuPMIntersectionPoint ) * m_iMaxIntersectionPoint ) / 1048576.0f;
	
	float fTotal = fPhotonMem + fIsectPointMem + fPMIsectPointMem;
	
	GLogManager::logging( LOG_INFO, "------------------ SGRTx2 GPU PHOTON MAPPING INFO ----------------------------" );
	
	GLogManager::logging( LOG_INFO, " -> Photon Memory Buffer : MaxPhoton ( %d ) ( %f MB )", m_iMaxPhotonSize, fPhotonMem );
	GLogManager::logging( LOG_INFO, " -> Intersection Point Buffer : %d ( %f MB )", m_iMaxIntersectionPoint, fIsectPointMem );
	GLogManager::logging( LOG_INFO, " -> PM Intersection Point Buffer : %d ( %f MB )", m_iMaxIntersectionPoint, fPMIsectPointMem );
	GLogManager::logging( LOG_INFO, " -> Total Allocated GPU Memory : ( %f MB )", fTotal );
									
	GLogManager::logging( LOG_INFO, "------------------------------------------------------------------------------" );
	
}
