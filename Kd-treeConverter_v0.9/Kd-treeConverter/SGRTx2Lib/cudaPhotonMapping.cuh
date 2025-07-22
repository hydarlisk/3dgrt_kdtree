#ifndef _CUDAPHOTONMAPPING_H_
#define _CUDAPHOTONMAPPING_H_

/**
 *	[경고] 
 *
 *	절대 float3, float2 를 한 구조체내에서 혼합해서 사용하지 마라 !! 
 *	float3 는 align 규칙이 적용되지 않으나 float2 는 8bytes align 규칙이 사용되기 때문에
 *	NVCC 컴파이러에서는 뜻하지 않는 padding 이 일어날 수 있다. cuda 내부 코드에 float2,4 선언이
 *	그렇게 되어 있음.
 *
 *	이렇게 컴파일된 것과 Visual C++ 등 시스템 컴파일러에서 컴파일된 구조체의 형태가
 *	틀리므로 뻑이 날 수 있다. !!!!!
 *
 *	따라서 반드시 float3 는 float 하고만 쓰고 float4 는 float2 하고만 쓰라.
 *	절대 float3, float2 이나 float4, float3 를 혼용해서 쓰지 마라.
 *	CUDA 내에서만 사용하는 구조체는 상관이 없으나 시스템 컴파일러로 컴파일한 소스와
 *	같이 혼용해서 구조체를 공유하는 경우는 반드시 지켜야 한다. !!!!
 *  이는 int 나 uint 등 다른 type 도 같다. 만약 어쩔수 없이 bytes 를 맞추어야 하는 경우는
 *	그냥 float 배열을 쓰라. float[2], float[3] 이거는 align 옵션이 없으므로 두 컴파일러가
 *	같은 형식으로 구조체를 만든다.
 *
 *
 *	그렇지 않았을때. 무얼상상하든 그 이상의 삽질이 계속될 것이다.
 *	
 *	by graphicsian.
 */
 
#include "GError.h"
#include "cudaRenderPipeline.cuh"
#include <cuda_runtime.h>

/**
 *	Global Photon map. cuda 와 공유.
 *	반드시 element 순서는 아래와 같이 해야 하고,
 *	함부로 element 를 추가하지 말것!!
 */
typedef struct _cu_photon_ {
	float3 pos;
	float3 normal;
	float3 power;
	float3 dir;
} cuPhoton;

/**
 *	Photon mapping 을 위한 intersection point 추가 정보.
 *	intersection point 와 어떤 photon 들과 계산을 해야하는지 정보와
 *	density area.
 *	( intersection point 가 속한 grid 주변의 photon 들 )
 */
typedef struct _cuPMIntersectionPoint_ {

	float area;						//	현재 intersection point 의 density 계산을 위한 면적.
	int photonIndexOffset;			//	현재 intersection point 주변의 cell 을 가리킬 index 정보를 가지고 있는 메모리의 offset
	int photonIndexCount;			//	주변의 탐색해야할 photon 을 가리키는 정보는 photonOffset + photonCount.

	float power[3];
	
	/** 디버깅을 위한 데이터 */
	int totalPhotonCount;			//	지금 ray 를 계산할때 탐색한 총 photon 개수.
	int usedPhotonCount;			//	지금 ray 를 계산할때 실제로 사용된 photon 개수.

} cuPMIntersectionPoint;

/**
 *	photon info 의 위치를 가리킬 index. ray 에서
 *	자기와 연결된 photon 위치를 가리킬때 사용한다.
 */
typedef struct _photon_index_ {
	int offset;
	int count;
} cuPhotonIndex;

/**
 *	cuda 로 photon mapping 을 수행하는 걸 처리하는
 *	클래스.
 *
 *	by graphicsian.
 */
class cudaPhotonMapping {

private:
	int m_iMaxPhotonSize;								// photon 의 최대개수. pDevicePhoton memory 는 이 크기만큼 잡힌다.
									
	cuPhoton *m_pDevicePhotonMem;						// device 에서 photon 을 저장할 global memory.
	int *m_pDeviceIntResult;							// device 로부터 int 결과 1개를 받을 global memory.
	
	/** gathering 관련 */
	
	cuIntersectionPoint *m_pDeviceIsectPointMem;		// intersection point 와 gathering 된 photon 정보를 위한 메모리.	
	cuPMIntersectionPoint *m_pDevicePMIsectPointMem;	// intersection point 와 gathering 된 photon 정보를 위한 메모리.	
	int m_iMaxIntersectionPoint;						// intersection point 를 위한 공간.

public:
	cudaPhotonMapping();
	~cudaPhotonMapping();
	
	GError initialize( int emitPhoton, int maxBound );
					   
	GError uninitialize();
	GError photonMapclear();
	
	GError photonTracing( int emitPhoton, int maxBound, int randomSeed,
						  bool bSaveDirectPhoton,
						  cudaRenderPipeline *pRenderPipeline, 
						  cuPhoton *pOutHostMem, int *pOutTracedPhotonSize, int *pOutTracedBound,
						  bool faceCCW );
						  
	GError uploadIntersectionPoint( cuIntersectionPoint *pIsectPointData, int iCount );
	GError calDensityArea( cuPMIntersectionPoint *pPMIsectPoint, int iCount,
						   cuPhoton *pPhotonInfo, int photonCount,
						   cuPhotonIndex *pPhotonIndex, int photonIndexCount, float radius );
	GError photonGathering( cuPMIntersectionPoint *pPMIsectPoint, int iCount,
							cuPhoton *pPhotonInfo, int photonCount,
							cuPhotonIndex *pPhotonIndex, int photonIndexCount,  float radius );

	void printStatusInfo();
		
};




#endif

