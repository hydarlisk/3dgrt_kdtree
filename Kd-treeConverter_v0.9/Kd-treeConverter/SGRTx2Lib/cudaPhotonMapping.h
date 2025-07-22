#ifndef CU_PHOTONMAPPING_H
#define CU_PHOTONMAPPING_H

/** 
 *	CUDA 에 사용되는 구조체를 CPU 에서 값을 채우기
 *	위해서 cudaPhotonMapping.cuh 를 공유해야 하는데,
 *	cudaPhotonMapping.cuh 를 바로쓰면 cuda 에서만 사용되는
 *	함수가 있기 때문에 cpu 에서는 여기에 해당함수를 선언해야 한다.
 *	cuRayTracer.h 에 해당 선언이 있기 때문에 먼저 로드한다음에
 *	cudaPhotonMapping.cuh 를 로드한다.
 *
 *	by graphicsian.
 */

#include "cudaRenderPipeline.h"
#include "cudaPhotonMapping.cuh"

#endif
