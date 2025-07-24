#ifndef _CUDA_RENDER_PIPELINE_H_
#define _CUDA_RENDER_PIPELINE_H_

#include <cuda_runtime.h>
/** 
 *	CUDA 에 사용되는 구조체를 CPU 에서 값을 채우기
 *	위해서 cudaRenderPipeline.cuh 를 공유해야 하는데,
 *	cudaRenderPipeline.cuh 를 바로쓰면 cuda 에서만 사용되는
 *	함수가 있기 때문에 cpu 에서는 여기에 해당함수를
 *	선언한 다음 cudaRenderPipeline.cuh 를 로드한다.
 */

inline float fminf(const float a, const float b) { return (b > a) ? a : b; }
inline float fmaxf(const float a, const float b) { return (b < a) ? a : b; }
inline float int_as_float_H(const int a) { return *(float *)&(a); }
//__host__ __device__ inline int float_as_int(const float a) { return *(int *)&(a); }
//union {
//    float f;
//    int i;
//} u;
//u.f = x;
//return u.i;
//#include "cudaRenderPipeline.cuh"

#endif
