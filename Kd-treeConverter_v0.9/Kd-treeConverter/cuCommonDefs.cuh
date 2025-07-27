#pragma once
#ifndef _CUDA_RENDERCOMMON_CUH_
#define _CUDA_RENDERCOMMON_CUH_

/**
 *	CUDA Kernel 과 CPU 가 서로 주고받을 데이터 구조 및
 *	CUDA Kernel 에서 사용할 데이터 구조.
 *
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

#include "SGRTx2Lib/cuda_math.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#if defined(__CUDACC__)
	#define  __HOST__ 
#else
	#define __HOST__ __host__
#endif

//필요한 구조체 추출하여 정의
//cuRay, cuIntersectionPoint, cuKdNode, cuKdTree, cuCamera, cuTriangle

#define SHORT_STACK_DEPTH	7
#define USE_CULLING_OPTION						//	Back Face Culling 유무.

//심플 버전. bank conflict 고려 안함.
struct shortStack {
	unsigned _top, quant, baseOffset;
	__host__ __device__ shortStack() : _top(shortStackDepth - 1), quant(0) {}
	__device__ inline void init(const unsigned smem_baseOffset)
	{
		baseOffset = smem_baseOffset * (shortStackDepth);
	}
	__device__ inline cu_traceState top() { return smemBuffer[baseOffset + _top]; }
	//	__device__ inline void push(unsigned id, float t_min, float t_max) { 
	__device__ inline void push(unsigned id, float t_max) {
		//_top=(_top+1)%shortStackDepth; 
		if (++_top == shortStackDepth)
			_top = 0;
		quant = min(quant + 1, shortStackDepth);
		smemBuffer[baseOffset + _top].nodeID = id;
		//		smemBuffer[baseOffset + _top].tMin = t_min;		
		smemBuffer[baseOffset + _top].tMax = t_max;
	}
	__device__ inline int empty() { return quant == 0; }
	__device__ inline int full() { return quant == shortStackDepth; }
	__device__ inline void pop() {
		if (_top == 0)
			_top = shortStackDepth;
		--_top; --quant;
		//	_top=(_top-1)%shortStackDepth;	--quant; 
	}
};

/**
 *	추적할 ray 정보를 표현.
 *	self intersection 을 피하기 위해서 이전에 intersect 된 삼각형의 id 를 가지게 한다.
 *	EPSILON 으로 처리할수도 있지만, 삼각형이 조밀한 경우 문제가 될 수 있으므로 id 기반으로
 *	처리하자.
 */
typedef struct __align__(16) _curay
{
	float3 dir;
	float pad;			//	texture float4 로 데이터를 넘기므로 float 하나 padding.
	float3 pos;
	float pad2;			//	texture float4 로 데이터를 넘겨야 하므로 padding 해야 함.

	//float4 info;		//	info.x 가 previous triangle index 를 가지는것. 나머지는 padding.

	//__HOST__ __device__ inline void init() {
	//	info.x = int_as_float_H( -1 );
	//}
	//__HOST__ __device__ inline void setPrevTriIndex( int index ) {
	//	info.x = int_as_float_H( index );
	//}
	//__HOST__ __device__ inline int getPrevTriIndex() const 
	//{	return float_as_int( info.x );			}

	__HOST__ __device__ inline float2 get_dir_pos(unsigned i) const
	{
		float2 ret;
		switch (i) {
		case 0:     ret.x = pos.x; ret.y = dir.x; return ret;
		case 1:     ret.x = pos.y; ret.y = dir.y; return ret;
		default:    ret.x = pos.z; ret.y = dir.z; return ret;
		}
	}
	__HOST__ __device__ inline float3 dir_perm_x(void) const
	{
		return make_float3(dir.x, dir.y, dir.z);
	}
	__HOST__ __device__ inline float3 dir_perm_y(void) const
	{
		return make_float3(dir.y, dir.z, dir.x);
	}
	__HOST__ __device__ inline float3 dir_perm_z(void) const
	{
		return make_float3(dir.z, dir.x, dir.y);
	}
	__HOST__ __device__ inline float3 pos_perm_x(void) const
	{
		return make_float3(pos.x, pos.y, pos.z);
	}
	__HOST__ __device__ inline float3 pos_perm_y(void) const
	{
		return make_float3(pos.y, pos.z, pos.x);
	}
	__HOST__ __device__ inline float3 pos_perm_z(void) const
	{
		return make_float3(pos.z, pos.x, pos.y);
	}

} cuRay;

/**
 *	intersection point 정보. cuda 안에서
 *	shading 을 하기위해서 geometry 정보 자체를
 *	담는다. 절대 float3,2,4 를 혼용해서 쓰지 말것. 이유는 맨위를 보시라.
 */
typedef struct __align__(16)
{
	unsigned int triIndex;			//	obj list 안에서 몇 번째 삼각형인지.
	unsigned int objectIndex;		//	삼각형이 포함된 object index. object material 에 접근할때 필요.
	unsigned int rayIndex;			//	이 intersection point 가 어떤 ray 의 결과인지.
	//	(left,top) 부터 순서대로. SuperSampling 까지 포함해서 계산된 index 이다.
	unsigned int boundDepth;		//	이 ray 의 현재 bounding depth.

	float3 pos, dir, normal;		//	intersection point 정보.
	float u, v;						//	texture 좌표.
	float3 colorWeight;				//	이 intersection point 의 color 가 최종 이미지에 영향을 줄값.
	float shadowCount;				//	해당지역에 그림자가 생겼는지여부. adaptive sampling 을 위해서. 생기면 -1, 아니면 1
	short bTexture;					//	texture 유무.
	short bSelected;				//	선택된 지역인지 여부.

	__HOST__ __device__ inline void init()
	{
		triIndex = unsigned(-1);
	}
	__HOST__ __device__ inline bool isHit(void) const
	{
		return triIndex != unsigned(-1);
	}

} cuIntersectionPoint;

/**
 *	cuda 내에서 intersection point 를 체크하기 위해서 사용하는
 *	구조체.
 */
typedef struct __align__(16)
{
	unsigned int triIndex;			//	obj list 안에서 몇 번째 삼각형인지.
	unsigned short objectIndex;		//	삼각형이 포함된 object index. object material 에 접근할때 필요.
	unsigned short bSelected;		//	삼각형이 선택되었는지 여부. for selective adaptive supersampling.
	float tHit, beta, gamma;		//	

	__HOST__ __device__ inline void init(float _tHit = FLT_MAX)
	{
		triIndex = unsigned(-1);	tHit = _tHit;
	}
	__HOST__ __device__ inline bool isHit(void) const
	{
		return triIndex != unsigned(-1);
	}

} cuIntersectionCheck;

/**
 *	카메라 정보. constant memory 로 올림.
 */
typedef struct _camera_info {
	float3 eye;
	float3 u, v, n;
	float fnear;
	float3 startPoint;		// 왼쪽상단 ray 의 position.
	float stepX, stepY;		// ray 하나당 이동거리.
} cuCamera;

/**
 *	shading 및 ray reflection 등을 위한
 *	삼각형의 geometry 정보. texture 로 올려야하므로
 *	type 이 다 같아야 하며 float4 의 배수 단위여야 한다.
 *	꼭지점 pos 정보는 안올려도 된다. itersection result 에서 tHit 값을
 *	가지고 계산할 수 있으므로.
 *	element 순서는 고치지 말것!
 *	절대 float3,2,4 를 혼용해서 쓰지 말것. 이유는 맨위를 보시라.
 */
typedef struct __align__(16) {
	float3 n0, n1, n2;
	float u0, v0;
	float u1, v1;
	float u2, v2;
	float pad;
} cuTriangleGeometry;

typedef uint2 kdtreeNode;

typedef struct _cu_boundingbox_
{
	float4 min_max[2];
	//__HOST__ __device__ inline void setMin(const float4 &minbbox)
	//{	min_max[0] = minbbox;	}
	//__HOST__ __device__ inline void setMax(const float4 &maxbbox)
	//{	min_max[1] = maxbbox;	}

} cuBoundingBox;

#endif