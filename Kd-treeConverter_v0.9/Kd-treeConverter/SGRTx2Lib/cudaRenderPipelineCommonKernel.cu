#define M_PI 3.14159f

/** 
 *	ray check 너무 앞에서 만나면 무시하기 위한 epsilon 
 */
#define RAY_START_EPSILON		EPSILON3
#define BARYCENTRY_EPSILON		EPSILON7

#define EPSILON3 1e-3f
#define EPSILON4 1e-4f
#define EPSILON5 1e-5f
#define EPSILON7 1e-7f

#define AIR_INDEX	1.0f

#define BLOOMING_THREAD_DIM					128
#define SHADING_THREAD_DIM					128
#define PRIMARY_RAY_THREAD_DIM				128
#define INTERSECTION_THREAD_DIM				128

#define ADAPTIVE_THREADS					256
#define SUBPIXEL_CAPABILITY					64			// 주의 반드시 ADAPTIVE_THREADS / 4개를 써야한다.!!!

#define USE_CONTRAST_COMPARE							// contrast 로 비교할지, luminance 로 비교할지

#define COLOR_WEIGHT_THREADHOLD				0.01f
#define EDGE_THRESHOLD						0.3f
#define DETECTOR_NORMAL_THREADHOLD			0.4f

/**
 *	contrast 의 디폴트 threshold. Mitchell 이 언급한 0.4, 0.3, 0.6 을 기본
 *	threshold 로 한다. 이 default 비율을 기반으로 프로그램내에서는 scale factor
 *	하나로 이 전체 threshold 를 내렸다, 올렸다한다.
 *
 *	scalefactor 가 1.0 일때 3threshold 가 모두 1 이 넘게하기위해서.
 *	0.4 0.3 0.6 threshold 에 0.34 를 곱해서 1.36, 1.2, 2.04 가 되게 한다.
 *
 *	scalefactor 가 0.3 이 되면 0.4 0.3 0.6 이 된다.
 */
#define CONTRAST_RED_DEFAULT_THRESHOLD		1.36f
#define CONTRAST_GREEN_DEFAULT_THRESHOLD	1.02f
#define CONTRAST_BLUE_DEFAULT_THRESHOLD		2.04f


#define FLOAT_TO_ARRAY( X )	( (float*)(&(X)) )

#include <cuda_runtime.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include "cuda_math.h"

#include "GKDTreeNode.h"
#include "cudaRenderCommon.cuh"


/**----------------------------------------------------------------------------------------------
 *	사용할 데이터 및 구조체
 **---------------------------------------------------------------------------------------------*/
/**
 *	Texture 데이터
 */
texture<kdtreeNode, 1, cudaReadModeElementType> inKdTreeNodeTex;
texture<float4, 1, cudaReadModeElementType> inWaldTriangleTex;
texture<float4, 1, cudaReadModeElementType> inPlueckerTriangleTex;

texture<unsigned, 1, cudaReadModeElementType> inObjectOffsetListTex;
texture<float4, 1, cudaReadModeElementType> inRayListTex;
texture<float4, 1, cudaReadModeElementType> inTriangleGeometryTex;
texture<float4, 1, cudaReadModeElementType> inObjectMaterialTex;

/**
 *	Light 가 ray set 으로 주어진 경우 사용할 Texture 데이터.
 */
texture<float4, 1, cudaReadModeElementType> inLightRaySetTexture;

/**
 *	Light 를 sphere polygon 으로 추정할때 사용할 texture 데이터.
 */
texture<float4, 1, cudaReadModeElementType> inLightSphereTexture;

/**
 *	blooming filter texture
 */
texture<float, 1, cudaReadModeElementType> inBloomingFilterTexture;

/**
 *	frame buffer 를 texture 로 접근할때 사용. read 할때만.
 */
texture<float4, 1, cudaReadModeElementType> inSamplingMapTexture;
texture<float, 1, cudaReadModeElementType> inFrameBufferTexture;
texture<float, 1, cudaReadModeElementType> inFrameBuffer2Texture;
texture<int, 1, cudaReadModeElementType> inASBufferTexture;

/**
 *	Constant 로 관리하는 데이터.
 */
__constant__ cuScene g_SceneInfo;
__constant__ cuThreshold g_ThresholdInfo;

__constant__ cuCamera g_CameraInfo;
__constant__ unsigned shortStackDepth;
__constant__ cuBoundingBox g_SceneBBox;

/**
 *	setting 된 texture 정보를 위한 constant.
 */
__constant__ cuTextureRef g_TextureRefInfo[ CUDA_MAX_TEXTURE ];
__constant__ int g_TextureRefCount;

/**
 *	Light 정보. 최대 20개까지 등록가능하게.
 */
__constant__ cuLight constantLightInfo[ CUDA_MAX_LIGHT ];
__constant__ int constantLightCount;

/**
 *	adaptive sampling 에서 subpixel 이 사용할 index table
 */
__constant__ int constantIndexTablePattern[ 24 ];
__constant__ float constantPixelWeight[12];

/**
 *	Texture 최대 CUDA_MAX_TEXTURE 까지 등록가능하게.
 *	동적으로 Texture 메모리를 할당할 방법은 없는듯 하니
 *	하나의 texture 안에 여러개 texture 를 다 올려서 내부적으로
 *	처리한다.
 */
texture<uchar4, 2, cudaReadModeNormalizedFloat> inObjectTexture;

// stack의 element 형식.
typedef struct 
{
	unsigned nodeID; 
//	float tMin, tMax; 
	float tMax;
}cu_traceState;

/**
 *	shared memory 로 intersection point 에서 필요한
 *	정보를 access 하기 위한 구조체. 항상 bytes 수는 14bytes 로
 *	고정해야 한다.
 */
typedef struct
{
	float3 pos, dir, normal;		//	intersection point 정보.
	float u, v;						//	texture 좌표.
	unsigned int objectIndex;		//	object index
	unsigned int bTexture;			//	texture 유무.
	float pad;
} sharedIntersectPoint;

extern __shared__ sharedIntersectPoint smemintersectPoint[];

extern __shared__ float sharedMemory[];
extern __shared__ cu_traceState smemBuffer[];

/*
struct smemIntersect {
	unsigned baseOffset;
	__device__ inline void init(const unsigned smem_baseOffset) 
	{ baseOffset = smem_baseOffset; }
	__device__ inline sharedIntersectPoint &getIntersectPoint(void){
		return &smemintersectPoint[baseOffset];
	}
*/

// traceStack은 lmem을 쓰는 stack이다. 일반적인 스택의 역할을 수행한다.
struct traceStack {
	cu_traceState lmemStack[50];
    int _top;
    traceStack() : _top(-1) {}
	__device__ inline void init(void){_top = -1;}
	__device__ inline cu_traceState top() const { return lmemStack[_top]; }
//	__device__ inline void push(const unsigned id, const float t_min, const float t_max) { 
	__device__ inline void push(const unsigned id, const float t_max) { 
		++_top;
		lmemStack[_top].nodeID = id;		
//		lmemStack[_top].tMin = t_min;		
		lmemStack[_top].tMax = t_max;
	}
	__device__ inline void push(const cu_traceState &element) { 
		++_top;	
		lmemStack[_top] = element;		
	}
	__device__ inline int empty() const { return _top == -1; }
    __device__ inline void pop() { --_top; }
};

//푸쉬할 때 오버헤드가 있는 단점이 있달까.
struct cached_lmemStack {
	traceStack lmemStack;
	unsigned _top, cacheQuant, baseOffset;
	cached_lmemStack() : _top(shortStackDepth-1), cacheQuant(0) {}
	__device__ inline void init(const unsigned smem_baseOffset) 
	{ baseOffset = smem_baseOffset*shortStackDepth; }
	__device__ inline cu_traceState top() const { 
		if(cacheQuant == 0)
			return lmemStack.top();
		return smemBuffer[baseOffset + _top];	
	}
//	__device__ inline void push(unsigned id, float t_min, float t_max) { 
	__device__ inline void push(unsigned id, float t_max) { 
		if(++_top == shortStackDepth)
			_top = 0;
		if(cacheQuant == shortStackDepth) {
			lmemStack.push(smemBuffer[baseOffset + _top]);
		}
		cacheQuant = min(cacheQuant+1, shortStackDepth);
		smemBuffer[baseOffset + _top].nodeID = id;		
//		smemBuffer[baseOffset + _top].tMin = t_min;		
		smemBuffer[baseOffset + _top].tMax = t_max;
	}
	__device__ inline int empty() const {	return cacheQuant == 0 && lmemStack.empty(); }
	__device__ inline void pop() {//주의!! 다 쓴다음에 pop()할것
		if(cacheQuant == 0) {
			lmemStack.pop();
			return;
		}		
		if(_top == 0)
			_top = shortStackDepth;
		--_top; --cacheQuant; 	
	}
};


//심플 버전. bank conflict 고려 안함.
struct shortStack {
	unsigned _top, quant, baseOffset;
	shortStack() : _top(shortStackDepth-1), quant(0) {}
	__device__ inline void init(const unsigned smem_baseOffset) 
	{ baseOffset = smem_baseOffset*(shortStackDepth); }
	__device__ inline cu_traceState top() { return smemBuffer[baseOffset + _top];}
//	__device__ inline void push(unsigned id, float t_min, float t_max) { 
	__device__ inline void push(unsigned id, float t_max) { 
		//_top=(_top+1)%shortStackDepth; 
		if(++_top == shortStackDepth)
			_top = 0;
		quant=min(quant+1, shortStackDepth);	
		smemBuffer[baseOffset + _top].nodeID = id;		
//		smemBuffer[baseOffset + _top].tMin = t_min;		
		smemBuffer[baseOffset + _top].tMax = t_max;
	}
	__device__ inline int empty() { return quant == 0; }
	__device__ inline int full() { return quant == shortStackDepth; }
	__device__ inline void pop() {
		if(_top == 0)
			_top = shortStackDepth;
		--_top; --quant; 	
	//	_top=(_top-1)%shortStackDepth;	--quant; 
	}
};


/**----------------------------------------------------------------------------------------------
 *	사용할 데이터 및 구조체
 **---------------------------------------------------------------------------------------------*/



/**----------------------------------------------------------------------------------------------
 *	함수들
 **---------------------------------------------------------------------------------------------*/

inline cudaError_t checkError( const char* title )
{
	cudaError_t error = cudaGetLastError();
	
	if( error != cudaSuccess )	{	
		GLogManager::logging( LOG_FATAL, "[%s] %s", title, cudaGetErrorString( error ) );
	}

	return error;
}

/** 
 *	reflection 
 */
__device__ float3 reflection( float3 ray, float3 normal )
{
	/**
	 *	물체의 뒷면에 맞은경우 normal 을 뒤집는다.
	 */
	float rdotn = dot( ray, normal );
	return normalize( 2.0f * rdotn * normal - ray );
//	return normalize( 2.0f * normal * dot( ray, normal ) - ray );
}

__device__ float3 refraction( float3 dir, float3 normal, float refractionIndex )
{
	float ddotn, ddotn2, n_div_nt, n_div_nt2;
	float sqrt_part;
	float in_sqrt;
	float3 nextDir;

	dir = -1.0f * dir;

	if( refractionIndex == AIR_INDEX )
	{
		nextDir = dir;
		return nextDir;
	}

	ddotn = dot( normal, dir );

	if(ddotn == 0.)
	{
		nextDir = dir;
		return nextDir;
	}
	if(ddotn < 0.)
	//normal case (from air to obj.)
	{
		n_div_nt = AIR_INDEX / refractionIndex;
	}
	else
	//(from obj. to air)
	{
		n_div_nt = refractionIndex / AIR_INDEX;
	}

	ddotn2 = ddotn*ddotn;

	n_div_nt2 = n_div_nt*n_div_nt;
	//check in_sqrt : temperal code
	in_sqrt = 1.0f - n_div_nt2*(1.0f - ddotn2);
	in_sqrt = fabs(in_sqrt);
	sqrt_part = (float)sqrt(in_sqrt);
	
	if(n_div_nt < 1.0) {
		nextDir = n_div_nt * ( dir - normal * ddotn ) - normal * sqrt_part;
	}
	else {
		nextDir = n_div_nt * ( dir - normal * ddotn ) + normal * sqrt_part;
	}

	return normalize( nextDir );
}

/**
 *	Object Material 을 가져온다.
 */
/*
__device__ void getObjectMaterial( int objIndex, cuObjectMaterial &material )
{
	float4 temp = tex1Dfetch( inObjectMaterialTex, 5 * objIndex + 0 );
	material.ambient.x = temp.x; material.ambient.y = temp.y;
	material.ambient.z = temp.z; material.diffuse.x = temp.w;

	temp = tex1Dfetch( inObjectMaterialTex, 5 * objIndex + 1 );
	material.diffuse.y = temp.x; material.diffuse.z = temp.y; 
	material.specular.x = temp.z; material.specular.y = temp.w;
		
	temp = tex1Dfetch( inObjectMaterialTex, 5 * objIndex + 2 );
	material.specular.z = temp.x; material.emission.x = temp.y;
	material.emission.y = temp.z; material.emission.z = temp.w;

	temp = tex1Dfetch( inObjectMaterialTex, 5 * objIndex + 3 );
	material.reflection = temp.x; material.transparency = temp.y; 
	material.roughness = temp.z; material.refractionIndex = temp.w;

	temp = tex1Dfetch( inObjectMaterialTex, 5 * objIndex + 4 );
	material.textureNumber = temp.x; material.light = temp.y; 
	material.iObjectID = temp.z;
}
*/
__device__ void getObjectMaterial( int objIndex, cuObjectMaterial &material )
{
	/** 
	 *	object material texture 에서 데이터를 로드해 온다. 
	 *	float4 형태의 texture 이므로 가져와서 재조합한다.
	 */
	float4 temp = tex1Dfetch( inObjectMaterialTex, 4 * objIndex + 0 );
	material.ambient_emission.x = temp.x; material.ambient_emission.y = temp.y;
	material.ambient_emission.z = temp.z; material.diffuse.x = temp.w;

	temp = tex1Dfetch( inObjectMaterialTex, 4 * objIndex + 1 );
	material.diffuse.y = temp.x; material.diffuse.z = temp.y; 
	material.specular.x = temp.z; material.specular.y = temp.w;
		
	temp = tex1Dfetch( inObjectMaterialTex, 4 * objIndex + 2 );
	material.specular.z = temp.x; material.reflection = temp.y; 
	material.transparency = temp.z; material.roughness = temp.w;

	temp = tex1Dfetch( inObjectMaterialTex, 4 * objIndex + 3 );
	material.refractionIndex = temp.x;	material.textureNumber = temp.y; 
	material.light = temp.z; 	material.iObjectID = temp.w;
}

/**
 *	Texture Collection Data 에서 textureNumber 의 u, v 에 해당하는 
 *	데이터를 가져온다. u, v 값은( 0, 1.0 ) 로 wrapping 한다음에
 *	texutre 를 접근하기 위해서는 실제 이미지 좌표로 변환한다.
 *	( 한 texture 안에 여러 texture 가 있기 때문 )
 *	return 값은 texture 값을 가져왔는지 여부.
 */
__device__ int fetchTexture( int textureNumber, float u, float v, float3 &color )
{
	if ( g_SceneInfo.bEnableTexture == false || 
		 textureNumber >= g_TextureRefCount || textureNumber < 0 ) return 0;

	if ( u != 0.0f ) { u = u - (int)u; if ( u == 0.0f ) u = 1.0f; }
	if ( v != 0.0f ) { v = v - (int)v; if ( v == 0.0f ) v = 1.0f; }
	if ( u < 0.0f ) u += 1.0f;
	if ( v < 0.0f ) v += 1.0f;

	u = (float) g_TextureRefInfo[ textureNumber ].x + u * (float) ( g_TextureRefInfo[ textureNumber ].width - 1 );
	v = (float) g_TextureRefInfo[ textureNumber ].y + v * (float) ( g_TextureRefInfo[ textureNumber ].height - 1 );
	
	float4 temp = tex2D( inObjectTexture, u, v );
	color.x = temp.x;
	color.y = temp.y;
	color.z = temp.z;
	
	return 1;
}

/**
 *	low-discrepancy sampling.
 */
__device__ float radicalInverse( int i, int base ) {

	float val = 0;
	float invBase = 1.0f / base;
	float invBi = invBase;

	while ( i > 0 ) {
		int d_i = ( i % base );
		val += d_i * invBi;
		i /= base;
		invBi *= invBase;
	}

	return val;
	
}

__device__ inline bool 
BoundsRayIntersect( const cuBoundingBox &box, const cuRay &ray, float &tmin, float &tmax )
{
	float l1 = 0.0f; 
	float l2 = 0.0f; 

//	if ( ray.dir.x != 0.0f ) {
		l1 = __fdividef( box.min_max[0].x - ray.pos.x, ray.dir.x );
		l2 = __fdividef( box.min_max[1].x - ray.pos.x, ray.dir.x );
		tmin = fmaxf( fminf( l1,l2 ), tmin );
		tmax = fminf( fmaxf( l1,l2 ), tmax );
//	}

//	if ( ray.dir.y != 0.0f ) {
		l1 = __fdividef( box.min_max[0].y - ray.pos.y, ray.dir.y );
		l2 = __fdividef( box.min_max[1].y - ray.pos.y, ray.dir.y );
		tmin = fmaxf( fminf( l1, l2 ), tmin );
		tmax = fminf( fmaxf( l1, l2 ), tmax );
//	}

//	if ( ray.dir.z != 0.0f ) {
		l1 = __fdividef( box.min_max[0].z - ray.pos.z, ray.dir.z );
		l2 = __fdividef( box.min_max[1].z - ray.pos.z, ray.dir.z );
		tmin = fmaxf( fminf( l1, l2 ), tmin );
		tmax = fminf( fmaxf( l1, l2 ), tmax );
//	}

	return ( ( tmax >= tmin ) & ( tmax >= 0.f ) );
}

/**
 *	현재 thread id 를 이용해서 처리할 ray id 를 구한다.
 */
__device__ inline int samplingRayID( int startOffset )
{
//	return blockIdx.x * blockDim.x + threadIdx.x + startOffset;
	return umul24(blockIdx.x , blockDim.x) + threadIdx.x + startOffset;
}

__device__ int samplingRayID_BlockGrouping( int startOffset )
{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = umul24(blockIdx.x , blockDim.x) + threadIdx.x;
	int y = umul24(blockIdx.y , blockDim.y) + threadIdx.y; 
	
	return y * g_SceneInfo.iResolutionX + x;
}

/**
 *	현재 thread id 를 이용해서 처리할 image 상의 index 를 구한다.
 */
__device__ int samplingImageIndex( int startOffset )
{
//	return blockIdx.x * blockDim.x + threadIdx.x + startOffset;
	return umul24(blockIdx.x , blockDim.x) + threadIdx.x + startOffset;
}
__device__ int samplingImageIndex_BlockGrouping( int startOffset )
{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int x = umul24(blockIdx.x , blockDim.x) + threadIdx.x;
	int y = umul24(blockIdx.y , blockDim.y) + threadIdx.y; 
	
	return y * g_SceneInfo.iResolutionX + x;
}

/**
 *	Pluecker Triangle Intersection.
 */
__device__ inline void PlueckerIntersection( const cuRay &ray, const int id,
											 cuIntersectionCheck &hit,
											 const float t_near, const float t_far,
											 bool faceCCW, bool bCulling )
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

	#ifdef USE_CULLING_OPTION

		if ( !tri.isTransparent() && bCulling == 1 ) {
			t = dot( dir, tri.normal );
			if ( faceCCW == 1 && t < 0.0f ) return;
			if ( faceCCW != 1 && t > 0.0f ) return;
		}

	#endif

		t = __fdividef( dot( tri.normal, tri.p0 ), dot( dir, tri.normal ) );
		if ( ( hit.tHit <= t ) | ( t < t_near - EPSILON4 ) | ( t > t_far + EPSILON4 ) ) return;
	
		float in = 1.0f / ( temp.x + temp.y + temp.z );

		hit.tHit = t;
		hit.beta = temp.y * in;
		hit.gamma = temp.x * in;
		hit.triIndex = id;
		hit.objectIndex = tri.getObjectIndex();
		//hit.bSelected = tri.isSelection();

	}
}

/**
 *	intersection point 데이터를 생성한다.
 */
__device__ void makeIntersectionPoint( cuRay* pRay, cuIntersectionCheck *pHit, 
									   cuIntersectionPoint *pCurrIsectResult )
{
	/**
	 *	구조체 cuTriangleGeometry 를 texture 로
	 *	로딩한것에서 값을 가져온다. texture는 float4 로 만들었기
	 *	때문에 구조체의 값을 가져오기 위해서 계산을 잘해야 한다.
	 */
	float3 n0, n1, n2;
	float2 uv0, uv1, uv2;
	float4 temp;
	
	temp = tex1Dfetch( inTriangleGeometryTex, 4 * pHit->triIndex + 0 );
	n0.x = temp.x; n0.y = temp.y; n0.z = temp.z; n1.x = temp.w;
	temp = tex1Dfetch( inTriangleGeometryTex, 4 * pHit->triIndex + 1 );
	n1.y = temp.x; n1.z = temp.y; n2.x = temp.z; n2.y = temp.w;
	temp = tex1Dfetch( inTriangleGeometryTex, 4 * pHit->triIndex + 2 );
	n2.z = temp.x; uv0.x = temp.y; uv0.y = temp.z; uv1.x = temp.w;
	temp = tex1Dfetch( inTriangleGeometryTex, 4 * pHit->triIndex + 3 );
	uv1.y = temp.x; uv2.x = temp.y; uv2.y = temp.z;
	
	/** 
	 *	삼각형 정보, ray 정보 채움.
	 */
	pCurrIsectResult->triIndex = pHit->triIndex; 
	pCurrIsectResult->objectIndex = pHit->objectIndex;
	pCurrIsectResult->bSelected = pHit->bSelected;
	pCurrIsectResult->dir.x = -pRay->dir.x;
	pCurrIsectResult->dir.y = -pRay->dir.y;
	pCurrIsectResult->dir.z = -pRay->dir.z;

	temp.w = 1.0f - pHit->beta - pHit->gamma;

	/**
	 *	boundDepth, rayIndex 와 colorWeight 정보는 ray 를 생성할때
	 *	기록해 두었으므로	여기서 업데이트 하면 안된다.
	 */
	//pCurrIsectResult->boundDepth;
	//pCurrIsectResult->rayIndex;
	//pCurrIsectResult->colorWeight;
	
	/**
	 *	position과 barycentric normal 을 계산한다.
	 */
	pCurrIsectResult->pos.x = pRay->pos.x + pHit->tHit * pRay->dir.x;
	pCurrIsectResult->pos.y = pRay->pos.y + pHit->tHit * pRay->dir.y;
	pCurrIsectResult->pos.z = pRay->pos.z + pHit->tHit * pRay->dir.z;

	pCurrIsectResult->normal = normalize( 
									n0 * ( temp.w ) +
									n1 * ( pHit->beta ) + n2 * ( pHit->gamma ) );
	float2 tex = uv0 * ( temp.w ) + uv1 * ( pHit->beta ) + uv2 * ( pHit->gamma );

	/** 지금은 텍스쳐 좌표는 2차원 값만 씀. */
	pCurrIsectResult->u = tex.x;
	pCurrIsectResult->v = tex.y;
}

/**
 *	Pluecker Triangle Intersection.
 */
__device__ inline void PlueckerShadowIntersection( const cuRay &ray, const int id,
												   cuIntersectionCheck &hit,
												   const float t_near, const float t_far )
{
	cuPlueckerTriangleInfo tri;
	float4 temp;
	float3 dir;
	float3 result;
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

	result.x = dot( dir, cross( tri.p1, tri.p0 ) );
	result.y = dot( dir, cross( tri.p0, tri.p2 ) );
	result.z = dot( dir, cross( tri.p2, tri.p1 ) );
	
	if ( ( result.x >= 0.0f && result.y >= 0.0f && result.z >= 0.0f ) ||
		( result.x <= 0.0f && result.y <= 0.0f && result.z <= 0.0f ) ) {

		if ( tri.isTransparent() )
			return;

		t = __fdividef( dot( tri.normal, tri.p0 ), dot( dir, tri.normal ) );
		if ( ( hit.tHit <= t ) | ( t < t_near - EPSILON4 ) | ( t > t_far + EPSILON4 ) ) return;
	

		float in = 1.0f / ( result.x + result.y + result.z );

		hit.tHit = t;
		hit.beta = result.y * in;
		hit.gamma = result.x * in;
		hit.triIndex = id;
		hit.objectIndex = tri.getObjectIndex();
		//hit.bSelected = tri.isSelection();

	}
}


/**
 *	WALD Intersection method.
 */
__device__ inline void MultipassIntersectRoutine( const cuRay &ray, const int id, cuIntersectionCheck &hit, 
										 const float t_near, const float t_far, bool faceCCW, bool bCulling  )
{
	cuWaldTriangleInfo tri;
	tri.internal0 = tex1Dfetch( inWaldTriangleTex, 3 * id );
	tri.internal1 = tex1Dfetch( inWaldTriangleTex, 3 * id + 1 );
	tri.internal2 = tex1Dfetch( inWaldTriangleTex, 3 * id + 2 );

	cuWaldTriangleInfo::perm_t p = tri.get_perm( ray );
	const float dot = ( tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z );
	//p.pos.x = ( tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z );
	const float denum = ( p.dir.x + tri.n_u() * p.dir.y + tri.n_v() * p.dir.z );
	const float t = __fdividef( dot, denum );
	//const float t = __fdividef( p.pos.x, denum );
	
	if ( isnan( t ) ) return;
	if ( ( hit.tHit <= t ) || ( t < t_near - EPSILON4 ) || ( t > t_far + EPSILON4 ) ) return;
	
	/**
	 *	culling 옵션이 있고, object 가 transparent 하지 않다면
	 *	앞면인지 뒷면인지 체크. 뒷면에 맞은거면 hit 처리 안함.
	 */
	if ( !tri.isTransparent() && bCulling == 1 ) {
		float value = p.dir.x + p.dir.y * tri.n_u() + p.dir.z * tri.n_v();
		if ( !tri.isPositiveDir() ) value = -value;
		if ( faceCCW == 1 && value < 0.0f ) return;
		if ( faceCCW != 1 && value > 0.0f ) return;
	}

	const float hu = p.pos.y + t * p.dir.y - tri.vert_ku();
	const float hv = p.pos.z + t * p.dir.z - tri.vert_kv();
	const float beta = hv * tri.b_nu() + hu * tri.b_nv();
	const float gamma = hu * tri.c_nu() + hv * tri.c_nv();
	
	/** 삼각형의 edge 와 부딪힐때, 수치오차가 있으므로 epsilon 을 좀 준다. */
	if ( isnan( beta * gamma ) ) return;
	if ( ( beta < 0.f - BARYCENTRY_EPSILON ) | ( gamma < 0.f - BARYCENTRY_EPSILON ) | ( ( 1.0f - beta - gamma ) < 0.0f - BARYCENTRY_EPSILON ) ) return;

	hit.tHit = t;
	hit.beta = beta;
	hit.gamma = gamma;
	hit.triIndex = id;
	hit.objectIndex = tri.getObjectIndex();
	hit.bSelected = tri.isSelection();

}

/**
 *	WALD Intersection method.
 */
__device__ inline void IntersectRoutine( const cuRay &ray, const int id, cuIntersectionCheck &hit, 
										 const float t_near, const float t_far, bool faceCCW, bool bCulling  )
{
	cuWaldTriangleInfo tri;
	tri.internal0 = tex1Dfetch( inWaldTriangleTex, 3 * id );
	tri.internal1 = tex1Dfetch( inWaldTriangleTex, 3 * id + 1 );
	tri.internal2 = tex1Dfetch( inWaldTriangleTex, 3 * id + 2 );

	cuWaldTriangleInfo::perm_t p = tri.get_perm( ray );
	//const float dot = ( tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z );
	p.pos.x = ( tri.n_d() - p.pos.x - tri.n_u() * p.pos.y - tri.n_v() * p.pos.z );
	const float denum = ( p.dir.x + tri.n_u() * p.dir.y + tri.n_v() * p.dir.z );
	//const float t = __fdividef( dot, denum );
	const float t = __fdividef( p.pos.x, denum );
	
	if ( isnan( t ) ) return;
	if ( ( hit.tHit <= t ) || ( t < t_near - EPSILON4 ) || ( t > t_far + EPSILON4 ) ) return;
	
	/**
	 *	culling 옵션이 있고, object 가 transparent 하지 않다면
	 *	앞면인지 뒷면인지 체크. 뒷면에 맞은거면 hit 처리 안함.
	 */
/*
#ifdef USE_CULLING_OPTION
	if ( !tri.isTransparent() && bCulling == 1 ) {
		float value = p.dir.x + p.dir.y * tri.n_u() + p.dir.z * tri.n_v();
		if ( !tri.isPositiveDir() ) value = -value;
		if ( faceCCW == 1 && value < 0.0f ) return;
		if ( faceCCW != 1 && value > 0.0f ) return;
	}
#endif
*/
	const float hu = p.pos.y + t * p.dir.y - tri.vert_ku();
	const float hv = p.pos.z + t * p.dir.z - tri.vert_kv();
	const float beta = hv * tri.b_nu() + hu * tri.b_nv();
	const float gamma = hu * tri.c_nu() + hv * tri.c_nv();
	
	/** 삼각형의 edge 와 부딪힐때, 수치오차가 있으므로 epsilon 을 좀 준다. */
	if ( isnan( beta * gamma ) ) return;
	if ( ( beta < 0.f - BARYCENTRY_EPSILON ) | ( gamma < 0.f - BARYCENTRY_EPSILON ) | ( ( 1.0f - beta - gamma ) < 0.0f - BARYCENTRY_EPSILON ) ) return;

	hit.tHit = t;
	hit.beta = beta;
	hit.gamma = gamma;
	hit.triIndex = id;
	hit.objectIndex = tri.getObjectIndex();
	hit.bSelected = tri.isSelection();

}


/** 
 *	shadow ray 의 intersection 체크 
 *	transparent 한 물체면 통과. 물체의 hit 여부만 알면된다. 
 *
 *	TODO: 나중에 HIT 여부만 체크하는 루틴으로 고치자.
 */
__device__ inline void
IntersectShadowRoutine( const cuRay &ray, const int id, 
				 cuIntersectionCheck &hit, const float t_near, const float t_far )
{
	cuWaldTriangleInfo tri;
	tri.internal2 = tex1Dfetch( inWaldTriangleTex, 3 * id + 2 );

	/**
	 *	transparent 하다면 hit 처리 안함.
	 */
	if ( tri.isTransparent() == 1 )
		return;

	tri.internal1 = tex1Dfetch( inWaldTriangleTex, 3 * id + 1 );
	tri.internal0 = tex1Dfetch( inWaldTriangleTex, 3 * id );	

	cuWaldTriangleInfo::perm_t p = tri.get_perm(ray);
	const float dot = (tri.n_d() - p.pos.x - tri.n_u()*p.pos.y - tri.n_v()*p.pos.z);
	const float denum = (p.dir.x + tri.n_u()*p.dir.y + tri.n_v()*p.dir.z);
	const float t = __fdividef(dot, denum);
	
	if ( isnan( t ) ) return;
	if ( ( hit.tHit <= t ) || ( t < t_near - EPSILON4 ) || ( t > t_far + EPSILON4 ) ) return;
	
	
	
	const float hu = p.pos.y + t*p.dir.y - tri.vert_ku();
	const float hv = p.pos.z + t*p.dir.z - tri.vert_kv();
	const float beta = hv*tri.b_nu() + hu*tri.b_nv();
	const float gamma = hu*tri.c_nu() + hv*tri.c_nv();
	
	/** 삼각형의 edge 와 부딪힐때, 수치오차가 있으므로 epsilon 을 좀 준다. */
	if ( isnan( beta * gamma ) ) return;
	if ( ( beta < 0.f - BARYCENTRY_EPSILON ) | ( gamma < 0.f - BARYCENTRY_EPSILON ) | ( ( 1.0f - beta - gamma ) < 0.0f - BARYCENTRY_EPSILON ) ) return;

	hit.tHit = t;
	hit.beta = beta;
	hit.gamma = gamma;
	hit.triIndex = id;
	hit.objectIndex = tri.getObjectIndex();
	hit.bSelected = tri.isSelection();

}

/**
 *	intersection point 를 찾는다.
 */
__device__ void intersect( cuRay &currRay, cuIntersectionCheck &intersectionCheck, bool faceCCW, bool bCulling )
{
	float t_near = 0.0f, t_far = FLT_MAX;

	if ( BoundsRayIntersect( g_SceneBBox, currRay, t_near, t_far ) ) {

		cached_lmemStack stack;
		const unsigned smem_baseOffset = umul24(threadIdx.y , blockDim.x) + threadIdx.x;
		stack.init( smem_baseOffset );

		kdtreeNode node = tex1Dfetch( inKdTreeNodeTex, 0 );
		
		while( true ) {

			while( !IS_LEAF( node ) ) {

				//const int axis = SPLIT_AXIS( node );
				//const float splitPos = SPLIT_POS( node );
				const float2 pos_dir = currRay.get_dir_pos( SPLIT_AXIS( node ) ); 
				//const float dir = pos_dir.y;			
				const float t_split = __fdividef( SPLIT_POS( node ) - pos_dir.x, pos_dir.y );
				const unsigned sign = signbit( pos_dir.y );
				const unsigned childOffset = FIRST_CHILD_OFFSET( node );
				unsigned idx = childOffset + (sign^( t_split <= t_near ));

				//if ( t_split <= t_near ) idx = childOffset + (sign^1);
				if ( t_near < t_split && t_split < t_far ) {
//					stack.push( childOffset + ( sign ^ 1 ), t_split, t_far );
					stack.push( childOffset + ( sign ^ 1 ), t_far );
					t_far = t_split;
				}
				node = tex1Dfetch( inKdTreeNodeTex, idx );
			}	
			
			unsigned baseOffset = OBJECTLIST_OFFSET( node );
			int objectSize = OBJECT_SIZE( node ) + baseOffset;
			for ( ; baseOffset < objectSize ; baseOffset++ ) {
				const unsigned objListOffset = tex1Dfetch( inObjectOffsetListTex, baseOffset );
				/** intersection 에서 제외할 삼각형일때 */
				//if ( currRay.getPrevTriIndex() == objListOffset ) continue;

				#if INTERSECTION_METHOD == 0
					IntersectRoutine( currRay, objListOffset, intersectionCheck, t_near, t_far, faceCCW, bCulling );
				#elif INTERSECTION_METHOD == 1
					PlueckerIntersection( currRay, objListOffset, intersectionCheck, t_near, t_far, faceCCW, bCulling );
				#endif

			}
			if( stack.empty() | ( intersectionCheck.tHit <= t_far ) )
				break;			
			const cu_traceState &trace = stack.top();
			t_near = t_far;
			node = tex1Dfetch(inKdTreeNodeTex, trace.nodeID);
//			t_near = trace.tMin;			
			t_far = trace.tMax;		
			stack.pop();

		}
	}
	
}


/**
 *	intersection point 를 찾는다.
 */
__device__ void MultipassIntersect( cuRay &currRay, cuIntersectionCheck &intersectionCheck, bool faceCCW, bool bCulling )
{
	float t_near = 0.0f, t_far = FLT_MAX;

	if ( BoundsRayIntersect( g_SceneBBox, currRay, t_near, t_far ) ) {

		cached_lmemStack stack;
		const unsigned smem_baseOffset = umul24(threadIdx.y , blockDim.x) + threadIdx.x;
		stack.init( smem_baseOffset );

		kdtreeNode node = tex1Dfetch( inKdTreeNodeTex, 0 );
		
		while( true ) {

			while( !IS_LEAF( node ) ) {

				//const int axis = SPLIT_AXIS( node );
				//const float splitPos = SPLIT_POS( node );
				const float2 pos_dir = currRay.get_dir_pos( SPLIT_AXIS( node ) ); 
				//const float dir = pos_dir.y;			
				const float t_split = __fdividef( SPLIT_POS( node ) - pos_dir.x, pos_dir.y );
				const unsigned sign = signbit( pos_dir.y );
				const unsigned childOffset = FIRST_CHILD_OFFSET( node );
				unsigned idx = childOffset + (sign^( t_split <= t_near ));

				//if ( t_split <= t_near ) idx = childOffset + (sign^1);
				if ( t_near < t_split && t_split < t_far ) {
//					stack.push( childOffset + ( sign ^ 1 ), t_split, t_far );
					stack.push( childOffset + ( sign ^ 1 ), t_far );
					t_far = t_split;
				}
				node = tex1Dfetch( inKdTreeNodeTex, idx );
			}	
			
			unsigned baseOffset = OBJECTLIST_OFFSET( node );
			int objectSize = OBJECT_SIZE( node ) + baseOffset;
			for ( ; baseOffset < objectSize ; baseOffset++ ) {
				const unsigned objListOffset = tex1Dfetch( inObjectOffsetListTex, baseOffset );
				MultipassIntersectRoutine( currRay, objListOffset, intersectionCheck, t_near, t_far, faceCCW, bCulling );
			}
			if( stack.empty() | ( intersectionCheck.tHit <= t_far ) )
				break;			
			const cu_traceState &trace = stack.top();
			t_near = t_far;
			node = tex1Dfetch(inKdTreeNodeTex, trace.nodeID);
//			t_near = trace.tMin;			
			t_far = trace.tMax;		
			stack.pop();

		}
	}
	
}






/**
 *	shadow ray 의 intersection 체크.
 */
__device__ void intersectShadow( cuRay &currRay, cuIntersectionCheck &intersectionCheck, float maxt )
{
	float t_scene_near = 0.0f, t_scene_far = maxt;	
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
			for ( ; baseOffset < objectSize ; baseOffset++ ) {
				const unsigned objListOffset = tex1Dfetch( inObjectOffsetListTex, baseOffset );
				/** intersection 에서 제외할 삼각형일때 */
				//if ( currRay.getPrevTriIndex() == objListOffset ) continue;
				
				#if INTERSECTION_METHOD == 0
					IntersectShadowRoutine( currRay, objListOffset, intersectionCheck, t_near, t_far );
				#elif INTERSECTION_METHOD == 1
					PlueckerShadowIntersection( currRay, objListOffset, intersectionCheck, t_near, t_far );
				#endif

				if ( intersectionCheck.tHit <= t_far )
					return;
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
 *	light 가 보이는지 shadow ray 를 쏴서 체크.
 *	일단은 최적화 없이 기존 ray check 기능을 사용해서 체크.
 */
__device__ bool checkVisibility( float3 pos, float3 lightPos )
{
	cuRay currRay;
	
	/** light 까지의 t 를 계산 */
	float3 dir = lightPos - pos;
	float maxt = length( dir );

	//float3 dir = normalize( lightPos - pos );	
	dir.x = __fdividef(dir.x, maxt);
	dir.y = __fdividef(dir.y, maxt);
	dir.z = __fdividef(dir.z, maxt);
	
	currRay.dir = make_float3( dir.x, dir.y, dir.z );
	currRay.pos = make_float3( pos.x + dir.x * RAY_START_EPSILON, 
							   pos.y + dir.y * RAY_START_EPSILON, 
							   pos.z + dir.z * RAY_START_EPSILON );
	
	cuIntersectionCheck currIsectCheck;
	currIsectCheck.init();

	/** shadow ray */
	intersectShadow( currRay, currIsectCheck, maxt );

	/** hit 를 했다는거는 light 를 만나기전에 물체를 만난것이다. */
	return !currIsectCheck.isHit();		
}


/**
 *	light 가 보이는지 shadow ray 를 쏴서 체크.
 *	일단은 최적화 없이 기존 ray check 기능을 사용해서 체크.
 */
__device__ bool checkVisibility_ForSelective( float3 pos, float3 lightPos, bool &bSelected )
{
	cuRay currRay;
	
	/** light 까지의 t 를 계산 */
	float3 dir = lightPos - pos;
	float maxt = length( dir );

	//float3 dir = normalize( lightPos - pos );	
	dir.x = __fdividef(dir.x, maxt);
	dir.y = __fdividef(dir.y, maxt);
	dir.z = __fdividef(dir.z, maxt);
	
	currRay.dir = make_float3( dir.x, dir.y, dir.z );
	currRay.pos = make_float3( pos.x + dir.x * RAY_START_EPSILON, 
							   pos.y + dir.y * RAY_START_EPSILON, 
							   pos.z + dir.z * RAY_START_EPSILON );
	
	cuIntersectionCheck currIsectCheck;
	currIsectCheck.init();

	/** shadow ray */
	intersectShadow( currRay, currIsectCheck, maxt );
	
	bSelected = currIsectCheck.bSelected;

	/** hit 를 했다는거는 light 를 만나기전에 물체를 만난것이다. */
	return !currIsectCheck.isHit();		
}
