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

#include "GKDTreeNode.h"
#include "GError.h"
#include "cuda_runtime.h"
#include "device_functions.h"

#define SHORT_STACK_DEPTH	7					
#define CUDA_MAX_LIGHT		10
#define CUDA_MAX_TEXTURE	150

#define OBJECT_MAX_ID		65535				//	Scene 안에 들어갈 수 있는 최대 ObjectID 번호 크기

#define USE_CULLING_OPTION						//	Back Face Culling 유무.
#define USE_SHADOW_RAY							//	Shadow 기능 지원 여부.

#define INTERSECTION_METHOD				0				//	Wald Method
//#define INTERSECTION_METHOD			1			//	Pluecker Method

#if defined(__CUDACC__)
	#define  __HOST__ 
#else
	#define __HOST__ __host__
#endif

inline float unsigned_as_float(const unsigned a) { return *(float *)&(a); }
inline unsigned float_as_unsigned(const float a) { return *(unsigned *)&(a); }

typedef struct _cu_boundingbox_
{
	float4 min_max[2];
	//__HOST__ __device__ inline void setMin(const float4 &minbbox)
	//{	min_max[0] = minbbox;	}
	//__HOST__ __device__ inline void setMax(const float4 &maxbbox)
	//{	min_max[1] = maxbbox;	}
	
} cuBoundingBox;

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
	//	info.x = int_as_float( -1 );
	//}
	//__HOST__ __device__ inline void setPrevTriIndex( int index ) {
	//	info.x = int_as_float( index );
	//}
	//__HOST__ __device__ inline int getPrevTriIndex() const 
	//{	return float_as_int( info.x );			}
	
	__HOST__ __device__ inline float2 get_dir_pos(unsigned i) const 
	{	float2 ret;
		switch(i) {
			case 0:     ret.x = pos.x; ret.y = dir.x; return ret;
			case 1:     ret.x = pos.y; ret.y = dir.y; return ret;
			default:    ret.x = pos.z; ret.y = dir.z; return ret;
		}
	}
	__HOST__ __device__ inline float3 dir_perm_x(void) const
	{	return make_float3(dir.x, dir.y, dir.z);	}
	__HOST__ __device__ inline float3 dir_perm_y(void) const
	{	return make_float3(dir.y, dir.z, dir.x);	}
	__HOST__ __device__ inline float3 dir_perm_z(void) const
	{	return make_float3(dir.z, dir.x, dir.y);	}
	__HOST__ __device__ inline float3 pos_perm_x(void) const
	{	return make_float3(pos.x, pos.y, pos.z);	}
	__HOST__ __device__ inline float3 pos_perm_y(void) const
	{	return make_float3(pos.y, pos.z, pos.x);	}
	__HOST__ __device__ inline float3 pos_perm_z(void) const
	{	return make_float3(pos.z, pos.x, pos.y);	}
	
} cuRay;

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

/**
 *	각 object 의 material. 여러개의 삼각형이
 *	하나의 object material 을 공유한다.
 *	이 object material 을 참조하기 위한 index 는
 *	cuTriangleInfo 구조체 안의 internal2.w 이다.
 *	texture 로 올려야하므로
 *	type 이 다 같아야 하며 float4 의 배수 단위여야 한다.
 *	element 순서는 고치지 말것!
 *	절대 float3,2,4 를 혼용해서 쓰지 말것. 이유는 맨위를 보시라.
 */
typedef struct __align__(16) cu_object_material {

	float3 ambient_emission;		//	ambient color ( r, g, b )
	float3 diffuse;					//	diffuse color ( r, g, b )
	float3 specular;				//	specular color. ( r, g, b )
//	float3 emission;				//	emit color.

	float reflection;				//	반사확률
	float transparency;				//	투과확률
	
	float roughness;				//	roughness.
	float refractionIndex;			//	굴절률
	float textureNumber;			//	texture 가 존재한다면 texture 번호. -1 이면 없는것.
									//	float 로 형태로 주고받으므로. int_as_float, float_as_int 로 상호변환.
	float light;					//	광원인지 여부. 0.0 이면 광원아님. 그 이외의 값이면 광원.
	float iObjectID;				//	물체의 고유번호. int_as_float, float_as_int 로 상호변환.
//	float pad;
		
} cuObjectMaterial;

/**
 *	intersection 을 계산하기 위한 triangle 정보. pluecker 방법
 */
struct __align__(16) cuPlueckerTriangleInfo  {

	float3 p0, p1, p2;				// p1 의 x,y,z 는 삼각형 꼭지점1 정보. 각 p1 p2 p3 의 w 는 normal 정보.
	float3 normal;
	float3 attrib;					// 삼각형 속성.
	float pad;						// padding.

	__HOST__ __device__ inline int getObjectIndex(void) const
	{	return float_as_int( attrib.x );	}

	__HOST__ __device__ inline int isTransparent(void) const
	{	return float_as_int( attrib.y );	}
};

/**
 *	intersection 을 계산하기 위한 triangle 정보. wald 방법
 *	internal2.w 를 object index 를 위한 값으로 사용한다.
 */
struct  __align__(16) cuWaldTriangleInfo  {

	float4 internal0, internal1, internal2;

	__HOST__ __device__ unsigned k() const { return float_as_int( internal0.x ); }
	__HOST__ __device__ float n_u() const { return internal0.y; }
	__HOST__ __device__ float n_v() const { return internal0.z; }
	__HOST__ __device__ float n_d() const { return internal0.w; }
	__HOST__ __device__ float vert_ku() const { return internal1.x; }
	__HOST__ __device__ float vert_kv() const { return internal1.y; }
	__HOST__ __device__ float b_nu() const { return internal1.z; }
	__HOST__ __device__ float b_nv() const { return internal1.w; }
	__HOST__ __device__ float c_nu() const { return internal2.x; }
	__HOST__ __device__ float c_nv() const { return internal2.y; }

	__HOST__ __device__ bool isTransparent() const { 
		return ( float_as_int( internal2.z ) == 1 || float_as_int( internal2.z ) == -1 ); 
	}

	/** wald 방법에서 n' 를 구할때 나눈값이 양수인지 여부. */
	__HOST__ __device__ bool isPositiveDir() const { return ( float_as_int( internal2.z ) > 0 ); }

	struct perm_t { float3 dir, pos; };
	__HOST__ __device__ inline perm_t get_perm(const cuRay &ray) const {
		perm_t perm;
		unsigned const axis = k();
		switch(axis) 
		{
		case 0:
			perm.dir = ray.dir_perm_x();
			perm.pos = ray.pos_perm_x();
			return perm;				
		case 1:		
			perm.dir = ray.dir_perm_y();
			perm.pos = ray.pos_perm_y();
			return perm;				
		default:				
			perm.dir = ray.dir_perm_z();
			perm.pos = ray.pos_perm_z();
			return perm;
		}
	}

	/** selective supersampling 을 위해서.. 하위3바이트는 object id, 상위 1byte 는 물체선택여부 */
	__HOST__ __device__ inline int getObjectIndex(void) const
	{	return ( float_as_int( internal2.w ) & 0x00ffffff );	}

	/** selective supersampling 을 위해서.. 하위3바이트는 object id, 상위 1byte 는 물체선택여부 */
	__HOST__ __device__ inline int isSelection(void) const
	{	return ( ( float_as_int( internal2.w ) & 0xff000000 ) > 0 );	}
};

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
	{	return triIndex != unsigned(-1); }
	
} cuIntersectionPoint;

/**
 *	Adaptive Sampling 을 위한 구조체. 
 */
typedef struct
{
	float  primaryAttr;					// primary hit object id, shadow 개수, 4bit selection 여부/하위 4bit texture 유무.
	float3 primaryNormal;
	float  secondaryAttr;				// secondary hit object id, shadow 개수, texture 유무.		
	float3 secondaryNormal;
} cuSamplingMap;

#define MAKE_PIXEL_ATTR_ASFLOAT( OID, SHADOW, SELECTED, TEXTURE )		( int_as_float( OID << 16 | SHADOW << 8 | SELECTED << 4 | TEXTURE ) )
#define GET_OID_ATTR( ATTR )						( ATTR >> 16 )
#define GET_TEXTURE_ATTR( ATTR )					( ATTR & 0x000000f )
#define GET_SELECTED_ATTR( ATTR )					( ATTR & 0x00000f0 )
#define GET_SHADOW_ATTR( ATTR )						(( ATTR & 0x0000ff00 ) >> 8)

#define MAKE_ACTIVE_SUBPIXEL( INDEX, CORNER )		( INDEX << 8 | CORNER )
#define GET_SUBPIXEL_INDEX( DATA )					( DATA >> 8 )
#define GET_SUBPIXEL_CORNER( DATA )					( DATA & 0x000000ff )

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
	
	__HOST__ __device__ inline void init(float _tHit=FLT_MAX)
	{	triIndex = unsigned(-1);	tHit = _tHit;	}
	__HOST__ __device__ inline bool isHit(void) const
	{	return triIndex != unsigned(-1); }
	
} cuIntersectionCheck;

/**
 *	Texture reference 정보.
 *	여러장의 texture 를 2d array 로 같이 올려서
 *	사용하게끔 처리하기 위해서 사용할 구조체.
 *
 *	2d array 내에서 해당 texture 의 위치에 대한 정보.
 */
typedef struct _cu_texutre_ref {
	int x, y, width, height;
} cuTextureRef;
 
/**
 *	Cuda 로 업로드할 Texture Data.
 */
typedef struct _cu_texutre_ {
	int width, height;
	unsigned char *pData;
} cuTexture;

/** 
 *	Light Type 정보.
 */
typedef enum _light_type {
	cuPointLight,
	cuRaySetLight,
	cuRectLight,
	cuVirtualLight,
} cuLightType;

/**
 *	light ray set.
 */
typedef struct _light_ray_set {
	float3 pos;
	float3 dir;
	float power;
	float padd;
} cuLightRaySet;

/**
 *	Light 모양을 sphere 로 추정할때 사용할 sphere 정보.
 */
typedef struct _light_sphere {
	float4 circle;
} cuLightSphere;

/**
 *	Light 정보. 
 *	Constant Memory 로 전송한다.
 *	절대 float3,2,4 를 혼용해서 쓰지 말것. 이유는 맨위를 보시라.
 *	여기는 float3 과 float 로만 쓰자.
 */
typedef struct _cu_light_ {

	int iObjectID;				// 고유번호. ( 광원도 하나의 물체이므로 유일한 고유번호 있음 )
	cuLightType lightType;

	/** direct illum 시 계산할 light 인지와 photon 을 emit 할 light 인지 여부. */
	bool bUseDirect;
	bool bUsePhoton;
	
	/**
	 *	이 light 로 부터 emit 시킬 photon 개수.
	 *	전체 scene 에 emit 시킬 photon 개수를 각 light 가 나누어서 뿌린다.
	 *	 scene 안의 light 들의 intensity 비율에 따라서 
	 *	각 light 에서 뿌릴 photon 개수를 결정한다.
	 */
	int iEmitPhoton;

	/**
	 *	한번 photon 을 뿌릴때, 처음 photon 부터 0,... 이렇게 index 를
	 *	붙인다고 할때, 현재 light 가 뿌릴 photon 의 시작 index.
	 *	이 light 에서는 iPhotonStartIndex 부터 iPhotonStartIndex + iEmitPhoton 에
	 *	해당하는 photon 을 뿌려야 한다.
	 */
	int iPhotonStartIndex;
	
	/**
	 *	이 light 에서 나가는 photon 한개의 power;
	 */
	float fOnePhotonPower;

	union {
		/** 
		 *	point light 일때 정보.
		 */
		struct {
			float3 pos;
			float3 color;
			float intensity;
		};
		
		/**
		 *	ray set light 일때 정보.
		 *	randomMode 가 0 이면 rayset 데이터를 차례대로generate. 
		 *	1 이면 rayset data 내에서 random 하게 선택.
		 */
		struct {
			int randomMode;	
			int iStartIndexInRaySetData;
			int iRaySetCount;
		};
	};

} cuLight;

/**
 *	Scene 정보. constant memory 로 올림.
 *	절대 float3,2,4 를 혼용해서 쓰지 말것. 이유는 맨위를 보시라.
 *	여기는 float3 과 float 로만 쓰자. bytes align 은 맞출것.
 */
typedef struct _cu_scene_info {
	
	float3 globalAmbient;						//	global ambient.
	int iResolutionX, iResolutionY;				//	resolution.
	int iSuperSamplingX, iSuperSamplingY;		//	한 픽셀당 sampling.
	int iBlockSizeX, iBlockSizeY;				//	한 block 당 쓰레드 size.
	int bEnableShadow;						
	int bEnableLocalShading;
	int iShadowRay;
	int bEnableTexture;
	int pad1, pad2, pad3;

} cuScene;

typedef struct _cu_thresholds {

	float primaryOIDRegionColorThreshold;
	float primaryNormalRegionColorThreshold;
	float primaryShadowRegionColorThreshold;
	float primaryTextureRegionColorThreshold;
	float secondaryOIDRegionColorThreshold;
	float secondaryNormalRegionColorThreshold;
	float secondaryShadowRegionColorThreshold;
	float secondaryTextureRegionColorThreshold;
	float etcRegionColorThreshold;
	float onlyColorThreshold;
	float pad1, pad2;

} cuThreshold;

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

#endif
