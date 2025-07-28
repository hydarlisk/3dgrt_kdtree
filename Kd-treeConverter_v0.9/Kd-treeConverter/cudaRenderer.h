#pragma once

// C++ 코드에서 참조할 구조체들
#include "Kd-treeConverter.h"
#include "OpenGLStuffs.h"


typedef uint2 kdtreeNode;

// stack의 element 형식.
typedef struct
{
	unsigned nodeID;
	//	float tMin, tMax; 
	float tMax;
}cu_traceState;

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
	//	float 로 형태로 주고받으므로. int_as_float_H, float_as_int 로 상호변환.
	float light;					//	광원인지 여부. 0.0 이면 광원아님. 그 이외의 값이면 광원.
	float iObjectID;				//	물체의 고유번호. int_as_float_H, float_as_int 로 상호변환.
	//	float pad;

} cuObjectMaterial;

/**
 * @brief CompositeObject를 사용하여 CUDA 렌더링을 시작하는 메인 함수.
 * @param object 렌더링할 CompositeObject (Kd-tree 포함).
 * @param camera 현재 카메라 정보.
 * @param width 렌더링할 프레임버퍼의 너비.
 * @param height 렌더링할 프레임버퍼의 높이.
 * @param out_framebuffer [out] 렌더링 결과가 저장될 Host 메모리의 프레임버퍼 포인터.
 * @param is_done [out] 렌더링 완료 여부를 나타내는 플래그.
 */
void launchCudaRender(
    const CompositeObject& object,
    const Camera& camera,
    int width,
    int height,
    float*& out_framebuffer,
    bool& is_done
);

#if 1
// -----------------------------------------------------------
// kdtree node
// -----------------------------------------------------------
	// macros for extracting node information
	#define IS_LEAF(node)				(((node).x & 3) == 3)
	#define IS_ROPE(node)				(((node).x & 4) == 4)

	#define SPLIT_AXIS(node)			( (node).x & 3)
	#define FIRST_CHILD_OFFSET(node)	( (node).x >> 3)
	#define SECOND_CHILD_OFFSET(node)	(((node).x >> 3) + 1)

	#define SPLIT_POS(node)				(*(float *)&((node).y))
	#define OBJECT_SIZE(node)			( (node).x >> 3)

	#define OBJECTLIST_OFFSET(node)		( (node).y)
	#define ROPE_NODE_OFFSET(node)		( (node).y)
#endif