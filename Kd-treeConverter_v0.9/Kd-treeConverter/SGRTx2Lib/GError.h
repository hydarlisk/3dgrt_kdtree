#pragma once

/**
 *	에러코드.
 */
typedef enum {

	errorNo = 0,					//	에러없음.
	errorUnknown,					//	Unknown 에러.

	errorUnsupportedRenderer,
	errorNoRenderer,

	/**
	 *	Object 관련 에러.
	 */
	errorInvalidObject,				//	Invalid Object.

	/**
	 *	File 관련 에러.
	 */
	errorFileNotFound,				//	File Not Found 에러.
	errorFileDataError,				//	File Data Error.
	errorFileInvalidSceneInfo,		//	Scene Info Error
	errorFileInvalidCamera,			//	Camera Info Error
	errorFileInvalidLight,			//	Light Info Error
	errorFileInvalidVersion,		//	Version Error
	errorFileInvalidObject,			//	Object Data Error
	errorFileInvalidGlobal,			//	Global Data Error

	/**
	 *	Rendering 관련 에러
	 */
	errorNoScene,					//	Scene 이 없다.
	errorKDTree,					//	kdtree 생성실패.
	errorRendering,					//	rendering 에러.
	errorOverflowRayCount,			//	ray 개수가 너무 많다.
	errorNoLight,					//	light 가 없다.

	errorOverflowMaxDepth,			//	최대 depth 를 넘어섰다.
	
	/**
	 *	Cuda ray tracing error
	 */
	errorCudaGeneratePrimaryRay,

	/**
	 *	Cuda Error
	 */
	errorCudaError,					//	Cuda 관련 에러.
	
	/**
	 *	Cuda Photon Mapping Error
	 */
	errorCudaLightError,			//	light upload 에러.
	errorCudaPhotonAllocError,		//	photon 저장소 할당 에러.
	errorphotonTracingError,	//	tracing 에러.
	errorCudaFreeError,				//	자원해제에러.
	errorNotEnoughRayMem,			//	photon tracing 을 위한 ray 공간이 부족.
	errorResultError,
	errorPhotonBoundingError,

	errorMPIError,

	/**
	 *	Cuda Error
	 */
	errorThreadError,				//	Thread 관련 에러.

} GError;
