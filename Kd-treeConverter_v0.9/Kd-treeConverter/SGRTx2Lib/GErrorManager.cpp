#include ".\gerrormanager.h"

GErrorManager::GErrorManager(void)
{
}

GErrorManager::~GErrorManager(void)
{
}

const char* GErrorManager::getGErrorString( GError error )
{
	switch( error ) {
		case errorNo:
			return "No Error";
		case errorUnknown:
			return "Unknown Error";

		case errorInvalidObject:
			return "errorInvalidObject";
		case errorUnsupportedRenderer:
			return "errorUnsupportedRenderer";


		case errorFileNotFound:
			return "File Not Found";
		case errorFileDataError:
			return "errorFileDataError";
		case errorFileInvalidSceneInfo:
			return "errorFileInvalidSceneInfo";
		case errorFileInvalidCamera:
			return "errorFileInvalidCamera";
		case errorFileInvalidLight:
			return "errorFileInvalidLight";
		case errorFileInvalidVersion:
			return "errorFileInvalidVersion";
		case errorFileInvalidObject:
			return "errorFileInvalidObject";
		case errorFileInvalidGlobal:
			return "errorFileInvalidGlobal";
		case errorKDTree:
			return "errorKDTree";
		case errorNoScene:
			return "errorNoScene";
		case errorOverflowRayCount:
			return "errorOverflowRayCount";
		case errorOverflowMaxDepth:
			return "errorOverflowMaxDepth";
		case errorNoLight:
			return "errorNoLight";
		case errorRendering:
			return "errorRendering";

		case errorCudaError:
			return "errorCudaError";
		case errorCudaLightError:
			return "errorCudaLightError";
		case errorCudaPhotonAllocError:
			return "errorCudaPhotonAllocError";
		case errorCudaFreeError:
			return "errorCudaFreeError";
		case errorphotonTracingError:
			return "errorphotonTracingError";
		case errorNotEnoughRayMem:
			return "errorNotEnoughRayMem";

		case errorCudaGeneratePrimaryRay:
			return "errorCudaGeneratePrimaryRay";
		case errorResultError:
			return "errorResultError";
		case errorPhotonBoundingError:
			return "errorPhotonBoundingError";

		case errorMPIError:
			return "errorMPIError";
			
	}

	return "Unknown Error";
}