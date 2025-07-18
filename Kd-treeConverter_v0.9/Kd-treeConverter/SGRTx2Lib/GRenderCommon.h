#pragma once

#include "GBase.h"
#include "GScene.h"
#include "cudaRenderPipeline.h"

/**
 *	공통으로 사용할 수 있는 함수들
 *
 *	by graphicsian.
 */
class GRenderCommon
{
public:
	GRenderCommon(void);
	~GRenderCommon(void);

	static cuLight* makeCudaLightInfo( GScene *pScene, int *pCount, 
									   cudaRenderPipeline *pCudaRenderPipeline );
};
