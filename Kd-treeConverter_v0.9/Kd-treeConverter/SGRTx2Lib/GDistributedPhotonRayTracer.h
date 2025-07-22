#pragma once

#include "GBase.h"
#include "GRenderer.h"
#include "GScene.h"
#include "GPhotonMapRayTracer.h"

/**
 *	MPICH2 를 사용한 분산렌더링.
 *	PHOTON MAPPING 과 RayTracing 을 가지고 수행한다.
 *
 *	by graphicsian.
 */

/** 
 *	렌더링시 root 가 각 node 에게 전송할
 *	render config. 절대 pointer 변수는 없어야 함.
 */
typedef struct _render_env_info_ {
	
	/** 
	 *	camera 정보. camera 안에서
	 *	value 변수들만 각 node 에서 대입될 것이다.
	 */
	GCamera camera;
	/**
	 *	scene resolution, sampling, shadow ray on/off
	 */
	int resolutionX, resolutionY;
	int superSamplingX, superSamplingY;
	bool bEnableShadow;

} GDistributedRenderConfig;

class GRenderSystem;
class GDistributedPhotonRayTracer : public GRenderer
{
private:
	GRenderSystem *m_pDistRenderManager;
	GPhotonMapRayTracer *m_pPhotonMapRayTracer;

public:
	GDistributedPhotonRayTracer( GRenderSystem *pDistManager,
								 GPhotonMappingOption *pOption );

	~GDistributedPhotonRayTracer(void);

private:

	virtual bool isOpenGLPipeline() { return false; }
	virtual bool isDistributed() { return true; }
	virtual GError rendering( GScene* pScene, bool isDebug );

protected:
	GDistributedRenderConfig makeRenderConfig( GScene *pScene );
	void setRenderConfig( GDistributedRenderConfig &config, GScene *pScene );

};
