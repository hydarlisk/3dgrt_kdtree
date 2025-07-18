#pragma once

#include "GBase.h"
#include "GScene.h"
#include "GImageBuffer.h"

/**
 * 렌더링에 관한 세팅을 각각에 대한 변수로 갖지 않고, flag 로 관리함.
 * 각 세팅은 함수로만 접근할 수 있음.
*/
enum RENDER_SETTING { G_LIGHTING = 1, G_SHADOW = 2, G_MATERIAL = 4, G_VERTEX_COLOR = 8, G_INDIRECT_ILLUMINATION = 16, G_DIRECT_ILLUMINATION = 32, G_SCENE_HIDE = 64,  };

/**
 *	렌더러 기본 클래스.
 *	각 렌더러는 반드시 이 클래스를 상속받아야 한다.
 *	rendering option 은 보편적인 option 들로 구성되었으므로
 *	지원하지 않으면 구현안해도 된다.
 *
 *	by graphicsian
 */
class GRenderer
{
private:
	/**
	 * RENDER_SETTING 의 flag 를 같는 변수.
	 * m_bEnableShadow 등의 변수들이 사라지고 flag 형식으로 관리함.
	*/
	int m_RenderSetting;

public:
	GRenderer();
	virtual ~GRenderer(void);

	virtual bool isOpenGLPipeline() = 0;
	virtual bool isDistributed() = 0;
	virtual GError rendering( GScene* pScene, bool isDebug ) = 0;
	virtual GError preRendering( GScene* pScene, bool isDebug ) { return errorNo; };
	virtual GError postRendering( GScene* pScene, bool isDebug ) { return errorNo; };

	// 이 함수들은 사용 안되는 것이 좋음.
	//int getRenderSetting() { return m_RenderSetting; }
	//void setRenderSetting( int setting ) { m_RenderSetting = setting; }

	void enableLighting( bool flag = true );
	void enableShadow( bool flag = true );
	void enableMaterial( bool flag = true );
	void enableVertexColor( bool flag = true );
	void enableIndirectIllum( bool flag = true );
	void enableDirectIllum( bool flag = true );
	void enableSceneHide( bool flag = true );

	bool isEnableLighting();
	bool isEnableShadow();
	bool isEnableMaterial();
	bool isEnableVertexColor();
	bool isEnableIndirectIllum();
	bool isEnableDirectIllum();
	bool isEnableSceneHide();
};






