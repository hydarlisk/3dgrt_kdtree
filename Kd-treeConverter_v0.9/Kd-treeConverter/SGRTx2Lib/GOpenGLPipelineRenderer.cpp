#include "GOpenGLPipelineRenderer.h"

GOpenGLPipelineRenderer::GOpenGLPipelineRenderer()
	: m_pScene(NULL)
{
	//enum RENDER_SETTING { G_LIGHTING = 1, G_SHADOW = 2, G_MATERIAL = 4, G_VERTEX_COLOR = 8, G_INDIRECT_ILLUMINATION = 16, G_DIRECT_ILLUMINATION = 32, G_SCENE = 64,  };
	GRenderer::enableLighting();
	GRenderer::enableShadow();
	GRenderer::enableMaterial();
	//m_RenderSetting = G_VERTEX_COLOR;
}

GOpenGLPipelineRenderer::~GOpenGLPipelineRenderer(void)
{
}

GError GOpenGLPipelineRenderer::rendering( GScene* pScene, bool isDebug )
{
	GError error;
	error = initialize( pScene );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't initialize rendering\n" );
		return error;
	}

	return error;
}

void GOpenGLPipelineRenderer::uninitialize()
{
}

GError GOpenGLPipelineRenderer::initialize( GScene *pScene )
{
	return errorNo;
}

