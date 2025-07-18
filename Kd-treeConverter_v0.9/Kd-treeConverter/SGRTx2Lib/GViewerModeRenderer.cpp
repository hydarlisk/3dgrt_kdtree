#include "GViewerModeRenderer.h"

GViewerModeRenderer::GViewerModeRenderer()
	: m_pScene(NULL)
{
	//! Render Vertex Color Only
	GRenderer::enableVertexColor();
}

GViewerModeRenderer::~GViewerModeRenderer(void)
{
}

GError GViewerModeRenderer::rendering( GScene* pScene, bool isDebug )
{
	GError error;
	error = initialize( pScene );
	if ( error != errorNo ) {
		GLogManager::logging( LOG_ERROR, "can't initialize rendering\n" );
		return error;
	}

	return error;
}

void GViewerModeRenderer::uninitialize()
{
}

GError GViewerModeRenderer::initialize( GScene *pScene )
{
	return errorNo;
}

