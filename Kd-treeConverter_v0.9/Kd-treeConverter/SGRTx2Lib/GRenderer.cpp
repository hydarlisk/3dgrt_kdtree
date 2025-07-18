#include "GRenderer.h"
#include "GSceneManager.h"

GRenderer::GRenderer(void)
{
	m_RenderSetting = 0;
}

GRenderer::~GRenderer(void)
{
}

void GRenderer::enableLighting( bool flag )
{
	if( flag )
	{
		m_RenderSetting |= G_LIGHTING;
	}
	else if( m_RenderSetting & G_LIGHTING )
	{
		m_RenderSetting -= G_LIGHTING;
	}
}

void GRenderer::enableShadow( bool flag )
{
	if( flag )
	{
		m_RenderSetting |= G_SHADOW;
	}
	else if( m_RenderSetting & G_SHADOW )
	{
		m_RenderSetting -= G_SHADOW;
	}
}

void GRenderer::enableMaterial( bool flag )
{
	if( flag )
	{
		m_RenderSetting |= G_MATERIAL;
	}
	else if( m_RenderSetting & G_MATERIAL )
	{
		m_RenderSetting -= G_MATERIAL;
	}
}

void GRenderer::enableVertexColor( bool flag )
{
	if( flag )
	{
		m_RenderSetting |= G_VERTEX_COLOR;
	}
	else if( m_RenderSetting & G_VERTEX_COLOR )
	{
		m_RenderSetting -= G_VERTEX_COLOR;
	}
}

void GRenderer::enableIndirectIllum( bool flag )
{
	if( flag )
	{
		m_RenderSetting |= G_INDIRECT_ILLUMINATION;
	}
	else if( m_RenderSetting & G_INDIRECT_ILLUMINATION )
	{
		m_RenderSetting -= G_INDIRECT_ILLUMINATION;
	}
}
	
void GRenderer::enableDirectIllum( bool flag )
{
	if( flag )
	{
		m_RenderSetting |= G_DIRECT_ILLUMINATION;
	}
	else if( m_RenderSetting & G_DIRECT_ILLUMINATION )
	{
		m_RenderSetting -= G_DIRECT_ILLUMINATION;
	}
}
void GRenderer::enableSceneHide( bool flag )
{
	if( flag )
	{
		m_RenderSetting |= G_SCENE_HIDE;
	}
	else if( m_RenderSetting & G_SCENE_HIDE )
	{
		m_RenderSetting -= G_SCENE_HIDE;
	}
}

bool GRenderer::isEnableLighting()
{
	return (m_RenderSetting & G_LIGHTING) > 0;
}

bool GRenderer::isEnableShadow()
{ 
	return (m_RenderSetting & G_SHADOW) > 0;
}


bool GRenderer::isEnableMaterial()
{
	return (m_RenderSetting & G_MATERIAL) > 0;
}

bool GRenderer::isEnableVertexColor()
{
	return (m_RenderSetting & G_VERTEX_COLOR) > 0;
}

bool GRenderer::isEnableIndirectIllum()
{
	return (m_RenderSetting & G_INDIRECT_ILLUMINATION) > 0;
}
	
bool GRenderer::isEnableDirectIllum()
{
	return (m_RenderSetting & G_DIRECT_ILLUMINATION) > 0;
}

bool GRenderer::isEnableSceneHide()
{
	return (m_RenderSetting & G_SCENE_HIDE) > 0;
}