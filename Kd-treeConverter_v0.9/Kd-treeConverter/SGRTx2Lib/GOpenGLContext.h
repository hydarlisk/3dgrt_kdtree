#pragma once

#include "GBase.h"
#include "gl/gl.h"

/**
 *	Window 계열에서 OpenGL Context 를 관리하는 클래스
 *	by graphicsian
 */
class  GOpenGLContext
{
private:
	/**++++++++++++++++++++++++++++++++++++++++++++++++
	 *	OpenGL Handle 
	 *++++++++++++++++++++++++++++++++++++++++++++++++*/
	HWND		m_hWnd;
	HDC			m_hDC;
	HGLRC		m_hGLRC;

public:
	GOpenGLContext(void);
	~GOpenGLContext(void);

	BOOL initOpenGL( HWND hWnd, int width, int height, BOOL isDoubleBuffering );
	BOOL setDisplay( int x, int y, int width, int height );
	BOOL setCurrentContext();
	BOOL removeCurrentContext();
	BOOL openGlSetPixelFormat( BOOL isDoubleBuffering );
	BOOL uninitOpenGL();
	BOOL swapBuffer();
};
