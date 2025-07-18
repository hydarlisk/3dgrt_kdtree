#include "GOpenGLContext.h"

GOpenGLContext::GOpenGLContext(void)
{
}

GOpenGLContext::~GOpenGLContext(void)
{
}


/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *	MFC용 OpenGL 초기화
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
BOOL GOpenGLContext::initOpenGL( HWND hwnd, int width, int height, BOOL isDoubleBuffering )
{
	m_hWnd = hwnd;
	m_hDC = ::GetDC( m_hWnd );

	if ( !openGlSetPixelFormat( isDoubleBuffering ) )
		return FALSE;

	m_hGLRC = wglCreateContext( m_hDC );

	if ( !setDisplay( 0, 0, width, height ) )
		return FALSE;

	return TRUE;
}

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *	화면 해상도를 결정한다.
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
BOOL GOpenGLContext::setDisplay( int x, int y, int width, int height )
{
	setCurrentContext();
	
	glViewport( x, y, width, height );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();

	removeCurrentContext();

	return TRUE;
}

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *	OpenGL Main Context 를 m_hGLRC 로 한다.
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
BOOL GOpenGLContext::setCurrentContext()
{
	wglMakeCurrent( m_hDC, m_hGLRC );
	return TRUE;
}

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *	OpenGL Main Context 해제
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
BOOL GOpenGLContext::removeCurrentContext()
{
	wglMakeCurrent( NULL, NULL );
	return TRUE;
}

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *	OpenGl 을 윈도우 DC 와 연결한다.
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
BOOL GOpenGLContext::openGlSetPixelFormat( BOOL isDoubleBuffering )
{
	PIXELFORMATDESCRIPTOR pfd;
	memset( &pfd, 0x00, sizeof( pfd ) );

	pfd.nSize = sizeof( pfd );
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
	if ( isDoubleBuffering )
		pfd.dwFlags |= PFD_DOUBLEBUFFER;

	pfd.cColorBits = 32;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cDepthBits = 8;
	pfd.iLayerType = PFD_MAIN_PLANE;

	int nPixelFormat = ChoosePixelFormat( m_hDC, &pfd );

	return SetPixelFormat( m_hDC, nPixelFormat, &pfd );
}

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *	OpenGl 해제
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
BOOL GOpenGLContext::uninitOpenGL()
{
	wglMakeCurrent( NULL, NULL );
	
	if ( m_hDC != 0 ) {
		::ReleaseDC( m_hWnd, m_hDC );
	}
	if ( m_hGLRC != 0 )	{
		wglDeleteContext( m_hGLRC );
		m_hGLRC = 0;
	}
	return TRUE;
}

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *	Swap
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
BOOL GOpenGLContext::swapBuffer()
{
	return SwapBuffers( m_hDC );
}