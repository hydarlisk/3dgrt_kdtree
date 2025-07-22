#pragma once

#include "GBase.h"
#include "GColor.h"

/**
 *	Rendering 결과를 저장할 Image Buffer
 *	RGB 를 각각 float 형으로 저장한다.
 *
 *	by graphicsian.
 */
class GImageBuffer
{
private:
	float *m_pBuffer;
	int m_iWidth, m_iHeight;
	int m_iWidthCount;

public:
	GImageBuffer();
	GImageBuffer( int width, int height );
	~GImageBuffer(void);

	void setColor( int x, int y, GColor &color );
	void setColor( int x, int y, float r, float g, float b );
	void getColor( int x, int y, float &r, float &g, float &b );
	void addColor( int x, int y, GColor &color );
	void add( GImageBuffer* addImage );
	void clear();

	float* getBuffer();
	void copy( float* src );

	int getWidth();
	int getHeight();

	void bloomming( float bloomRadius, float bloomWeight );
	void gammaCorrection( float gamma, float gain );

	bool loadImage( const char* szFileName );
	bool saveImage( const char* filename );
	double calPSNR( GImageBuffer *pDest );

	GImageBuffer *makeDiffImage( GImageBuffer *dest );

};
