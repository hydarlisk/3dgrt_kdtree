#pragma once

#include "GBase.h"
#include "GColor.h"

/**
 *	Texture 하나의 정보.
 *
 *	by graphicsian
 */
class GTextureManager;
class GTexture
{
private:
	int m_TextureID;
	char m_szFileName[1024];
	unsigned char *m_pTextureData;
	int m_iWidth, m_iHeight;
	int m_iChannels;
	int m_iPadding;

	bool m_bLoad;
	bool m_bHasTexture;
	bool m_bLoadFail;

public:
	GTexture( int id );
	virtual ~GTexture(void);

	void setTextureID( int id );
	int getTextureID();
	void setFileName( const char *name );
	const char* getFileName();

	bool isLoaded();
	unsigned char* getTextureData();
	int getWidth();
	int getHeight();
	bool loadTexture();
	bool loadImageFile();
	GColor getTexel(float u, float v);
	void getTexel4(__m128 *uv, _sse_vec &texcolor);
};
