#pragma once

#include "GBase.h"
#include "GTexture.h"
#include "GCriticalSection.h"

#include <vector>

using namespace std;

/**
 *	Renderer 안에서 사용하는 Texture 를 관리하는
 *	클래스. FreeImage Library 를 사용해서 jpg 를
 *	읽어들인다.
 *
 *	by graphicsian.
 */
class  GTextureManager
{
private:
	static GTextureManager g_TextureManager;

private:
	vector<GTexture*> m_TextureList;

public:
	GTextureManager(void);
	virtual ~GTextureManager(void);

	static GTextureManager* getInstance();

	void clear();
	int addTexture( const char* imagefile );
	GTexture* getTexture( int textureID );
	void loadAllTexture();
	GTexture* getTextureByIndex( int index );
	int getTextureCount();

	GCriticalSection	m_TextureListCS;
};
