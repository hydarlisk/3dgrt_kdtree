#include ".\gtexturemanager.h"

GTextureManager GTextureManager::g_TextureManager;

GTextureManager::GTextureManager(void)
{
}

GTextureManager::~GTextureManager(void)
{
	clear();
}

void GTextureManager::clear()
{
	for ( int i = 0; i < (int) m_TextureList.size(); ++i ) {
		delete m_TextureList[i];
	}
	m_TextureList.clear();
}

GTextureManager* GTextureManager::getInstance()
{
	return &g_TextureManager;
}

/**
 *	Texture 를 추가하고, texture 고유번호를 리턴해준다.
 *	만약 동일한 texture 라면 기존에 있는 번호를 리턴.
 */
int GTextureManager::addTexture( const char* filename )
{
	m_TextureListCS.lock();
	int nTextureListSize = (int) m_TextureList.size();

	int textureID = nTextureListSize;
	for ( int i = 0; i < (int) m_TextureList.size(); ++i ) {
		if ( _stricmp( m_TextureList[ i ]->getFileName(), filename ) == 0 ) {
			textureID = i;
			break;
		}
	}

	if (textureID == nTextureListSize) {
		GTexture *pTexture = new GTexture( textureID );
		pTexture->setFileName( filename );

		m_TextureList.push_back( pTexture );
	}
	m_TextureListCS.unlock();

	return textureID;
}

/**
 *	textureID 에 해당하는 texture 객체 리턴.
 */	
GTexture* GTextureManager::getTexture( int textureID )
{
	if ( textureID < 0 || textureID >= (int) m_TextureList.size() )
		return NULL;

	return m_TextureList[ textureID ];
}

int GTextureManager::getTextureCount()
{
	return (int) m_TextureList.size();
}

GTexture* GTextureManager::getTextureByIndex( int index )
{
	return m_TextureList[ index ];
}

void GTextureManager::loadAllTexture()
{
	GTexture *pTexture = NULL;

	for ( int i = 0; i < (int) m_TextureList.size(); ++i ) {
		pTexture = m_TextureList[ i ];
		pTexture->loadTexture();
	}
}
