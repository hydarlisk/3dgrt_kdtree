#include "GTexture.h"
#include "GTextureManager.h"
#include "FreeImage.h"
#include "GClassMacro.h"

#include "SSE_math.h"

GTexture::GTexture( int id )
{
	m_szFileName[0] = 0x00;
	m_TextureID = id;

	m_pTextureData = NULL;
	m_iWidth = 0;
	m_iHeight = 0;
	m_iChannels = 0;
    m_iPadding = 0;
	m_bLoad = false;
	m_bLoadFail = false;
}

GTexture::~GTexture(void)
{
	if ( m_pTextureData != NULL )
		free( m_pTextureData );
}

void GTexture::setFileName( const char *name )
{
	strncpy( m_szFileName, name, 1000 );
	m_szFileName[ 1000 ] = 0x00;
}

const char* GTexture::getFileName()
{
	return m_szFileName;
}

bool GTexture::isLoaded()
{
	return m_bLoad;
}

unsigned char* GTexture::getTextureData()
{
	return m_pTextureData;
}

int GTexture::getWidth()
{
	return m_iWidth;
}

int GTexture::getHeight()
{
	return m_iHeight;
}

void GTexture::setTextureID( int id )
{
	m_TextureID = id;
}

int GTexture::getTextureID()
{
	return m_TextureID;
}

bool GTexture::loadTexture()
{
	m_bLoad = loadImageFile();
	m_bLoadFail = !m_bLoad;

	return m_bLoad;
}

/**
 *	이미지를 로드한다. 
 *	alpha 값이 없더라도 강제로 채운다.
 */
bool GTexture::loadImageFile()
{
	FIBITMAP* temp, *fibitMap;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	int pitch = 0;
	
	/**
	 *	File Load.
	 */
	fif = FreeImage_GetFileType( m_szFileName, 0 );
	if( FIF_UNKNOWN == fif ) {
		GLogManager::logging( LOG_INFO, "image file [%s] type is unknown.", m_szFileName );
		return false;
	}

	temp = FreeImage_Load( fif, m_szFileName, 0 );
	if( temp == NULL ) {
		GLogManager::logging( LOG_INFO, "image file [%s] failed to load.", m_szFileName );
		return false;
	}

	m_iWidth = FreeImage_GetWidth( temp );
	m_iHeight = FreeImage_GetHeight( temp );

	/**
	 * 강제로 RGBA Format 으로 맞춘다.
	 */
	fibitMap = FreeImage_ConvertTo32Bits( temp );
	FreeImage_Unload( temp );

	m_iChannels = 4;

	pitch = FreeImage_GetPitch( fibitMap );
	m_iPadding = pitch - FreeImage_GetLine( fibitMap );
	BYTE* data = FreeImage_GetBits( fibitMap );
	
	m_pTextureData = (BYTE*) malloc ( FreeImage_GetPitch( fibitMap ) * m_iHeight );
	memcpy( m_pTextureData, data, FreeImage_GetPitch( fibitMap ) * m_iHeight );

	/** BGR 로 되어 있으므로 RGB 로 변경 */
	for ( int j = 0; j < m_iHeight; ++j ) {
		for ( int i = 0; i < m_iWidth; ++i ) {
			*( m_pTextureData + j * pitch + i * 4 + 0 ) = *( data + j * pitch + i * 4 + 2 );
			*( m_pTextureData + j * pitch + i * 4 + 1 ) = *( data + j * pitch + i * 4 + 1 );
			*( m_pTextureData + j * pitch + i * 4 + 2 ) = *( data + j * pitch + i * 4 + 0 );
			*( m_pTextureData + j * pitch + i * 4 + 3 ) = *( data + j * pitch + i * 4 + 3 );
		}
	}

	FreeImage_Unload( fibitMap );

	GLogManager::logging( LOG_INFO, "[ Texture : %s, %d x %d, pitch=%d, padding=%d ] OK.", 
					m_szFileName, m_iWidth, m_iHeight, pitch, m_iPadding );
	
	return true;
}

#define COLOR_SAMPLES 4
GColor GTexture::getTexel(float u, float v)
{
	int i;

	float wx = u - (int)u;
	float wy = v - (int)v;

	if ( wx < 0 ) wx = 1.0f + wx;
	if ( wy < 0 ) wy = 1.0f + wy;

	wx *= m_iWidth;
	wy *= m_iHeight;

	int xtex = (int)(wx);
	int ytex = (int)(wy);
	int xtexp = min(xtex+1, m_iWidth-1);
	int ytexp = min(ytex+1, m_iHeight-1);

	float tex0[COLOR_SAMPLES], tex1[COLOR_SAMPLES], tex2[COLOR_SAMPLES], tex3[COLOR_SAMPLES];
	float rTexR = 1.0f/255.0f;

	int width_line = m_iWidth*COLOR_SAMPLES;
	int tcoord0 = xtex	*COLOR_SAMPLES + ytex *width_line;
	int tcoord1 = xtexp	*COLOR_SAMPLES + ytex *width_line;
	int tcoord2 = xtex	*COLOR_SAMPLES + ytexp*width_line;
	int tcoord3 = xtexp	*COLOR_SAMPLES + ytexp*width_line;

	for(i=0; i<COLOR_SAMPLES; i++) {
		tex0[i] = m_pTextureData[tcoord0+i]	* rTexR;
		tex1[i] = m_pTextureData[tcoord1+i]	* rTexR;
		tex2[i] = m_pTextureData[tcoord2+i]	* rTexR;
		tex3[i] = m_pTextureData[tcoord3+i]	* rTexR;
	}

	GColor texcolor0(tex0[0], tex0[1], tex0[2]);
	GColor texcolor1(tex1[0], tex1[1], tex1[2]);
	GColor texcolor2(tex2[0], tex2[1], tex2[2]);
	GColor texcolor3(tex3[0], tex3[1], tex3[2]);

	wx -= xtex;
	wy -= ytex;

	//texcolor0 = texcolor0*(1.0f-wx) + texcolor1*wx;
	//texcolor2 = texcolor2*(1.0f-wx) + texcolor3*wx;
	//texcolor0 = texcolor0*(1.0f-wy) + texcolor2*wy;
	GColor _tcol_a, _tcol_b;
	_GCOL_vMULF(_tcol_a, texcolor0, (1.0f-wx));
	_GCOL_vMULF(_tcol_b, texcolor1, wx);
	_GCOL_vADD(texcolor0, _tcol_a, _tcol_b);
	_GCOL_vMULF(_tcol_a, texcolor2, (1.0f-wx));
	_GCOL_vMULF(_tcol_b, texcolor3, wx);
	_GCOL_vADD(texcolor2, _tcol_a, _tcol_b);
	_GCOL_vMULF(_tcol_a, texcolor0, (1.0f-wy));
	_GCOL_vMULF(_tcol_b, texcolor2, wy);
	_GCOL_vADD(texcolor0, _tcol_a, _tcol_b);

	return  texcolor0;
}

void GTexture::getTexel4(__m128 *uv, _sse_vec &texcolor)
{
	int i;

	union { __m128 wx4; float wx[4]; };
	union { __m128 wy4; float wy[4]; };

	wx4 = _mm_sub_ps(uv[0], _mm_cvtepi32_ps(_mm_cvttps_epi32(uv[0])));
	wy4 = _mm_sub_ps(uv[1], _mm_cvtepi32_ps(_mm_cvttps_epi32(uv[1])));
	
	wx4 = _mm_add_ps(wx4, sse_update(_mm_set1_ps(1), _mm_setzero_ps(), _mm_cmplt_ps(wx4, _mm_setzero_ps())));
	wy4 = _mm_add_ps(wy4, sse_update(_mm_set1_ps(1), _mm_setzero_ps(), _mm_cmplt_ps(wy4, _mm_setzero_ps())));
	wx4 = _mm_mul_ps(wx4, _mm_set1_ps((float)m_iWidth));
	wy4 = _mm_mul_ps(wy4, _mm_set1_ps((float)m_iHeight));

	__m128i xtex = _mm_cvttps_epi32(wx4);
	__m128i ytex = _mm_cvttps_epi32(wy4);
	__m128i xtexp = sse_imin_v2(_mm_add_epi32(xtex, _mm_set1_epi32(1)), _mm_set1_epi32(m_iWidth-1));
	__m128i ytexp = sse_imin_v2(_mm_add_epi32(ytex, _mm_set1_epi32(1)), _mm_set1_epi32(m_iHeight-1));

	__m128i width_line = _mm_set1_epi32(m_iWidth*COLOR_SAMPLES);

	union { __m128i tcoord0_4; int tcoord0[4]; };
	union { __m128i tcoord1_4; int tcoord1[4]; };
	union { __m128i tcoord2_4; int tcoord2[4]; };
	union { __m128i tcoord3_4; int tcoord3[4]; };

	tcoord0_4 = _mm_add_epi32(sse_imul_v2(xtex,  _mm_set1_epi32(COLOR_SAMPLES)), sse_imul_v2(ytex,  width_line));
	tcoord1_4 = _mm_add_epi32(sse_imul_v2(xtexp, _mm_set1_epi32(COLOR_SAMPLES)), sse_imul_v2(ytex,  width_line));
	tcoord2_4 = _mm_add_epi32(sse_imul_v2(xtex,  _mm_set1_epi32(COLOR_SAMPLES)), sse_imul_v2(ytexp, width_line));
	tcoord3_4 = _mm_add_epi32(sse_imul_v2(xtexp, _mm_set1_epi32(COLOR_SAMPLES)), sse_imul_v2(ytexp, width_line));

	_sse_float tex0[4], tex1[4], tex2[4], tex3[4];
	__m128 rTexR = _mm_set1_ps(0.00392157f);	// 1.0f/255.0f

	for (i = 0; i < 4; i++) {
		tex0[i].v4 = _mm_mul_ps(rTexR, _mm_setr_ps(m_pTextureData[tcoord0[i]], m_pTextureData[tcoord0[i]+1], m_pTextureData[tcoord0[i]+2], 0));
		tex1[i].v4 = _mm_mul_ps(rTexR, _mm_setr_ps(m_pTextureData[tcoord1[i]], m_pTextureData[tcoord1[i]+1], m_pTextureData[tcoord1[i]+2], 0));
		tex2[i].v4 = _mm_mul_ps(rTexR, _mm_setr_ps(m_pTextureData[tcoord2[i]], m_pTextureData[tcoord2[i]+1], m_pTextureData[tcoord2[i]+2], 0));
		tex3[i].v4 = _mm_mul_ps(rTexR, _mm_setr_ps(m_pTextureData[tcoord3[i]], m_pTextureData[tcoord3[i]+1], m_pTextureData[tcoord3[i]+2], 0));
	}

	wx4 = _mm_sub_ps(wx4, _mm_cvtepi32_ps(xtex));
	wy4 = _mm_sub_ps(wy4, _mm_cvtepi32_ps(ytex));

	for (i = 0; i < 4; i++) {
		tex0[i].v4 = _mm_add_ps(_mm_mul_ps(tex0[i].v4, _mm_set1_ps(1 - wx[i])), _mm_mul_ps(tex1[i].v4, _mm_set1_ps(wx[i])));
		tex2[i].v4 = _mm_add_ps(_mm_mul_ps(tex2[i].v4, _mm_set1_ps(1 - wx[i])), _mm_mul_ps(tex3[i].v4, _mm_set1_ps(wx[i])));
		tex0[i].v4 = _mm_add_ps(_mm_mul_ps(tex0[i].v4, _mm_set1_ps(1 - wy[i])), _mm_mul_ps(tex2[i].v4, _mm_set1_ps(wy[i])));
	}

	texcolor.x4 = _mm_setr_ps(tex0[0].x, tex0[1].x, tex0[2].x, tex0[3].x);
	texcolor.y4 = _mm_setr_ps(tex0[0].y, tex0[1].y, tex0[2].y, tex0[3].y);
	texcolor.z4 = _mm_setr_ps(tex0[0].z, tex0[1].z, tex0[2].z, tex0[3].z);
}