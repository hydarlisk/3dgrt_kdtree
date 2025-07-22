#include ".\gimagebuffer.h"
#include <math.h>
#include "FreeImage.h"

GImageBuffer::GImageBuffer()
{
	m_pBuffer = NULL;
}

GImageBuffer::GImageBuffer( int width, int height )
{
	m_pBuffer = NULL;
	m_iWidth = width;
	m_iHeight = height;
	m_iWidthCount = m_iWidth * 3;
	m_pBuffer = (float*) malloc( sizeof( float ) * m_iWidthCount * m_iHeight );
	memset( m_pBuffer, 0x00, m_iWidthCount * m_iHeight );
}

GImageBuffer::~GImageBuffer(void)
{
	if ( m_pBuffer )
		free( m_pBuffer );
}
	
void GImageBuffer::setColor( int x, int y, GColor &color )
{
	*( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 0 ) = min( 1.0f, color.r );
	*( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 1 ) = min( 1.0f, color.g );
	*( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 2 ) = min( 1.0f, color.b );
}

void GImageBuffer::addColor( int x, int y, GColor &color )
{
	float* pColor = m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x;

	color.r += pColor[0];
	color.g += pColor[1];
	color.b += pColor[2];

	pColor[0] = min( 1.0f, color.r );
	pColor[1] = min( 1.0f, color.g );
	pColor[2] = min( 1.0f, color.b );
}

void GImageBuffer::setColor( int x, int y, float r, float g, float b )
{
	*( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 0 ) = min( 1.0f, r );
	*( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 1 ) = min( 1.0f, g );
	*( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 2 ) = min( 1.0f, b );
}

void GImageBuffer::getColor( int x, int y, float &r, float &g, float &b )
{
	r = *( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 0 );
	g = *( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 1 );
	b = *( m_pBuffer + ( m_iHeight - y - 1 ) * m_iWidthCount + 3 * x + 2 );
}

void GImageBuffer::clear()
{
	memset( m_pBuffer, 0x00, sizeof( float ) * m_iWidthCount * m_iHeight );
}

float* GImageBuffer::getBuffer()
{
	return m_pBuffer;
}

void GImageBuffer::copy( float* src )
{
	memcpy( m_pBuffer, src, sizeof( float ) * m_iWidthCount * m_iHeight );
}

void GImageBuffer::add( GImageBuffer* addImage )
{
	float* buffer = addImage->getBuffer();

	for ( int i = 0; i < m_iWidthCount * m_iHeight; ++i ) {
		m_pBuffer[ i ] += buffer[ i ];
	}
}

int GImageBuffer::getWidth()
{
	return m_iWidth;
}

int GImageBuffer::getHeight()
{
	return m_iHeight;
}

/**
 * Dr.차득현군의 Halogen light 소스에서 가져와 사용했음을 밝히는 바입니다.
 * bloomming
 */
void GImageBuffer::bloomming( float bloomRadius, float bloomWeight )
{
	// Compute image-space extent of bloom effect
	int bloomSupport, bloomWidth;
	int i, j, x, y, x0, x1, y0, y1, offset;
	int by, bx, dx, dy, dist2, bloomOffset;

	// Initialize bloom filter table
	float *bloomFilter, *bloomImage, dist, sumWt, wt;
	int nPix = m_iWidth * m_iHeight;
	
	bloomSupport = (int)( bloomRadius * max( m_iWidth , m_iHeight ) );
	bloomWidth = bloomSupport / 2;
	bloomFilter = (float *)malloc( bloomWidth * bloomWidth * sizeof( float ) );
	for ( i = 0; i < bloomWidth * bloomWidth; ++i ) {
		dist = (float) sqrt( (float)( i ) ) / (float)( bloomWidth );
		bloomFilter[i] = (float) pow( max( 0.f, 1.f - dist ), 4.f );
	}
	// Apply bloom filter to image pixels
	bloomImage = (float *) malloc( sizeof( float ) * 3 * nPix );
	
	for ( i = 0; i < nPix * 3; i++ )
		bloomImage[i] = 0.0f;

	for ( y = 0; y < m_iHeight; ++y ) {
		for ( x = 0; x < m_iWidth; ++x ) {
			// Compute bloom for pixel _(x,y)_
			// Compute extent of pixels contributing bloom
			x0 = max( 0, x - bloomWidth );
			x1 = min( x + bloomWidth, m_iWidth - 1 );
			y0 = max( 0, y - bloomWidth );
			y1 = min( y + bloomWidth, m_iHeight - 1 );
			offset = y * m_iWidth + x;
			sumWt = 0.;
			for ( by = y0; by <= y1; ++by )
				for ( bx = x0; bx <= x1; ++bx ) {
					// Accumulate bloom from pixel $(bx,by)$
					dx = x - bx; dy = y - by;
					if ( dx == 0 && dy == 0 ) continue;
					dist2 = dx * dx + dy * dy;
					if ( dist2 < bloomWidth * bloomWidth ) {
						bloomOffset = bx + by * m_iWidth;
						wt = bloomFilter[ dist2 ];
						sumWt += wt;
						for ( j = 0; j < 3; ++j )
							bloomImage[ 3 * offset + j ] += wt * m_pBuffer[ 3 * bloomOffset + j ];
					}
				}
				bloomImage[ 3 * offset  ] /= sumWt;
				bloomImage[ 3 * offset + 1 ] /= sumWt;
				bloomImage[ 3 * offset + 2 ] /= sumWt;
		}
	}
	// Mix bloom effect into each pixel
	for ( i = 0; i < 3 * nPix; ++i )
		m_pBuffer[ i ] = (1.f-bloomWeight) * m_pBuffer[i] + bloomWeight * bloomImage[i];		
	// Free memory allocated for bloom effect
	free( bloomFilter );
	free( bloomImage );
}

/**
 * Dr.차득현군의 Halogen light 소스에서 가져와 사용했음을 밝히는 바입니다.
 * gammaCorrection
 */
void GImageBuffer::gammaCorrection( float gamma, float gain )
{
	int i, j, offset;
	float invgamma;

	invgamma = 1.0f / gamma;

	for ( i = 0; i < m_iHeight; ++i ) {
		for ( j = 0; j < m_iWidth; ++j ) {
			offset = i * m_iWidth + j;
			offset *= 3;
			m_pBuffer[ offset ]		= (float)pow( m_pBuffer[offset], invgamma) * gain;
			m_pBuffer[ offset + 1 ]	= (float)pow( m_pBuffer[offset+1], invgamma) * gain;
			m_pBuffer[ offset + 2 ]	= (float)pow( m_pBuffer[offset+2], invgamma) * gain;
		}
	}
}


/**
 *	이미지를 로드한다. 
 */
bool GImageBuffer::loadImage( const char* szFileName )
{
	FIBITMAP* fibitMap;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	int pitch = 0;
	
	/**
	 *	File Load.
	 */
	fif = FreeImage_GetFileType( szFileName, 0 );
	if( FIF_UNKNOWN == fif ) {
		GLogManager::logging( LOG_INFO, "image file [%s] type is unknown.", szFileName );
		return false;
	}

	fibitMap = FreeImage_Load( fif, szFileName, 0 );
	if( fibitMap == NULL ) {
		GLogManager::logging( LOG_INFO, "image file [%s] failed to load.", szFileName );
		return false;
	}

	m_iWidth = FreeImage_GetWidth( fibitMap );
	m_iHeight = FreeImage_GetHeight( fibitMap );
	m_iWidthCount = m_iWidth * 3;
	if ( m_pBuffer != NULL ) {
		free( m_pBuffer );
	}
	
	pitch = FreeImage_GetPitch( fibitMap );
	BYTE* data = FreeImage_GetBits( fibitMap );
	
	m_pBuffer = (float*) malloc ( sizeof( float ) * m_iWidthCount * m_iHeight );

	/** BGR 로 되어 있으므로 RGB 로 변경 */
	for ( int j = 0; j < m_iHeight; ++j ) {
		for ( int i = 0; i < m_iWidth; ++i ) {
			*( m_pBuffer + j * m_iWidthCount + i * 3 + 0 ) = *( data + j * pitch + i * 3 + 2 ) / 255.0f;
			*( m_pBuffer + j * m_iWidthCount + i * 3 + 1 ) = *( data + j * pitch + i * 3 + 1 ) / 255.0f;
			*( m_pBuffer + j * m_iWidthCount + i * 3 + 2 ) = *( data + j * pitch + i * 3 + 0 ) / 255.0f;
		}
	}

	FreeImage_Unload( fibitMap );

	GLogManager::logging( LOG_INFO, "[ Image : %s, %d x %d, pitch=%d ] OK.", 
					szFileName, m_iWidth, m_iHeight, pitch );
	
	return true;
}

/**
 *	이미지를 저장한다.
 */
bool GImageBuffer::saveImage( const char* filename )
{
	FIBITMAP *bitmap = FreeImage_Allocate( m_iWidth, m_iHeight, 24 );
	RGBQUAD rgb;
	float r, g, b;
	bool bSuccess = true;
	
	for( int y = 0; y < m_iHeight; ++y ) {

		for ( int x = 0; x < m_iWidth; ++x ) {
			
			getColor( x, y, r, g, b );

			rgb.rgbRed = (unsigned char)( min( 255.0, r * 255.0f ) );
			rgb.rgbGreen = (unsigned char)( min( 255.0, g * 255.0f ) );
			rgb.rgbBlue = (unsigned char)( min( 255.0, b * 255.0f ) );
			
			FreeImage_SetPixelColor( bitmap, x, y, &rgb );

		}

	}
	
	FreeImage_FlipVertical( bitmap );
	
	if ( !FreeImage_Save( FIF_BMP, bitmap, filename, 0 ) ) {
		GLogManager::logging( LOG_FATAL, "result image file %s save error", filename );
		bSuccess = false;
	}
	
	FreeImage_Unload( bitmap );

	return bSuccess;
}

GImageBuffer *GImageBuffer::makeDiffImage( GImageBuffer *dest )
{
	if ( m_iWidth != dest->getWidth() || m_iHeight != dest->getHeight() )
		return NULL;

	GImageBuffer *newImage = new GImageBuffer( m_iWidth, m_iHeight );

	for( int j = 0; j < m_iHeight; ++j ) {
		for ( int i = 0; i < m_iWidth; ++i ) {
			*( newImage->getBuffer() + m_iWidth * 3 * j + i * 3 + 0 ) 
				= fabs( *( m_pBuffer + m_iWidth * 3 * j + i * 3 + 0 ) - *( dest->getBuffer() + m_iWidth * 3 * j + i * 3 + 0 ) );
			*( newImage->getBuffer() + m_iWidth * 3 * j + i * 3 + 1 ) 
				= fabs( *( m_pBuffer + m_iWidth * 3 * j + i * 3 + 1 ) - *( dest->getBuffer() + m_iWidth * 3 * j + i * 3 + 1 ) );
			*( newImage->getBuffer() + m_iWidth * 3 * j + i * 3 + 2 ) 
				= fabs( *( m_pBuffer + m_iWidth * 3 * j + i * 3 + 2 ) - *( dest->getBuffer() + m_iWidth * 3 * j + i * 3 + 2 ) );
		}
	}

	return newImage;
}

/**
 *	PSNR 계산.
 */
double GImageBuffer::calPSNR( GImageBuffer *pDest )
{
	if ( m_iWidth != pDest->getWidth() || m_iHeight != pDest->getHeight() )
		return -1.0f;
	
	//int r0, g0, b0, r1, g1, b1;
	int srcColor, destColor;

	double mse = 0.0f;
	double psnr = 0.0f;

	for ( int j = 0; j < m_iHeight; ++j ) {
		for ( int i = 0; i < 3 * m_iWidth; ++i ) {
			srcColor = int(*( m_pBuffer + m_iWidth * 3 * j + i ) * 255.0f);
			destColor = int(*( pDest->getBuffer() + m_iWidth * 3 * j + i ) * 255.0f);
			mse += ( srcColor - destColor ) * ( srcColor - destColor );
		}
	}

	mse = mse / (double)( m_iWidth * m_iHeight * 3 );
	if ( mse == 0.0f )
		psnr = 100000.0f;
	else psnr = 20 * log10( 255.0f / sqrt( mse ) );

	//for ( int j = 0; j < m_iHeight; ++j ) {
	//	for ( int i = 0; i < m_iWidth; ++i ) {
	//		r0 = *( m_pBuffer + m_iWidth * 3 * j + i * 3 + 0 ) * 255.0f;
	//		g0 = *( m_pBuffer + m_iWidth * 3 * j + i * 3 + 1 ) * 255.0f;
	//		b0 = *( m_pBuffer + m_iWidth * 3 * j + i * 3 + 2 ) * 255.0f;

	//		r1 = *( pDest->getBuffer() + m_iWidth * 3 * j + i * 3 + 0 ) * 255.0f;
	//		g1 = *( pDest->getBuffer() + m_iWidth * 3 * j + i * 3 + 1 ) * 255.0f;
	//		b1 = *( pDest->getBuffer() + m_iWidth * 3 * j + i * 3 + 2 ) * 255.0f;

	//		mse += ( r0 - r1 ) * ( r0 - r1 ) + ( g0 - g1 ) * ( g0 - g1 ) + ( b0 - b1 ) * ( b0 - b1 );
	//	}
	//}

	//mse = mse / (double)( m_iWidth * m_iHeight );
	//if ( mse == 0.0f )
	//	psnr = 100000.0f;
	//else psnr = 20 * log10( sqrt( 3 * 255.0 * 255.0 ) / sqrt( mse ) );

	return psnr;
}