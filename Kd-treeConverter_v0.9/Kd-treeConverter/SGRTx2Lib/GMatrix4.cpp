//--------------------------------------------------------------------------//
//																			//
//	4x4 행렬																//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#include "GMatrix4.h"

GMatrix4::GMatrix4(void)
{
	for ( int i = 0; i < 16; ++i )
		matrix[i] = 0.0;
}

GMatrix4::~GMatrix4(void)
{
}

GMatrix4::GMatrix4( const GMatrix4 &mat )
{
	for ( int i = 0; i < 16; ++i )
	{
		matrix[i] = mat.matrix[i];
	}
}

BOOL GMatrix4::operator== ( const GMatrix4 &mat )
{
	for ( int i = 0; i < 16; ++i )
	{
		if ( matrix[i] != mat.matrix[i] )
			return FALSE;
	}

	return TRUE;
}

void GMatrix4::operator= ( const GMatrix4 &mat )
{
	for ( int i = 0; i < 16; ++i )
	{
		matrix[i] = mat.matrix[i];
	}
}

GMatrix4 GMatrix4::operator+ ( const GMatrix4 &mat )
{
	GMatrix4 newmat;

	for ( int i = 0; i < 16; ++i )
	{
		newmat.matrix[i] = matrix[i] + mat.matrix[i];
	}

	return newmat;
}

GMatrix4 GMatrix4::operator- ( const GMatrix4 &mat )
{
	GMatrix4 newmat;

	for ( int i = 0; i < 16; ++i )
	{
		newmat.matrix[i] = matrix[i] - mat.matrix[i];
	}

	return newmat;
}

/** 
 *	벡터를 4개로 해서 계산했다가 다시 3개만 리턴.
 */
GVector GMatrix4::operator* ( const GVector &vec )
{
	float input[4] = { vec.x, vec.y, vec.z, 1.0f };
	float result[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	
	for( int i = 0; i < 4; ++i ) {
		for( int j = 0; j < 4; ++j ) {
			result[i] += MATRIX( matrix, i, j ) *
						 input[j];
		}
	}

	return GVector( result[0], result[1], result[2] );
}

/** 
 *	벡터를 4개로 해서 계산했다가 다시 3개만 리턴.
 */
GPoint GMatrix4::operator* ( const GPoint &point )
{
	float input[4] = { point.x, point.y, point.z, 1.0f };
	float result[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	
	for( int i = 0; i < 4; ++i ) {
		for( int j = 0; j < 4; ++j ) {
			result[i] += MATRIX( matrix, i, j ) *
						 input[j];
		}
	}

	return GPoint( result[0], result[1], result[2] );
}

GMatrix4 GMatrix4::operator* ( const GMatrix4 &mat )
{
	GMatrix4 newmat;

	for( int i = 0; i < 4; ++i )
	{
		for( int j = 0; j < 4; ++j )
		{
			MATRIX( newmat.matrix, i, j ) = 0;
			for ( int k = 0; k < 4; ++k )
			{
				MATRIX( newmat.matrix, i, j ) += MATRIX( matrix, i, k ) *
												 MATRIX( mat.matrix, k, j );
			}
		}
	}

	return newmat;
}

void GMatrix4::SetMatrix( GVector v0, GVector v1, GVector v2 )
{
	MATRIX( matrix, 0, 0 ) = v0.x;
	MATRIX( matrix, 0, 1 ) = v1.x;
	MATRIX( matrix, 0, 2 ) = v2.x;
	MATRIX( matrix, 0, 3 ) = 0.0f;
	MATRIX( matrix, 1, 0 ) = v0.y;
	MATRIX( matrix, 1, 1 ) = v1.y;
	MATRIX( matrix, 1, 2 ) = v2.y;
	MATRIX( matrix, 1, 3 ) = 0.0f;
	MATRIX( matrix, 2, 0 ) = v0.z;
	MATRIX( matrix, 2, 1 ) = v1.z;
	MATRIX( matrix, 2, 2 ) = v2.z;
	MATRIX( matrix, 2, 3 ) = 0.0f;
	MATRIX( matrix, 3, 0 ) = 0.0f;
	MATRIX( matrix, 3, 1 ) = 0.0f;
	MATRIX( matrix, 3, 2 ) = 0.0f;
	MATRIX( matrix, 3, 3 ) = 1.0f;
}

void GMatrix4::SetMatrix( float x0, float x1, float x2, float x3,
							float x4, float x5, float x6, float x7,
							float x8, float x9, float x10, float x11,
							float x12, float x13, float x14, float x15 )
{
	MATRIX( matrix, 0, 0 ) = x0;
	MATRIX( matrix, 0, 1 ) = x1;
	MATRIX( matrix, 0, 2 ) = x2;
	MATRIX( matrix, 0, 3 ) = x3;
	MATRIX( matrix, 1, 0 ) = x4;
	MATRIX( matrix, 1, 1 ) = x5;
	MATRIX( matrix, 1, 2 ) = x6;
	MATRIX( matrix, 1, 3 ) = x7;
	MATRIX( matrix, 2, 0 ) = x8;
	MATRIX( matrix, 2, 1 ) = x9;
	MATRIX( matrix, 2, 2 ) = x10;
	MATRIX( matrix, 2, 3 ) = x11;
	MATRIX( matrix, 3, 0 ) = x12;
	MATRIX( matrix, 3, 1 ) = x13;
	MATRIX( matrix, 3, 2 ) = x14;
	MATRIX( matrix, 3, 3 ) = x15;
}

float GMatrix4::GetElement( int i, int j )
{
	return MATRIX( matrix, i, j );
}

void GMatrix4::SetElement( int i, int j, float value )
{
	MATRIX( matrix, i, j ) = value;
}

void GMatrix4::identity()
{
	SetMatrix( 1.0f, 0.0f, 0.0f, 0.0f,
			   0.0f, 1.0f, 0.0f, 0.0f,
			   0.0f, 0.0f, 1.0f, 0.0f,
			   0.0f, 0.0f, 0.0f, 1.0f );
}

GMatrix4 GMatrix4::createTranslateMatrix( float x, float y, float z )
{
	GMatrix4 matrix;
	matrix.SetMatrix( 1.0f, 0.0f, 0.0f, x,
					  0.0f, 1.0f, 0.0f, y,
					  0.0f, 0.0f, 1.0f, z,
					  0.0f, 0.0f, 0.0f, 1.0f );
	return matrix;
}

GMatrix4 GMatrix4::transpose()
{
	GMatrix4 mat;

	for ( int i = 0; i < 4; ++i ) {
		for ( int j = 0; j < 4; ++j ) {
			mat.SetElement( j, i, GetElement( i, j ) );
		}
	}

	return mat;
}
