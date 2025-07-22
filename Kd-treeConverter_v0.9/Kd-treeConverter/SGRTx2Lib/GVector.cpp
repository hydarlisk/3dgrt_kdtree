//--------------------------------------------------------------------------//
//																			//
//	Vector 클래스															//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#include "GVector.h"
#include <math.h>

void GVector::setVector( float sx, float sy, float sz, float sw )
{
	x = sx;
	y = sy;
	z = sz;
	w = sw;
}

BOOL GVector::operator==( const GVector &vector )
{
	if ( x == vector.x && y == vector.y && z == vector.z && w == vector.w )
		return TRUE;
	return FALSE;
}

GVector GVector::operator+ ( const GVector &vector ) const
{
	return GVector( x + vector.x, y + vector.y, z + vector.z, w + vector.w );
}

GVector GVector::operator- ( const GVector &vector ) const
{
	return GVector( x - vector.x, y - vector.y, z - vector.z, w - vector.w );
}

GVector GVector::operator- () const
{
	return GVector( -x, -y, -z, -w );
}

void GVector::operator+= ( const GVector &vector )
{
	x += vector.x;
	y += vector.y;
	z += vector.z;
	w += vector.w;
}

void GVector::operator-= ( const GVector &vector )
{
	x -= vector.x;
	y -= vector.y;
	z -= vector.z;
	w -= vector.w;
}

float& GVector::operator[] ( int index )
{
	return m_Vector[ index ];
}

float GVector::getElement( int index )
{
	return m_Vector[ index ];
}

float GVector::innerProduct( const GVector &vector )
{
	return x * vector.x + y * vector.y + z * vector.z;
}

GVector GVector::outerProduct( const GVector &vector )
{
	return GVector( y * vector.z - z * vector.y,
		z * vector.x - x * vector.z,
		x * vector.y - y * vector.x );
}

GVector GVector::normalize() const
{
	float length = sqrt( x * x + y * y + z * z );
	if ( length == 0.0f )
		return GVector( 0.0f, 0.0f, 0.0f );
	float invLength = 1 / length;
	return GVector( x * invLength, y * invLength, z * invLength );
}

float GVector::length()
{
	return sqrt( x * x + y * y + z * z );
}

bool GVector::isZero()
{
	if( fabs(x) < G_EPSILON && fabs(y) < G_EPSILON && fabs(z) < G_EPSILON )
		return true;
	return false;
}

const float* GVector::getVector()
{
	return m_Vector;
}
