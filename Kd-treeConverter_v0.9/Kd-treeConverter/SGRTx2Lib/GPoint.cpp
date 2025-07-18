//--------------------------------------------------------------------------//
//																			//
//	Vector 클래스															//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#include "GPoint.h"
#include "GVector.h"
#include <math.h>


void GPoint::setPoint( float x, float y, float z )
{
	this->x = x;
	this->y = y;
	this->z = z;
}

void GPoint::operator-= ( const GPoint &point )
{
	x -= point.x;
	y -= point.y;
	z -= point.z;
}

void GPoint::operator+= ( const GPoint &point )
{
	x += point.x;
	y += point.y;
	z += point.z;
}

float& GPoint::operator[] ( int index )
{
	return data[ index ];
}

GPoint GPoint::operator+ ( const GPoint &point ) const
{
	return GPoint( x + point.x, y + point.y, z + point.z );
}

GPoint GPoint::operator+ ( const GVector &vector ) const
{
	return GPoint( x + vector.x, y + vector.y, z + vector.z );
}

GPoint GPoint::operator- ( const GPoint &point ) const
{
	return GPoint( x - point.x, y - point.y, z - point.z );
}
const float *GPoint::getPoint()
{
	return data;
}

float GPoint::innerProduct( const GPoint &point )
{
	return x * point.x + y * point.y + z * point.z;
}

GPoint GPoint::outerProduct( const GPoint &point )
{
	return GPoint( y * point.z - z * point.y,
		z * point.x - x * point.z,
		x * point.y - y * point.x );
}

GPoint GPoint::normalize() const
{
	float length = sqrt( x * x + y * y + z * z );
	if ( length == 0.0f )
		return GPoint( 0.0f, 0.0f, 0.0f );
	float invLength = 1 / length;
	return GPoint( x * invLength, y * invLength, z * invLength );
}

float GPoint::length()
{
	return sqrt( x * x + y * y + z * z );
}

bool GPoint::isZero()
{
	if( fabs(x) < G_EPSILON && fabs(y) < G_EPSILON && fabs(z) < G_EPSILON )
		return true;
	return false;
}