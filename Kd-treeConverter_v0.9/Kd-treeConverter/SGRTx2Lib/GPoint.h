//--------------------------------------------------------------------------//
//																			//
//	Point Å¬·¡½º															//
//																			//
//	Copyright (c) 2005  ÁøºÀÁØ	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#pragma once

#include "GBase.h"

class GVector;
class GPoint
{
public:
	union {
		struct { float x, y, z;	};
		struct { float data[3];	};
	};

public:
	GPoint(void)                               : x(0.0f),     y(0.0f),     z(0.0f) { };
	GPoint( float _x, float _y, float _z )     : x(_x),       y(_y),       z(_z)   { };
	GPoint( float v[4] )                       : x(v[0]),     y(v[1]),     z(v[2]) { };
	virtual ~GPoint(void)	{}

	const float *getPoint();
	void setPoint( float x, float y, float z );

	float& operator[] ( int index );

	float	innerProduct( const GPoint &point );		// ³»Àû
	GPoint	outerProduct( const GPoint &point );		// ¿ÜÀû
	GPoint	normalize() const;							// Á¤±ÔÈ­
	float	length();									// ±æÀÌ
	bool	isZero();

	GPoint operator+ ( const GPoint &point ) const;			
	GPoint operator+ ( const GVector &vector ) const;
	GPoint operator- ( const GPoint &point ) const;	

	friend GPoint operator * ( const GPoint& v, const float f ) { return GPoint( v.x * f, v.y * f, v.z * f ); }	// ½ºÄ®¶ó °ö
	friend GPoint operator * ( const float f, const GPoint& v ) { return GPoint( v.x * f, v.y * f, v.z * f ); }	// ½ºÄ®¶ó °ö
	friend GPoint operator / ( const GPoint& v, const float f ) { float invf = 1/f; return GPoint( v.x * invf, v.y * invf, v.z * invf ); }	// ½ºÄ®¶ó ³ª´°¼À

	void operator-= ( const GPoint &point );			// »¬¼À
	void operator+= ( const GPoint &point );			// µ¡¼À
	GPoint& operator *=( const float f )  { x *= f; y *= f; z *= f; return *this; }	                            // ½ºÄ®¶ó °ö
	GPoint& operator /=( const float f )  { float invf = 1/f; x *= invf; y *= invf; z *= invf; return *this; }	// ½ºÄ®¶ó ³ª´°¼À
};
