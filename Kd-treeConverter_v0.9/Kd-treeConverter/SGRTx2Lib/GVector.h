//--------------------------------------------------------------------------//
//																			//
//	Vector Å¬·¡½º															//
//																			//
//	Copyright (c) 2005  ÁøºÀÁØ	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#pragma once

#include "GBase.h"
#include "GPoint.h"

class GVector
{
public:
	union {
		struct { float x, y, z, w;  };
		struct { float m_Vector[4]; };
	};

public:
	GVector(void)                                            : x(0.0f),     y(0.0f),     z(0.0f),     w(1.0f) { };
	GVector( const GVector &vector )                         : x(vector.x), y(vector.y), z(vector.z), w(1.0f) { };
	GVector( const GPoint &point )                           : x(point.x),  y(point.y),  z(point.z),  w(1.0f) { };
	GVector( float _x, float _y, float _z, float _w = 0.0f ) : x(_x),       y(_y),       z(_z),       w(_w)   { };
	GVector( float v[4] )                                    : x(v[0]),     y(v[1]),     z(v[2]),     w(v[3]) { };
	virtual ~GVector(void)	{}

	float*			GetPointer() { return m_Vector; }
	const float*	getVector();
	void setVector( float sx, float sy, float sz, float sw = 0.0f );	// º¤ÅÍ°ª ÀÔ·Â

	float& operator[] ( int index );
	float getElement( int index );

	float	innerProduct( const GVector &vector );		// ³»Àû
	GVector	outerProduct( const GVector &vector );		// ¿ÜÀû
	GVector	normalize() const;							// Á¤±ÔÈ­
	float	length();									// ±æÀÌ
	bool	isZero();

	BOOL    operator==( const GVector &vector );		// º¤ÅÍ°¡ °°ÀºÁö. º¤ÅÍÀÇ »óµîÀ» ÀÇ¹ÌÇÏ´Â°Ô ¾Æ´Ï¶ó ´Ü¼øÈ÷ ¼ººÐÀÌ ¸ðµÎ ÀÏÄ¡ÇØ¾ßÇÔ
	void    operator= ( const GVector &vector )                { this->x = vector.x; this->y = vector.y; this->z = vector.z; this->w = vector.w; } // ´ëÀÔ
	GVector operator+ ( const GVector &vector ) const;	// µ¡¼À
	GVector operator- ( const GVector &vector ) const;	// »¬¼À
	GVector operator- () const;							// »¬¼À

	friend GVector operator * ( const GVector& v, const float f ) { return GVector( v.x * f, v.y * f, v.z * f, v.w * f  ); }	// ½ºÄ®¶ó °ö
	friend GVector operator * ( const float f, const GVector& v ) { return GVector( v.x * f, v.y * f, v.z * f, v.w * f  ); }	// ½ºÄ®¶ó °ö
	friend GVector operator / ( const GVector& v, const float f ) { float invf = 1/f; return GVector( v.x * invf, v.y * invf, v.z * invf, v.w * invf ); }	// ½ºÄ®¶ó ³ª´°¼À

	void operator-= ( const GVector &vector );			// »¬¼À
	void operator+= ( const GVector &vector );			// µ¡¼À
	GVector& operator *=( const float f )  { x *= f; y *= f; z *= f; w *= f; return *this; }	                            // ½ºÄ®¶ó °ö
	GVector& operator /=( const float f )  { float invf = 1/f; x *= invf; y *= invf; z *= invf; w *= invf; return *this; }	// ½ºÄ®¶ó ³ª´°¼À
};
