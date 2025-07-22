//--------------------------------------------------------------------------//
//																			//
//	Point 클래스															//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
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

	float	innerProduct( const GPoint &point );		// 내적
	GPoint	outerProduct( const GPoint &point );		// 외적
	GPoint	normalize() const;							// 정규화
	float	length();									// 길이
	bool	isZero();

	GPoint operator+ ( const GPoint &point ) const;			
	GPoint operator+ ( const GVector &vector ) const;
	GPoint operator- ( const GPoint &point ) const;	

	friend GPoint operator * ( const GPoint& v, const float f ) { return GPoint( v.x * f, v.y * f, v.z * f ); }	// 스칼라 곱
	friend GPoint operator * ( const float f, const GPoint& v ) { return GPoint( v.x * f, v.y * f, v.z * f ); }	// 스칼라 곱
	friend GPoint operator / ( const GPoint& v, const float f ) { float invf = 1/f; return GPoint( v.x * invf, v.y * invf, v.z * invf ); }	// 스칼라 나눗셈

	void operator-= ( const GPoint &point );			// 뺄셈
	void operator+= ( const GPoint &point );			// 덧셈
	GPoint& operator *=( const float f )  { x *= f; y *= f; z *= f; return *this; }	                            // 스칼라 곱
	GPoint& operator /=( const float f )  { float invf = 1/f; x *= invf; y *= invf; z *= invf; return *this; }	// 스칼라 나눗셈
};
