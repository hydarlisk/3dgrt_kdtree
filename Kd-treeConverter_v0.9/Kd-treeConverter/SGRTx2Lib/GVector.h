//--------------------------------------------------------------------------//
//																			//
//	Vector 클래스															//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
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
	void setVector( float sx, float sy, float sz, float sw = 0.0f );	// 벡터값 입력

	float& operator[] ( int index );
	float getElement( int index );

	float	innerProduct( const GVector &vector );		// 내적
	GVector	outerProduct( const GVector &vector );		// 외적
	GVector	normalize() const;							// 정규화
	float	length();									// 길이
	bool	isZero();

	BOOL    operator==( const GVector &vector );		// 벡터가 같은지. 벡터의 상등을 의미하는게 아니라 단순히 성분이 모두 일치해야함
	void    operator= ( const GVector &vector )                { this->x = vector.x; this->y = vector.y; this->z = vector.z; this->w = vector.w; } // 대입
	GVector operator+ ( const GVector &vector ) const;	// 덧셈
	GVector operator- ( const GVector &vector ) const;	// 뺄셈
	GVector operator- () const;							// 뺄셈

	friend GVector operator * ( const GVector& v, const float f ) { return GVector( v.x * f, v.y * f, v.z * f, v.w * f  ); }	// 스칼라 곱
	friend GVector operator * ( const float f, const GVector& v ) { return GVector( v.x * f, v.y * f, v.z * f, v.w * f  ); }	// 스칼라 곱
	friend GVector operator / ( const GVector& v, const float f ) { float invf = 1/f; return GVector( v.x * invf, v.y * invf, v.z * invf, v.w * invf ); }	// 스칼라 나눗셈

	void operator-= ( const GVector &vector );			// 뺄셈
	void operator+= ( const GVector &vector );			// 덧셈
	GVector& operator *=( const float f )  { x *= f; y *= f; z *= f; w *= f; return *this; }	                            // 스칼라 곱
	GVector& operator /=( const float f )  { float invf = 1/f; x *= invf; y *= invf; z *= invf; w *= invf; return *this; }	// 스칼라 나눗셈
};
