#pragma once

//--------------------------------------------------------------------------//
//																			//
//	Color 클래스															//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//
#include "GBase.h"
#include "SSE_common.h"

class  GColor
{
public:
	union {
		struct { float r, g, b, a; };
		struct { float m_Color[4]; };
		__m128 rgba;
	};

public:
	GColor( float _r, float _g, float _b, float _a = 1.0f ) : r(_r),   g(_g),   b(_b),   a(_a)   { };
	GColor( float f[3] )                                    : r(f[0]), g(f[1]), b(f[2])          { };
	GColor(void)                                            : r(0.0f), g(0.0f), b(0.0f), a(1.0f) { };
	~GColor(void);

	bool operator== ( const GColor& color );
	void operator+= ( const GColor& color );
	void operator*= ( const GColor& color );
	void operator-= ( const GColor& color );
	GColor operator- ( const GColor& color );

	friend GColor operator + ( const GColor& v1, const GColor& v2 ) { return GColor( v1.r + v2.r, v1.g + v2.g, v1.b + v2.b, v1.a + v2.a ); }
	friend GColor operator * ( const GColor& v, const float f )     { return GColor( v.r * f, v.g * f, v.b * f, v.a * f ); }
	friend GColor operator * ( const float f, const GColor& v )     { return GColor( v.r * f, v.g * f, v.b * f, v.a * f ); }
	friend GColor operator * ( const GColor& v1, const GColor& v2 ) { return GColor( v1.r * v2.r, v1.g * v2.g, v1.b * v2.b, v1.a * v2.a ); }

	const float* getColor();
	void setColor( const float color[] );
	void setColor( float r, float g, float b, float alpha );

};
