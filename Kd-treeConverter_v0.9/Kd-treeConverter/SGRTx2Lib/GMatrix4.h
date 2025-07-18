//--------------------------------------------------------------------------//
//																			//
//	4x4 Çà·Ä																//
//																			//
//	Copyright (c) 2005  ÁøºÀÁØ	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#pragma once

#include "GBase.h"
#include "GVector.h"
#include "GPoint.h"

#define MATRIX( NAME, X, Y ) NAME[ Y * 4 + X ]

//! Row major matrix
class  GMatrix4
{
public:
	float matrix[16];

public:
	GMatrix4(void);
	GMatrix4( const GMatrix4& mat );
	~GMatrix4(void);

	BOOL operator== ( const GMatrix4 &mat );
	void operator= ( const GMatrix4 &mat );

	GMatrix4 operator+ ( const GMatrix4 &mat );
	GMatrix4 operator- ( const GMatrix4 &mat );
	GMatrix4 operator* ( const GMatrix4 &mat );
	GVector operator* ( const GVector &vec );
	GPoint operator* ( const GPoint &point );

	void SetMatrix( GVector v0, GVector v1, GVector v2 );
	void SetMatrix( float x0, float x1, float x2, float x3,
					float x4, float x5, float x6, float x7,
					float x8, float x9, float x10, float x11,
					float x12, float x13, float x14, float x15 );

	float GetElement( int i, int j );
	void  SetElement( int i, int j, float value );
	void identity();
	GMatrix4 transpose();

	static GMatrix4 createTranslateMatrix( float x, float y, float z );

};
