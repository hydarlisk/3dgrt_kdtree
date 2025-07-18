//--------------------------------------------------------------------------//
//																			//
//	여러가지 변수 Type 선언													//
//																			//
//	Copyright (c) 2005  진봉준	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#ifndef GBASE_H
#define GBASE_H

#define RENDER914_VERSION	"Render914_Version_0.1"

#include <float.h>

/**
 *	윈도우에서는 클래스Lib 를 사용하기 위해서.
 */
#ifdef WIN32
	#include <windows.h>
	#define FILE_SEPARATOR	'\\'
#else
	#define FILE_SEPARATOR	'/'
#endif

#ifndef FALSE
	#define FALSE 0
#endif

#ifndef TRUE
	#define TRUE 1
#endif

#define G_TO_RADIAN 0.01745329f
#define G_TO_DEGREE 57.2958279f
#define G_PI 3.14159f
#define G_EPSILON 0.000001f
#define G_FLOAT_EQUAL_EPSILON 0.00001f
#define G_INFINITY FLT_MAX

#define IS_FLOAT_EQUAL( A, B )	\
		( (A) >= (B) - G_FLOAT_EQUAL_EPSILON && (A) <= (B) + G_FLOAT_EQUAL_EPSILON )

#include "GError.h"
#include "GErrorManager.h"
#include "GLogManager.h"
#include "GTimer.h"

#endif
