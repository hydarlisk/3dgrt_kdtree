//--------------------------------------------------------------------------//
//																			//
//	�������� ���� Type ����													//
//																			//
//	Copyright (c) 2005  ������	( sonagi21@naver.com )						//
//																			//
//--------------------------------------------------------------------------//

#ifndef GBASE_H
#define GBASE_H

#define RENDER914_VERSION	"Render914_Version_0.1"

#include <float.h>

/**
 *	�����쿡���� Ŭ����Lib �� ����ϱ� ���ؼ�.
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

#define CUDA_SAFE_CALL(call)                                                   \
do {                                                                           \
    cudaError_t err = call;                                                   \
    if (cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
} while (0)


#endif
