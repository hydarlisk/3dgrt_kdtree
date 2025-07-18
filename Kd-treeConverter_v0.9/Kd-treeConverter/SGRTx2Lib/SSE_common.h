#pragma once

//#ifdef WIN32
//	#define WIN32_LEAN_AND_MEAN
//	#include <windows.h>
//#endif

// Predefine Keyword for Aligned Data
#if defined (__GNUC__) && defined(__unix__)
	#define ALIGN_DATA(x) __attribute__((aligned(x)))
#elif defined (WIN32)
	#define ALIGN_DATA(x) __declspec(align(x))
#endif

// SSE intrinsic
#ifdef WIN32
	#include <intrin.h>
#else
	#include <emmintrin.h>      // __m128 data type and SSE2 functions
#endif


typedef __declspec(align(16)) struct _sse_vec_t {
	union {
		struct { __m128 x4,   y4,   z4;		};
		struct { float  x[4], y[4], z[4];	};
		__m128  v4[3];
		float f[12];
	};					// 48
} _sse_vec;

typedef __declspec(align(16)) struct _sse_vecF_t {
	union {
		struct { __m128 x4,   y4,   z4,   w4;		};
		struct { float  x[4], y[4], z[4], w[4];		};
		__m128  v4[4];
		float f[16];
	};					// 48
} _sse_vecF;

typedef __declspec(align(16)) struct _sse_float_t {
	union {
		struct { float x, y, z, w; };
		__m128  v4;
		float f[4];
	};					// 16
} _sse_float;

typedef __declspec(align(16)) struct _sse_uint_t {
	union {
		struct { UINT x, y, z, w; };
		__m128i v4;
		UINT f[4];
	};					// 16
} _sse_uint;