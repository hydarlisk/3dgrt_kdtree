#ifndef _SSE_MATH_H_
#define _SSE_MATH_H_

#include <math.h>
#include "SSE_common.h"

#pragma warning ( disable : 4244 )

#define RAY_START_EPSILON	1e-4f
#define EPSILON				1e-3f

static float rcp_Int2Float_Table[17] = {
	1, 1, 0.5f, 0.333333f, 0.25f, 0.2f, 0.166667f,  0.142857f, 0.125f, 0.111111f,
	0.1f, 0.0909091f, 0.0833333f, 0.0769231f, 0.0714286f, 0.0666667f, 0.0625f};
#define rcp_int2float(x) rcp_Int2Float_Table[(x)]

static const __m128  sse_true   = (__m128&)(_mm_set1_epi32(0xffffffff));
static const __m128  sse_false  = _mm_setzero_ps();
static const __m128i sse_itrue  = (_mm_set1_epi32(0xffffffff));
static const __m128i sse_ifalse = _mm_setzero_si128();

//---------------------------------------------------------
// radical inverse function for low-discrepancy sampling
//---------------------------------------------------------
inline double sgrtRadicalInverse(int i, int base) {
	double val = 0;
	double invBase = 1.0/base;
	double invBi = invBase;

	while (i > 0) {
			// Compute next digit of radical inverse
			int d_i = (i % base);
			val += d_i * invBi;
			i /= base;
			invBi *= invBase;
	}
	return val;
}

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------

#define cpu_fset1	sse_fset1

inline float cpu_fdot(const _sse_float& a, const _sse_float& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;	// 3개합
}

inline void cpu_inverse(_sse_float& e, const _sse_float& n)
{
	e.x = 1.0f / n.x;
	e.y = 1.0f / n.y;
	e.z = 1.0f / n.z;
	e.w = 1.0f / n.w;
}

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------

inline __m128 operator+(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
inline __m128 operator-(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
inline __m128 operator*(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
inline const __m128& operator+=(__m128& a, __m128 b) { return (a =_mm_add_ps(a, b)); }
inline const __m128& operator-=(__m128& a, __m128 b) { return (a =_mm_sub_ps(a, b)); }
inline const __m128& operator*=(__m128& a, __m128 b) { return (a =_mm_mul_ps(a, b)); }

//#define DECLARE_EPI32_CONST(name) \
  //extern const _MM_ALIGN16 int _epi32_##name[4];

//DECLARE_EPI32_CONST(sign_mask)

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------
static const float _NEGMASK = 0.0f * -1.0f;
static const __m128   _MASKSIGN_ = _mm_set1_ps(_NEGMASK);
static const __m128i _iMASKSIGN_ = (__m128i&)_MASKSIGN_;

#define _mm_abs_ps(vec)     _mm_andnot_ps(_MASKSIGN_,vec)
#define _mm_neg_ps(vec)     _mm_xor_ps(_MASKSIGN_,vec)
#define _mm_abs_epi32(vec)  _mm_andnot_si128(_iMASKSIGN_,vec)
#define _mm_neg_epi32(vec)  _mm_xor_si128(_iMASKSIGN_,vec)
#define _mm_not_ps(vec)	    _mm_xor_ps(sse_true,vec)
#define _mm_not_epi32(vec)	_mm_xor_si128(sse_itrue,vec)



#define VECTOR4UI_SHUFFLE_MASK(p,q,r,s) (((p<<6)|(q<<4)|(r<<2)|(s)))
#define vector4ui_swizzle(v,p,q,r,s) ((_mm_shuffle_epi32( (v), ((s)<<6)|((r)<<4)|((q)<<2)|(p))))
inline __m128i sse_imul(const __m128i v1, const __m128i v2)
{
    __m128i a1=vector4ui_swizzle(v1,1,0,3,2);
    __m128i b1=vector4ui_swizzle(v2,1,0,3,2);
    __m128i mul1=_mm_mul_epu32(v1,v2);
    __m128i mul2=_mm_mul_epu32(a1,b1);
    __m128i composite=_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(mul1),_mm_castsi128_ps(mul2),VECTOR4UI_SHUFFLE_MASK(0,2,0,2)));
    return (vector4ui_swizzle(composite,1,3,0,2));
}

inline __m128i sse_imul_v2(const __m128i a, const __m128i b)
{
  return _mm_or_si128(
    _mm_and_si128( _mm_mul_epu32(a,b),
                   _mm_setr_epi32(0xffffffff,0,0xffffffff,0)),
    _mm_slli_si128(
         _mm_and_si128(
                  _mm_mul_epu32(_mm_srli_si128(a,4),_mm_srli_si128(b,4)),
                  _mm_setr_epi32(0xffffffff,0,0xffffffff,0)),
	4));
}

inline __m128i sse_imin_v2(const __m128i a, const __m128i b)
{
	__m128i t = _mm_cmpgt_epi32(a,b);
	return _mm_or_si128( _mm_and_si128(t,b), _mm_andnot_si128(t,a));
}

inline __m128i sse_imax_v2(const __m128i a, const __m128i b)
{
	__m128i t = _mm_cmpgt_epi32(a,b);
	return _mm_or_si128( _mm_andnot_si128(t,b), _mm_and_si128(t,a));
}
// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------

inline __m128 sse_update(const __m128 a, const __m128 b, const __m128 mask)
{
	return _mm_or_ps(_mm_and_ps(a,mask),
		_mm_andnot_ps(mask,b));
}

inline __m128i sse_update(const __m128i a, const __m128i b, const __m128i mask)
{
	return _mm_or_si128(_mm_and_si128(a,mask),
		_mm_andnot_si128(mask,b));
}

inline __m128 sse_inverse(const __m128 n)
{
	const __m128 rcp = _mm_rcp_ps(n);
	//return SSE_Update4(_mm_sub_ps(_mm_add_ps(rcp,rcp), _mm_mul_ps(_mm_mul_ps(rcp,rcp),n)), _mm_set1_ps(100000), _mm_cmpneq_ps(n, _mm_setzero_ps()));
	return _mm_sub_ps(_mm_add_ps(rcp,rcp), _mm_mul_ps(_mm_mul_ps(rcp,rcp),n));
}

inline __m128 sse_rsqrt( const __m128 v )
{
	const __m128 nr = _mm_rsqrt_ps( v );
	const __m128 muls = _mm_mul_ps( _mm_mul_ps( v, nr ), nr );
	return _mm_mul_ps( _mm_mul_ps( _mm_set1_ps(0.5f), nr ), _mm_sub_ps( _mm_set1_ps(3), muls ) );
}

inline __m128 fastxovery( const __m128 v1, const __m128 v2 )
{
	const __m128 n = _mm_rcp_ps( v2 );
	return _mm_mul_ps( v1, _mm_sub_ps( _mm_add_ps( n, n ), _mm_mul_ps( _mm_mul_ps( v2, n ), n ) ) );
}

inline float sse_dot( const __m128 v1, const __m128 v2 )
{
	union u { __m128 m; float f[4]; } x;
	x.m = _mm_mul_ps(v1, v2);
	return x.f[0] + x.f[1] + x.f[2] + x.f[3];	// 4개합
}

inline float sse_fdot(const _sse_float& a, const _sse_float& b)
{
	union u { __m128 m; float f[4]; } x;
	x.m = _mm_mul_ps(a.v4, b.v4);
	return x.f[0] + x.f[1] + x.f[2];			// 3개합
}

inline _sse_float sse_fset1(const float* v)
{
	_sse_float t;
	t.x = v[0];
	t.y = v[1];
	t.z = v[2];
	return t;
}

inline _sse_float sse_fset1(float x, float y, float z)
{
	_sse_float t;
	t.x = x;
	t.y = y;
	t.z = z;
	return t;
}

inline _sse_uint sse_iset1(unsigned int x, unsigned int y, unsigned int z)
{
	_sse_uint t;
	t.x = x;
	t.y = y;
	t.z = z;
	return t;
}

inline _sse_uint sse_iset1(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
	_sse_uint t;
	t.x = x;
	t.y = y;
	t.z = z;
	t.w = w;
	return t;
}

inline _sse_float cpu_fmul(const _sse_float& a, const float& b)
{
	_sse_float t;
	t.x = a.x *b;
	t.y = a.y *b;
	t.z = a.z *b;
	return t;
}

inline _sse_float cpu_fmul(const float& a, const _sse_float& b)
{
	_sse_float t;
	t.x = b.x *a;
	t.y = b.y *a;
	t.z = b.z *a;
	return t;
}

inline _sse_float cpu_fadd(const _sse_float& a, const _sse_float& b)
{
	_sse_float t;
	t.x = a.x + b.x;
	t.y = a.y + b.y;
	t.z = a.z + b.z;
	return t;
}

inline _sse_float cpu_fsub(const _sse_float& a, const _sse_float& b)
{
	_sse_float t;
	t.x = a.x - b.x;
	t.y = a.y - b.y;
	t.z = a.z - b.z;
	return t;
}

inline _sse_float sse_fmul(const _sse_float& a, const float& b)
{
	_sse_float t;
	t.v4 = _mm_mul_ps(a.v4, _mm_set1_ps(b));
	return t;
}

inline _sse_float sse_fmul(const float& a, const _sse_float& b)
{
	_sse_float t;
	t.v4 = _mm_mul_ps(b.v4, _mm_set1_ps(a));
	return t;
}

inline _sse_float sse_fadd(const _sse_float& a, const _sse_float& b)
{
	_sse_float t;
	t.v4 = _mm_add_ps(a.v4, b.v4);
	return t;
}

inline _sse_float sse_fsub(const _sse_float& a, const _sse_float& b)
{
	_sse_float t;
	t.v4 = _mm_sub_ps(a.v4, b.v4);
	return t;
}

#define sse_vdot(a,b)		_mm_add_ps(_mm_add_ps(_mm_mul_ps((a).x4, (b).x4), _mm_mul_ps((a).y4, (b).y4)), _mm_mul_ps((a).z4, (b).z4))
inline __m128 sse_vlength2(const _sse_vec& v)
{
	return sse_vdot(v,v);
}

inline __m128 sse_vlength(const _sse_vec& v)
{
	return _mm_sqrt_ps(sse_vdot(v,v));
}

inline _sse_vec sse_vupdate(const _sse_vec& a, const _sse_vec& b, const __m128 mask)
{
	_sse_vec t;
	t.x4 = sse_update(a.x4, b.x4, mask);
	t.y4 = sse_update(a.y4, b.y4, mask);
	t.z4 = sse_update(a.z4, b.z4, mask);
	return t;
}

inline _sse_vec sse_vsigned(const _sse_vec& v)
{
	_sse_vec t;
	t.x4 = _mm_neg_ps(v.x4);
	t.y4 = _mm_neg_ps(v.y4);
	t.z4 = _mm_neg_ps(v.z4);
	return t;
}

inline _sse_vec sse_vmul(const _sse_vec& a, const _sse_vec& b)
{
	_sse_vec t;
	t.x4 = _mm_mul_ps(a.x4, b.x4);
	t.y4 = _mm_mul_ps(a.y4, b.y4);
	t.z4 = _mm_mul_ps(a.z4, b.z4);
	return t;
}

inline _sse_vec sse_vmul(const __m128 a, const _sse_vec& b)
{
	_sse_vec t;
	t.x4 = _mm_mul_ps(a, b.x4);
	t.y4 = _mm_mul_ps(a, b.y4);
	t.z4 = _mm_mul_ps(a, b.z4);
	return t;
}

inline _sse_vec sse_vmul(const _sse_vec& a, const __m128 b)
{
	_sse_vec t;
	t.x4 = _mm_mul_ps(a.x4, b);
	t.y4 = _mm_mul_ps(a.y4, b);
	t.z4 = _mm_mul_ps(a.z4, b);
	return t;
}

inline _sse_vec sse_vadd(const _sse_vec& a, const _sse_vec& b)
{
	_sse_vec t;
	t.x4 = _mm_add_ps(a.x4, b.x4);
	t.y4 = _mm_add_ps(a.y4, b.y4);
	t.z4 = _mm_add_ps(a.z4, b.z4);
	return t;
}

inline _sse_vec sse_vsub(const _sse_vec& a, const _sse_vec& b)
{
	_sse_vec t;
	t.x4 = _mm_sub_ps(a.x4, b.x4);
	t.y4 = _mm_sub_ps(a.y4, b.y4);
	t.z4 = _mm_sub_ps(a.z4, b.z4);
	return t;
}

inline _sse_vec sse_vnorm(const _sse_vec& v)
{
	__m128 invLen = sse_rsqrt(sse_vdot(v, v));
	_sse_vec t = sse_vmul(v, invLen);
	return t;
}

inline _sse_vec sse_vset1 (float n)
{
	_sse_vec t;
	t.x4 = _mm_set1_ps(n);
	t.y4 = _mm_set1_ps(n);
	t.z4 = _mm_set1_ps(n);
	return t;
}

inline _sse_vec sse_vset1 (float x, float y, float z)
{
	_sse_vec t;
	t.x4 = _mm_set1_ps(x);
	t.y4 = _mm_set1_ps(y);
	t.z4 = _mm_set1_ps(z);
	return t;
}

inline _sse_vec sse_vset1 (float v[4])
{
	_sse_vec t;
	t.x4 = _mm_set1_ps(v[0]);
	t.y4 = _mm_set1_ps(v[1]);
	t.z4 = _mm_set1_ps(v[2]);
	return t;
}

inline _sse_vec sse_vset1 (__m128 v)
{
	_sse_vec t;
	t.x4 = v;
	t.y4 = v;
	t.z4 = v;
	return t;
}

inline _sse_vec sse_vset1 (float v0[4], float v1[4], float v2[4], float v3[4])
{
	_sse_vec t;
	t.x4 = _mm_setr_ps(v0[0], v1[0], v2[0], v3[0]);
	t.y4 = _mm_setr_ps(v0[1], v1[1], v2[1], v3[1]);
	t.z4 = _mm_setr_ps(v0[2], v1[2], v2[2], v3[2]);
	return t;
}

inline _sse_vec sse_vset1 (float v00, float v01, float v02,
						   float v10, float v11, float v12,
						   float v20, float v21, float v22,
						   float v30, float v31, float v32)
{
	_sse_vec t;
	t.x4 = _mm_setr_ps(v00, v10, v20, v30);
	t.y4 = _mm_setr_ps(v01, v11, v21, v31);
	t.z4 = _mm_setr_ps(v02, v12, v22, v32);
	return t;
}

inline _sse_vec sse_vmax(const _sse_vec& a, const _sse_vec& b)
{
	_sse_vec t;
	t.x4 = _mm_max_ps(a.x4, b.x4);
	t.y4 = _mm_max_ps(a.y4, b.y4);
	t.z4 = _mm_max_ps(a.z4, b.z4);
	return t;
}

inline __m128 sse_set1_i(unsigned int n)
{
	union u { __m128 m; unsigned int f[4]; } x;
	x.f[0] = x.f[1] = x.f[2] = x.f[3] = n;
	return x.m;
}

inline __m128 sse_set_f(float src[4])
{
	union u { __m128 m; float f[4]; } x;
	x.f[0] = src[0]; x.f[1] = src[1]; x.f[2] = src[2]; x.f[3] = src[3];
	return x.m;
}

inline void sse_get_f(float dst[4], const __m128 src)
{
	union u { __m128 m; float f[4]; } x;
	x.m = src;
	dst[0] = x.f[0];	dst[1] = x.f[1];	dst[2] = x.f[2];	dst[3] = x.f[3];
}

inline void sse_get_i(int dst[4], const __m128 src)
{
	union u { __m128 m; int f[4]; } x;
	x.m = src;
	dst[0] = x.f[0];	dst[1] = x.f[1];	dst[2] = x.f[2];	dst[3] = x.f[3];
}

inline void sse_get_f(float dst[4][3], const _sse_vec& src)
{
	union u { __m128 m; float f[4]; } x;

	x.m = src.x4;		// x 성분
	dst[0][0] = x.f[0];		dst[1][0] = x.f[1];		dst[2][0] = x.f[2];		dst[3][0] = x.f[3];

	x.m = src.y4;		// y 성분
	dst[0][1] = x.f[0];		dst[1][1] = x.f[1];		dst[2][1] = x.f[2];		dst[3][1] = x.f[3];

	x.m = src.z4;		// z 성분
	dst[0][2] = x.f[0];		dst[1][2] = x.f[1];		dst[2][2] = x.f[2];		dst[3][2] = x.f[3];
}

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------

// ----------------------------------
// Power functions (SIMD)
// ----------------------------------

	// A) Schlick Reflection Model ; x^n = x / (n - nx +x)
	inline __m128 sse_pow_Schlick( const __m128& base, const __m128& exponent)
	{
		__m128 denom = _mm_mul_ps( exponent, base);
		denom = _mm_sub_ps( exponent, denom);
		denom = _mm_add_ps( base, denom);
		return _mm_mul_ps( base, _mm_rcp_ps(denom));
	}

	// B) Polynomial Approach
	#define EXP_POLY_DEGREE 3
	#define POLY0(x, c0) _mm_set1_ps(c0)
	#define POLY1(x, c0, c1) _mm_add_ps(_mm_mul_ps(POLY0(x, c1), x), _mm_set1_ps(c0))
	#define POLY2(x, c0, c1, c2) _mm_add_ps(_mm_mul_ps(POLY1(x, c1, c2), x), _mm_set1_ps(c0))
	#define POLY3(x, c0, c1, c2, c3) _mm_add_ps(_mm_mul_ps(POLY2(x, c1, c2, c3), x), _mm_set1_ps(c0))
	#define POLY4(x, c0, c1, c2, c3, c4) _mm_add_ps(_mm_mul_ps(POLY3(x, c1, c2, c3, c4), x), _mm_set1_ps(c0))
	#define POLY5(x, c0, c1, c2, c3, c4, c5) _mm_add_ps(_mm_mul_ps(POLY4(x, c1, c2, c3, c4, c5), x), _mm_set1_ps(c0))

	inline __m128 exp2f4(__m128 x)
	{
	   __m128i ipart;
	   __m128 fpart, expipart, expfpart;

	   x = _mm_min_ps(x, _mm_set1_ps( 129.00000f));
	   x = _mm_max_ps(x, _mm_set1_ps(-126.99999f));

	   /* ipart = int(x - 0.5) */
	   ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _mm_set1_ps(0.5f)));

	   /* fpart = x - ipart */
	   fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));

	   /* expipart = (float) (1 << ipart) */
	   expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _mm_set1_epi32(127)), 23));

	   /* minimax polynomial fit of 2**x, in range [-0.5, 0.5[ */
	#if EXP_POLY_DEGREE == 5
	   expfpart = POLY5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
	#elif EXP_POLY_DEGREE == 4
	   expfpart = POLY4(fpart, 1.0000026f, 6.9300383e-1f, 2.4144275e-1f, 5.2011464e-2f, 1.3534167e-2f);
	#elif EXP_POLY_DEGREE == 3
	   expfpart = POLY3(fpart, 9.9992520e-1f, 6.9583356e-1f, 2.2606716e-1f, 7.8024521e-2f);
	#elif EXP_POLY_DEGREE == 2
	   expfpart = POLY2(fpart, 1.0017247f, 6.5763628e-1f, 3.3718944e-1f);
	#else
	#error
	#endif

	   return _mm_mul_ps(expipart, expfpart);
	}

	#define LOG_POLY_DEGREE 5

	inline __m128 log2f4(__m128 x)
	{
	   __m128i exp = _mm_set1_epi32(0x7F800000);
	   __m128i mant = _mm_set1_epi32(0x007FFFFF);

	   __m128 one = _mm_set1_ps( 1.0f);

	   __m128i i = _mm_castps_si128(x);

	   __m128 e = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(_mm_and_si128(i, exp), 23), _mm_set1_epi32(127)));

	   __m128 m = _mm_or_ps(_mm_castsi128_ps(_mm_and_si128(i, mant)), one);

	   __m128 p;

	   /* Minimax polynomial fit of log2(x)/(x - 1), for x in range [1, 2[ */
	#if LOG_POLY_DEGREE == 6
	   p = POLY5( m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f,  3.1821337e-1f, -3.4436006e-2f);
	#elif LOG_POLY_DEGREE == 5
	   p = POLY4(m, 2.8882704548164776201f, -2.52074962577807006663f, 1.48116647521213171641f, -0.465725644288844778798f, 0.0596515482674574969533f);
	#elif LOG_POLY_DEGREE == 4
	   p = POLY3(m, 2.61761038894603480148f, -1.75647175389045657003f, 0.688243882994381274313f, -0.107254423828329604454f);
	#elif LOG_POLY_DEGREE == 3
	   p = POLY2(m, 2.28330284476918490682f, -1.04913055217340124191f, 0.204446009836232697516f);
	#else
	#error
	#endif

	   /* This effectively increases the polynomial degree by one, but ensures that log2(1) == 0*/
	   p = _mm_mul_ps(p, _mm_sub_ps(m, one));

	   return _mm_add_ps(p, e);
	}

	static inline __m128
	sse_pow_Polynomial(__m128 x, __m128 y)
	{
	   return exp2f4(_mm_mul_ps(log2f4(x), y));
	}


// ----------------------------------
// Power functions (no SIMD)
// ----------------------------------

	// A) Schlick Reflection Model ; x^n = x / (n - nx +x)
	inline float _Pow_Schlick_( float base, float exponent)
	{
		float denom = exponent - exponent * base + base;
		return base * 1/denom;
	}

	// B) 
	static float shift23  = (1<<23);
	static float OOshift23= 1.0/(1<<23);

	inline float _Log2_IEEE_FP_(float i) {
		float LogBodge=0.346607f;
		float x;
		float y;
		x=*(int *)&i;
		x*= OOshift23; // 1/pow(2,23);
		x=x-127;

		y=x-floorf(x);
		y=(y-y*y)*LogBodge;
		return x+y;
	}

	inline float _Pow2_IEEE_FP_(float i) {
		float PowBodge=0.33971f;
		float x;
		float y=i-floorf(i);
		y=(y-y*y)*PowBodge;

		x=i+127-y;
		x*= shift23; // pow(2,23);
		*(int*)&x=(int)x;
		return x;
	}

	inline float _Pow_IEEE_FP_(float a, float b) {
		return _Pow2_IEEE_FP_(b*_Log2_IEEE_FP_(a));
	}


// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------

inline int _Log2Int_(float v) {
	return ((*(int *) &v) >> 23) - 127;			// 이거 구현 맞는 거냐? 라꾸야
}

inline int _Round2Int_(double val) {
	#define _doublemagic                    double (6755399441055744.0)
	//2^52 * 1.5,  uses limited precision to floor
	val             = val + _doublemagic;
	return ((long*)&val)[0];
}

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------
// 아래 memset 구현의 성능 효과는 잘 모르겠음.. 

#define SSE_MMREG_SIZE 16
#define SSE_MIN_LEN 0x40

inline void *zeromem_sse( void *dst, size_t size )
{
	void   *ret   = dst;
	char   *p_dst = (char*)dst;

	if( size >= SSE_MIN_LEN )
	{
		register size_t delta;
		if( ( delta = ((size_t)p_dst) & ( SSE_MMREG_SIZE - 1 ) ) != 0 ) /* Align destinition */
		{
			delta = SSE_MMREG_SIZE - delta;
			size -= delta;
			memset( p_dst, 0, delta );
			p_dst += delta;
		}

		//Get number of even blocks and save reminder
		size_t i = size >> 6;
		size &= 63;

		for( ; i; --i, p_dst += 64 )
		{
			__m128 xmm0, xmm1, xmm2, xmm3;
			xmm0 = _mm_setzero_ps();
			xmm1 = _mm_setzero_ps();
			xmm2 = _mm_setzero_ps();
			xmm3 = _mm_setzero_ps();
			_mm_stream_ps( (float*)p_dst     , xmm0 );
			_mm_stream_ps( (float*)p_dst + 4 , xmm1 );
			_mm_stream_ps( (float*)p_dst + 8 , xmm2 );
			_mm_stream_ps( (float*)p_dst + 12, xmm3 );
		}
	}

	if( size )
		memset( p_dst, 0, size );

	return ret;
}

inline void _nontemporal_copy_(char* outbuff, char* inbuff, int size) {
	const int step = 64; // a handy unroll factor, equal to WC buffer size
	while(size > step) {
		_mm_prefetch(inbuff + 320, _MM_HINT_NTA); // non-temporal
		__m128i A = _mm_loadu_si128((__m128i*) (inbuff + 0));
		__m128i B = _mm_loadu_si128((__m128i*) (inbuff + 16));
		__m128i C = _mm_loadu_si128((__m128i*) (inbuff + 32));
		__m128i D = _mm_loadu_si128((__m128i*) (inbuff + 48));
		// destination must be 16-byte aligned for streaming store!
		_mm_stream_si128((__m128i*) (outbuff + 0), A);
		_mm_stream_si128((__m128i*) (outbuff + 16), B);
		_mm_stream_si128((__m128i*) (outbuff + 32), C);
		_mm_stream_si128((__m128i*) (outbuff + 48), D);
		inbuff += step;
		outbuff += step;
		size -= step;
	}
	_mm_mfence(); // ensure last WC buffers get flushed to memory
	if(size) memcpy(outbuff, inbuff, size);
}

inline void nontemporal_copy (void* outbuff, void* inbuff, int size) 
{
	_nontemporal_copy_((char*)outbuff, (char*)inbuff, size);
}

#endif // _SSE_MATH_H_