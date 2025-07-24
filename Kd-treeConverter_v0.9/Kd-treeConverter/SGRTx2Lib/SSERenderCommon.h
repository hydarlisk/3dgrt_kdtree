
#ifndef _SSE_RENDERCOMMON_H_
#define _SSE_RENDERCOMMON_H_


#include "SSE_common.h"
#include "GKDTreeNode.h"

// ------------------------------------------------------------------
// BVH
struct TMIntCandidate {
	//! pointer to the index of the first vertex
	const unsigned int* pFirstIndex;
	//! intersection point in barycentric coordinates
	GVector vBaryP;
	//! distance from ray origin to intersection point
	float t;
};

typedef __declspec(align(16)) struct _sse_4x4_raypacket_bvh_t {
	_sse_vec d[4];							// 48 *4
	_sse_vec o;						     	// 48 	
} _sse_4x4_raypacket_bvh;
// BVH
// ------------------------------------------------------------------

typedef float Pixel;
typedef kdtreeNode  KdTreeNode2;
typedef GScene		Scene;
typedef GVector		vector3;


#define AXIS_SIZE	3		// x, y, z
#define SIMD_SIZE	4		// Vec4


// -----------------------------------------------------------
// TriAccel2
// -----------------------------------------------------------
class GPolygonObject;
class TriAccel2
{
public:
	// plane
	float n_u;		// normal.u / normal.k
	float n_v;		// normal.v / normal.k
	float n_d;		// constant of plane equation
	struct {
		unsigned int k: 2;				// projection dimension
		unsigned int isTransparent: 1;	// transparent material 
		unsigned int mbox: 29;			// transparent material 
	};

	// line equation for line ac
	float b_nu;
	float b_nv;
	float b_d;
	GPolygonObject *pObject;

	// line equation for line ab
	float c_nu;
	float c_nv;
	float c_d;
	int indexInObject;

	GPoint			N;						// 16	T.T
	//float area;
};

class TriAccel_P
{
public:
	float n_u;		// normal.u / normal.k
	float n_v;		// normal.v / normal.k
	float n_d;		// N' dot P

	float p_u;		// Pu
	float p_v;		// Pv
	int   k;		// k

	float e0u;		// edge0.u
	float e0v;		// edge0.v
	float e1u;		// edge1.u
	float e1v;		// edge1.v

	float rcp_area;
	int isTransparent;
	int mbox;

	GPoint			N;						// 16	T.T
};

// -----------------------------------------------------------
// Color
// -----------------------------------------------------------
class Color
{
public:
	Color() : r( 0.0f ), g( 0.0f ), b( 0.0f ) {};
	Color( float a_R, float a_G, float a_B ) : r( a_R ), g( a_G ), b( a_B ) {};
	Color( float a_RGB[3] ) : r( a_RGB[0] ), g( a_RGB[1] ), b( a_RGB[2] ) {};
	void Set( float a_R, float a_G, float a_B ) { r = a_R; g = a_G; b = a_B; }
	void operator += ( const Color& a_V ) { r += a_V.r; g += a_V.g; b += a_V.b; }
	void operator += ( Color* a_V ) { r += a_V->r; g += a_V->g; b += a_V->b; }
	void operator -= ( const Color& a_V ) { r -= a_V.r; g -= a_V.g; b -= a_V.b; }
	void operator -= ( Color* a_V ) { r -= a_V->r; g -= a_V->g; b -= a_V->b; }
	void operator *= ( const float f ) { r *= f; g *= f; b *= f; }
	void operator *= ( const Color& a_V ) { r *= a_V.r; g *= a_V.g; b *= a_V.b; }
	void operator *= ( Color* a_V ) { r *= a_V->r; g *= a_V->g; b *= a_V->b; }
	Color operator- () const { return Color( -r, -g, -b ); }
	friend Color operator + ( const Color& v1, const Color& v2 ) { return Color( v1.r + v2.r, v1.g + v2.g, v1.b + v2.b ); }
	friend Color operator - ( const Color& v1, const Color& v2 ) { return Color( v1.r - v2.r, v1.g - v2.g, v1.b - v2.b ); }
	friend Color operator + ( const Color& v1, Color* v2 ) { return Color( v1.r + v2->r, v1.g + v2->g, v1.b + v2->b ); }
	friend Color operator - ( const Color& v1, Color* v2 ) { return Color( v1.r - v2->r, v1.g - v2->g, v1.b - v2->b ); }
	friend Color operator * ( const Color& v, const float f ) { return Color( v.r * f, v.g * f, v.b * f ); }
	friend Color operator * ( const Color& v1, const Color& v2 ) { return Color( v1.r * v2.r, v1.g * v2.g, v1.b * v2.b ); }
	friend Color operator * ( const float f, const Color& v ) { return Color( v.r * f, v.g * f, v.b * f ); }
	union
	{
		__m128 rgba;
		struct { float r, g, b, a; };
	};
};


// -----------------------------------------------------------
// RayPacket class definition
// -----------------------------------------------------------
typedef __declspec(align(16)) struct _sse_1x1_raypacket_t {
	_sse_float d;						// 16
	_sse_float o;						// 16
	int RayId;							// 4
	char Depth;							// 1
	char padd[11];						// 11
} _sse_1x1_raypacket;


typedef struct __declspec(align(128)) _sse_2x2_raypacket_t {
	_sse_vec d;								// 48
	_sse_vec o;								// 48
	int RayId;								// 4
	unsigned __int16 xmask, ymask, zmask;	// 6
	char RayWay;							// 1
	char Depth;								// 1
	char padd[4];							// 4

	__forceinline bool IsCoherent() {
		return (((xmask == 0)||(xmask == 15)) && ((ymask == 0)||(ymask == 15)) &&
				((zmask == 0)||(zmask == 15)));
	};
} _sse_2x2_raypacket;

typedef struct __declspec(align(128)) _sse_2x2_rayfrustum_t {
	_sse_2x2_raypacket rp;

	// frustum culling
	__m128 tfar;							// 16
	__m128 torg0;							// 16
	__m128 torg1;							// 16
} _sse_2x2_rayfrustum;

typedef __declspec(align(16)) struct _sse_4x4_raypacket_t {
	_sse_vec d[4];							// 48 *4
	_sse_vec o[4];							// 48 *4
	_sse_vec di[4];							// 48 *4
	int RayId;								// 4
	char RayWay;							// 1
	char Depth;								// 1
	unsigned __int16 xmask, ymask, zmask;	// 6
	__forceinline bool IsCoherent()	{
		return (((xmask == 0)||(xmask == 65535)) && ((ymask == 0)||(ymask == 65535)) &&
				((zmask == 0)||(zmask == 65535)));
	};
	int maxdirsign;							// 4	 -- frustum culling
} _sse_4x4_raypacket;

typedef __declspec(align(16)) struct _sse_8x8_raypacket_t {
	_sse_vec d[16];							// 48 *16
	_sse_vec o[16];							// 48 *16
	_sse_vec di[16];						// 48 *16
	int RayId;								// 4
	char RayWay;							// 1
	char Depth;								// 1
	unsigned __int64 xmask, ymask, zmask;	// 6
	__forceinline bool IsCoherent()	{
		return (((xmask == 0)||(xmask == _UI64_MAX)) && ((ymask == 0)||(ymask == _UI64_MAX)) &&
				((zmask == 0)||(zmask == _UI64_MAX)));
	};
	int maxdirsign;							// 4	 -- frustum culling
} _sse_8x8_raypacket;

// -----------------------------------------------------------
// Ray mask
// -----------------------------------------------------------
typedef __declspec(align(16)) struct raymask2x2_t {
	union { __m128 mask4; __m128i imask4; unsigned int mask[4]; };			// 16
} _sse_2x2_raymask;

typedef __declspec(align(16)) struct raymask4x4_t {
	union { __m128 mask4[4]; __m128i imask4[4]; unsigned int mask[16]; };	// 64
} _sse_4x4_raymask;

// -----------------------------------------------------------
// kdtree isect
// -----------------------------------------------------------
typedef struct __declspec(align(16)) _sse_1x1_isect_t
{
	float dist;					// 4
	int   tacc;					// 4
	unsigned int addr;			// 4

	float u;					// 4
	float v;					// 4

	char pad[12];				// 12

	GColor color;				// 16
	_sse_float n;				// 16

	struct _adss_measure_t {
		int pri_oid;		// 4
		int sec_oid;		// 4
		int pri_shadow;		// 4
		int sec_shadow;		// 4
		int pri_texture;	// 4
		int sec_texture;	// 4
		char pad[8];		// 8
	_sse_float n2;				// 16
	} ads;

	GColor weight;					// 16
} _sse_1x1_isect;


typedef struct __declspec(align(128)) _sse_2x2_isect_t
{
	union { __m128  dist4;	float dist[4]; };					// 16
	union { __m128i tacc4; __m128 ftacc; int tacc[4]; };		// 16
	union { __m128  u4;		float u[4]; };						// 16
	union { __m128  v4;		float v[4]; };						// 16
	union { __m128i addr4; unsigned int addr[4]; };				// 16
	_sse_vec n;						// 48
	GColor color[4];				// 16 * 4
} _sse_2x2_isect;

typedef struct __declspec(align(16))  _sse_4x4_isect_t
{
	union { __m128  dist4[4];	float dist[16]; };						// 64
	union { __m128i tacc4[4]; __m128 ftacc[4]; int tacc[16]; };			// 64
	union { __m128i addr4[4]; unsigned int addr[16]; };					// 64

	union { __m128  u4[4];		float u[16]; };							// 64
	union { __m128  v4[4];		float v[16]; };							// 64

	GColor color[16];													// 16 * 16 (256)
	_sse_vec n[4];														// 48 * 4 (192)

	struct _adss_measure_t {
		union { __m128i pri_oid4[4];    __m128 pri_oidf4[4];    int pri_oid[16]; };			// 64
		union { __m128i sec_oid4[4];    __m128 sec_oidf4[4];    int sec_oid[16]; };			// 64
		union { __m128i pri_shadow4[4]; __m128 pri_shadowf4[4]; int pri_shadow[16]; };		// 64
		union { __m128i sec_shadow4[4]; __m128 sec_shadowf4[4]; int sec_shadow[16]; };		// 64
		union { __m128i pri_texture4[4]; __m128 pri_texturef4[4]; int pri_texture[16]; };	// 64
		union { __m128i sec_texture4[4]; __m128 sec_texturef4[4]; int sec_texture[16]; };	// 64
	_sse_vec n2[4];														// 48 * 4 (192)
	} ads;

	GColor weight;					// 16
	float fTime[16];
} _sse_4x4_isect;

// -----------------------------------------------------------
// kdtree traversal stack
// -----------------------------------------------------------
typedef struct __declspec(align(16)) _sse_1x1_kdstack_t
{
	float t_far_;										// 4
	float t_near;										// 4
	KdTreeNode2* node;									// 4
	unsigned int depth;									// 4
} _sse_1x1_kdstack;

typedef struct __declspec(align(64)) _sse_2x2_kdstack_t
{
	union { __m128 t_far_4; float t_far_[4]; };			// 16
	union { __m128 t_near4; float t_near[4]; };			// 16
	KdTreeNode2* node;									// 4
	unsigned int depth;									// 4
	char pad[8];										// 8
} _sse_2x2_kdstack;

typedef struct __declspec(align(16)) _sse_4x4_kdstack_t
{
	union { __m128 t_far_4[4]; float t_far_[16]; };		// 64
	union { __m128 t_near4[4]; float t_near[16]; };		// 64
	KdTreeNode2* node;									// 4
	unsigned int depth;									// 4
	char pad[8];										// 8
} _sse_4x4_kdstack;


// ------------------------------------------------------------------------------------
// SSERenderPipelineQ 관련 구조체
// ------------------------------------------------------------------------------------
	__declspec(align(16)) struct _sse_2x2_isectQ
	{
		union { __m128  dist4;	float dist[4]; };					// 16
		union { __m128i tacc4; __m128 ftacc; int tacc[4]; };		// 16
		union { __m128  u4;		float u[4]; };						// 16
		union { __m128  v4;		float v[4]; };						// 16
		union { __m128i addr4; unsigned int addr[4]; };				// 16
		_sse_vec n;						// 48
		GColor color[4];				// 16 * 4
		GColor weight[4];				// 16 * 4
	};

	__declspec(align(16)) struct _sse_4x4_isectQ
	{
		union { __m128  dist4[4];	float dist[16]; };						// 64
		union { __m128i tacc4[4]; __m128 ftacc[4]; int tacc[16]; };			// 64
		union { __m128i addr4[4]; unsigned int addr[16]; };					// 64

		union { __m128  u4[4];		float u[16]; };							// 64
		union { __m128  v4[4];		float v[16]; };							// 64

		GColor color[16];													// 16 * 16 (256)
		_sse_vec n[4];														// 48 * 4 (192)

		struct _adss_measure_t {
			union { __m128i pri_oid4[4];    __m128 pri_oidf4[4];    int pri_oid[16]; };			// 64
			union { __m128i sec_oid4[4];    __m128 sec_oidf4[4];    int sec_oid[16]; };			// 64
			union { __m128i pri_shadow4[4]; __m128 pri_shadowf4[4]; int pri_shadow[16]; };		// 64
			union { __m128i sec_shadow4[4]; __m128 sec_shadowf4[4]; int sec_shadow[16]; };		// 64
			union { __m128i pri_texture4[4]; __m128 pri_texturef4[4]; int pri_texture[16]; };	// 64
			union { __m128i sec_texture4[4]; __m128 sec_texturef4[4]; int sec_texture[16]; };	// 64
		_sse_vec n2[4];														// 48 * 4 (192)
		} ads;

		GColor weight[16];				// 16 * 16

		float fTime[16];
	};

	__declspec(align(16)) struct _sse_2x2_traceData
	{
		_sse_2x2_raypacket	*rp;
		_sse_2x2_isectQ		*is;
		_sse_2x2_raymask	*rm;
		_sse_2x2_raymask	*tm;
		int id;
		_sse_2x2_traceData	*next;
	};

	__declspec(align(16)) struct _sse_4x4_traceData
	{
		_sse_4x4_raypacket	*rp;
		_sse_4x4_isectQ		*is;
		_sse_4x4_raymask	*rm;
		_sse_4x4_raymask	*tm;
		int id;
		_sse_4x4_traceData	*next;
	};
// ------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------
// SSERayQueue 관련 구조체
// ------------------------------------------------------------------------------------
__declspec(align(16)) struct _sse_1x1_rayitemData
{
	_sse_1x1_raypacket		*rp;
	_sse_1x1_rayitemData	*next;
	unsigned int addr;									// 4
	GColor weight;
};

__declspec(align(16)) struct _sse_2x2_rayitemData
{
	_sse_2x2_raypacket		*rp;
	_sse_2x2_rayitemData	*next;
	union { __m128i addr4; unsigned int addr[4]; };		// 16
	GColor weight;
};

__declspec(align(16)) struct _sse_4x4_rayitemData
{
	_sse_4x4_raypacket		*rp;
	_sse_4x4_rayitemData	*next;
	union { __m128i addr4[4]; unsigned int addr[16]; };	// 64
	GColor weight;
};

__declspec(align(4)) struct _tableitemData
{
	char flag;
};

// -----------------------------------------------------------
// Detect_ADPSS_measure
// -----------------------------------------------------------
__declspec(align(16)) struct Detect_ADPSS_measure
{
	GColor oColor;
	_sse_float ray_d;
	int		pri_oid;				// Index of object hitted by primary ray
	int		sec_oid;				// Index of object hitted by secondary ray
	_sse_float	pri_normal;	
	_sse_float	sec_normal;
	int		pri_shadow;				// is shadow
	int		sec_shadow;				// is shadow
	int		pri_texture;			// is texture
	int		sec_texture;			// is texture
};

// -----------------------------------------------------------
// for tiny
// -----------------------------------------------------------
typedef struct RMASK4_t {
	union { __m128 f[4]; __m128i i[4]; };
} RMASK4;

typedef struct RDATA4_t {
	__m128 f[4];
} RDATA4;

typedef struct RSSEFLOAT4_t {
	_sse_float v[4];
} RSSEFLOAT4;

typedef struct RSSEUINT4_t {
	_sse_uint v[4];
} RSSEUINT4;

typedef struct RSSEVEC4_t {
	_sse_vec v[4];
} RSSEVEC4;

typedef struct RMASK_t {
	union { __m128 f; __m128i i; };
} RMASK;

typedef struct RDATA_t {
	__m128 f;
} RDATA;

typedef struct RSSEFLOAT_t {
	_sse_float v;
} RSSEFLOAT;

typedef struct RSSEUINT_t {
	_sse_uint v;
} RSSEUINT;

typedef struct RSSEVEC_t {
	_sse_vec v;
} RSSEVEC;

#endif
