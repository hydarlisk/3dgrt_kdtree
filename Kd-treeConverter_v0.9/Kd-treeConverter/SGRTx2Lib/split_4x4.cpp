// -----------------------------------------------------------
// raytracer.cpp
// 2008 - oipini
// -----------------------------------------------------------
#include "GScene.h"
#include "GTexture.h"
#include "GTextureManager.h"
#include "GRenderSystem.h"

#include <stdio.h>
#include "SSE_math.h"
#include "SSERenderPipeline.h"

#include "GlobalOption.h"

#pragma warning ( disable : 4068 )
#pragma warning ( disable : 949 )

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

static const unsigned int modulo[] =  {0,1,2,0,1};
#define ku modulo[k+1]
#define kv modulo[k+2]

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::InitPacket
//		Initialize a packet for tracing
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::Split_InitPkt4x4(int nIdx)
{
	int i;
	_sse_4x4_raypacket	*rp = &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is = &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	// Clear initial color & mask
	memset( &rm->mask,  255, 64 );
	memset( &is->color, 0, 1024 );

	// Normalize ray's direction vector
	for ( i = 3; i >= 0; i-- ) {
		const __m128 v1 = sse_rsqrt( _mm_add_ps( _mm_add_ps( _mm_mul_ps( rp->d[i].x4, rp->d[i].x4 ), 
						  _mm_mul_ps( rp->d[i].y4, rp->d[i].y4 ) ), _mm_mul_ps( rp->d[i].z4, rp->d[i].z4 ) ) );
		rp->d[i].x4 = _mm_mul_ps( rp->d[i].x4, v1 );
		rp->d[i].y4 = _mm_mul_ps( rp->d[i].y4, v1 );
		rp->d[i].z4 = _mm_mul_ps( rp->d[i].z4, v1 );
	}

	// Pre-calculate reciprocal of ray's direction
	for ( i = 3; i >= 0; i-- ) {
		rp->di[i].x4 = sse_inverse(rp->d[i].x4);
		rp->di[i].y4 = sse_inverse(rp->d[i].y4);
		rp->di[i].z4 = sse_inverse(rp->d[i].z4);
	}

	// Pre-calculate dir_mask ray's direction vector
	rp->xmask =  _mm_movemask_ps( rp->d[0].x4 )       + (_mm_movemask_ps( rp->d[1].x4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].x4 ) << 8) + (_mm_movemask_ps( rp->d[3].x4 ) << 12);
	rp->ymask =  _mm_movemask_ps( rp->d[0].y4 )       + (_mm_movemask_ps( rp->d[1].y4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].y4 ) << 8) + (_mm_movemask_ps( rp->d[3].y4 ) << 12);
	rp->zmask =  _mm_movemask_ps( rp->d[0].z4 )       + (_mm_movemask_ps( rp->d[1].z4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].z4 ) << 8) + (_mm_movemask_ps( rp->d[3].z4 ) << 12);


	const float x_abs = fabsf(rp->d[0].x[0]);
	const float y_abs = fabsf(rp->d[0].y[0]);
	const float z_abs = fabsf(rp->d[0].z[0]);
	const int maxabsdir     = (x_abs > y_abs)?  ((x_abs > z_abs)?0:2) : ((y_abs > z_abs)?1:2);
	const int maxabsdirsign = (((float*)&rp->d[0].v4[maxabsdir])[0] < 0);
	rp->maxdirsign = maxabsdir | (maxabsdirsign? (1 << 2):0);

	if (m_RunStatics == 1) {
		if (rp->Depth == 0)	{ G_PR++; }
		else				{ G_RR++; }
	}
}

void SSERenderPipeline::Split_InitPkt4x4_ShwRay(void)
{
	int i;
	_sse_4x4_raypacket	*rp = &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*is = &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*rm	= &m_ShadowRMask4x4[0];

	// Clear intersection information
	for ( i = 3; i >= 0; i-- ) {
		is->tacc4[i] = _mm_setzero_si128();	// shadow rays : only tacc use
	}

	// Normalize ray's direction vector
	for ( i = 3; i >= 0; i-- ) {
		const __m128 v1 = sse_rsqrt( _mm_add_ps( _mm_add_ps( _mm_mul_ps( rp->d[i].x4, rp->d[i].x4 ), 
						  _mm_mul_ps( rp->d[i].y4, rp->d[i].y4 ) ), _mm_mul_ps( rp->d[i].z4, rp->d[i].z4 ) ) );
		rp->d[i].x4 = _mm_mul_ps( rp->d[i].x4, v1 );
		rp->d[i].y4 = _mm_mul_ps( rp->d[i].y4, v1 );
		rp->d[i].z4 = _mm_mul_ps( rp->d[i].z4, v1 );
	}

	// Pre-calculate reciprocal of ray's direction
	for ( i = 3; i >= 0; i-- ) {
		rp->di[i].x4 = sse_inverse(rp->d[i].x4);
		rp->di[i].y4 = sse_inverse(rp->d[i].y4);
		rp->di[i].z4 = sse_inverse(rp->d[i].z4);
	}

	// Pre-calculate dir_mask ray's direction vector
	rp->xmask =  _mm_movemask_ps( rp->d[0].x4 )       + (_mm_movemask_ps( rp->d[1].x4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].x4 ) << 8) + (_mm_movemask_ps( rp->d[3].x4 ) << 12);
	rp->ymask =  _mm_movemask_ps( rp->d[0].y4 )       + (_mm_movemask_ps( rp->d[1].y4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].y4 ) << 8) + (_mm_movemask_ps( rp->d[3].y4 ) << 12);
	rp->zmask =  _mm_movemask_ps( rp->d[0].z4 )       + (_mm_movemask_ps( rp->d[1].z4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].z4 ) << 8) + (_mm_movemask_ps( rp->d[3].z4 ) << 12);

	if (m_RunStatics == 1) { G_SR++; }
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
static const __m128 neginf4 = _mm_set_ps1(-FLT_MAX);
static const __m128 posinf4 = _mm_set_ps1( FLT_MAX);



bool SSERenderPipeline::Split_Isect4x4_PlaneTest_PriRay( TriAccel &acc, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, int temp )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const unsigned int k	= acc.k;

	__m128 nd[4];

	for (j = 3; j >= 0; j--) {
		nd[j]	=_mm_add_ps(rp->d[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->d[j].v4[ku]),
							_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->d[j].v4[kv])));
		f.f[j]	=_mm_sub_ps(_mm_set_ps1(acc.n_d),
				 _mm_add_ps(rp->o[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->o[j].v4[ku]),
							_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->o[j].v4[kv]))));
		f.f[j]	=_mm_mul_ps(f.f[j],sse_inverse(nd[j]));
	}

	__m128 mask_sum = _mm_setzero_ps();
	for (j = 3; j >= 0; j--) {
		Mask_Hit.f[j]	= _mm_and_ps(Mask_Hit.f[j],
			_mm_and_ps(_mm_cmpge_ps(is->dist4[j],f.f[j]),	_mm_cmpgt_ps(f.f[j],_mm_set1_ps(EPSILON))));	// eps < f <= Hit4.dist
		mask_sum	= _mm_or_ps (Mask_Hit.f[j], mask_sum);
	}

	if (_mm_movemask_ps(mask_sum)==0) return false;

	return true;
}

bool SSERenderPipeline::Split_Isect4x4_TriUVTest_PriRay( TriAccel &acc, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, RDATA4 &lambda, RDATA4 &mue, int check_aperture )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const unsigned int k	= acc.k;

	__m128 hu[4], hv[4];

#if FRUSTUM_CULLING == ON
	if (check_aperture == 1) {
#endif
	__m128 mask_sum = _mm_setzero_ps();
	for (j = 3; j >= 0; j--) {
		hu[j]		= _mm_add_ps(rp->o[j].v4[ku], _mm_mul_ps(f.f[j],rp->d[j].v4[ku]));
		hv[j]		= _mm_add_ps(rp->o[j].v4[kv], _mm_mul_ps(f.f[j],rp->d[j].v4[kv]));

		lambda.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.b_nv)));
		lambda.f[j]	= _mm_add_ps(lambda.f[j],_mm_set_ps1(acc.b_d));
		mue.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.c_nv)));
		mue.f[j]	= _mm_add_ps(mue.f[j],   _mm_set_ps1(acc.c_d));

		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmpgt_ps(lambda.f[j],_mm_setzero_ps()));
		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmpgt_ps(mue.f[j],   _mm_setzero_ps()));
		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmple_ps(_mm_add_ps(lambda.f[j],mue.f[j]), _mm_set1_ps(1)));
		mask_sum	= _mm_or_ps (Mask_Hit.f[j], mask_sum);
	}
	if (_mm_movemask_ps(mask_sum)==0) return false;
#if FRUSTUM_CULLING == ON
	} else {
	for (j = 3; j >= 0; j--) {
		hu[j]		= _mm_add_ps(rp->o[j].v4[ku], _mm_mul_ps(f.f[j],rp->d[j].v4[ku]));
		hv[j]		= _mm_add_ps(rp->o[j].v4[kv], _mm_mul_ps(f.f[j],rp->d[j].v4[kv]));

		lambda.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.b_nv)));
		lambda.f[j]	= _mm_add_ps(lambda.f[j],_mm_set_ps1(acc.b_d));
		mue.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.c_nv)));
		mue.f[j]	= _mm_add_ps(mue.f[j],   _mm_set_ps1(acc.c_d));
	}
	}
#endif

	return true;
}

bool SSERenderPipeline::Split_Isect4x4_PlaneTest_SecRay( TriAccel &acc, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const unsigned int k	= acc.k;

	__m128 nd[4];

	for (j = 3; j >= 0; j--) {
		nd[j]	=_mm_add_ps(rp->d[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->d[j].v4[ku]),
							_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->d[j].v4[kv])));
		f.f[j]	=_mm_sub_ps(_mm_set_ps1(acc.n_d),
				 _mm_add_ps(rp->o[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->o[j].v4[ku]),
							_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->o[j].v4[kv]))));
		f.f[j]	=_mm_mul_ps(f.f[j],sse_inverse(nd[j]));
	}

	__m128 mask_sum = _mm_setzero_ps();
	for (j = 3; j >= 0; j--) {
		Mask_Hit.f[j]	= _mm_and_ps(Mask_Hit.f[j],
			_mm_and_ps(_mm_cmpge_ps(is->dist4[j],f.f[j]),	_mm_cmpgt_ps(f.f[j],_mm_set1_ps(EPSILON))));	// eps < f <= Hit4.dist
		mask_sum	= _mm_or_ps (Mask_Hit.f[j], mask_sum);
	}

	if (_mm_movemask_ps(mask_sum)==0) return false;
	return true;
}

bool SSERenderPipeline::Split_Isect4x4_TriUVTest_SecRay( TriAccel &acc, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, RDATA4 &lambda, RDATA4 &mue )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const unsigned int k	= acc.k;

	__m128 hu[4], hv[4];

	__m128 mask_sum = _mm_setzero_ps();
	for (j = 3; j >= 0; j--) {
		hu[j]		= _mm_add_ps(rp->o[j].v4[ku], _mm_mul_ps(f.f[j],rp->d[j].v4[ku]));
		hv[j]		= _mm_add_ps(rp->o[j].v4[kv], _mm_mul_ps(f.f[j],rp->d[j].v4[kv]));

		lambda.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.b_nv)));
		lambda.f[j]	= _mm_add_ps(lambda.f[j],_mm_set_ps1(acc.b_d));
		mue.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.c_nv)));
		mue.f[j]	= _mm_add_ps(mue.f[j],   _mm_set_ps1(acc.c_d));

		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmpgt_ps(lambda.f[j],_mm_setzero_ps()));
		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmpgt_ps(mue.f[j],   _mm_setzero_ps()));
		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmple_ps(_mm_add_ps(lambda.f[j],mue.f[j]), _mm_set1_ps(1)));
		mask_sum	= _mm_or_ps (Mask_Hit.f[j], mask_sum);
	}
	if (_mm_movemask_ps(mask_sum)==0) return false;
	return true;
}


bool SSERenderPipeline::Split_Isect4x4_PlaneTest_ShwRay( TriAccel &acc, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*rm	= &m_ShadowRMask4x4[0];

	const unsigned int k	= acc.k;

	__m128 nd[4];

	for (j = 3; j >= 0; j--) {
		nd[j]	=_mm_add_ps(rp->d[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->d[j].v4[ku]),
							_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->d[j].v4[kv])));
		f.f[j]	=_mm_sub_ps(_mm_set_ps1(acc.n_d),
				 _mm_add_ps(rp->o[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->o[j].v4[ku]),
							_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->o[j].v4[kv]))));
		f.f[j]	=_mm_mul_ps(f.f[j],sse_inverse(nd[j]));
	}

	__m128 mask_sum = _mm_setzero_ps();
	for (j = 3; j >= 0; j--) {
		Mask_Hit.f[j]	= _mm_and_ps(Mask_Hit.f[j],
			_mm_and_ps(_mm_cmpge_ps(is->dist4[j],f.f[j]),	_mm_cmpgt_ps(f.f[j],_mm_set1_ps(EPSILON))));	// eps < f <= Hit4.dist
		mask_sum	= _mm_or_ps (Mask_Hit.f[j], mask_sum);
	}

	if (_mm_movemask_ps(mask_sum)==0) return false;
	return true;
}

bool SSERenderPipeline::Split_Isect4x4_TriUVTest_ShwRay( TriAccel &acc, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, RDATA4 &lambda, RDATA4 &mue )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*rm	= &m_ShadowRMask4x4[0];

	const unsigned int k	= acc.k;

	__m128 hu[4], hv[4];

	__m128 mask_sum = _mm_setzero_ps();
	for (j = 3; j >= 0; j--) {
		hu[j]		= _mm_add_ps(rp->o[j].v4[ku], _mm_mul_ps(f.f[j],rp->d[j].v4[ku]));
		hv[j]		= _mm_add_ps(rp->o[j].v4[kv], _mm_mul_ps(f.f[j],rp->d[j].v4[kv]));

		lambda.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.b_nv)));
		lambda.f[j]	= _mm_add_ps(lambda.f[j],_mm_set_ps1(acc.b_d));
		mue.f[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.c_nv)));
		mue.f[j]	= _mm_add_ps(mue.f[j],   _mm_set_ps1(acc.c_d));

		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmpgt_ps(lambda.f[j],_mm_setzero_ps()));
		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmpgt_ps(mue.f[j],   _mm_setzero_ps()));
		Mask_Hit.f[j] = _mm_and_ps(Mask_Hit.f[j], _mm_cmple_ps(_mm_add_ps(lambda.f[j],mue.f[j]), _mm_set1_ps(1)));
		mask_sum	= _mm_or_ps (Mask_Hit.f[j], mask_sum);
	}
	if (_mm_movemask_ps(mask_sum)==0) return false;
	return true;
}

void SSERenderPipeline::Split_FCull_Init( __m128 &term1, __m128 &term2, const int baseOffset, int nIdx )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	// 절대값이 큰 축 결정
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	const int I0 = rp->maxdirsign & 3;
	const int I1 = (I0<2)? I0+1 : 0, I2 = (I1<2)? I1+1 : 0;
	const int maxabsdir = I0;
	const int negmaxabsdir = rp->maxdirsign & (1 << 2);

	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	// BBox 셋팅
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	//m_bbox = m_Data->m_TriObjList[m_Data->m_TriOffList[baseOffset]]->m_BBox;
	//for (i = baseOffset; i < baseOffset+nObjs; i++) {
	//	int triID = m_Data->m_TriOffList[i];
	//	m_bbox += m_Data->m_TriObjList[triID]->m_BBox;
	//}

	//float bminf = m_bbox.getMin()[I0], bmaxf = m_bbox.getMax()[I0];
	float bminf = *(float*)&m_Data->m_TriOffList[baseOffset+I0], bmaxf = *(float*)&m_Data->m_TriOffList[baseOffset+3+I0];
	if (bminf == bmaxf) bminf -= 1.0f;							// bbox 가 ray 방향과 수직인 평면인 경우
	__m128 bmin =_mm_set_ps1(bminf), bmax =_mm_set_ps1(bmaxf);

	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	// frustum 의 y00~z11 의 좌표 결정
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	__m128 r0min[2], r0max[2], r1min[2], r1max[2], org0[3];
	union { float f[4]; __m128 v4; } orgmin; orgmin.v4 = posinf4;
	union { float f[4]; __m128 v4; } orgmax; orgmax.v4 = neginf4;

	for (j = 3; j >= 0; j--) {
		const __m128 di = rp->di[j].v4[I0];

		org0[0] = rp->o[j].x4; org0[1] = rp->o[j].y4; org0[2] = rp->o[j].z4;
		orgmin.v4 = _mm_min_ps(orgmin.v4 , org0[I0]);
		orgmax.v4 = _mm_max_ps(orgmax.v4,  org0[I0]);

		const __m128 dir0[] = {rp->d[j].v4[I1], rp->d[j].v4[I2]};
		const __m128 tentr = (bmin-org0[I0]) * di, texit = (bmax-org0[I0]) * di;
		if (j == 3) {
			r0max[0] = r0min[0] = org0[I1] + tentr * dir0[0];
			r0max[1] = r0min[1] = org0[I2] + tentr * dir0[1];
			r1max[0] = r1min[0] = org0[I1] + texit * dir0[0];
			r1max[1] = r1min[1] = org0[I2] + texit * dir0[1];
		} else {
			r0min[0] =_mm_min_ps (r0min[0] , org0[I1] + tentr * dir0[0]);
			r0max[0] =_mm_max_ps (r0max[0] , org0[I1] + tentr * dir0[0]);
			r0min[1] =_mm_min_ps (r0min[1] , org0[I2] + tentr * dir0[1]);
			r0max[1] =_mm_max_ps (r0max[1] , org0[I2] + tentr * dir0[1]);
			r1min[0] =_mm_min_ps (r1min[0] , org0[I1] + texit * dir0[0]);
			r1max[0] =_mm_max_ps (r1max[0] , org0[I1] + texit * dir0[0]);
			r1min[1] =_mm_min_ps (r1min[1] , org0[I2] + texit * dir0[1]);
			r1max[1] =_mm_max_ps (r1max[1] , org0[I2] + texit * dir0[1]);
		}
	}

	_MM_TRANSPOSE4_PS(r0min[0], r0min[1], r1min[0], r1min[1]);
	const __m128 emin = _mm_min_ps( _mm_min_ps(r0min[0], r0min[1]),
		_mm_min_ps(r1min[0], r1min[1]));
	_MM_TRANSPOSE4_PS(r0max[0], r0max[1], r1max[0], r1max[1]);
	const __m128 emax = _mm_max_ps( _mm_max_ps(r0max[0], r0max[1]),
		_mm_max_ps(r1max[0], r1max[1]));

	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	// frustum 4개 면
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	const __m128 mix1 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(3,2,1,0));
	const __m128 mix2 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(1,0,3,2));
	const __m128 xminmax =_mm_shuffle_ps(bmin, bmax, _MM_SHUFFLE(0,0,0,0));
	const __m128 xmaxmin =_mm_shuffle_ps(bmax, bmin, _MM_SHUFFLE(0,0,0,0));

	float term0 = bmaxf - bminf;
	term2 = mix1 - mix2;
	term1 = mix1*(xminmax - xmaxmin) - xminmax*term2;
	term2 *= _mm_set_ps1(1/term0); term1 *= _mm_set_ps1(1/term0);

#if 1
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	// 모서리 광선
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	const __m128 ext1 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(2,0,2,0)); // I1 values
	const __m128 ext2 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(3,1,3,1)); // I2 values

	__m128 entr[3], extr[3]; // 2 AA rectangles forming frustum
	entr[I0] = bmin;
	entr[I1] =_mm_shuffle_ps(ext1, ext1, _MM_SHUFFLE(0,2,2,0));
	entr[I2] =_mm_shuffle_ps(ext2, ext2, _MM_SHUFFLE(2,2,0,0));
	extr[I0] = bmax;
	extr[I1] =_mm_shuffle_ps(ext1, ext1, _MM_SHUFFLE(1,3,3,1));
	extr[I2] =_mm_shuffle_ps(ext2, ext2, _MM_SHUFFLE(3,3,1,1));

	_sse_2x2_rayfrustum	*frp	= &m_FrustomRayPk2x2[0];
	frp->rp.o.x4 = entr[0]; frp->rp.o.y4 = entr[1]; frp->rp.o.z4 = entr[2];
	frp->rp.d.x4 = extr[0] - entr[0];
	frp->rp.d.y4 = extr[1] - entr[1];
	frp->rp.d.z4 = extr[2] - entr[2];

	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	// Near / Far culling
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	__m128 di = sse_inverse(frp->rp.d.v4[I0]);
	orgmin.v4 = _mm_set1_ps(__min( __min(orgmin.f[0], orgmin.f[1]), __min(orgmin.f[1], orgmin.f[2])));
	orgmax.v4 = _mm_set1_ps(__max( __max(orgmax.f[0], orgmax.f[1]), __max(orgmax.f[1], orgmax.f[2])));

	if (negmaxabsdir) {
		frp->tfar = neginf4;	frp->torg0 = (orgmax.v4 - bmin) * di;	frp->torg1 = (orgmin.v4 - bmin) * di;
	} else {
		frp->tfar = posinf4;	frp->torg0 = (orgmin.v4 - bmin) * di;	frp->torg1 = (orgmax.v4 - bmin) * di;
	}
#endif 
}

bool SSERenderPipeline::Split_FCull_TriEdge( int &check_aperture, const TriAccel &acc, const int negmaxabsdir )
{
	_sse_2x2_rayfrustum	*frp	= &m_FrustomRayPk2x2[0];

	__m128 den;

	const unsigned int k	= acc.k;

	den	= frp->rp.d.v4[k] + _mm_set_ps1(acc.n_u) * frp->rp.d.v4[ku] + _mm_set_ps1(acc.n_v) * frp->rp.d.v4[kv];

	int densign = _mm_movemask_ps(den);
	if (densign == 0 || densign == 0xf ) {
		__m128 beam_tnom = _mm_set_ps1(acc.n_d) - 
				(frp->rp.o.v4[k] + _mm_set_ps1(acc.n_u) * frp->rp.o.v4[ku] + _mm_set_ps1(acc.n_v) * frp->rp.o.v4[kv]);
		if (_mm_movemask_ps(_mm_xor_ps(beam_tnom, den)) == 0xf) return false;

		const int flags = (negmaxabsdir)?0:0xf;
		__m128 t0 = beam_tnom * sse_inverse(den);
		if (_mm_movemask_ps(_mm_cmpgt_ps(t0, frp->tfar))  == flags) return false;
		if (_mm_movemask_ps(_mm_cmplt_ps(t0, frp->torg0)) == flags) return false;


		//den_lt_0 = _mm_cmplt_ps(den, _mm_setzero_ps());


		if ( 1 ) {
			RDATA lambda, mue;
			__m128 hu, hv;

			hu		= frp->rp.o.v4[ku] + t0 * frp->rp.d.v4[ku];
			hv		= frp->rp.o.v4[kv] + t0 * frp->rp.d.v4[kv];

			lambda.f	= hu * _mm_set_ps1(acc.b_nu) + hv * _mm_set_ps1(acc.b_nv);
			lambda.f	= lambda.f + _mm_set_ps1(acc.b_d);

			mue.f		= hu * _mm_set_ps1(acc.c_nu) + hv * _mm_set_ps1(acc.c_nv);
			mue.f		= mue.f    + _mm_set_ps1(acc.c_d);

			__m128 valid = _mm_cmpgt_ps(lambda.f, _mm_setzero_ps());
			if (_mm_movemask_ps(valid)==0) return false;

			__m128 valid0 = _mm_cmpgt_ps(mue.f, _mm_setzero_ps());
			if (_mm_movemask_ps(valid0)==0) return false;

			__m128 valid1 = _mm_cmple_ps(_mm_add_ps(lambda.f,mue.f), _mm_set1_ps(1));
			if (_mm_movemask_ps(valid1)==0) return false;

			valid =_mm_and_ps(valid, _mm_and_ps(valid0, valid1));
			check_aperture = (_mm_movemask_ps(valid) != 0xf);
		}


	}
	return true;
}



void SSERenderPipeline::Split_Isect4x4__PriRay( const KdTreeNode *node, int nIdx, int temp )
{
	int i, j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	if (nObjs == 0) return;

PriIsect_FtnCall_Count++;
#if FRUSTUM_CULLING == ON
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	// 절대값이 큰 축 결정
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	const int I0 = rp->maxdirsign & 3;
	const int I1 = (I0<2)? I0+1 : 0, I2 = (I1<2)? I1+1 : 0;
	const int maxabsdir = I0;
	const int negmaxabsdir = rp->maxdirsign & (1 << 2);

	__m128 term1, term2;
	Split_FCull_Init(term1, term2, baseOffset, nIdx);

	for (i = baseOffset+6; i < baseOffset+6+nObjs; i++) {
#else
	for (i = baseOffset; i < baseOffset+nObjs; i++) {
#endif
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		RMASK4 Mask_Hit;

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { continue; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (acc.isTransparent || !m_bBackFaceCulling) {
			for (j = 3; j >= 0; j--) {
				Mask_Hit.f[j] = rm->mask4[j];
			}
		} else {
			_sse_vec tri_N;
			tri_N.x4 = _mm_set1_ps(acc.N.x);
			tri_N.y4 = _mm_set1_ps(acc.N.y);
			tri_N.z4 = _mm_set1_ps(acc.N.z);
			for (j = 3; j >= 0; j--) {
				Mask_Hit.f[j] = _mm_and_ps(rm->mask4[j],
					_mm_cmpgt_ps(sse_vdot(rp->d[j], tri_N), _mm_setzero_ps()));
			}
		}

		PriIsect_TriChk_Count++;

	int check_aperture = 0;

#if FRUSTUM_CULLING == ON
		GPoint	A, B, C;
		m_Data->m_TriObjList[triID]->getPoint(A, B, C);
		const __m128 d0 = term1 + _mm_set1_ps(A[I0]) * term2 + _mm_set_ps(-A[I2], -A[I1], A[I2], A[I1]);
		const __m128 d1 = term1 + _mm_set1_ps(B[I0]) * term2 + _mm_set_ps(-B[I2], -B[I1], B[I2], B[I1]);
		const __m128 d2 = term1 + _mm_set1_ps(C[I0]) * term2 + _mm_set_ps(-C[I2], -C[I1], C[I2], C[I1]);
		if (_mm_movemask_ps(d0) & _mm_movemask_ps(d1) & _mm_movemask_ps(d2)) {
			PriIsect_FC_Cull_Count++;
			continue;
		}

		check_aperture = 2;
		if (!Split_FCull_TriEdge(check_aperture, acc, negmaxabsdir)) continue;
#endif

		acc.mbox = rp->RayId;

		// -----------------------------------------------------------------------------
		// ==>> Plane test (t)
		// -----------------------------------------------------------------------------
		RDATA4 f;
		if (!Split_Isect4x4_PlaneTest_PriRay(acc, nIdx, Mask_Hit, f, check_aperture)) continue;

		// -----------------------------------------------------------------------------
		// ==>> Triangle boundaries test (u, v)
		// -----------------------------------------------------------------------------
		RDATA4 lambda, mue;
		if (!Split_Isect4x4_TriUVTest_PriRay(acc, nIdx, Mask_Hit, f, lambda, mue, check_aperture)) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
		for (j = 3; j >= 0; j--) {
			is->u4[j]		= sse_update(lambda.f[j], is->u4[j],	Mask_Hit.f[j]);
			is->v4[j]		= sse_update(mue.f[j],    is->v4[j],	Mask_Hit.f[j]);
			is->dist4[j]	= sse_update(f.f[j],    is->dist4[j],	Mask_Hit.f[j]);
			is->tacc4[j]	= _mm_or_si128( _mm_andnot_si128( Mask_Hit.i[j], is->tacc4[j] ),
											_mm_and_si128   ( Mask_Hit.i[j], tacc4 ) );
		}
	}
}

void SSERenderPipeline::Split_Isect4x4__SecRay( const KdTreeNode *node, int nIdx )
{
	int i, j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		RMASK4 Mask_Hit;

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { continue; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (acc.isTransparent || !m_bBackFaceCulling) {
			for (j = 3; j >= 0; j--) {
				Mask_Hit.f[j] = rm->mask4[j];
			}
		} else {
			_sse_vec tri_N;
			tri_N.x4 = _mm_set1_ps(acc.N.x);
			tri_N.y4 = _mm_set1_ps(acc.N.y);
			tri_N.z4 = _mm_set1_ps(acc.N.z);
			for (j = 3; j >= 0; j--) {
				Mask_Hit.f[j] = _mm_and_ps(rm->mask4[j],
					_mm_cmpgt_ps(sse_vdot(rp->d[j], tri_N), _mm_setzero_ps()));
			}
		}

		// -----------------------------------------------------------------------------
		// ==>> Plane test (t)
		// -----------------------------------------------------------------------------
		RDATA4 f;
		if (!Split_Isect4x4_PlaneTest_SecRay(acc, nIdx, Mask_Hit, f)) continue;

		// -----------------------------------------------------------------------------
		// ==>> Triangle boundaries test (u, v)
		// -----------------------------------------------------------------------------
		RDATA4 lambda, mue;
		if (!Split_Isect4x4_TriUVTest_SecRay(acc, nIdx, Mask_Hit, f, lambda, mue)) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
		for (j = 3; j >= 0; j--) {
			is->u4[j]		= sse_update(lambda.f[j], is->u4[j],	Mask_Hit.f[j]);
			is->v4[j]		= sse_update(mue.f[j],    is->v4[j],	Mask_Hit.f[j]);
			is->dist4[j]	= sse_update(f.f[j],    is->dist4[j],	Mask_Hit.f[j]);
			is->tacc4[j]	= _mm_or_si128( _mm_andnot_si128( Mask_Hit.i[j], is->tacc4[j] ),
											_mm_and_si128   ( Mask_Hit.i[j], tacc4 ) );
		}
	}
}

void SSERenderPipeline::Split_Isect4x4__ShwRay( const KdTreeNode *node, int temp )
{
	int i, j;

	_sse_4x4_raypacket	*rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*rm	= &m_ShadowRMask4x4[0];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		RMASK4 Mask_Hit;

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { continue; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		// ---------------------------------------------------------------
		// 투명한 물체는 투과
		// ---------------------------------------------------------------
		if (acc.isTransparent) continue;
		for (j = 3; j >= 0; j--) {
			Mask_Hit.f[j] = rm->mask4[j];
		}

		// -----------------------------------------------------------------------------
		// ==>> Plane test (t)
		// -----------------------------------------------------------------------------
		RDATA4 f;
		if (!Split_Isect4x4_PlaneTest_ShwRay(acc, temp, Mask_Hit, f)) continue;

		// -----------------------------------------------------------------------------
		// ==>> Triangle boundaries test (u, v)
		// -----------------------------------------------------------------------------
		RDATA4 lambda, mue;
		if (!Split_Isect4x4_TriUVTest_ShwRay(acc, temp, Mask_Hit, f, lambda, mue)) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
		for (j = 3; j >= 0; j--) {
			is->u4[j]		= sse_update(lambda.f[j], is->u4[j],	Mask_Hit.f[j]);
			is->v4[j]		= sse_update(mue.f[j],    is->v4[j],	Mask_Hit.f[j]);
			is->dist4[j]	= sse_update(f.f[j],    is->dist4[j],	Mask_Hit.f[j]);
			is->tacc4[j]	= _mm_or_si128( _mm_andnot_si128( Mask_Hit.i[j], is->tacc4[j] ),
											_mm_and_si128   ( Mask_Hit.i[j], tacc4 ) );
		}
	}
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::Split_Trace4x4__PriRay( int nIdx, int temp )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	// ---------------------------------------------------------------------------
	// 유효한 Ray 없는 경우 return
	// ---------------------------------------------------------------------------
	if (_mm_movemask_ps(_mm_or_ps(_mm_or_ps(_mm_or_ps(rm->mask4[0], rm->mask4[1]), rm->mask4[2]), rm->mask4[3])) == 0) return;

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist4[0]	= sse_update(_mm_set1_ps(100000), is->dist4[0], rm->mask4[0]);
	is->dist4[1]	= sse_update(_mm_set1_ps(100000), is->dist4[1], rm->mask4[1]);
	is->dist4[2]	= sse_update(_mm_set1_ps(100000), is->dist4[2], rm->mask4[2]);
	is->dist4[3]	= sse_update(_mm_set1_ps(100000), is->dist4[3], rm->mask4[3]);
	is->tacc4[0]	= sse_update(_mm_setzero_si128(), is->tacc4[0], rm->imask4[0]);
	is->tacc4[1]	= sse_update(_mm_setzero_si128(), is->tacc4[1], rm->imask4[1]);
	is->tacc4[2]	= sse_update(_mm_setzero_si128(), is->tacc4[2], rm->imask4[2]);
	is->tacc4[3]	= sse_update(_mm_setzero_si128(), is->tacc4[3], rm->imask4[3]);

	// ray direction
	const unsigned int* ray_dir = &raydir[rp->RayWay][0][0];	// Get precomputed the traversal order (front/back)
																//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// Active Mask
	__m128 Mask_Active4[4] = { rm->mask4[0], rm->mask4[1], rm->mask4[2], rm->mask4[3] };

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	__m128 t_near4[4], t_far_4[4];
	t_near4[0] = t_near4[1]	= t_near4[2] = t_near4[3]	= _mm_setzero_ps();				//	_mm_setzero_ps();
	t_far_4[0] = t_far_4[1]	= t_far_4[2] = t_far_4[3]	= _mm_set1_ps(100000);			//	_mm_set_ps1(INFINITY);

	{
		__m128 scenebox_min_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.x);
		__m128 scenebox_min_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.y);
		__m128 scenebox_min_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.z);
		__m128 scenebox_max_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.x);
		__m128 scenebox_max_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.y);
		__m128 scenebox_max_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.z);

		for (j = 3; j >= 0; j--) {
			__m128 l1, l2;
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_x, rp->o[j].x4), rp->di[j].x4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_x, rp->o[j].x4), rp->di[j].x4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_y, rp->o[j].y4), rp->di[j].y4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_y, rp->o[j].y4), rp->di[j].y4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_z, rp->o[j].z4), rp->di[j].z4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_z, rp->o[j].z4), rp->di[j].z4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
		}
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const __m128 node_split4 = _mm_set_ps1(SPLIT_POS(*node));
			const unsigned int dim	= SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			__m128 d[4];
			
				d[0]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[0].v4[dim]), rp->di[0].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[0], d[0]), Mask_Active4[0]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[0], d[0]), Mask_Active4[0]));
				d[1]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[1].v4[dim]), rp->di[1].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[1], d[1]), Mask_Active4[1]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[1], d[1]), Mask_Active4[1]));
				d[2]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[2].v4[dim]), rp->di[2].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[2], d[2]), Mask_Active4[2]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[2], d[2]), Mask_Active4[2]));
				d[3]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[3].v4[dim]), rp->di[3].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[3], d[3]), Mask_Active4[3]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[3], d[3]), Mask_Active4[3]));

				node = BackSideSon;
				if (d_near == 0)	continue;	// traverse the back  child
				node = FrontSideSon;
				if (d_far == 0)		continue;	// traverse the front child

									if (IS_LEAF(*FrontSideSon) && OBJECT_SIZE(*FrontSideSon) == 0) {
										node = BackSideSon;
										for (j = 3; j >= 0; j--) {
											t_near4[j]		= _mm_max_ps(t_near4[j], d[j]);
											Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
										}
										continue;
									} else
									if (IS_LEAF(*BackSideSon) && OBJECT_SIZE(*BackSideSon) == 0) {
										node = FrontSideSon;
										for (j = 3; j >= 0; j--) {
											t_far_4[j]		= _mm_min_ps(t_far_4[j],d[j]);
											Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
										}
										continue;
									}

				for (j = 3; j >= 0; j--) {
					m_Stack4x4[stackIndex].t_far_4[j] = t_far_4[j];
					m_Stack4x4[stackIndex].t_near4[j] = _mm_max_ps(t_near4[j], d[j]);
					t_far_4[j]		= _mm_min_ps(t_far_4[j],d[j]);
					Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
				}
				m_Stack4x4[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		Split_Isect4x4__PriRay(node, nIdx, temp);

		// Termination test
		if ((_mm_movemask_ps( _mm_or_ps( _mm_or_ps( _mm_or_ps( 
			_mm_cmpgt_ps( is->dist4[0], t_far_4[0] ),	_mm_cmpgt_ps( is->dist4[1], t_far_4[1] ) ), 
			_mm_cmpgt_ps( is->dist4[2], t_far_4[2] ) ), _mm_cmpgt_ps( is->dist4[3], t_far_4[3] ) ) ) == 0) || (stackIndex == 0)) break;

		// Stack pop
		--stackIndex;
		node			= m_Stack4x4[stackIndex].node;
		t_near4[0]		= m_Stack4x4[stackIndex].t_near4[0];
		t_near4[1]		= m_Stack4x4[stackIndex].t_near4[1];
		t_near4[2]		= m_Stack4x4[stackIndex].t_near4[2];
		t_near4[3]		= m_Stack4x4[stackIndex].t_near4[3];
		t_far_4[0]		= m_Stack4x4[stackIndex].t_far_4[0];
		t_far_4[1]		= m_Stack4x4[stackIndex].t_far_4[1];
		t_far_4[2]		= m_Stack4x4[stackIndex].t_far_4[2];
		t_far_4[3]		= m_Stack4x4[stackIndex].t_far_4[3];
		Mask_Active4[0] = _mm_cmple_ps(t_near4[0], t_far_4[0]);
		Mask_Active4[1] = _mm_cmple_ps(t_near4[1], t_far_4[1]);
		Mask_Active4[2] = _mm_cmple_ps(t_near4[2], t_far_4[2]);
		Mask_Active4[3] = _mm_cmple_ps(t_near4[3], t_far_4[3]);
	}
}

void SSERenderPipeline::Split_Trace4x4__SecRay( int nIdx )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	// ---------------------------------------------------------------------------
	// 유효한 Ray 없는 경우 return
	// ---------------------------------------------------------------------------
	if (_mm_movemask_ps(_mm_or_ps(_mm_or_ps(_mm_or_ps(rm->mask4[0], rm->mask4[1]), rm->mask4[2]), rm->mask4[3])) == 0) return;

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist4[0]	= sse_update(_mm_set1_ps(100000), is->dist4[0], rm->mask4[0]);
	is->dist4[1]	= sse_update(_mm_set1_ps(100000), is->dist4[1], rm->mask4[1]);
	is->dist4[2]	= sse_update(_mm_set1_ps(100000), is->dist4[2], rm->mask4[2]);
	is->dist4[3]	= sse_update(_mm_set1_ps(100000), is->dist4[3], rm->mask4[3]);
	is->tacc4[0]	= sse_update(_mm_setzero_si128(), is->tacc4[0], rm->imask4[0]);
	is->tacc4[1]	= sse_update(_mm_setzero_si128(), is->tacc4[1], rm->imask4[1]);
	is->tacc4[2]	= sse_update(_mm_setzero_si128(), is->tacc4[2], rm->imask4[2]);
	is->tacc4[3]	= sse_update(_mm_setzero_si128(), is->tacc4[3], rm->imask4[3]);

	// ray direction
	const unsigned int* ray_dir = &raydir[rp->RayWay][0][0];	// Get precomputed the traversal order (front/back)
																//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// Active Mask
	__m128 Mask_Active4[4] = { rm->mask4[0], rm->mask4[1], rm->mask4[2], rm->mask4[3] };

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	__m128 t_near4[4], t_far_4[4];
	t_near4[0] = t_near4[1]	= t_near4[2] = t_near4[3]	= _mm_setzero_ps();				//	_mm_setzero_ps();
	t_far_4[0] = t_far_4[1]	= t_far_4[2] = t_far_4[3]	= _mm_set1_ps(100000);			//	_mm_set_ps1(INFINITY);

	{
		__m128 scenebox_min_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.x);
		__m128 scenebox_min_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.y);
		__m128 scenebox_min_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.z);
		__m128 scenebox_max_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.x);
		__m128 scenebox_max_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.y);
		__m128 scenebox_max_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.z);

		for (j = 3; j >= 0; j--) {
			__m128 l1, l2;
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_x, rp->o[j].x4), rp->di[j].x4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_x, rp->o[j].x4), rp->di[j].x4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_y, rp->o[j].y4), rp->di[j].y4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_y, rp->o[j].y4), rp->di[j].y4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_z, rp->o[j].z4), rp->di[j].z4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_z, rp->o[j].z4), rp->di[j].z4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
		}
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const __m128 node_split4 = _mm_set_ps1(SPLIT_POS(*node));
			const unsigned int dim	= SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			__m128 d[4];
			
				d[0]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[0].v4[dim]), rp->di[0].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[0], d[0]), Mask_Active4[0]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[0], d[0]), Mask_Active4[0]));
				d[1]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[1].v4[dim]), rp->di[1].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[1], d[1]), Mask_Active4[1]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[1], d[1]), Mask_Active4[1]));
				d[2]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[2].v4[dim]), rp->di[2].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[2], d[2]), Mask_Active4[2]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[2], d[2]), Mask_Active4[2]));
				d[3]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[3].v4[dim]), rp->di[3].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[3], d[3]), Mask_Active4[3]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[3], d[3]), Mask_Active4[3]));

				node = BackSideSon;
				if (d_near == 0)	continue;	// traverse the back  child
				node = FrontSideSon;
				if (d_far == 0)		continue;	// traverse the front child

									if (IS_LEAF(*FrontSideSon) && OBJECT_SIZE(*FrontSideSon) == 0) {
										node = BackSideSon;
										for (j = 3; j >= 0; j--) {
											t_near4[j]		= _mm_max_ps(t_near4[j], d[j]);
											Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
										}
										continue;
									} else
									if (IS_LEAF(*BackSideSon) && OBJECT_SIZE(*BackSideSon) == 0) {
										node = FrontSideSon;
										for (j = 3; j >= 0; j--) {
											t_far_4[j]		= _mm_min_ps(t_far_4[j],d[j]);
											Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
										}
										continue;
									}

				for (j = 3; j >= 0; j--) {
					m_Stack4x4[stackIndex].t_far_4[j] = t_far_4[j];
					m_Stack4x4[stackIndex].t_near4[j] = _mm_max_ps(t_near4[j], d[j]);
					t_far_4[j]		= _mm_min_ps(t_far_4[j],d[j]);
					Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
				}
				m_Stack4x4[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		Split_Isect4x4__SecRay(node, nIdx);

		// Termination test
		if ((_mm_movemask_ps( _mm_or_ps( _mm_or_ps( _mm_or_ps( 
			_mm_cmpgt_ps( is->dist4[0], t_far_4[0] ),	_mm_cmpgt_ps( is->dist4[1], t_far_4[1] ) ), 
			_mm_cmpgt_ps( is->dist4[2], t_far_4[2] ) ), _mm_cmpgt_ps( is->dist4[3], t_far_4[3] ) ) ) == 0) || (stackIndex == 0)) break;

		// Stack pop
		--stackIndex;
		node			= m_Stack4x4[stackIndex].node;
		t_near4[0]		= m_Stack4x4[stackIndex].t_near4[0];
		t_near4[1]		= m_Stack4x4[stackIndex].t_near4[1];
		t_near4[2]		= m_Stack4x4[stackIndex].t_near4[2];
		t_near4[3]		= m_Stack4x4[stackIndex].t_near4[3];
		t_far_4[0]		= m_Stack4x4[stackIndex].t_far_4[0];
		t_far_4[1]		= m_Stack4x4[stackIndex].t_far_4[1];
		t_far_4[2]		= m_Stack4x4[stackIndex].t_far_4[2];
		t_far_4[3]		= m_Stack4x4[stackIndex].t_far_4[3];
		Mask_Active4[0] = _mm_cmple_ps(t_near4[0], t_far_4[0]);
		Mask_Active4[1] = _mm_cmple_ps(t_near4[1], t_far_4[1]);
		Mask_Active4[2] = _mm_cmple_ps(t_near4[2], t_far_4[2]);
		Mask_Active4[3] = _mm_cmple_ps(t_near4[3], t_far_4[3]);
	}
}

void SSERenderPipeline::Split_Trace4x4__ShwRay( int temp )
{
	int j;

	_sse_4x4_raypacket	*rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*rm	= &m_ShadowRMask4x4[0];

	// ---------------------------------------------------------------------------
	// 유효한 Ray 없는 경우 return
	// ---------------------------------------------------------------------------
	//if (_mm_movemask_ps(_mm_or_ps(_mm_or_ps(_mm_or_ps(rm->mask4[0], rm->mask4[1]), rm->mask4[2]), rm->mask4[3])) == 0) return;

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist4[0]	= sse_update(_mm_set1_ps(100000), is->dist4[0], rm->mask4[0]);
	is->dist4[1]	= sse_update(_mm_set1_ps(100000), is->dist4[1], rm->mask4[1]);
	is->dist4[2]	= sse_update(_mm_set1_ps(100000), is->dist4[2], rm->mask4[2]);
	is->dist4[3]	= sse_update(_mm_set1_ps(100000), is->dist4[3], rm->mask4[3]);
	is->tacc4[0]	= sse_update(_mm_setzero_si128(), is->tacc4[0], rm->imask4[0]);
	is->tacc4[1]	= sse_update(_mm_setzero_si128(), is->tacc4[1], rm->imask4[1]);
	is->tacc4[2]	= sse_update(_mm_setzero_si128(), is->tacc4[2], rm->imask4[2]);
	is->tacc4[3]	= sse_update(_mm_setzero_si128(), is->tacc4[3], rm->imask4[3]);

	// ray direction
	const unsigned int* ray_dir = &raydir[rp->RayWay][0][0];	// Get precomputed the traversal order (front/back)
																//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// Active Mask
	__m128 Mask_Active4[4] = { rm->mask4[0], rm->mask4[1], rm->mask4[2], rm->mask4[3] };

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	__m128 t_near4[4], t_far_4[4];
	t_near4[0] = t_near4[1]	= t_near4[2] = t_near4[3]	= _mm_setzero_ps();				//	_mm_setzero_ps();
	t_far_4[0] = t_far_4[1]	= t_far_4[2] = t_far_4[3]	= _mm_set1_ps(100000);			//	_mm_set_ps1(INFINITY);

	{
		__m128 scenebox_min_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.x);
		__m128 scenebox_min_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.y);
		__m128 scenebox_min_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.z);
		__m128 scenebox_max_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.x);
		__m128 scenebox_max_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.y);
		__m128 scenebox_max_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.z);

		for (j = 3; j >= 0; j--) {
			__m128 l1, l2;
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_x, rp->o[j].x4), rp->di[j].x4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_x, rp->o[j].x4), rp->di[j].x4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_y, rp->o[j].y4), rp->di[j].y4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_y, rp->o[j].y4), rp->di[j].y4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_z, rp->o[j].z4), rp->di[j].z4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_z, rp->o[j].z4), rp->di[j].z4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
		}
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const __m128 node_split4 = _mm_set_ps1(SPLIT_POS(*node));
			const unsigned int dim	= SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			__m128 d[4];
			
				d[0]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[0].v4[dim]), rp->di[0].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[0], d[0]), Mask_Active4[0]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[0], d[0]), Mask_Active4[0]));
				d[1]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[1].v4[dim]), rp->di[1].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[1], d[1]), Mask_Active4[1]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[1], d[1]), Mask_Active4[1]));
				d[2]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[2].v4[dim]), rp->di[2].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[2], d[2]), Mask_Active4[2]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[2], d[2]), Mask_Active4[2]));
				d[3]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[3].v4[dim]), rp->di[3].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[3], d[3]), Mask_Active4[3]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[3], d[3]), Mask_Active4[3]));

				node = BackSideSon;
				if (d_near == 0)	continue;	// traverse the back  child
				node = FrontSideSon;
				if (d_far == 0)		continue;	// traverse the front child

									if (IS_LEAF(*FrontSideSon) && OBJECT_SIZE(*FrontSideSon) == 0) {
										node = BackSideSon;
										for (j = 3; j >= 0; j--) {
											t_near4[j]		= _mm_max_ps(t_near4[j], d[j]);
											Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
										}
										continue;
									} else
									if (IS_LEAF(*BackSideSon) && OBJECT_SIZE(*BackSideSon) == 0) {
										node = FrontSideSon;
										for (j = 3; j >= 0; j--) {
											t_far_4[j]		= _mm_min_ps(t_far_4[j],d[j]);
											Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
										}
										continue;
									}

				for (j = 3; j >= 0; j--) {
					m_Stack4x4[stackIndex].t_far_4[j] = t_far_4[j];
					m_Stack4x4[stackIndex].t_near4[j] = _mm_max_ps(t_near4[j], d[j]);
					t_far_4[j]		= _mm_min_ps(t_far_4[j],d[j]);
					Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
				}
				m_Stack4x4[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		Split_Isect4x4__ShwRay(node, 0);

		// Termination test
		if ((_mm_movemask_ps( _mm_or_ps( _mm_or_ps( _mm_or_ps( 
			_mm_cmpgt_ps( is->dist4[0], t_far_4[0] ),	_mm_cmpgt_ps( is->dist4[1], t_far_4[1] ) ), 
			_mm_cmpgt_ps( is->dist4[2], t_far_4[2] ) ), _mm_cmpgt_ps( is->dist4[3], t_far_4[3] ) ) ) == 0) || (stackIndex == 0)) break;

		// Stack pop
		--stackIndex;
		node			= m_Stack4x4[stackIndex].node;
		t_near4[0]		= m_Stack4x4[stackIndex].t_near4[0];
		t_near4[1]		= m_Stack4x4[stackIndex].t_near4[1];
		t_near4[2]		= m_Stack4x4[stackIndex].t_near4[2];
		t_near4[3]		= m_Stack4x4[stackIndex].t_near4[3];
		t_far_4[0]		= m_Stack4x4[stackIndex].t_far_4[0];
		t_far_4[1]		= m_Stack4x4[stackIndex].t_far_4[1];
		t_far_4[2]		= m_Stack4x4[stackIndex].t_far_4[2];
		t_far_4[3]		= m_Stack4x4[stackIndex].t_far_4[3];
		Mask_Active4[0] = _mm_cmple_ps(t_near4[0], t_far_4[0]);
		Mask_Active4[1] = _mm_cmple_ps(t_near4[1], t_far_4[1]);
		Mask_Active4[2] = _mm_cmple_ps(t_near4[2], t_far_4[2]);
		Mask_Active4[3] = _mm_cmple_ps(t_near4[3], t_far_4[3]);
	}
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void SSERenderPipeline::Split_Shading4x4 (const int nIdx) {

	int i, x, y;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	// --------------------------------------------------------------
	// 유효한 Ray 없는 경우 return
	// --------------------------------------------------------------
	union { __m128i iAvailMask4; __m128 fAvailMask4; };
	iAvailMask4 = _mm_or_si128(	_mm_or_si128(	_mm_or_si128(
		_mm_and_si128(rm->imask4[0], _mm_cmpgt_epi32(is->tacc4[0], _mm_setzero_si128())),
		_mm_and_si128(rm->imask4[1], _mm_cmpgt_epi32(is->tacc4[1], _mm_setzero_si128()))),
		_mm_and_si128(rm->imask4[2], _mm_cmpgt_epi32(is->tacc4[2], _mm_setzero_si128()))),
		_mm_and_si128(rm->imask4[3], _mm_cmpgt_epi32(is->tacc4[3], _mm_setzero_si128())));
	if (_mm_movemask_ps(fAvailMask4) == 0) return;

	_sse_vec hit_p[4];
	int n_refl = 0;
	int n_refr = 0;

	_sse_vec global_ambt;

	_sse_vec	mat_cAmbt[4], mat_cDiff[4], mat_cSpec[4], mat_cEmit[4];
	_sse_float	mat_fRough[4];
	_sse_float	mat_fRefl[4], mat_fRefr[4], mat_fRIdx[4];
	_sse_vec	mat_cTex[4];
	_sse_uint	obj_num[4];

	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	// Setup
	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	Split_Shading4x4__Setup(nIdx, hit_p, global_ambt, mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit,
		mat_fRough, mat_fRefl, mat_fRefr, mat_fRIdx, mat_cTex, obj_num, n_refl, n_refr);

	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	// Local Shading
	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	if ( m_bIsEnableLocalShading ) {
		Split_Shading4x4__LocalShading(nIdx, hit_p, global_ambt, mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit,
			mat_fRough, mat_fRefl, mat_fRefr, mat_fRIdx, mat_cTex, obj_num);
	} else {
		i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
			if (rm->mask[i])
			is->color[i] = GColor(mat_cTex[y].x[x], mat_cTex[y].y[x], mat_cTex[y].z[x]);
		}
	}

	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	// Secondary Ray Generation
	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	if (nIdx < m_iMaxReflectionDepth) {
		Split_Shading4x4_RayGeneration_SecRay(nIdx, hit_p,
		mat_fRefl, mat_fRefr, mat_fRIdx, mat_cTex, n_refl, n_refr);
	}
}

void SSERenderPipeline::Split_Shading4x4__Setup (const int nIdx, _sse_vec *hit_p,
	_sse_vec &global_ambt,
	_sse_vec *mat_cAmbt, _sse_vec *mat_cDiff, _sse_vec *mat_cSpec, _sse_vec *mat_cEmit,
	_sse_float *mat_fRough,	_sse_float *mat_fRefl, _sse_float *mat_fRefr, _sse_float *mat_fRIdx,
	_sse_vec *mat_cTex,
	_sse_uint *obj_num,
	int &n_refl, int &n_refr)
{
	int i, x, y;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	hit_p[0].x4 = _mm_add_ps(rp->o[0].x4, _mm_mul_ps(rp->d[0].x4, is->dist4[0]));
	hit_p[0].y4 = _mm_add_ps(rp->o[0].y4, _mm_mul_ps(rp->d[0].y4, is->dist4[0]));
	hit_p[0].z4 = _mm_add_ps(rp->o[0].z4, _mm_mul_ps(rp->d[0].z4, is->dist4[0]));
	hit_p[1].x4 = _mm_add_ps(rp->o[1].x4, _mm_mul_ps(rp->d[1].x4, is->dist4[1]));
	hit_p[1].y4 = _mm_add_ps(rp->o[1].y4, _mm_mul_ps(rp->d[1].y4, is->dist4[1]));
	hit_p[1].z4 = _mm_add_ps(rp->o[1].z4, _mm_mul_ps(rp->d[1].z4, is->dist4[1]));
	hit_p[2].x4 = _mm_add_ps(rp->o[2].x4, _mm_mul_ps(rp->d[2].x4, is->dist4[2]));
	hit_p[2].y4 = _mm_add_ps(rp->o[2].y4, _mm_mul_ps(rp->d[2].y4, is->dist4[2]));
	hit_p[2].z4 = _mm_add_ps(rp->o[2].z4, _mm_mul_ps(rp->d[2].z4, is->dist4[2]));
	hit_p[3].x4 = _mm_add_ps(rp->o[3].x4, _mm_mul_ps(rp->d[3].x4, is->dist4[3]));
	hit_p[3].y4 = _mm_add_ps(rp->o[3].y4, _mm_mul_ps(rp->d[3].y4, is->dist4[3]));
	hit_p[3].z4 = _mm_add_ps(rp->o[3].z4, _mm_mul_ps(rp->d[3].z4, is->dist4[3]));

	n_refl = 0;
	n_refr = 0;
	int n_hit = 0;
	int n_tex = 0;

	GColor global_ambient = m_globalAmbient;
	global_ambt  = sse_vset1(global_ambient.r, global_ambient.g, global_ambient.b);


#if 1
	__m128i _test_iTriID4 = _mm_set1_epi32(is->tacc[0]-1);
	union { __m128i iTemp4; __m128 fTemp4; };
	union { __m128i iTriID4[4]; unsigned int iTriID[16]; __m128 fTriID4[4]; float fTriID[16]; };

	int nCoherenceChk  = 0xf;
	for (y = 3; y >= 0; y--) {
		iTriID4[y] = _mm_sub_epi32(is->tacc4[y], _mm_set1_epi32(1));
		iTemp4 = _mm_cmpeq_epi32(iTriID4[y], _test_iTriID4);
		nCoherenceChk &= _mm_movemask_ps(fTemp4);
	}

	if (nCoherenceChk == 0xf && rm->mask[0] && is->tacc[0] != 0) {
		const int triID      = iTriID[0];
		GObject   *pObject   = m_Data->m_TriObjList[triID]->m_pObject;
		GMaterial *pMaterial = pObject->getMaterial();

		const GColor ambient	= pMaterial->m_Ambient;
		const GColor diffuse	= pMaterial->m_Diffuse;
		const GColor specular	= pMaterial->m_Specular;
		const GColor emission	= pMaterial->m_Emission;
		const float  reflection	= pMaterial->m_fReflection;
		const float  refraction	= pMaterial->m_fTransparency;
		const float  refrIndex	= pMaterial->m_fRefractionIndex;
		const UINT   object_num	= pObject->m_iObjectNumber;

		// Get object color
		GTexture  *pTexture = NULL;
		GColor texColor;
		if ( m_bIsUseTexture ) {
			pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
		}

		// 16개가 다 같으니까 한번만 불르면 되는데, 왜 4번 불러?????????????

		for (y = 3; y >= 0; y--) {
				is->ads.pri_oid4[y] = sse_update( _mm_set1_epi32(object_num), is->ads.pri_oid4[y], rm->imask4[y]);

				mat_cAmbt[y].x4 = _mm_set1_ps(ambient.r);  mat_cDiff[y].x4 = _mm_set1_ps(diffuse.r);  mat_cSpec[y].x4 = _mm_set1_ps(specular.r);  mat_cEmit[y].x4 = _mm_set1_ps(emission.r);
				mat_cAmbt[y].y4 = _mm_set1_ps(ambient.g);  mat_cDiff[y].y4 = _mm_set1_ps(diffuse.g);  mat_cSpec[y].y4 = _mm_set1_ps(specular.g);  mat_cEmit[y].y4 = _mm_set1_ps(emission.g);
				mat_cAmbt[y].z4 = _mm_set1_ps(ambient.b);  mat_cDiff[y].z4 = _mm_set1_ps(diffuse.b);  mat_cSpec[y].z4 = _mm_set1_ps(specular.b);  mat_cEmit[y].z4 = _mm_set1_ps(emission.b);
				mat_fRough[y].v4 = _mm_set1_ps(0.7);//pMaterial->getRoughness());
				mat_fRefl[y].v4 = _mm_set1_ps(reflection);
				mat_fRefr[y].v4 = _mm_set1_ps(refraction);
				mat_fRIdx[y].v4 = _mm_set1_ps(refrIndex);
				obj_num[y].v4 = _mm_set1_epi32(object_num);

				__m128 beta  = is->u4[y];
				__m128 gamma = is->v4[y];
				__m128 alpha = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1), beta), gamma);

				_sse_vec normal;
				m_Data->m_TriObjList[triID]->calBarycentricNormal( alpha, beta, gamma, &normal );
				is->n[y].x4 = normal.x4;
				is->n[y].y4 = normal.y4;
				is->n[y].z4 = normal.z4;

				if (pTexture && pTexture->isLoaded()) {
					is->ads.pri_texture4[y] = _mm_set1_epi32(1);
					__m128 uv[2];
					m_Data->m_TriObjList[triID]->calBarycentricUV( alpha, beta, gamma, uv );
					pTexture->getTexel4( uv, mat_cTex[y] );
					n_tex++;
				} else {
					is->ads.pri_texture4[y] = _mm_set1_epi32(0);
					mat_cTex[y] = mat_cDiff[y];
				}
		}

		if (m_RunStatics == 1) {
			int mask = 0;
			mask = _mm_movemask_ps(rm->mask4[0]);	n_hit += ((mask & 1) + ((mask & 2) >> 1) + ((mask & 4) >> 2) + ((mask & 8) >> 3));
			mask = _mm_movemask_ps(rm->mask4[1]);	n_hit += ((mask & 1) + ((mask & 2) >> 1) + ((mask & 4) >> 2) + ((mask & 8) >> 3));
			mask = _mm_movemask_ps(rm->mask4[2]);	n_hit += ((mask & 1) + ((mask & 2) >> 1) + ((mask & 4) >> 2) + ((mask & 8) >> 3));
			mask = _mm_movemask_ps(rm->mask4[3]);	n_hit += ((mask & 1) + ((mask & 2) >> 1) + ((mask & 4) >> 2) + ((mask & 8) >> 3));
			n_refr = (refraction>0)?n_hit:0;
			n_refl = (reflection>0)?n_hit:0;
		} else {
			n_hit = 16;
			n_refr = (refraction>0)?n_hit:0;
			n_refl = (reflection>0)?n_hit:0;
		}
	} else {	// triangle coherence : not case
#else
	{
#endif
		
	i = 15;
	for ( y = 3; y >= 0; y-- ) {
	for ( x = 3; x >= 0; x-- ) {
		if (rm->mask[i] && is->tacc[i] != 0) {
				const int triID      = is->tacc[i]-1;
				GObject   *pObject   = m_Data->m_TriObjList[triID]->m_pObject;
				GMaterial *pMaterial = pObject->getMaterial();

				const GColor ambient	= pMaterial->m_Ambient;
				const GColor diffuse	= pMaterial->m_Diffuse;
				const GColor specular	= pMaterial->m_Specular;
				const GColor emission	= pMaterial->m_Emission;
				const float  reflection	= pMaterial->m_fReflection;
				const float  refraction	= pMaterial->m_fTransparency;
				const float  refrIndex	= pMaterial->m_fRefractionIndex;
				const UINT   object_num	= pObject->m_iObjectNumber;

				is->ads.pri_oid[i] = object_num;

				mat_cAmbt[y].x[x] = ambient.r;  mat_cDiff[y].x[x] = diffuse.r;  mat_cSpec[y].x[x] = specular.r;  mat_cEmit[y].x[x] = emission.r;
				mat_cAmbt[y].y[x] = ambient.g;  mat_cDiff[y].y[x] = diffuse.g;  mat_cSpec[y].y[x] = specular.g;  mat_cEmit[y].y[x] = emission.g;
				mat_cAmbt[y].z[x] = ambient.b;  mat_cDiff[y].z[x] = diffuse.b;  mat_cSpec[y].z[x] = specular.b;  mat_cEmit[y].z[x] = emission.b;
				mat_fRough[y].f[x] = 0.7;//pMaterial->getRoughness();
				mat_fRefl[y].f[x] = reflection;
				mat_fRefr[y].f[x] = refraction;
				mat_fRIdx[y].f[x] = refrIndex;
				obj_num[y].f[x] = object_num;

				n_refl += ((reflection > 0)?1:0);
				n_refr += ((refraction > 0)?1:0);

				GVector N = m_Data->m_TriObjList[triID]->calBarycentricNormal(1-is->u[i]-is->v[i], is->u[i], is->v[i]);
				is->n[y].x[x] = N.x;
				is->n[y].y[x] = N.y;
				is->n[y].z[x] = N.z;

				// Get object color
				GColor texColor;
				if ( m_bIsUseTexture ) {
					GTexture  *pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
					if (pTexture && pTexture->isLoaded()) {
						is->ads.pri_texture[i] = 1;
						GPoint point = m_Data->m_TriObjList[triID]->calBarycentricUV( 1-is->u[i]-is->v[i], is->u[i], is->v[i] );
						float u = point.x;
						float v = point.y;
						texColor = pTexture->getTexel( u, v );
						n_tex++;
					} else {
						is->ads.pri_texture[i] = 0;
						texColor = diffuse;
					}
				} else {
					texColor = diffuse;
				}
				mat_cTex[y].x[x] = texColor.r;
				mat_cTex[y].y[x] = texColor.g;
				mat_cTex[y].z[x] = texColor.b;

				n_hit++;
		}
		i--;
	}
	}


	} // triangle coherence : if end

	// 통계치
	if (m_RunStatics == 1) {
		if (rp->Depth == 0) {
			m_pf_Hit_SpecPnt_PR += (n_refl + n_refr);
			m_pf_Hit_DiffPnt_PR += (n_hit - n_refl - n_refr);
			m_pf_TexRef_PR += n_tex;
		} else {
			m_pf_Hit_SpecPnt_RR += (n_refl + n_refr);
			m_pf_Hit_DiffPnt_RR += (n_hit - n_refl - n_refr);
			m_pf_TexRef_RR += n_tex;
		}
	}
}

void SSERenderPipeline::Split_Shading4x4__LocalShading (const int nIdx, _sse_vec *hit_p,
	_sse_vec  &global_ambt,
	_sse_vec *mat_cAmbt, _sse_vec *mat_cDiff, _sse_vec *mat_cSpec, _sse_vec *mat_cEmit,
	_sse_float *mat_fRough,	_sse_float *mat_fRefl, _sse_float *mat_fRefr, _sse_float *mat_fRIdx,
	_sse_vec *mat_cTex,
	_sse_uint *obj_num)
{

	int i, x, y;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	int _pf_shading = 0;

	_sse_vec oColor[4];
	_sse_vec N[4], R[4], L[4];

	union { __m128 shadingmask[4]; __m128i ishadingmask[4]; };			// Ray Hit(o) & RayMask(o)
	union { __m128 islightmask[4]; __m128i iislightmask[4]; };			// Ray Hit object is current light?
	union { __m128 noshadowmask[4]; __m128i inoshadowmask[4];   };		// No  shadow
	union { __m128 yeshadowmask[4]; __m128i iyeshadowmask[4]; };		// Yes shadow
	union { __m128 isisectmask;    __m128i iisisectmask; };				// ShadowRay Hit(o)
	union { __m128 noisectmask;    __m128i inoisectmask; };				// ShadowRay Hit(x)
	union { __m128 shadowmask[4];  __m128i ishadowmask[4]; };			// Yes shaodw

	union { __m128 _pf_shadingmask[4];  __m128i _pf_ishadingmask[4]; };	// Shading Point
	union { __m128 _pf_shadowmask[4];   __m128i _pf_ishadowmask[4]; };	// Shadow  Point

	for (i = 3; i >= 0; i--) {
		ishadingmask[i] = _mm_cmpgt_epi32(is->tacc4[i], _mm_setzero_si128());
		shadingmask[i]  = _mm_and_ps(shadingmask[i], rm->mask4[i]);
		shadowmask[i]   = _mm_setzero_ps();
		_pf_shadingmask[i]	= _mm_setzero_ps();
		_pf_shadowmask[i]	= _mm_setzero_ps();
	}

	_sse_4x4_raypacket	*shadow_rp;
	_sse_4x4_isect		*shadow_is;
	_sse_4x4_raymask	*shadow_rm;
	if ( m_bIsEnableShadow ) {
		shadow_rp	= &m_ShadowRayPk4x4[0];
		shadow_is	= &m_ShadowIsect4x4[0];
		shadow_rm	= &m_ShadowRMask4x4[0];
		memcpy(shadow_rp->o, hit_p, sizeof(_sse_vec)*4);
	}

	for (i = 3; i >= 0; i--) {
		// Shading 에서, 투명한 물체일때, Normal 과 dir 의 dot 이 < 0 이라면 normal 을 뒤짚는다.
		__m128 mask = _mm_and_ps(
				_mm_cmpgt_ps(mat_fRefr[i].v4, _mm_setzero_ps()),
				_mm_cmpgt_ps(sse_vdot(is->n[i], rp->d[i]), _mm_setzero_ps()));
		N[i] = sse_vupdate(sse_vsigned(is->n[i]), is->n[i], mask);
		R[i] = sse_vnorm(sse_vadd(sse_vmul(_mm_mul_ps(_mm_set1_ps(-2),  sse_vdot(N[i], rp->d[i])), N[i]), rp->d[i]));

		// Background color
		oColor[i] = sse_vset1(0.0f);

		// Ambient color
		oColor[i] = sse_vupdate(sse_vmul(global_ambt, mat_cAmbt[i]), oColor[i], shadingmask[i]);

		// Emission color
		oColor[i] = sse_vupdate(sse_vadd(oColor[i], mat_cEmit[i]) ,oColor[i], shadingmask[i]);
	}

	// Diffuse & Specular color
	const vector<GLight*>* pLightList = m_Scene->getLightList();
	for ( int lx = 0; lx < (int) pLightList->size(); ++lx ) {	GLight* pLight = (*pLightList)[ lx ];
		// Point Light 만 일단 지원
		if ( pLight->getLightType() != typePointLight )  continue;

		GColor   lightColor = pLight->getLightColor();
		_sse_vec lColor     = sse_vset1(lightColor.r, lightColor.g, lightColor.b);
		GPoint   lightPos   = pLight->getPosition();
		_sse_vec lPos       = sse_vset1(lightPos.x, lightPos.y, lightPos.z);

		// 광원 자기자신인 경우
		for (i = 3; i >= 0; i--) {
			iislightmask[i] = _mm_cmpeq_epi32(obj_num[i].v4, _mm_set1_epi32(pLight->m_iObjectNumber));
			oColor[i] = sse_vupdate(sse_vadd(oColor[i], sse_vmul(lColor, sse_vset1(pLight->getIntensity()))), oColor[i], _mm_and_ps(islightmask[i], shadingmask[i]));
		}

		// 그림자 확인
		if ( m_bIsEnableShadow ) {
			Split_Shading4x4_RayGeneration_ShwRay(hit_p, &lightPos, shadingmask);

			for (i = 3; i >= 0; i--) {
				__m128 lDist = sse_vlength(sse_vsub(lPos, hit_p[i]));

				// shadow 관련 visible 조건
				// 1) shadingmask[i]           : 물체와 교점있는 것
				// 2) shadow_is->tacc4[i] > 0  : shadow ray 가 교차점이 뒤에 존재하거나 아니면 거리가 거의 가깝거나
				iisisectmask = _mm_cmpgt_epi32(shadow_is->tacc4[i], _mm_setzero_si128());
				inoisectmask = _mm_cmpeq_epi32(shadow_is->tacc4[i], _mm_setzero_si128());

				noshadowmask[i] = _mm_or_ps(
						noisectmask,
						_mm_or_ps(
							_mm_cmplt_ps(_mm_abs_ps(_mm_sub_ps(lDist, shadow_is->dist4[i])), _mm_set1_ps(1.f*EPSILON)),
							_mm_cmpgt_ps(shadow_is->dist4[i], lDist)));
				yeshadowmask[i] = _mm_andnot_ps(noshadowmask[i], _mm_andnot_ps(islightmask[i], shadingmask[i]));
				noshadowmask[i] = _mm_and_ps(noshadowmask[i], _mm_andnot_ps(islightmask[i], shadingmask[i]));

				// Phong shading
				L[i] = sse_vnorm(sse_vsub(lPos, hit_p[i]));

				oColor[i] = sse_vupdate(sse_vadd(oColor[i], sse_vadd(
					sse_vmul(sse_vmul(mat_cTex[i],  lColor), sse_vset1(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], N[i])))),
					sse_vmul(sse_vmul(mat_cSpec[i], lColor), sse_vset1(sse_pow_Schlick(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], R[i])), mat_fRough[i].v4)))
						)), oColor[i], noshadowmask[i]);
				// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
				//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );

				_pf_shadingmask[i] = _mm_or_ps(_pf_shadingmask[i], noshadowmask[i]);
				_pf_shadowmask[i]  = _mm_or_ps(_pf_shadowmask[i],  yeshadowmask[i]);
				shadowmask[i] = _mm_add_ps(shadowmask[i], _mm_and_ps(_mm_set1_ps(1), yeshadowmask[i]));
			}
			_pf_shading+=4;
		} else {
			for (i = 3; i >= 0; i--) {
				noshadowmask[i] = _mm_andnot_ps(islightmask[i], shadingmask[i]);

				// Phong shading
				L[i] = sse_vnorm(sse_vsub(lPos, hit_p[i]));

				oColor[i] = sse_vupdate(sse_vadd(oColor[i], sse_vadd(
					sse_vmul(sse_vmul(mat_cTex[i],  lColor), sse_vset1(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], N[i])))),
					sse_vmul(sse_vmul(mat_cSpec[i], lColor), sse_vset1(sse_pow_Schlick(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], R[i])), mat_fRough[i].v4)))
						)), oColor[i], noshadowmask[i]);
				// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
				//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );

				_pf_shadingmask[i] = _mm_or_ps(_pf_shadingmask[i], noshadowmask[i]);
				shadowmask[i] = _mm_setzero_ps();
			}
			_pf_shading+=4;
		}
	}


	if ( rp->Depth < m_iMaxReflectionDepth ) {
		for (i = 3; i >= 0; i--) {
			oColor[i] = sse_vupdate(
				sse_vmul(oColor[i], sse_vsub(sse_vsub(sse_vset1(1.0f), sse_vset1(mat_fRefl[i].v4)), sse_vset1(mat_fRefr[i].v4))),
				oColor[i], shadingmask[i]);
		}
	}

	// 추후 adaptive samplling 용
	//if ( rp->Depth < 2 ) {
	//	for (i = 3; i >= 0; i--) {
	//		is->ads.pri_shadowf4[i] = _mm_and_ps(shadowmask[i], rm->mask4[i]);
	//	}
	//}

	i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
		if (rm->mask[i])
		is->color[i] = GColor(oColor[y].x[x], oColor[y].y[x], oColor[y].z[x]);
	}

	if ( m_RunStatics == 1 ) {
		int mask;
		int _pf_shading_pnt = 0;
		int _pf_shadow_pnt  = 0;
		for (i = 3; i >= 0; i--) {
			mask = _mm_movemask_ps(_pf_shadingmask[i]);
			_pf_shading_pnt += ((mask & 1) + ((mask & 2) >> 1) + ((mask & 4) >> 2) + ((mask & 8) >> 3));
			mask = _mm_movemask_ps(_pf_shadowmask[i]);
			_pf_shadow_pnt  += ((mask & 1) + ((mask & 2) >> 1) + ((mask & 4) >> 2) + ((mask & 8) >> 3));
		}
		if ( rp->Depth == 0 ) {
			m_pf_Hit_ShadCnt_PR += _pf_shading;			// CMI-6, _pf_shading 으로 연산횟수 카운트
			m_pf_Hit_ShadPnt_PR += _pf_shading_pnt;		// CMI-6, masking 으로 직접 조사
			m_pf_Hit_ShwPnt_ALL += _pf_shadow_pnt;		// RPI-9, masking 으로 직접 조사
		} else {
			m_pf_Hit_ShadCnt_RR += _pf_shading;			// CMI-6, _pf_shading 으로 연산횟수 카운트
			m_pf_Hit_ShadPnt_RR += _pf_shading_pnt;		// CMI-6, masking 으로 직접 조사
		}
		
		m_pf_Hit_ShwCnt_PR;								// 현재 사용되지 않음
	}
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// -----------------------------------------------------------
// SSERenderPipeline::RayGeneration
//		Generate & Trace Ray
// -----------------------------------------------------------
void SSERenderPipeline::Split_Shading4x4_RayGeneration_SecRay (const int nIdx, _sse_vec *hit_p,
	_sse_float *mat_fRefl, _sse_float *mat_fRefr, _sse_float *mat_fRIdx,
	_sse_vec *mat_cTex,	int n_refl, int n_refr)
{
	int i, x, y;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	if (rp->Depth < m_iMaxReflectionDepth) {
		// ---------------------------------------------------------------------------
		// reflection
		// ---------------------------------------------------------------------------
		if (n_refl > 0) {
			__m128 dot_i4[4];

			const int nNextIdx = nIdx+1;

			_sse_4x4_raypacket	*refl_rp	= &m_RayPk4x4[nNextIdx];
			_sse_4x4_isect		*refl_is	= &m_Isect4x4[nNextIdx];
			_sse_4x4_raymask	*refl_rm	= &m_RMask4x4[nNextIdx];

			dot_i4[0] = sse_vdot(rp->d[0], is->n[0]);
			dot_i4[1] = sse_vdot(rp->d[1], is->n[1]);
			dot_i4[2] = sse_vdot(rp->d[2], is->n[2]);
			dot_i4[3] = sse_vdot(rp->d[3], is->n[3]);

			refl_rp->d[0].x4 = _mm_sub_ps(rp->d[0].x4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[0], is->n[0].x4)));
			refl_rp->d[0].y4 = _mm_sub_ps(rp->d[0].y4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[0], is->n[0].y4)));
			refl_rp->d[0].z4 = _mm_sub_ps(rp->d[0].z4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[0], is->n[0].z4)));
			refl_rp->d[1].x4 = _mm_sub_ps(rp->d[1].x4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[1], is->n[1].x4)));
			refl_rp->d[1].y4 = _mm_sub_ps(rp->d[1].y4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[1], is->n[1].y4)));
			refl_rp->d[1].z4 = _mm_sub_ps(rp->d[1].z4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[1], is->n[1].z4)));
			refl_rp->d[2].x4 = _mm_sub_ps(rp->d[2].x4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[2], is->n[2].x4)));
			refl_rp->d[2].y4 = _mm_sub_ps(rp->d[2].y4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[2], is->n[2].y4)));
			refl_rp->d[2].z4 = _mm_sub_ps(rp->d[2].z4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[2], is->n[2].z4)));
			refl_rp->d[3].x4 = _mm_sub_ps(rp->d[3].x4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[3], is->n[3].x4)));
			refl_rp->d[3].y4 = _mm_sub_ps(rp->d[3].y4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[3], is->n[3].y4)));
			refl_rp->d[3].z4 = _mm_sub_ps(rp->d[3].z4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4[3], is->n[3].z4)));
			Split_InitPkt4x4( nNextIdx );

			refl_rp->o[0].x4 = _mm_add_ps(hit_p[0].x4, _mm_mul_ps(refl_rp->d[0].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[0].y4 = _mm_add_ps(hit_p[0].y4, _mm_mul_ps(refl_rp->d[0].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[0].z4 = _mm_add_ps(hit_p[0].z4, _mm_mul_ps(refl_rp->d[0].z4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[1].x4 = _mm_add_ps(hit_p[1].x4, _mm_mul_ps(refl_rp->d[1].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[1].y4 = _mm_add_ps(hit_p[1].y4, _mm_mul_ps(refl_rp->d[1].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[1].z4 = _mm_add_ps(hit_p[1].z4, _mm_mul_ps(refl_rp->d[1].z4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[2].x4 = _mm_add_ps(hit_p[2].x4, _mm_mul_ps(refl_rp->d[2].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[2].y4 = _mm_add_ps(hit_p[2].y4, _mm_mul_ps(refl_rp->d[2].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[2].z4 = _mm_add_ps(hit_p[2].z4, _mm_mul_ps(refl_rp->d[2].z4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[3].x4 = _mm_add_ps(hit_p[3].x4, _mm_mul_ps(refl_rp->d[3].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[3].y4 = _mm_add_ps(hit_p[3].y4, _mm_mul_ps(refl_rp->d[3].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o[3].z4 = _mm_add_ps(hit_p[3].z4, _mm_mul_ps(refl_rp->d[3].z4, _mm_set1_ps(RAY_START_EPSILON)));

			refl_rp->Depth = rp->Depth+1;

			_sse_4x4_raymask SecRay_mask4;
			SecRay_mask4.mask4[0] = _mm_and_ps( _mm_and_ps( rm->mask4[0], (__m128&)_mm_cmpgt_epi32(is->tacc4[0], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefl[0].v4, _mm_setzero_ps()));
			SecRay_mask4.mask4[1] = _mm_and_ps( _mm_and_ps( rm->mask4[1], (__m128&)_mm_cmpgt_epi32(is->tacc4[1], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefl[1].v4, _mm_setzero_ps()));
			SecRay_mask4.mask4[2] = _mm_and_ps( _mm_and_ps( rm->mask4[2], (__m128&)_mm_cmpgt_epi32(is->tacc4[2], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefl[2].v4, _mm_setzero_ps()));
			SecRay_mask4.mask4[3] = _mm_and_ps( _mm_and_ps( rm->mask4[3], (__m128&)_mm_cmpgt_epi32(is->tacc4[3], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefl[3].v4, _mm_setzero_ps()));

			unsigned int b;
			if (refl_rp->IsCoherent()) {
				refl_rp->RayWay = (refl_rp->xmask & 1) + (refl_rp->ymask & 2) + (refl_rp->zmask & 4);
				refl_rm->mask4[0] = SecRay_mask4.mask4[0];
				refl_rm->mask4[1] = SecRay_mask4.mask4[1];
				refl_rm->mask4[2] = SecRay_mask4.mask4[2];
				refl_rm->mask4[3] = SecRay_mask4.mask4[3];
				Split_Trace4x4__SecRay(nNextIdx);
				Split_Shading4x4(nNextIdx);
				i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
					if (refl_is->tacc[i] && refl_rm->mask[i]) {
						is->color[i] += mat_fRefl[y].f[x] * refl_is->color[i] * GColor(mat_cTex[y].x[x], mat_cTex[y].y[x], mat_cTex[y].z[x]);
					}
				}
			} else {
				// detect octants and render one by one
				bool b_quad[8] = { false, false, false, false, false, false, false, false };
				for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
					int q = ((refl_rp->xmask & b)?1:0) + ((refl_rp->ymask & b)?2:0) + ((refl_rp->zmask & b)?4:0);
					b_quad[q] = true;
				}
				for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
					for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
						int q = ((refl_rp->xmask & b)?1:0) + ((refl_rp->ymask & b)?2:0) + ((refl_rp->zmask & b)?4:0);
						if (q == cq) refl_rm->mask[i] = 0xffffffff; else refl_rm->mask[i] = 0;
					}
					refl_rm->mask4[0] = _mm_and_ps( refl_rm->mask4[0], SecRay_mask4.mask4[0] );
					refl_rm->mask4[1] = _mm_and_ps( refl_rm->mask4[1], SecRay_mask4.mask4[1] );
					refl_rm->mask4[2] = _mm_and_ps( refl_rm->mask4[2], SecRay_mask4.mask4[2] );
					refl_rm->mask4[3] = _mm_and_ps( refl_rm->mask4[3], SecRay_mask4.mask4[3] );
					refl_rp->RayWay = cq;
					Split_Trace4x4__SecRay(nNextIdx);
				}
				refl_rm->mask4[0] = SecRay_mask4.mask4[0];
				refl_rm->mask4[1] = SecRay_mask4.mask4[1];
				refl_rm->mask4[2] = SecRay_mask4.mask4[2];
				refl_rm->mask4[3] = SecRay_mask4.mask4[3];
				Split_Shading4x4(nNextIdx);
				i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
					if (refl_is->tacc[i] && refl_rm->mask[i]) {
						is->color[i] += mat_fRefl[y].f[x] * refl_is->color[i] * GColor(mat_cTex[y].x[x], mat_cTex[y].y[x], mat_cTex[y].z[x]);
					}
				}
			}
			if (rp->Depth == 0) {
				for (y = 3; y >= 0; y--) {
					is->ads.sec_oid4[y]     = refl_is->ads.pri_oid4[y];
					is->ads.sec_shadow4[y]  = refl_is->ads.pri_shadow4[y];
					is->ads.sec_texture4[y] = refl_is->ads.pri_texture4[y];
					is->ads.n2[y]           = refl_is->n[y];
				}
			}
		}

		// ---------------------------------------------------------------------------
		// refraction
		// ---------------------------------------------------------------------------
		if (n_refr > 0) {
			__m128 dot_i4[4];
			__m128 dot_r4[4];
			__m128 n_div_nt4[4];

			const int nNextIdx = nIdx+1;

			_sse_4x4_raypacket	*refr_rp	= &m_RayPk4x4[nNextIdx];
			_sse_4x4_isect		*refr_is	= &m_Isect4x4[nNextIdx];
			_sse_4x4_raymask	*refr_rm	= &m_RMask4x4[nNextIdx];

			dot_i4[0] = sse_vdot(rp->d[0], is->n[0]);
			dot_i4[1] = sse_vdot(rp->d[1], is->n[1]);
			dot_i4[2] = sse_vdot(rp->d[2], is->n[2]);
			dot_i4[3] = sse_vdot(rp->d[3], is->n[3]);

			__m128 mask[4];
			mask[0]		 = _mm_cmplt_ps(dot_i4[0], _mm_setzero_ps());
			mask[1]		 = _mm_cmplt_ps(dot_i4[1], _mm_setzero_ps());
			mask[2]		 = _mm_cmplt_ps(dot_i4[2], _mm_setzero_ps());
			mask[3]		 = _mm_cmplt_ps(dot_i4[3], _mm_setzero_ps());

			n_div_nt4[0] = _mm_or_ps ( _mm_and_ps(mask[0], fastxovery(_mm_set1_ps(AIR_INDEX), mat_fRIdx[0].v4)),
									_mm_andnot_ps(mask[0], fastxovery(mat_fRIdx[0].v4, _mm_set1_ps(AIR_INDEX))));
			n_div_nt4[1] = _mm_or_ps ( _mm_and_ps(mask[1], fastxovery(_mm_set1_ps(AIR_INDEX), mat_fRIdx[1].v4)),
									_mm_andnot_ps(mask[1], fastxovery(mat_fRIdx[1].v4, _mm_set1_ps(AIR_INDEX))));
			n_div_nt4[2] = _mm_or_ps ( _mm_and_ps(mask[2], fastxovery(_mm_set1_ps(AIR_INDEX), mat_fRIdx[2].v4)),
									_mm_andnot_ps(mask[2], fastxovery(mat_fRIdx[2].v4, _mm_set1_ps(AIR_INDEX))));
			n_div_nt4[3] = _mm_or_ps ( _mm_and_ps(mask[3], fastxovery(_mm_set1_ps(AIR_INDEX), mat_fRIdx[3].v4)),
									_mm_andnot_ps(mask[3], fastxovery(mat_fRIdx[3].v4, _mm_set1_ps(AIR_INDEX))));

			dot_r4[0] = _mm_sqrt_ps(_mm_abs_ps(
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps( _mm_mul_ps(n_div_nt4[0], n_div_nt4[0]), 
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps(dot_i4[0],dot_i4[0]))	))));
			dot_r4[1] = _mm_sqrt_ps(_mm_abs_ps(
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps( _mm_mul_ps(n_div_nt4[1], n_div_nt4[1]), 
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps(dot_i4[1],dot_i4[1]))	))));
			dot_r4[2] = _mm_sqrt_ps(_mm_abs_ps(
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps( _mm_mul_ps(n_div_nt4[2], n_div_nt4[2]), 
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps(dot_i4[2],dot_i4[2]))	))));
			dot_r4[3] = _mm_sqrt_ps(_mm_abs_ps(
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps( _mm_mul_ps(n_div_nt4[3], n_div_nt4[3]), 
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps(dot_i4[3],dot_i4[3]))	))));

			for (i = 3; i >= 0; i--) {
			mask[i] = _mm_cmplt_ps(n_div_nt4[i], _mm_set1_ps(1));
			refr_rp->d[i].x4 = _mm_add_ps( _mm_mul_ps(n_div_nt4[i], _mm_sub_ps(rp->d[i].x4, _mm_mul_ps(is->n[i].x4, dot_i4[i]))),  sse_update(_mm_neg_ps(_mm_mul_ps(is->n[i].x4, dot_r4[i])), _mm_mul_ps(is->n[i].x4, dot_r4[i]), mask[i]));
			refr_rp->d[i].y4 = _mm_add_ps( _mm_mul_ps(n_div_nt4[i], _mm_sub_ps(rp->d[i].y4, _mm_mul_ps(is->n[i].y4, dot_i4[i]))),  sse_update(_mm_neg_ps(_mm_mul_ps(is->n[i].y4, dot_r4[i])), _mm_mul_ps(is->n[i].y4, dot_r4[i]), mask[i]));
			refr_rp->d[i].z4 = _mm_add_ps( _mm_mul_ps(n_div_nt4[i], _mm_sub_ps(rp->d[i].z4, _mm_mul_ps(is->n[i].z4, dot_i4[i]))),  sse_update(_mm_neg_ps(_mm_mul_ps(is->n[i].z4, dot_r4[i])), _mm_mul_ps(is->n[i].z4, dot_r4[i]), mask[i]));
			}
			Split_InitPkt4x4( nNextIdx );

			refr_rp->o[0].x4 = _mm_add_ps(hit_p[0].x4, _mm_mul_ps(refr_rp->d[0].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[0].y4 = _mm_add_ps(hit_p[0].y4, _mm_mul_ps(refr_rp->d[0].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[0].z4 = _mm_add_ps(hit_p[0].z4, _mm_mul_ps(refr_rp->d[0].z4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[1].x4 = _mm_add_ps(hit_p[1].x4, _mm_mul_ps(refr_rp->d[1].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[1].y4 = _mm_add_ps(hit_p[1].y4, _mm_mul_ps(refr_rp->d[1].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[1].z4 = _mm_add_ps(hit_p[1].z4, _mm_mul_ps(refr_rp->d[1].z4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[2].x4 = _mm_add_ps(hit_p[2].x4, _mm_mul_ps(refr_rp->d[2].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[2].y4 = _mm_add_ps(hit_p[2].y4, _mm_mul_ps(refr_rp->d[2].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[2].z4 = _mm_add_ps(hit_p[2].z4, _mm_mul_ps(refr_rp->d[2].z4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[3].x4 = _mm_add_ps(hit_p[3].x4, _mm_mul_ps(refr_rp->d[3].x4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[3].y4 = _mm_add_ps(hit_p[3].y4, _mm_mul_ps(refr_rp->d[3].y4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o[3].z4 = _mm_add_ps(hit_p[3].z4, _mm_mul_ps(refr_rp->d[3].z4, _mm_set1_ps(RAY_START_EPSILON)));

			refr_rp->Depth = rp->Depth+1;

			_sse_4x4_raymask SecRay_mask4;
			SecRay_mask4.mask4[0] = _mm_and_ps( _mm_and_ps( rm->mask4[0], (__m128&)_mm_cmpgt_epi32(is->tacc4[0], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefr[0].v4, _mm_setzero_ps()));
			SecRay_mask4.mask4[1] = _mm_and_ps( _mm_and_ps( rm->mask4[1], (__m128&)_mm_cmpgt_epi32(is->tacc4[1], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefr[1].v4, _mm_setzero_ps()));
			SecRay_mask4.mask4[2] = _mm_and_ps( _mm_and_ps( rm->mask4[2], (__m128&)_mm_cmpgt_epi32(is->tacc4[2], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefr[2].v4, _mm_setzero_ps()));
			SecRay_mask4.mask4[3] = _mm_and_ps( _mm_and_ps( rm->mask4[3], (__m128&)_mm_cmpgt_epi32(is->tacc4[3], _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefr[3].v4, _mm_setzero_ps()));

			unsigned int b;
			if (refr_rp->IsCoherent()) {
				refr_rp->RayWay = (refr_rp->xmask & 1) + (refr_rp->ymask & 2) + (refr_rp->zmask & 4);
				refr_rm->mask4[0] = SecRay_mask4.mask4[0];
				refr_rm->mask4[1] = SecRay_mask4.mask4[1];
				refr_rm->mask4[2] = SecRay_mask4.mask4[2];
				refr_rm->mask4[3] = SecRay_mask4.mask4[3];
				Split_Trace4x4__SecRay(nNextIdx);
				Split_Shading4x4(nNextIdx);
				i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
					if (refr_is->tacc[i] && refr_rm->mask[i]) {
						is->color[i] += mat_fRefr[y].f[x] * refr_is->color[i] * GColor(mat_cTex[y].x[x], mat_cTex[y].y[x], mat_cTex[y].z[x]);
					}
				}
			} else {
				// detect octants and render one by one
				bool b_quad[8] = { false, false, false, false, false, false, false, false };
				for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
					int q = ((refr_rp->xmask & b)?1:0) + ((refr_rp->ymask & b)?2:0) + ((refr_rp->zmask & b)?4:0);
					b_quad[q] = true;
				}
				for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
					for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
						int q = ((refr_rp->xmask & b)?1:0) + ((refr_rp->ymask & b)?2:0) + ((refr_rp->zmask & b)?4:0);
						if (q == cq) refr_rm->mask[i] = 0xffffffff; else refr_rm->mask[i] = 0;
					}
					refr_rm->mask4[0] = _mm_and_ps( refr_rm->mask4[0], SecRay_mask4.mask4[0] );
					refr_rm->mask4[1] = _mm_and_ps( refr_rm->mask4[1], SecRay_mask4.mask4[1] );
					refr_rm->mask4[2] = _mm_and_ps( refr_rm->mask4[2], SecRay_mask4.mask4[2] );
					refr_rm->mask4[3] = _mm_and_ps( refr_rm->mask4[3], SecRay_mask4.mask4[3] );
					refr_rp->RayWay = cq;
					Split_Trace4x4__SecRay(nNextIdx);
				}
				refr_rm->mask4[0] = SecRay_mask4.mask4[0];
				refr_rm->mask4[1] = SecRay_mask4.mask4[1];
				refr_rm->mask4[2] = SecRay_mask4.mask4[2];
				refr_rm->mask4[3] = SecRay_mask4.mask4[3];
				Split_Shading4x4(nNextIdx);
				i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
					if (refr_is->tacc[i] && refr_rm->mask[i]) {
						is->color[i] += mat_fRefr[y].f[x] * refr_is->color[i] * GColor(mat_cTex[y].x[x], mat_cTex[y].y[x], mat_cTex[y].z[x]);
					}
				}
			}
			if (rp->Depth == 0) {
				for (y = 3; y >= 0; y--) {
					is->ads.sec_oid4[y]     = refr_is->ads.pri_oid4[y];
					is->ads.sec_shadow4[y]  = refr_is->ads.pri_shadow4[y];
					is->ads.sec_texture4[y] = refr_is->ads.pri_texture4[y];
					is->ads.n2[y]           = refr_is->n[y];
				}
			}
		}
	}
}

void SSERenderPipeline::Split_Shading4x4_RayGeneration_ShwRay(const _sse_vec objectPos[], const GPoint* lightPos, const __m128 shadingmask[]) {

	_sse_4x4_raypacket	*shadow_rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*shadow_is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*shadow_rm	= &m_ShadowRMask4x4[0];

	_sse_vec lPos = sse_vset1(lightPos->x, lightPos->y, lightPos->z);

	shadow_rp->d[0] = sse_vsub(lPos, objectPos[0]);
	shadow_rp->d[1] = sse_vsub(lPos, objectPos[1]);
	shadow_rp->d[2] = sse_vsub(lPos, objectPos[2]);
	shadow_rp->d[3] = sse_vsub(lPos, objectPos[3]);
	Split_InitPkt4x4_ShwRay();

	shadow_rp->o[0] = sse_vadd(objectPos[0], sse_vmul(shadow_rp->d[0], sse_vset1(RAY_START_EPSILON)));
	shadow_rp->o[1] = sse_vadd(objectPos[1], sse_vmul(shadow_rp->d[1], sse_vset1(RAY_START_EPSILON)));
	shadow_rp->o[2] = sse_vadd(objectPos[2], sse_vmul(shadow_rp->d[2], sse_vset1(RAY_START_EPSILON)));
	shadow_rp->o[3] = sse_vadd(objectPos[3], sse_vmul(shadow_rp->d[3], sse_vset1(RAY_START_EPSILON)));

	unsigned int i, b;
	// coherence 체크 겸 ray dir 결정 (q = 8방향중하나)
	if (shadow_rp->IsCoherent()) {
		shadow_rp->RayWay = (shadow_rp->xmask & 1) + (shadow_rp->ymask & 2) + (shadow_rp->zmask & 4);
		shadow_rm->mask4[0] = shadingmask[0];
		shadow_rm->mask4[1] = shadingmask[1];
		shadow_rm->mask4[2] = shadingmask[2];
		shadow_rm->mask4[3] = shadingmask[3];
		Split_Trace4x4__ShwRay(0);
	} else {
		// detect octants and render one by one
		bool b_quad[8] = { false, false, false, false, false, false, false, false };
		for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
			int q = ((shadow_rp->xmask & b)?1:0) + ((shadow_rp->ymask & b)?2:0) + ((shadow_rp->zmask & b)?4:0);
			b_quad[q] = true;
		}
		for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
			for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
				int q = ((shadow_rp->xmask & b)?1:0) + ((shadow_rp->ymask & b)?2:0) + ((shadow_rp->zmask & b)?4:0);
				if (q == cq) shadow_rm->mask[i] = 0xffffffff; else shadow_rm->mask[i] = 0;
			}
			shadow_rm->mask4[0] = _mm_and_ps(shadow_rm->mask4[0], shadingmask[0]);
			shadow_rm->mask4[1] = _mm_and_ps(shadow_rm->mask4[1], shadingmask[1]);
			shadow_rm->mask4[2] = _mm_and_ps(shadow_rm->mask4[2], shadingmask[2]);
			shadow_rm->mask4[3] = _mm_and_ps(shadow_rm->mask4[3], shadingmask[3]);
			shadow_rp->RayWay = cq;
			Split_Trace4x4__ShwRay(0);
		}
	}
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// -----------------------------------------------------------
// SSERenderPipeline::RenderTiles
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Split_Render4x4__PriRay( int nJobID )
{
	int yTileStart = 0;
	int xTileEnd = (m_Resolution.x) >> 2;
	int yTileEnd = (m_Resolution.y) >> 2;
	int tx, ty;
	int i;

	// =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+
	// Thread Setting
	const float fTileSize = yTileEnd * m_fThreadRcpJobSize;

	if (nJobID == -1) {									// No thread
		yTileStart = 0;
		yTileEnd;
	} else if (nJobID == (m_iThreadJobSize - 1)) {		// This job is last one.
		yTileStart	= (int(fTileSize *  nJobID    ));
		yTileEnd;
	} else {											// This job is not last one.
		yTileStart	= (int(fTileSize *  nJobID    ));
		yTileEnd	= (int(fTileSize * (nJobID +1)));
	}
	// =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+

	// start spawning rays
	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[0];
	_sse_4x4_isect		*is	= &m_Isect4x4[0];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[0];
	_sse_4x4_raypacket	tpos;		// target position for ray casting (pixel center)
	_sse_4x4_raypacket	delta4;		// delta4 for next target position between packet4x4
	_sse_4x4_raypacket	jpos;		// jittered position for sampling

	const union { unsigned int f[4]; __m128i v4; } addr_delta = { {4, 4, 4, 4} };

	const __m128 ray_o_x4 = _mm_load1_ps( &m_Origin.x );
	const __m128 ray_o_y4 = _mm_load1_ps( &m_Origin.y );
	const __m128 ray_o_z4 = _mm_load1_ps( &m_Origin.z );
	const __m128 delta_x4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.x ));
	const __m128 delta_y4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.y ));
	const __m128 delta_z4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.z ));

	for ( i = 0; i < 4; i++ ) {
		rp->o[i].x4    = ray_o_x4;	rp->o[i].y4    = ray_o_y4;	rp->o[i].z4    = ray_o_z4;
		delta4.d[i].x4 = delta_x4;	delta4.d[i].y4 = delta_y4;	delta4.d[i].z4 = delta_z4;
	}

	const int iCastSeq16_x[16] = { 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3 };
	const int iCastSeq16_y[16] = { 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3 };
	const float fSampSeq_x[5][16] = { { 0, }, { 0, }, { -0.25f, +0.25f, }, { -0.333f, 0, 0.333f, }, { -0.375f, -0.125f, 0.125f, 0.375f, }};
	const float fSampSeq_y[5][16] = { { 0, }, { 0, }, { -0.25f, +0.25f, }, { -0.333f, 0, 0.333f, }, { -0.375f, -0.125f, 0.125f, 0.375f, }};
	const float rcpSuperSampling_x = 1.0f / m_SuperSampling.x;
	const float rcpSuperSampling_y = 1.0f / m_SuperSampling.y;
	const float fXjitter = 0.5f * rcpSuperSampling_x;	// jitterRate = 0.5f
	const float fYjitter = 0.5f * rcpSuperSampling_y;	// jitterRate = 0.5f
	const float fSampWeight = rcpSuperSampling_x * rcpSuperSampling_y;

	for ( ty = yTileStart; ty < yTileEnd; ty++ ) {
		unsigned int i = 0;

		// -----------------------------------------------------------------------
		// tpos (Ray 를 쏠 방향지점) 계산
		// -----------------------------------------------------------------------
		// m_LeftUp : image screen 위쪽 왼편 모서리의 pixel 중심 으로 이미 셋팅 되어 있음
		for(i = 0; i < 4; i++) {
			// tpos.d = m_LeftUp + m_DX * (float)iCastSeq4x4_x[i] - m_DY * (float)(ty * 4 + iCastSeq4x4_y[i]);
			tpos.d[i].x4 = _mm_add_ps(m_LeftUp4->x4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->x4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->x4, _mm_add_ps(_mm_set1_ps(ty<<2),m_CastSeq4x4_y4[i]))));
			tpos.d[i].y4 = _mm_add_ps(m_LeftUp4->y4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->y4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->y4, _mm_add_ps(_mm_set1_ps(ty<<2),m_CastSeq4x4_y4[i]))));
			tpos.d[i].z4 = _mm_add_ps(m_LeftUp4->z4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->z4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->z4, _mm_add_ps(_mm_set1_ps(ty<<2),m_CastSeq4x4_y4[i]))));

			// is->addr[i] = iCastSeq4x4_x[i] + (m_Height - 1 - (ty * 4 + iCastSeq4x4_y[i])) * m_Width;
			// m_Tmp2_Addr = (m_Height - 1 - iCastSeq4x4_y[i]) * m_Width
			is->addr4[i] = _mm_add_epi32(m_iCastSeq4x4_x4[i], _mm_sub_epi32(m_Tmp2_Addr4[i], _mm_set1_epi32((m_Width*ty)<<2)));
		}

		for ( tx = 0; tx < xTileEnd; tx++ ) {
			unsigned int i, b;

			Color o_color[16];
			for (int nSampX = 0; nSampX < m_SuperSampling.x; nSampX++) {
			for (int nSampY = 0; nSampY < m_SuperSampling.y; nSampY++) {

				if (m_SuperSampling.x == 1) {
					jpos = tpos;
				} else {
					if ( m_Scene->isEnableJittering() ) {
						#if 1		// regular jitter
							GVector samp_D;
							samp_D  = m_DX * (fSampSeq_x[m_SuperSampling.x][nSampX] + ( (float)sgrtRadicalInverse(nSampX,3)*2.0f*fXjitter - fXjitter))
									- m_DY * (fSampSeq_y[m_SuperSampling.y][nSampY] + ( (float)sgrtRadicalInverse(nSampY,5)*2.0f*fYjitter - fYjitter));
							for ( i = 0; i < 4; i++ ) {
								jpos.d[i]	= sse_vadd(tpos.d[i], sse_vset1(samp_D.x, samp_D.y, samp_D.z));
							}
						#else
							GVector samp_D[16];
							for (i = 0; i < 16; i++) {
								samp_D[i]  =  m_DX * (nSampX + sgrtRadicalInverse((((tx<<2) + iCastSeq16_x[i]) << 8) + nSampX,3)) * rcpSuperSampling
											- m_DY * (nSampY + sgrtRadicalInverse((((ty<<2) + iCastSeq16_y[i]) << 8) + nSampY,5)) * rcpSuperSampling;
							}
							for ( i = 0; i < 4; i++ ) {
								jpos.d[i]	= sse_vadd(tpos.d[i], sse_vset1(samp_D[4*i+0].Array, samp_D[4*i+1].Array, samp_D[4*i+2].Array, samp_D[4*i+3].Array));
							}
						#endif
					} else {
						GVector samp_D;
						samp_D  = m_DX * fSampSeq_x[m_SuperSampling.x][nSampX]
								- m_DY * fSampSeq_y[m_SuperSampling.y][nSampY];
						for ( i = 0; i < 4; i++ ) {
							jpos.d[i]	= sse_vadd(tpos.d[i], sse_vset1(samp_D.x, samp_D.y, samp_D.z));
						}
					}
				}

				// -----------------------------------------------------------------------
				// Ray packet 을 셋팅 - 시작점(ray_o_x4, ray_o_y4, ray_o_z4) ~ 끝점(tpos)
				// -----------------------------------------------------------------------
				for ( i = 0; i < 4; i++ ) {
					rp->d[i].x4 = _mm_sub_ps( jpos.d[i].x4, ray_o_x4 );
					rp->d[i].y4 = _mm_sub_ps( jpos.d[i].y4, ray_o_y4 );
					rp->d[i].z4 = _mm_sub_ps( jpos.d[i].z4, ray_o_z4 );
				}
				rp->Depth = 0;
				Split_InitPkt4x4( 0 );	// direction vector normalize 등

				// -----------------------------------------------------------------------
				// Coherence 체크 후 rendering
				// -----------------------------------------------------------------------
				if (rp->IsCoherent()) {
					rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
					Split_Trace4x4__PriRay(0, 0);
					Split_Shading4x4(0);
				} else {
					// detect octants and render one by one
					bool b_quad[8] = { false, false, false, false, false, false, false, false };
					for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
						int q = ((rp->xmask & b)?1:0) + ((rp->ymask & b)?2:0) + ((rp->zmask & b)?4:0);
						b_quad[q] = true;
					}
					for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
						for ( b = 1, i = 0; i < 16; i++, b <<= 1 ) {
							int q = ((rp->xmask & b)?1:0) + ((rp->ymask & b)?2:0) + ((rp->zmask & b)?4:0);
							if (q == cq) rm->mask[i] = 0xffffffff; else rm->mask[i] = 0;
						}
						rp->RayWay = cq;
						Split_Trace4x4__PriRay(0, 0);
					}
					memset( &rm->mask4, 255, 16 * 4 );
					Split_Shading4x4(0);
				}

				// -----------------------------------------------------------------------
				// Copy color info to Memory
				// -----------------------------------------------------------------------
				for ( i = 0; i < 16; i++ ) {
					o_color[i].rgba = _mm_add_ps(o_color[i].rgba, is->color[i].rgba);
//					o_color[i].rgba = _mm_add_ps(o_color[i].rgba, _mm_set1_ps(is->fTime[i]));
				}

			}}	// Loops of (nSampleX * nSampleY)

			for ( i = 0; i < 16; i++ ) {
				//_mm_store_ps(m_Dest+3*is->addr[i], 
				//	_mm_add_ps(_mm_load_ps(m_Dest+3*is->addr[i]),
				//	_mm_mul_ps(o_color[i].rgba, _mm_setr_ps(fSampWeight, fSampWeight, fSampWeight, 0))));
				m_Dest[3*(is->addr[i])]   = o_color[i].r * fSampWeight;
				m_Dest[3*(is->addr[i])+1] = o_color[i].g * fSampWeight;
				m_Dest[3*(is->addr[i])+2] = o_color[i].b * fSampWeight;
			}

			// Render tile (tpos) 의 위치를 이동
			for ( i = 0; i < 4; i++ ) {
				tpos.d[i].x4 = _mm_add_ps( tpos.d[i].x4, delta4.d[i].x4 );
				tpos.d[i].y4 = _mm_add_ps( tpos.d[i].y4, delta4.d[i].y4 );
				tpos.d[i].z4 = _mm_add_ps( tpos.d[i].z4, delta4.d[i].z4 );
				is->addr4[i] = _mm_add_epi32( is->addr4[i], addr_delta.v4 );
			}
		}
	}
}
