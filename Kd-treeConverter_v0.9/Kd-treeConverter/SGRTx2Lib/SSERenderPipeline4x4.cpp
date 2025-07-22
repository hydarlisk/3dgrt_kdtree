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
void SSERenderPipeline::InitPacket4x4(int nIdx)
{
	COUNT_STATE( nIdx, RAY_COUNT, 1 );
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

	// Pre-calculate dir_mask ray's direction vector
	rp->xmask =  _mm_movemask_ps( rp->d[0].x4 )       + (_mm_movemask_ps( rp->d[1].x4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].x4 ) << 8) + (_mm_movemask_ps( rp->d[3].x4 ) << 12);
	rp->ymask =  _mm_movemask_ps( rp->d[0].y4 )       + (_mm_movemask_ps( rp->d[1].y4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].y4 ) << 8) + (_mm_movemask_ps( rp->d[3].y4 ) << 12);
	rp->zmask =  _mm_movemask_ps( rp->d[0].z4 )       + (_mm_movemask_ps( rp->d[1].z4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].z4 ) << 8) + (_mm_movemask_ps( rp->d[3].z4 ) << 12);
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

void SSERenderPipeline::IsectPacket4x4( const KdTreeNode *node, int nIdx )
{
	int i, j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	COUNT_STATE( nIdx, VISITED_LEAF_NODE, 1 );
	if( nObjs == 0 )
	{COUNT_STATE( nIdx, EMPTY_NODE, 1 );}
#if 0

	//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//=//
	if (nObjs == 0) return;

	BBoxTime.start();
	*m_bbox = GBoundingBox(GVector(), GVector());
	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int triID = m_Data->m_TriOffList[i];
		*m_bbox += m_Data->m_TriObjList[triID]->m_BBox;
	}

	BBoxTime.end();
	BBoxFreq += BBoxTime.getElapsedFreq();
	BBoxCount += 1;

	CullTime.start();

	float x_abs = fabsf(rp->d[0].x[0]);
	float y_abs = fabsf(rp->d[0].y[0]);
	float z_abs = fabsf(rp->d[0].z[0]);
	const int I0 = (x_abs > y_abs)?  ((x_abs > z_abs)?0:2) : ((y_abs > z_abs)?1:2);
	const int I1 = (I0<2)? I0+1 : 0, I2 = (I1<2)? I1+1 : 0;

	float bminf = m_bbox->getMin()[I0], bmaxf = m_bbox->getMax()[I0];

	__m128 bmin =_mm_set_ps1(bminf), bmax =_mm_set_ps1(bmaxf);
	__m128 r0min[2], r0max[2], r1min[2], r1max[2], org0[3];
	__m128 orgmin = posinf4, orgmax = neginf4;

	for (j = 3; j >= 0; j--) {
		const __m128 di = rp->d[j].v4[I0];

		org0[0] = rp->o[j].x4; org0[1] = rp->o[j].y4; org0[2] = rp->o[j].z4;
		orgmin = _mm_min_ps(orgmin , org0[I0]);
		orgmax = _mm_max_ps(orgmax,  org0[I0]);

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


	const __m128 ext1 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(2,0,2,0)); // I1 values
	const __m128 ext2 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(3,1,3,1)); // I2 values

	__m128 entr[3], extr[3]; // 2 AA rectangles forming frustum
	entr[I0] = bmin;
	entr[I1] =_mm_shuffle_ps(ext1, ext1, _MM_SHUFFLE(0,2,2,0));
	entr[I2] =_mm_shuffle_ps(ext2, ext2, _MM_SHUFFLE(2,2,0,0));
	extr[I0] = bmax;
	extr[I1] =_mm_shuffle_ps(ext1, ext1, _MM_SHUFFLE(1,3,3,1));
	extr[I2] =_mm_shuffle_ps(ext2, ext2, _MM_SHUFFLE(3,3,1,1));

	const __m128 mix1 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(3,2,1,0));
	const __m128 mix2 =_mm_shuffle_ps(emin, emax, _MM_SHUFFLE(1,0,3,2));
	const __m128 xminmax =_mm_shuffle_ps(bmin, bmax, _MM_SHUFFLE(0,0,0,0));
	const __m128 xmaxmin =_mm_shuffle_ps(bmax, bmin, _MM_SHUFFLE(0,0,0,0));

	float term0 = bmaxf - bminf;
	__m128 term2 = mix1 - mix2;
	__m128 term1 = mix1*(xminmax - xmaxmin) - xminmax*term2;
	term2 *= _mm_set_ps1(1/term0); term1 *= _mm_set_ps1(1/term0);

	_sse_2x2_raypacket	*frp	= &m_FrustomRayPk2x2[0];
	frp->o.x4 = entr[0]; frp->o.y4 = entr[1]; frp->o.z4 = entr[2];
	frp->d.x4 = extr[0] - entr[0];
	frp->d.y4 = extr[1] - entr[1];
	frp->d.z4 = extr[2] - entr[2];

//	__m128 di = 

#endif

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		union { __m128 Mask_Hit[4]; __m128i iMask_Hit[4]; };

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { 
			COUNT_STATE( nIdx, MAILBOXED_TRIANGLE_COUNT, 1 );
			continue;
		}
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		COUNT_STATE( nIdx, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (acc.isTransparent || !m_Scene->isBackFaceCulling()) {
			for (j = 3; j >= 0; j--) {
				Mask_Hit[j] = rm->mask4[j];
			}
		} else {
			_sse_vec tri_N;
			tri_N.x4 = _mm_set1_ps(acc.N.x);
			tri_N.y4 = _mm_set1_ps(acc.N.y);
			tri_N.z4 = _mm_set1_ps(acc.N.z);
			for (j = 3; j >= 0; j--) {
				Mask_Hit[j] = _mm_and_ps(rm->mask4[j],
					_mm_cmpgt_ps(sse_vdot(rp->d[j], tri_N), _mm_setzero_ps()));
			}
		}

		// -----------------------------------------------------------------------------
		// ==>> Plane test (t)
		// -----------------------------------------------------------------------------
		const unsigned int k	= acc.k;

		__m128 nd[4], f[4];

		for (j = 3; j >= 0; j--) {
			nd[j]	=_mm_add_ps(rp->d[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->d[j].v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->d[j].v4[kv])));
			f[j]	=_mm_sub_ps(_mm_set_ps1(acc.n_d),
					 _mm_add_ps(rp->o[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->o[j].v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->o[j].v4[kv]))));
			f[j]	=_mm_mul_ps(f[j],sse_inverse(nd[j]));
		}

		__m128 mask_sum = _mm_setzero_ps();
		for (j = 3; j >= 0; j--) {
			Mask_Hit[j]	= _mm_and_ps(Mask_Hit[j],
				_mm_and_ps(_mm_cmpge_ps(is->dist4[j],f[j]),	_mm_cmpgt_ps(f[j],_mm_set1_ps(EPSILON))));	// eps < f <= Hit4.dist
			mask_sum	= _mm_or_ps (Mask_Hit[j], mask_sum);
		}
		if (_mm_movemask_ps(mask_sum)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> Triangle boundaries test (u, v)
		// -----------------------------------------------------------------------------
		__m128 hu[4], hv[4];
		__m128 lambda[4], mue[4];

		mask_sum = _mm_setzero_ps();
		for (j = 3; j >= 0; j--) {
			hu[j]		= _mm_add_ps(rp->o[j].v4[ku], _mm_mul_ps(f[j],rp->d[j].v4[ku]));
			hv[j]		= _mm_add_ps(rp->o[j].v4[kv], _mm_mul_ps(f[j],rp->d[j].v4[kv]));

			lambda[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.b_nv)));
			lambda[j]	= _mm_add_ps(lambda[j],_mm_set_ps1(acc.b_d));
			mue[j]		= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.c_nv)));
			mue[j]		= _mm_add_ps(mue[j],   _mm_set_ps1(acc.c_d));

			Mask_Hit[j]	= _mm_and_ps(Mask_Hit[j], _mm_cmpgt_ps(lambda[j],_mm_setzero_ps()));
			Mask_Hit[j]	= _mm_and_ps(Mask_Hit[j], _mm_cmpgt_ps(mue[j],   _mm_setzero_ps()));
			Mask_Hit[j] = _mm_and_ps(Mask_Hit[j], _mm_cmple_ps(_mm_add_ps(lambda[j],mue[j]), _mm_set1_ps(1)));
			mask_sum	= _mm_or_ps (Mask_Hit[j], mask_sum);
		}
		if (_mm_movemask_ps(mask_sum)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
		for (j = 3; j >= 0; j--) {
			is->u4[j]		= sse_update(lambda[j], is->u4[j],		Mask_Hit[j]);
			is->v4[j]		= sse_update(mue[j],    is->v4[j],		Mask_Hit[j]);
			is->dist4[j]	= sse_update(f[j],      is->dist4[j],	Mask_Hit[j]);
			is->tacc4[j]	= _mm_or_si128( _mm_andnot_si128( iMask_Hit[j], is->tacc4[j] ),
											_mm_and_si128   ( iMask_Hit[j], tacc4 ) );
		}
	}
}


//#define ku_ ((modulo[k+1])>(modulo[k+2]))?(modulo[k+2]):(modulo[k+1])
//#define kv_ ((modulo[k+1])>(modulo[k+2]))?(modulo[k+1]):(modulo[k+2])
#define ku_ modulo[k+1]
#define kv_ modulo[k+2]
// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::IsectPacket_P( const KdTreeNode *node, int nIdx )
{
	int i, j;

	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[nIdx];
	_sse_4x4_isect		*is	= &m_Isect4x4[nIdx];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int       triID = m_Data->m_TriOffList[i];
		TriAccel_P &acc = m_Data->m_TriAccList_P[triID];

		union { __m128 Mask_Hit[4]; __m128i iMask_Hit[4]; };

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { continue; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (acc.isTransparent || !m_Scene->isBackFaceCulling()) {
			for (j = 0; j < 4; j++) {
				Mask_Hit[j] = rm->mask4[j];
			}
		} else {
			_sse_vec tri_N;
			tri_N.x4 = _mm_set1_ps(acc.N.x);
			tri_N.y4 = _mm_set1_ps(acc.N.y);
			tri_N.z4 = _mm_set1_ps(acc.N.z);
			for (j = 0; j < 4; j++) {
				Mask_Hit[j] = _mm_and_ps(rm->mask4[j],
					_mm_cmpgt_ps(sse_vdot(rp->d[j], tri_N), _mm_setzero_ps()));
			}
		}

		const unsigned int k	= acc.k;

		__m128 det[4], det_t[4], det_u[4], det_v[4];
		__m128 nrv, nru, du, dv, dw, ou, ov, ow, po_u, po_v;
		__m128 tmpdet0, tmpdet1;

		__m128 mask_sum = _mm_setzero_ps();
		for (j = 0; j < 4; j++) {
			ou   = rp->o[j].v4[ku_];
			ov   = rp->o[j].v4[kv_];
			ow   = rp->o[j].v4[k];
			du   = rp->d[j].v4[ku_];
			dv   = rp->d[j].v4[kv_];
			dw   = rp->d[j].v4[k];

			po_u = _mm_sub_ps(_mm_set1_ps(acc.p_u), ou);		// P.u - O.u
			po_v = _mm_sub_ps(_mm_set1_ps(acc.p_v), ov);		// P.v - O.v

			// det_t = n_p - (O.u * N.u + O.v * N.v + O.w)
			det_t[j] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(acc.n_u), ou),
											 _mm_mul_ps(_mm_set1_ps(acc.n_v), ov)), ow);	// O.u * N.u + O.v * N.v + O.w
			det_t[j] = _mm_sub_ps(_mm_set1_ps(acc.n_d), det_t[j]);

			// det = D.u * N.u + D.v * N.v + dw
			det[j]   = _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(acc.n_u), du),
											 _mm_mul_ps(_mm_set1_ps(acc.n_v), dv)), dw);	// D.u * N.u + D.v * N.v + D.w

			// Du = D.u * det_t - (P.u - O.u) * det
			nru = _mm_mul_ps(po_u, det[j]);				// (P.u - O.u) * det
			du  = _mm_mul_ps(du, det_t[j]);				// D.u * det_t
			du  = _mm_sub_ps(du, nru);

			// Dv = D.v * det_t - (P.v - O.v) * det
			nrv = _mm_mul_ps(po_v, det[j]);				// (P.v - O.v) * det
			dv  = _mm_mul_ps(dv, det_t[j]);				// D.v * det_t
			dv  = _mm_sub_ps(dv, nrv);

			// det_u = (e1vDu - e1uDv)
			nru = _mm_mul_ps(_mm_set1_ps(acc.e1v), du);
			nrv = _mm_mul_ps(_mm_set1_ps(acc.e1u), dv);
			det_u[j] = _mm_sub_ps(nru, nrv);
			det_u[j] = _mm_mul_ps(det_u[j], _mm_set1_ps(acc.rcp_area));

			// det_v = (e0uDv - e0vDu)
			nru = _mm_mul_ps(_mm_set1_ps(acc.e0v), du);
			nrv = _mm_mul_ps(_mm_set1_ps(acc.e0u), dv);
			det_v[j] = _mm_sub_ps(nrv, nru);
			det_v[j] = _mm_mul_ps(det_v[j], _mm_set1_ps(acc.rcp_area));

			tmpdet0     = _mm_sub_ps(det[j], _mm_add_ps(det_u[j], det_v[j]));
			tmpdet0     = _mm_xor_ps(tmpdet0, det_u[j]);
			tmpdet1     = _mm_xor_ps(det_v[j], det_u[j]);

			Mask_Hit[j] = _mm_and_ps(_mm_cmpge_ps(_mm_or_ps(tmpdet0, tmpdet1), _mm_setzero_ps()), Mask_Hit[j]);
			mask_sum	= _mm_or_ps (Mask_Hit[j], mask_sum);
		}
		if (_mm_movemask_ps(mask_sum)==0) continue;

		mask_sum = _mm_setzero_ps();
		for (j = 0; j < 4; j++) {
			//__m128 det2 = _mm_mul_ps(det[j], det[j]);
			//Mask_Hit[j]	= _mm_and_ps(Mask_Hit[j],
			//	_mm_and_ps( _mm_cmplt_ps(_mm_mul_ps(det[j], det_t[j]), _mm_mul_ps(det2,is->dist4[j])),
			//				_mm_cmpgt_ps(_mm_mul_ps(det[j], det_t[j]), _mm_mul_ps(det2,_mm_set1_ps(EPSILON)))));	// eps < f <= Hit4.dist
			//mask_sum	= _mm_or_ps (Mask_Hit[j], mask_sum);
			__m128 r_det = sse_inverse(det[j]);
			Mask_Hit[j]	= _mm_and_ps(Mask_Hit[j],
				_mm_and_ps( _mm_cmplt_ps(_mm_mul_ps(r_det, det_t[j]), is->dist4[j]),
							_mm_cmpgt_ps(_mm_mul_ps(r_det, det_t[j]), _mm_set1_ps(EPSILON))));	// eps < f <= Hit4.dist
			mask_sum	= _mm_or_ps (Mask_Hit[j], mask_sum);
		}
		if (_mm_movemask_ps(mask_sum)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
		for (j = 0; j < 4; j++) {
			__m128 r_det = sse_inverse(det[j]);
			__m128 t     = _mm_mul_ps(r_det, det_t[j]);
			__m128 u     = _mm_mul_ps(r_det, det_u[j]);
			__m128 v     = _mm_mul_ps(r_det, det_v[j]);

			is->u4[j]		= sse_update(v, is->u4[j],		Mask_Hit[j]);
			//is->v4[j]		= sse_update(_mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1),u), v), is->v4[j],		Mask_Hit[j]);
			is->v4[j]		= sse_update(u, is->v4[j],		Mask_Hit[j]);
			is->dist4[j]	= sse_update(t, is->dist4[j],	Mask_Hit[j]);
			is->tacc4[j]	= _mm_or_si128( _mm_andnot_si128( iMask_Hit[j], is->tacc4[j] ),
											_mm_and_si128   ( iMask_Hit[j], tacc4 ) );
		}
	}
}

int test_tx, test_ty;
// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::TracePacket4x4( int nIdx )
{
	int i, j;

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

	// ray reciprocal direction
	_sse_vec rcpRayDir4[4];
	for (j = 3; j >= 0; j--) {
		for (i = 0; i < AXIS_SIZE; i++) {
			rcpRayDir4[j].v4[i] = sse_inverse(rp->d[j].v4[i]);
		}
	}

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
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_x, rp->o[j].x4), rcpRayDir4[j].x4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_x, rp->o[j].x4), rcpRayDir4[j].x4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_y, rp->o[j].y4), rcpRayDir4[j].y4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_y, rp->o[j].y4), rcpRayDir4[j].y4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_z, rp->o[j].z4), rcpRayDir4[j].z4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_z, rp->o[j].z4), rcpRayDir4[j].z4);
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
			
				d[0]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[0].v4[dim]), rcpRayDir4[0].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[0], d[0]), Mask_Active4[0]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[0], d[0]), Mask_Active4[0]));
				d[1]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[1].v4[dim]), rcpRayDir4[1].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[1], d[1]), Mask_Active4[1]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[1], d[1]), Mask_Active4[1]));
				d[2]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[2].v4[dim]), rcpRayDir4[2].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[2], d[2]), Mask_Active4[2]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[2], d[2]), Mask_Active4[2]));
				d[3]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[3].v4[dim]), rcpRayDir4[3].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[3], d[3]), Mask_Active4[3]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[3], d[3]), Mask_Active4[3]));

				node = BackSideSon;
				if (d_near == 0)
				{
					COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the back  child
				}
				node = FrontSideSon;
				if (d_far == 0)
				{
					COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the front child
				}

				COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN );

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
		IsectPacket4x4(node, nIdx);

		// Termination test
		if ((_mm_movemask_ps( _mm_or_ps( _mm_or_ps( _mm_or_ps( 
			_mm_cmpgt_ps( is->dist4[0], t_far_4[0] ),	_mm_cmpgt_ps( is->dist4[1], t_far_4[1] ) ), 
			_mm_cmpgt_ps( is->dist4[2], t_far_4[2] ) ), _mm_cmpgt_ps( is->dist4[3], t_far_4[3] ) ) ) == 0) || (stackIndex == 0)) break;

		COUNT_TREE_OPERATOR( nIdx, TRAVERSE_UP );
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
// SSERenderPipeline::InitPacket
//		Initialize a packet for tracing
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::InitShadowPacket4x4(void)
{
	COUNT_STATE( -1, RAY_COUNT, 1 );
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

	// Pre-calculate dir_mask ray's direction vector
	rp->xmask =  _mm_movemask_ps( rp->d[0].x4 )       + (_mm_movemask_ps( rp->d[1].x4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].x4 ) << 8) + (_mm_movemask_ps( rp->d[3].x4 ) << 12);
	rp->ymask =  _mm_movemask_ps( rp->d[0].y4 )       + (_mm_movemask_ps( rp->d[1].y4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].y4 ) << 8) + (_mm_movemask_ps( rp->d[3].y4 ) << 12);
	rp->zmask =  _mm_movemask_ps( rp->d[0].z4 )       + (_mm_movemask_ps( rp->d[1].z4 ) << 4) +
				(_mm_movemask_ps( rp->d[2].z4 ) << 8) + (_mm_movemask_ps( rp->d[3].z4 ) << 12);
}


// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::IsectShadowPacket4x4( const KdTreeNode *node )
{
	int i, j;

	_sse_4x4_raypacket	*rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*rm	= &m_ShadowRMask4x4[0];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	COUNT_STATE( -1, VISITED_LEAF_NODE, 1 );
	if( nObjs == 0 )
	{COUNT_STATE( -1, EMPTY_NODE, 1 );}

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) {
			COUNT_STATE( -1, MAILBOXED_TRIANGLE_COUNT, 1 );
			continue;
		}
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		COUNT_STATE( -1, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// 투명한 물체는 투과
		// ---------------------------------------------------------------
		if (acc.isTransparent) continue;

		union { __m128 Mask_Hit[4]; __m128i iMask_Hit[4]; };
		for (j = 3; j >= 0; j--) { Mask_Hit[j] = rm->mask4[j]; }

		// -----------------------------------------------------------------------------
		// ==>> Plane test (t)
		// -----------------------------------------------------------------------------
		const unsigned int k	= acc.k;

		__m128 nd[4], f[4];

		for (j = 3; j >= 0; j--) {
			nd[j]	=_mm_add_ps(rp->d[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->d[j].v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->d[j].v4[kv])));
			f[j]	=_mm_sub_ps(_mm_set_ps1(acc.n_d),
					 _mm_add_ps(rp->o[j].v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->o[j].v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->o[j].v4[kv]))));
			f[j]	=_mm_mul_ps(f[j],sse_inverse(nd[j]));
		}

		__m128 mask_sum = _mm_setzero_ps();
		for (j = 3; j >= 0; j--) {
			Mask_Hit[j]	= _mm_and_ps(_mm_cmpge_ps(is->dist4[j],f[j]),	_mm_cmpgt_ps(f[j],_mm_set1_ps(EPSILON)));	// eps < f <= Hit4.dist
			mask_sum	= _mm_or_ps (Mask_Hit[j], mask_sum);
		}
		if (_mm_movemask_ps(mask_sum)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> Triangle boundaries test (u, v)
		// -----------------------------------------------------------------------------
		__m128 hu[4], hv[4];
		__m128 lambda[4], mue[4];

		mask_sum = _mm_setzero_ps();
		for (j = 3; j >= 0; j--) {
			hu[j]		= _mm_add_ps(rp->o[j].v4[ku], _mm_mul_ps(f[j],rp->d[j].v4[ku]));
			hv[j]		= _mm_add_ps(rp->o[j].v4[kv], _mm_mul_ps(f[j],rp->d[j].v4[kv]));

			lambda[j]	= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.b_nv)));
			lambda[j]	= _mm_add_ps(lambda[j],_mm_set_ps1(acc.b_d));
			mue[j]		= _mm_add_ps(_mm_mul_ps(hu[j],_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv[j],_mm_set_ps1(acc.c_nv)));
			mue[j]		= _mm_add_ps(mue[j],   _mm_set_ps1(acc.c_d));

			Mask_Hit[j]	= _mm_and_ps(Mask_Hit[j], _mm_cmpgt_ps(lambda[j],_mm_setzero_ps()));
			Mask_Hit[j]	= _mm_and_ps(Mask_Hit[j], _mm_cmpgt_ps(mue[j],   _mm_setzero_ps()));
			Mask_Hit[j] = _mm_and_ps(Mask_Hit[j], _mm_cmple_ps(_mm_add_ps(lambda[j],mue[j]), _mm_set1_ps(1)));
			mask_sum	= _mm_or_ps (Mask_Hit[j], mask_sum);
		}
		if (_mm_movemask_ps(mask_sum)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
		for (j = 3; j >= 0; j--) {
			is->u4[j]		= sse_update(lambda[j], is->u4[j],		Mask_Hit[j]);
			is->v4[j]		= sse_update(mue[j],    is->v4[j],		Mask_Hit[j]);
			is->dist4[j]	= sse_update(f[j],      is->dist4[j],	Mask_Hit[j]);
			is->tacc4[j]	= _mm_or_si128( _mm_andnot_si128( iMask_Hit[j], is->tacc4[j] ),
											_mm_and_si128   ( iMask_Hit[j], tacc4 ) );
		}
	}
}


// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::TraceShadowPacket4x4( void )
{
	int i, j;

	_sse_4x4_raypacket	*rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*rm	= &m_ShadowRMask4x4[0];

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

	// ray reciprocal direction
	_sse_vec rcpRayDir4[4];
	for (j = 3; j >= 0; j--) {
		for (i = 0; i < AXIS_SIZE; i++) {
			rcpRayDir4[j].v4[i] = sse_inverse(rp->d[j].v4[i]);
		}
	}

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
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_x, rp->o[j].x4), rcpRayDir4[j].x4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_x, rp->o[j].x4), rcpRayDir4[j].x4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_y, rp->o[j].y4), rcpRayDir4[j].y4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_y, rp->o[j].y4), rcpRayDir4[j].y4);
			t_near4[j] = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4[j] );
			t_far_4[j] = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4[j] );
			l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_z, rp->o[j].z4), rcpRayDir4[j].z4);
			l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_z, rp->o[j].z4), rcpRayDir4[j].z4);
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
			
				d[0]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[0].v4[dim]), rcpRayDir4[0].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[0], d[0]), Mask_Active4[0]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[0], d[0]), Mask_Active4[0]));
				d[1]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[1].v4[dim]), rcpRayDir4[1].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[1], d[1]), Mask_Active4[1]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[1], d[1]), Mask_Active4[1]));
				d[2]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[2].v4[dim]), rcpRayDir4[2].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[2], d[2]), Mask_Active4[2]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[2], d[2]), Mask_Active4[2]));
				d[3]	=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o[3].v4[dim]), rcpRayDir4[3].v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4[3], d[3]), Mask_Active4[3]));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4[3], d[3]), Mask_Active4[3]));

				node = BackSideSon;
				if (d_near == 0)
				{
					COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the back  child
				}
				node = FrontSideSon;
				if (d_far == 0)
				{
					COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the front child
				}

				COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN );


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
		IsectShadowPacket4x4(node);

		// Termination test
		if ((_mm_movemask_ps( _mm_or_ps( _mm_or_ps( _mm_or_ps( 
			_mm_cmpgt_ps( is->dist4[0], t_far_4[0] ),	_mm_cmpgt_ps( is->dist4[1], t_far_4[1] ) ), 
			_mm_cmpgt_ps( is->dist4[2], t_far_4[2] ) ), _mm_cmpgt_ps( is->dist4[3], t_far_4[3] ) ) ) == 0) || (stackIndex == 0)) break;

		COUNT_TREE_OPERATOR( -1, TRAVERSE_UP );
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

// ---------------------------------------------------------------------------
// checkVisibility
// ---------------------------------------------------------------------------
void SSERenderPipeline::checkVisibility4x4(const _sse_vec objectPos[], const GPoint* lightPos, const __m128 shadingmask[]) {

	_sse_4x4_raypacket	*shadow_rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isect		*shadow_is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*shadow_rm	= &m_ShadowRMask4x4[0];

	_sse_vec lPos = sse_vset1(lightPos->x,  lightPos->y,  lightPos->z);

	shadow_rp->d[0] = sse_vsub(lPos, objectPos[0]);
	shadow_rp->d[1] = sse_vsub(lPos, objectPos[1]);
	shadow_rp->d[2] = sse_vsub(lPos, objectPos[2]);
	shadow_rp->d[3] = sse_vsub(lPos, objectPos[3]);
	InitShadowPacket4x4();

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
		TraceShadowPacket4x4();
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
			TraceShadowPacket4x4();
		}
	}
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void SSERenderPipeline::Shading4x4 (const int nIdx) {

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

	bool bIsEnableShadow       = m_bIsEnableShadow;
	bool bIsRunShadowChk       = false;
	bool bIsEnableLocalShading = m_bIsEnableLocalShading;
	bool bIsUseTexture         = m_bIsUseTexture;

	int n_refl = 0;
	int n_refr = 0;

	GColor global_ambient = m_Scene->getGlobalAmbient();
	_sse_vec global_ambt  = sse_vset1(global_ambient.r, global_ambient.g, global_ambient.b);

	_sse_vec	mat_cAmbt[4], mat_cDiff[4], mat_cSpec[4], mat_cEmit[4];
	_sse_float	mat_fRough[4];
	_sse_float	mat_fRefl[4], mat_fRefr[4], mat_fRIdx[4];
	_sse_vec	mat_cTex[4];
	_sse_uint	obj_num[4];


#if 0
	i = 15;
	for ( y = 3; y >= 0; y-- ) {
	for ( x = 3; x >= 0; x-- ) {
		if (rm->mask[i] && is->tacc[i] != 0) {
				const int triID      = is->tacc[i]-1;
				GVector N = m_Data->m_TriObjList[triID]->calBarycentricNormal(1-is->u[i]-is->v[i], is->u[i], is->v[i]);
				is->n[y].x[x] = N.x;
				is->n[y].y[x] = N.y;
				is->n[y].z[x] = N.z;

				const float shading = N.x * rp->d[y].x[x] + N.y * rp->d[y].y[x] + N.z * rp->d[y].z[x];
				is->color[i] = GColor(shading+0.5f, shading+0.5f, shading+0.5f);
		}
		i--;
	}
	}
	return;
#endif

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

		const GColor ambient    = pMaterial->m_Ambient;
		const GColor diffuse    = pMaterial->m_Diffuse;
		const GColor specular   = pMaterial->m_Specular;
		const GColor emission   = pMaterial->m_Emission;
		const float  reflection = pMaterial->m_fReflection;
		const float  refraction = pMaterial->m_fTransparency;
		const float  refrIndex  = pMaterial->m_fRefractionIndex;
		const UINT   object_num = pObject->m_iObjectNumber;

		// Get object color
		GTexture  *pTexture = NULL;
		GColor texColor;
		if ( bIsUseTexture ) {
			pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
		}

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
				} else {
					is->ads.pri_texture4[y] = _mm_set1_epi32(0);
					mat_cTex[y] = mat_cDiff[y];
				}
		}

		n_refr = (refraction>0)?16:0;
		n_refl = (reflection>0)?16:0;
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

				const GColor ambient    = pMaterial->m_Ambient;
				const GColor diffuse    = pMaterial->m_Diffuse;
				const GColor specular   = pMaterial->m_Specular;
				const GColor emission   = pMaterial->m_Emission;
				const float  reflection = pMaterial->m_fReflection;
				const float  refraction = pMaterial->m_fTransparency;
				const float  refrIndex  = pMaterial->m_fRefractionIndex;
				const UINT   object_num = pObject->m_iObjectNumber;

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
				if ( bIsUseTexture ) {
					GTexture  *pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
					if (pTexture && pTexture->isLoaded()) {
						is->ads.pri_texture[i] = 1;
						GPoint point = m_Data->m_TriObjList[triID]->calBarycentricUV( 1-is->u[i]-is->v[i], is->u[i], is->v[i] );
						float u = point.x;
						float v = point.y;
						texColor = pTexture->getTexel( u, v );
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
		}
		i--;
	}
	}

	} // triangle coherence : if end

	_sse_vec oColor[4];
	_sse_vec N[4], R[4], L[4];

	union { __m128 shadingmask[4]; __m128i ishadingmask[4]; };			// Ray Hit(o) & RayMask(o)
	union { __m128 islightmask[4]; __m128i iislightmask[4]; };			// Ray Hit object is current light?
	union { __m128 noshadowmask[4]; __m128i inoshadowmask[4]; };		// No shadow
	union { __m128 yeshadowmask[4]; __m128i iyeshadowmask[4]; };		// Yes shadow
	union { __m128 isisectmask;    __m128i iisisectmask; };				// ShadowRay Hit(o)
	union { __m128 noisectmask;    __m128i inoisectmask; };				// ShadowRay Hit(x)
	union { __m128 shadowmask[4];  __m128i ishadowmask[4]; };			// Yes shaodw

	for (i = 3; i >= 0; i--) {
		ishadingmask[i] = _mm_cmpgt_epi32(is->tacc4[i], _mm_setzero_si128());
		shadingmask[i]  = _mm_and_ps(shadingmask[i], rm->mask4[i]);
		shadowmask[i]   = _mm_setzero_ps();
	}

	if ( bIsEnableLocalShading ) {
		_sse_4x4_raypacket	*shadow_rp;
		_sse_4x4_isect		*shadow_is;
		_sse_4x4_raymask	*shadow_rm;
		if ( bIsEnableShadow ) {
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
			if ( pLight->isEnabled() != true ) continue;

			GColor   lightColor = pLight->getLightColor();
			_sse_vec lColor     = sse_vset1(lightColor.r, lightColor.g, lightColor.b);
			GPoint   lightPos   = pLight->getPosition();
			_sse_vec lPos       = sse_vset1(lightPos.x, lightPos.y, lightPos.z);

			// 광원 자기자신인 경우
			for (i = 3; i >= 0; i--) {
				iislightmask[i] = _mm_cmpeq_epi32(obj_num[i].v4, _mm_set1_epi32(pLight->getObjectNumber()));
				oColor[i] = sse_vupdate(sse_vadd(oColor[i], sse_vmul(lColor, sse_vset1(pLight->getIntensity()))), oColor[i], _mm_and_ps(islightmask[i], shadingmask[i]));
			}

			// 그림자 확인
			if ( bIsEnableShadow ) {
				checkVisibility4x4(hit_p, &lightPos, shadingmask);

				for (i = 3; i >= 0; i--) {
					__m128 lDist = sse_vlength(sse_vsub(lPos, hit_p[i]));

					// shadow 관련 visible 조건
					// 1) shadingmask           : 물체와 교점있는 것
					// 2) shadow_is->tacc4 > 0  : shadow ray 가 교차점이 뒤에 존재하거나 아니면 거리가 거의 가깝거나
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

					shadowmask[i] = _mm_add_ps(shadowmask[i], _mm_and_ps(_mm_set1_ps(1), yeshadowmask[i]));
				}
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

					shadowmask[i] = _mm_setzero_ps();
				}
			}
		}
	} else {
		for (i = 3; i >= 0; i--) {
			oColor[i] = mat_cTex[i];
		}
	}

	if ( rp->Depth < m_iMaxReflectionDepth) {
		for (i = 3; i >= 0; i--) {
			oColor[i] = sse_vupdate(
				sse_vmul(oColor[i], sse_vsub(sse_vsub(sse_vset1(1.0f), sse_vset1(mat_fRefl[i].v4)), sse_vset1(mat_fRefr[i].v4))),
				oColor[i], shadingmask[i]);
		}
	}

	if ( rp->Depth < 2 ) {
		for (i = 3; i >= 0; i--) {
			is->ads.pri_shadowf4[i] = _mm_and_ps(shadowmask[i], rm->mask4[i]);
		}
	}

	i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
		if (rm->mask[i])
		is->color[i] = GColor(oColor[y].x[x], oColor[y].y[x], oColor[y].z[x]);
	}

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
			InitPacket4x4( nNextIdx );

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
				refl_rm->mask4[0] = _mm_and_ps( _mm_and_ps(refl_rm->mask4[0], rm->mask4[0]), _mm_cmpgt_ps(mat_fRefl[0].v4, _mm_setzero_ps()));
				refl_rm->mask4[1] = _mm_and_ps( _mm_and_ps(refl_rm->mask4[1], rm->mask4[1]), _mm_cmpgt_ps(mat_fRefl[1].v4, _mm_setzero_ps()));
				refl_rm->mask4[2] = _mm_and_ps( _mm_and_ps(refl_rm->mask4[2], rm->mask4[2]), _mm_cmpgt_ps(mat_fRefl[2].v4, _mm_setzero_ps()));
				refl_rm->mask4[3] = _mm_and_ps( _mm_and_ps(refl_rm->mask4[3], rm->mask4[3]), _mm_cmpgt_ps(mat_fRefl[3].v4, _mm_setzero_ps()));
				TracePacket4x4( nNextIdx );
				Shading4x4(nNextIdx);
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
					TracePacket4x4(nNextIdx);
				}
				refl_rm->mask4[0] = SecRay_mask4.mask4[0];
				refl_rm->mask4[1] = SecRay_mask4.mask4[1];
				refl_rm->mask4[2] = SecRay_mask4.mask4[2];
				refl_rm->mask4[3] = SecRay_mask4.mask4[3];
				Shading4x4(nNextIdx);
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
			InitPacket4x4( nNextIdx );

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
				TracePacket4x4(nNextIdx);
				Shading4x4(nNextIdx);
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
					TracePacket4x4(nNextIdx);
				}
				refr_rm->mask4[0] = SecRay_mask4.mask4[0];
				refr_rm->mask4[1] = SecRay_mask4.mask4[1];
				refr_rm->mask4[2] = SecRay_mask4.mask4[2];
				refr_rm->mask4[3] = SecRay_mask4.mask4[3];
				Shading4x4(nNextIdx);
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

// -----------------------------------------------------------
// SSERenderPipeline::RenderPacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::RenderPacket4x4( const int nIdx )
{
	TracePacket4x4(nIdx);
	Shading4x4(nIdx);
}

// -----------------------------------------------------------
// SSERenderPipeline::RenderTiles
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Render4x4( int nJobID )
{
#if 0		// 0: Y방향으로 균등 분할하여 렌더링,  1 : 쓰레드 별로 mix 된 상태로 렌더링
	int xTileEnd = (m_Resolution.x) >> 2;
	int yTileEnd = (m_Resolution.y) >> 2;
	int nTileDelta = m_iThreadCount;
	int tx, ty, txy;
	int i;

	// start spawning rays
	_sse_4x4_raypacket	*rp	= &m_RayPk4x4[0];
	_sse_4x4_isect		*is	= &m_Isect4x4[0];
	_sse_4x4_raymask	*rm	= &m_RMask4x4[0];
	_sse_4x4_raypacket	tpos;		// target position for ray casting (pixel center)
	_sse_4x4_raypacket	delta4;		// delta4 for next target position between packet4x4
	_sse_4x4_raypacket	jpos;		// jittered position for sampling

	union { __m128i addr_delta4; unsigned int addr_delta[4]; };
	addr_delta4 = _mm_set1_epi32(4*nTileDelta);

	__m128 ray_o_x4 = _mm_load1_ps( &m_Origin.x );
	__m128 ray_o_y4 = _mm_load1_ps( &m_Origin.y );
	__m128 ray_o_z4 = _mm_load1_ps( &m_Origin.z );
	__m128 delta_x4 = _mm_mul_ps(_mm_set1_ps(4.0f*nTileDelta), _mm_load1_ps( &m_DX.x ));
	__m128 delta_y4 = _mm_mul_ps(_mm_set1_ps(4.0f*nTileDelta), _mm_load1_ps( &m_DX.y ));
	__m128 delta_z4 = _mm_mul_ps(_mm_set1_ps(4.0f*nTileDelta), _mm_load1_ps( &m_DX.z ));

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

	tx = nThreadID + xTileEnd;
	ty = -1;
	for ( txy = nThreadID; txy < xTileEnd * yTileEnd; txy += nTileDelta, tx += nTileDelta) {
		if (tx >= xTileEnd) {
			ty++;
			tx -= xTileEnd;

			// -----------------------------------------------------------------------
			// Set tpos
			// -----------------------------------------------------------------------
			for(i = 0; i < 4; i++) {
				// tpos.d = m_LeftUp + m_DX * (float)iCastSeq16_x[i] - m_DY * (float)(ty * 4 + iCastSeq16_y[i]);
				tpos.d[i].x4 = _mm_add_ps(m_LeftUp4->x4, _mm_sub_ps(
										  _mm_mul_ps(m_DX4->x4, _mm_add_ps(m_CastSeq4x4_x4[i],_mm_set1_ps(tx<<2))),
										  _mm_mul_ps(m_DY4->x4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));
				tpos.d[i].y4 = _mm_add_ps(m_LeftUp4->y4, _mm_sub_ps(
										  _mm_mul_ps(m_DX4->y4, _mm_add_ps(m_CastSeq4x4_x4[i],_mm_set1_ps(tx<<2))),
										  _mm_mul_ps(m_DY4->y4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));
				tpos.d[i].z4 = _mm_add_ps(m_LeftUp4->z4, _mm_sub_ps(
										  _mm_mul_ps(m_DX4->z4, _mm_add_ps(m_CastSeq4x4_x4[i],_mm_set1_ps(tx<<2))),
										  _mm_mul_ps(m_DY4->z4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));

				//is->addr[i] = iCastSeq16_x[i] + (m_Height - 1 - (ty * 4 + iCastSeq16_y[i])) * m_Width;
				is->addr4[i] = _mm_add_epi32(_mm_add_epi32(_mm_set1_epi32(tx*4), m_iCastSeq4x4_x4[i]), _mm_sub_epi32(m_Tmp2_Addr4[i], _mm_set1_epi32(m_Width*ty*4)));
			}
		}

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
								samp_D[i]  =  m_DX * (nSampX + sgrtRadicalInverse((((tx<<2) + iCastSeq16_x[i]) << 8) + nSampX,3)) * rcpSuperSampling_x
											- m_DY * (nSampY + sgrtRadicalInverse((((ty<<2) + iCastSeq16_y[i]) << 8) + nSampY,5)) * rcpSuperSampling_y;
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
				InitPacket4x4( 0 );	// direction vector normalize 등

				// -----------------------------------------------------------------------
				// Coherence 체크 후 rendering
				// -----------------------------------------------------------------------
				if (rp->IsCoherent()) {
					rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
					TracePacket4x4( 0 );
					Shading4x4( 0 );
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
						TracePacket4x4( 0 );
					}
					memset( &rm->mask4, 255, 16 * 4 );
					Shading4x4( 0 );
				}
				
				// -----------------------------------------------------------------------
				// Copy color info to Memory
				// -----------------------------------------------------------------------
				for ( i = 0; i < 16; i++ ) {
					o_color[i].rgba = _mm_add_ps(o_color[i].rgba, is->color[i].rgba);
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
				is->addr4[i] = _mm_add_epi32( is->addr4[i], addr_delta4 );
			}

	}
#else
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
								samp_D[i]  =  m_DX * (nSampX + sgrtRadicalInverse((((tx<<2) + iCastSeq16_x[i]) << 8) + nSampX,3)) * rcpSuperSampling_x
											- m_DY * (nSampY + sgrtRadicalInverse((((ty<<2) + iCastSeq16_y[i]) << 8) + nSampY,5)) * rcpSuperSampling_y;
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
				InitPacket4x4( 0 );	// direction vector normalize 등

				// -----------------------------------------------------------------------
				// Coherence 체크 후 rendering
				// -----------------------------------------------------------------------
				if (rp->IsCoherent()) {
					rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
					TracePacket4x4( 0 );
					Shading4x4( 0 );
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
						TracePacket4x4( 0 );
					}
					memset( &rm->mask4, 255, 16 * 4 );
					Shading4x4( 0 );
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
#endif
}

// -----------------------------------------------------------
// SSERenderPipeline::Render4x4_ADPSS_OnePass
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Render4x4_ADPSS_OnePass( int nJobID )
{
	int yTileStart = 0;
	int xTileEnd = (m_Resolution.x) >> 2;
	int yTileEnd = (m_Resolution.y) >> 2;
	int tx, ty;
	int x, y;
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

	for ( ty = yTileStart; ty < yTileEnd; ty++ ) {
		unsigned int i = 0;

		// -----------------------------------------------------------------------
		// Set tpos
		// -----------------------------------------------------------------------
		for(i = 0; i < 4; i++) {
			// tpos.d = m_LeftUp + m_DX * (float)iCastSeq16_x[i] - m_DY * (float)(ty * 4 + iCastSeq16_y[i]);
			tpos.d[i].x4 = _mm_add_ps(m_LeftUp4->x4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->x4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->x4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));
			tpos.d[i].y4 = _mm_add_ps(m_LeftUp4->y4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->y4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->y4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));
			tpos.d[i].z4 = _mm_add_ps(m_LeftUp4->z4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->z4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->z4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));

			//is->addr[i] = iCastSeq16_x[i] + (m_Height - 1 - (ty * 4 + iCastSeq16_y[i])) * m_Width;
			is->addr4[i] = _mm_add_epi32(m_iCastSeq4x4_x4[i], _mm_sub_epi32(m_Tmp2_Addr4[i], _mm_set1_epi32(m_Width*ty*4)));
		}

		for ( tx = 0; tx < xTileEnd; tx++ ) {
			unsigned int i, b;

			// -----------------------------------------------------------------------
			// Ray packet 을 셋팅 - 시작점(ray_o_x4, ray_o_y4, ray_o_z4) ~ 끝점(tpos)
			// -----------------------------------------------------------------------
			for ( i = 0; i < 4; i++ ) {
				rp->d[i].x4 = _mm_sub_ps( tpos.d[i].x4, ray_o_x4 );
				rp->d[i].y4 = _mm_sub_ps( tpos.d[i].y4, ray_o_y4 );
				rp->d[i].z4 = _mm_sub_ps( tpos.d[i].z4, ray_o_z4 );
			}
			rp->Depth = 0;
			InitPacket4x4( 0 );	// direction vector normalize 등

			// -----------------------------------------------------------------------
			// Coherence 체크 후 rendering
			// -----------------------------------------------------------------------
			if (rp->IsCoherent()) {
				rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
				TracePacket4x4( 0 );
				Shading4x4( 0 );
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
					TracePacket4x4( 0 );
				}
				memset( &rm->mask4, 255, 16 * 4 );
				Shading4x4( 0 );
			}
			
			// -----------------------------------------------------------------------
			// Copy color info to Memory
			// -----------------------------------------------------------------------
			for ( i = 0; i < 16; i++ ) {
				m_Dest[3*(is->addr[i])]   = is->color[i].r;
				m_Dest[3*(is->addr[i])+1] = is->color[i].g;
				m_Dest[3*(is->addr[i])+2] = is->color[i].b;
			}

			// -----------------------------------------------------------------------
			// Save informations for Adaptive SuperSampling
			// -----------------------------------------------------------------------
			i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
				m_ADPSS_data[is->addr[i]].ray_d.x = tpos.d[y].x[x];
				m_ADPSS_data[is->addr[i]].ray_d.y = tpos.d[y].y[x];
				m_ADPSS_data[is->addr[i]].ray_d.z = tpos.d[y].z[x];

				m_ADPSS_data[is->addr[i]].pri_oid = is->ads.pri_oid[i];			// Primary   hit Object ID
				m_ADPSS_data[is->addr[i]].sec_oid = is->ads.sec_oid[i];			// Secondary hit Object ID
				m_ADPSS_data[is->addr[i]].pri_shadow = is->ads.pri_shadow[i];	// Primary   hit Shadow
				m_ADPSS_data[is->addr[i]].sec_shadow = is->ads.sec_shadow[i];	// Secondary hit Shadow

				m_ADPSS_data[is->addr[i]].pri_normal.x = is->n[y].x[x];	// Primary   hit Object Normal
				m_ADPSS_data[is->addr[i]].pri_normal.y = is->n[y].y[x];
				m_ADPSS_data[is->addr[i]].pri_normal.z = is->n[y].z[x];

				m_ADPSS_data[is->addr[i]].sec_normal.x = is->ads.n2[y].x[x];	// Secondary hit Object Normal
				m_ADPSS_data[is->addr[i]].sec_normal.y = is->ads.n2[y].y[x];
				m_ADPSS_data[is->addr[i]].sec_normal.z = is->ads.n2[y].z[x];

				m_ADPSS_data[is->addr[i]].oColor.rgba = is->color[i].rgba;

				m_ADPSS_data[is->addr[i]].pri_texture = is->ads.pri_texture[i];
				m_ADPSS_data[is->addr[i]].sec_texture = is->ads.sec_texture[i];
			}

			// -----------------------------------------------------------------------
			// Render tile (tpos) 의 위치를 이동
			// -----------------------------------------------------------------------
			for ( i = 0; i < 4; i++ ) {
				tpos.d[i].x4 = _mm_add_ps( tpos.d[i].x4, delta4.d[i].x4 );
				tpos.d[i].y4 = _mm_add_ps( tpos.d[i].y4, delta4.d[i].y4 );
				tpos.d[i].z4 = _mm_add_ps( tpos.d[i].z4, delta4.d[i].z4 );
				is->addr4[i] = _mm_add_epi32( is->addr4[i], addr_delta.v4 );
			}
		}
	}
}

static __m128 fSampSeq_x[4] = { _mm_setr_ps(-0.375f, -0.125f, -0.375f, -0.125f),
								 _mm_setr_ps(0.125f, 0.375f, 0.125f, 0.375f),
								 _mm_setr_ps(-0.375f, -0.125f, -0.375f, -0.125f),
								 _mm_setr_ps(0.125f, 0.375f, 0.125f, 0.375f) };
static __m128 fSampSeq_y[4] = { _mm_setr_ps(-0.375f, -0.375f, -0.125f, -0.125f),
								 _mm_setr_ps(-0.375f, -0.375f, -0.125f, -0.125f),
								 _mm_setr_ps(0.125f, 0.125f, 0.375f, 0.375f),
								 _mm_setr_ps(0.125f, 0.125f, 0.375f, 0.375f) };



#define DETECTOR_PRIMARY_COLOR_THRESHOLD	0.0f
#define DETECTOR_SECONDARY_COLOR_THRESHOLD	0.0f
#define DETECTOR_TEXTURE_COLOR_THRESHOLD	0.5f

#define DETECTOR_SHADING_COLOR_THRESHOLD	0.7f

#define DETECTOR_NORMAL_THREADHOLD			0.4f

#define DETECTOR_COLOR_THRESHOLD			0.04f
#define DETECTOR_EDGE_COLOR_THRESHOLD		0.04f

// -----------------------------------------------------------
// SSERenderPipeline::Render4x4_ADPSS_Detection
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Render4x4_ADPSS_Detection( int nJobID )
{
	const int   nCompareType							= m_Scene->getAdaptiveSamplingCompareType();
	const float fPrimaryOIDRegionColorThreshold			= m_Scene->getPrimaryOIDRegionColorThreshold();
	const float fPrimaryNormalRegionColorThreshold		= m_Scene->getPrimaryNormalRegionColorThreshold();
	const float fPrimaryShadowRegionColorThreshold		= m_Scene->getPrimaryShadowRegionColorThreshold();
	const float fPrimaryTextureRegionColorThreshold		= m_Scene->getPrimaryTextureRegionColorThreshold();
	const float fSecondaryOIDRegionColorThreshold		= m_Scene->getSecondaryOIDRegionColorThreshold();
	const float fSecondaryNormalRegionColorThreshold	= m_Scene->getSecondaryNormalRegionColorThreshold();
	const float fSecondaryShadowRegionColorThreshold	= m_Scene->getSecondaryShadowRegionColorThreshold();
	const float fSecondaryTextureRegionColorThreshold	= m_Scene->getSecondaryTextureRegionColorThreshold();
	const float fEtcRegionColorThreshold				= m_Scene->getEtcRegionColorThreshold();

	const unsigned int wPixels = m_Resolution.x;
	const unsigned int hPixels = m_Resolution.y;
	int j;
	int pi, sub_pi;
	int xPi, yPi;

	int nPixelStart = 0;
	int nPixelEnd = wPixels * hPixels;

	// =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+
	// Thread Setting
	const float fPixelperThread = (wPixels) * (hPixels) * m_fThreadRcpJobSize;

	if (nJobID == -1) {									// No thread
		nPixelStart = 0;
		nPixelEnd;
	} else if (nJobID == (m_iThreadJobSize - 1)) {		// This job is last one.
		nPixelStart	= (int(fPixelperThread *  nJobID    ));
		nPixelEnd;
	} else {											// This job is not last one.
		nPixelStart	= (int(fPixelperThread *  nJobID    ));
		nPixelEnd	= (int(fPixelperThread * (nJobID +1)));
	}
	// =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+ =+

	union { __m128  Mask_PatternTest[4]; __m128i iMask_PatternTest[4]; };
	union { __m128i adjPixAddr4; unsigned int adjPixAddr[4]; };	
	union { __m128i adjPixData4; unsigned int adjPixData[4]; __m128 fadjPixData4; float fadjPixData[4]; };

	yPi = nPixelStart / wPixels;
	xPi = nPixelStart - yPi * wPixels;

	for (pi = nPixelStart; pi < nPixelEnd; pi++, xPi++) {
		if (xPi == wPixels)  { xPi = 0; yPi++; }

		int nSuperSample2x2_Count = 0;
		int nSuperSample2x2_Flag  = 0;

		// -----------------------------------------------------------------------------------------------
		// Pixel 의 difference value 생성
		// -----------------------------------------------------------------------------------------------
		float colordifference = 1.0f;
		if (1) {
			float xvalue = 0, yvalue = 0;

			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			union { __m128 grayScale4[2]; float grayScale[8]; };
			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
				//  grayScale       [0]		[1]	  
				//    0 1 2			O O O	x x x
				//    3 t 4			O t x	x t O
				//    5 6 7			x x x	O O O

			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			__m128 GrayScaleCoef4 = _mm_setr_ps(0.3f, 0.59f, 0.11f, 0);
			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
				// GrayScale = 0.3 * red + 0.59 * green + 0.11 * blue

			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			__m128 xSobelMask4[2], ySobelMask4[2];
			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
				//  xSobelMask4        ySobelMask4
				//   -1  0  1			 1  2  1
				//   -2  0  2			 O  0  0
				//   -1  0  1			-1 -2 -1
				xSobelMask4[0] = _mm_setr_ps( -1, 0, 1, -2 );
				xSobelMask4[1] = _mm_setr_ps(  2, -1, 0, 1 );
				ySobelMask4[0] = _mm_setr_ps( 1, 2, 1, 0 );
				ySobelMask4[1] = _mm_setr_ps( 0, -1, -2, -1 );


			union { __m128i tmpPixAddr4[2]; unsigned int tmpPixAddr[8]; };	
			tmpPixAddr4[0] = _mm_add_epi32(m_Sobel_AdjPixAddr4[0], _mm_set1_epi32(pi));
			tmpPixAddr4[1] = _mm_add_epi32(m_Sobel_AdjPixAddr4[1], _mm_set1_epi32(pi));

			if (xPi == 0 || xPi == wPixels -1 || yPi == 0 || yPi == hPixels -1) {
					float hx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
					float hy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
					for ( int j = -1; j <= 1; ++j ) {
					for ( int i = -1; i <= 1; ++i ) {
						if ( xPi + i >= 0 && xPi + i < (int)wPixels && 
							 yPi + j >= 0 && yPi + j < (int)hPixels ) {
							int index = (yPi + j) * wPixels + (xPi + i);
							float grayScale =   0.3f  * m_ADPSS_data[index].oColor.r +
												0.59f * m_ADPSS_data[index].oColor.g +
												0.11f * m_ADPSS_data[index].oColor.b;
							xvalue += hx[ j + 1 ][ i + 1 ] * grayScale;
							yvalue += hy[ j + 1 ][ i + 1 ] * grayScale;
						}
					}
					}
			} else {
				// grayScale[0~7] 에 GrayScale 값을 저장
				for (j = 0; j < 8; j++) {
					grayScale[j] = sse_dot(GrayScaleCoef4, m_ADPSS_data[tmpPixAddr[j]].oColor.rgba);
				}
				xvalue  = sse_dot(grayScale4[0], xSobelMask4[0]);
				xvalue += sse_dot(grayScale4[1], xSobelMask4[1]);
				yvalue  = sse_dot(grayScale4[0], ySobelMask4[0]);
				yvalue += sse_dot(grayScale4[1], ySobelMask4[1]);
			}
			colordifference = min( 1.0f, fabs( xvalue ) + fabs( yvalue ) );
		}
		// ----------------------------------------------------Pixel 의 difference value 생성-------------

		// Check 4 corner test among adjacent pixels
		for (sub_pi = 0; sub_pi < 4; sub_pi++) {
			// sub_pi :
			//	 [0]		[1]			[2]			[3]		
			//	 O O x		x O O		x x x		x x x	
			//	 O t x		x t O		O t x		x t O	
			//	 x x x		x x x		O O x		x O O	

			// 테두리 영역에 대한 처리
			if (xPi == 0          && (sub_pi & 1) == 0) continue;		// Left   margin
			if (xPi == wPixels -1 && (sub_pi & 1) == 1) continue;		// Right  margin
			if (yPi == 0          && (sub_pi & 2) == 2) continue;		// Bottom margin
			if (yPi == hPixels -1 && (sub_pi & 2) == 0) continue;		// Top    margin


			adjPixAddr4 = _mm_add_epi32(m_ADPSS_AdjPixAddr4[sub_pi], _mm_set1_epi32(pi));

			int nPatternTestResult = 0xf;

			bool flag = false;
			bool checkRegion = false;

			while (1) {

				float fColorThreshold = 0;
				/** 오직 하나의 threshold 로 전체이미지를 color 비교하는경우는 이것만 하고 break */
				if ( ( nCompareType & 512 ) == 512 ) {
					if ( colordifference > fColorThreshold ) {
						flag = true;
						break;
					}
				}

				/** primary oid 영역. */
				for (j = 0; j < 4; j++) {
					adjPixData[j] = m_ADPSS_data[adjPixAddr[j]].pri_oid;
				}
				iMask_PatternTest[sub_pi] = _mm_cmpeq_epi32(_mm_sub_epi32(adjPixData4, _mm_set1_epi32(adjPixData[0])), _mm_setzero_si128());	// 모두 같으면 '1'
				nPatternTestResult = _mm_movemask_ps(Mask_PatternTest[sub_pi]);
				if (nPatternTestResult != 0xf) {
					checkRegion = true;
					if ( ( nCompareType & 1 ) == 1 && colordifference > fPrimaryOIDRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** primary normal */
				for (j = 0; j < 4; j++) {
					if ((_mm_movemask_ps(_mm_cmpeq_ps(m_ADPSS_data[adjPixAddr[j]].pri_normal.v4, _mm_setzero_ps())) & 0x7) == 0x7) {
						m_ADPSS_data[adjPixAddr[j]].pri_normal.v4 = _mm_set1_ps(1.0f);
					}
					fadjPixData[j] = 
						sse_fdot(m_ADPSS_data[adjPixAddr[j]].pri_normal, m_ADPSS_data[adjPixAddr[0]].pri_normal);
				}
				Mask_PatternTest[sub_pi] = _mm_cmpgt_ps(fadjPixData4, _mm_set1_ps(0.5f));														// 모두 0.5 보다 크면 '1'
				nPatternTestResult = _mm_movemask_ps(Mask_PatternTest[sub_pi]);
				if (nPatternTestResult != 0xf) {
					checkRegion = true;
					if ( ( nCompareType & 2 ) == 2 && colordifference > fPrimaryNormalRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** primary shadow */
				for (j = 0; j < 4; j++) {
					adjPixData[j] = m_ADPSS_data[adjPixAddr[j]].pri_shadow;
				}
				iMask_PatternTest[sub_pi] = _mm_cmpeq_epi32(_mm_sub_epi32(adjPixData4, _mm_set1_epi32(adjPixData[0])), _mm_setzero_si128());	// 모두 같으면 '1'
				nPatternTestResult = _mm_movemask_ps(Mask_PatternTest[sub_pi]);
				if (nPatternTestResult != 0xf) {
					checkRegion = true;
					if ( ( nCompareType & 4 ) == 4 && colordifference > fPrimaryShadowRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** primary texture */
				if ( m_ADPSS_data[pi].pri_texture ) {
					checkRegion = true;
					if ( ( nCompareType & 8 ) == 8 && colordifference > fPrimaryTextureRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** secondary oid */
				for (j = 0; j < 4; j++) {
					adjPixData[j] = m_ADPSS_data[adjPixAddr[j]].sec_oid;
				}
				iMask_PatternTest[sub_pi] = _mm_cmpeq_epi32(_mm_sub_epi32(adjPixData4, _mm_set1_epi32(adjPixData[0])), _mm_setzero_si128());	// 모두 같으면 '1'
				nPatternTestResult = _mm_movemask_ps(Mask_PatternTest[sub_pi]);
				if (nPatternTestResult != 0xf) {
					checkRegion = true;
					if ( ( nCompareType & 16 ) == 16 && colordifference > fSecondaryOIDRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** secondary normal */
				for (j = 0; j < 4; j++) {
					if ((_mm_movemask_ps(_mm_cmpeq_ps(m_ADPSS_data[adjPixAddr[j]].sec_normal.v4, _mm_setzero_ps())) & 0x7) == 0x7) {
						m_ADPSS_data[adjPixAddr[j]].sec_normal.v4 = _mm_set1_ps(1.0f);
					}
					fadjPixData[j] = 
						sse_fdot(m_ADPSS_data[adjPixAddr[j]].sec_normal, m_ADPSS_data[adjPixAddr[0]].sec_normal);
				}
				Mask_PatternTest[sub_pi] = _mm_cmpgt_ps(fadjPixData4, _mm_set1_ps(0.5f));														// 모두 0.5 보다 크면 '1'
				nPatternTestResult = _mm_movemask_ps(Mask_PatternTest[sub_pi]);
				if (nPatternTestResult != 0xf) {
					checkRegion = true;
					if ( ( nCompareType & 32 ) == 32 && colordifference > fSecondaryNormalRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** secondary shadow */
				for (j = 0; j < 4; j++) {
					adjPixData[j] = m_ADPSS_data[adjPixAddr[j]].sec_shadow;
				}
				iMask_PatternTest[sub_pi] = _mm_cmpeq_epi32(_mm_sub_epi32(adjPixData4, _mm_set1_epi32(adjPixData[0])), _mm_setzero_si128());	// 모두 같으면 '1'
				nPatternTestResult = _mm_movemask_ps(Mask_PatternTest[sub_pi]);
				if (nPatternTestResult != 0xf) {
					checkRegion = true;
					if ( ( nCompareType & 64 ) == 64 && colordifference > fSecondaryShadowRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** secondary texture */
				if ( m_ADPSS_data[pi].sec_texture ) {
					checkRegion = true;
					if ( ( nCompareType & 128 ) == 128 && colordifference > fSecondaryTextureRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** target 영역이외에 대해서는 pixel color 로 비교 */
				if ( !checkRegion && ( nCompareType & 256 ) == 256 && colordifference > fEtcRegionColorThreshold ) {
					flag = true;
					break;
				}

				break;		// while(1) 
			}

			// ----------------------------------------------
			// Z) Super Sampling 지역 결정
			// ----------------------------------------------
			if ( flag ) {
				nSuperSample2x2_Flag |= 1 << sub_pi;
				nSuperSample2x2_Count++;
			}
		}	// sub_pi loop


		//if (xPi == 0 || xPi == wPixels -1 || yPi == 0  || yPi == hPixels -1) {
		//{
		//	m_Dest[3*pi]   = colordifference;
		//		m_Dest[3*pi+1] = colordifference;
		//		m_Dest[3*pi+2] = colordifference;

		//continue;
		//}

		//----------------------------------------
		// 샘플링 되는 지역만 표시
		//----------------------------------------
		if (m_Scene->isEnableSamplingDebugInfo() == true) {
			if ( nSuperSample2x2_Count > 0 ) {
				m_Dest[3*pi]   = 1;
				m_Dest[3*pi+1] = 1;
				m_Dest[3*pi+2] = 1;
			} else {
				m_Dest[3*pi]   = 0;
				m_Dest[3*pi+1] = 0;
				m_Dest[3*pi+2] = 0;
			}
			continue;
		}
		//----------------------------------------

		// Generate Ray-Packet of SuperSampling
		if ( nSuperSample2x2_Count > 0 ) {
			float fSampWeight = 0.0625f; // 1/16
			for (sub_pi = 0; sub_pi < 4; sub_pi++) {
				if (nSuperSample2x2_Flag & (1 << sub_pi)) {
					// 여기서 super sampleing 함
					m_RayT2x2->Set_Item(pi, sub_pi);
				}
			}
		}

		GColor o_color;
		o_color.rgba = _mm_mul_ps(_mm_set1_ps(1 - 0.25f * nSuperSample2x2_Count), m_ADPSS_data[pi].oColor.rgba);
		m_Dest[3*pi]   = o_color.r;
		m_Dest[3*pi+1] = o_color.g;
		m_Dest[3*pi+2] = o_color.b;

		// Primary hit obj 에 따른 이미지
		//m_Dest[3*pi] = 1.0f * m_ADPSS_data[pi].pri_oid / m_Scene->getObjectCount();
		//m_Dest[3*pi+1] = 0;
		//m_Dest[3*pi+2] = 0;
	}
}

// -----------------------------------------------------------
// SSERenderPipeline::Render4x4_ADPSS_Detection
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Render4x4_ADPSS_TwoPass( int nJobID )
{
	const unsigned int wPixels = m_Resolution.x;
	const unsigned int hPixels = m_Resolution.y;

	_sse_2x2_raypacket		*rp	= &m_RayPk2x2[0];
	_sse_2x2_isect			*is	= &m_Isect2x2[0];
	_sse_2x2_raymask		*rm	= &m_RMask2x2[0];
	_sse_2x2_raypacket	opos;		// origin position for ray casting (camera position)

	const __m128 ray_o_x4 = _mm_load1_ps( &m_Origin.x );
	const __m128 ray_o_y4 = _mm_load1_ps( &m_Origin.y );
	const __m128 ray_o_z4 = _mm_load1_ps( &m_Origin.z );

	opos.o.x4    = ray_o_x4;	opos.o.y4    = ray_o_y4;	opos.o.z4    = ray_o_z4;


	int j;

	const __m128 color_weight = { 0.0625f, 0.0625f, 0.0625f, 0.0625f };

	while ( 1 ) {
		int pi, sub_pi;
		if (m_RayT2x2->Get_Item(pi, sub_pi) == 0) break;

		// ray_o
		rp->o = opos.o;

		// ray_d = pixel_center + (m_DX * fSampSeq_x) - (m_DY * fSampSeq_y) 
		rp->d.x4 = _mm_add_ps(_mm_set1_ps(m_ADPSS_data[pi].ray_d.x),
					_mm_sub_ps(_mm_mul_ps(m_DX4->x4, fSampSeq_x[sub_pi]), _mm_mul_ps(m_DY4->x4, fSampSeq_y[sub_pi])));
		rp->d.y4 = _mm_add_ps(_mm_set1_ps(m_ADPSS_data[pi].ray_d.y),
					_mm_sub_ps(_mm_mul_ps(m_DX4->y4, fSampSeq_x[sub_pi]), _mm_mul_ps(m_DY4->y4, fSampSeq_y[sub_pi])));
		rp->d.z4 = _mm_add_ps(_mm_set1_ps(m_ADPSS_data[pi].ray_d.z),
					_mm_sub_ps(_mm_mul_ps(m_DX4->z4, fSampSeq_x[sub_pi]), _mm_mul_ps(m_DY4->z4, fSampSeq_y[sub_pi])));

		// jitter 시 아래 코드 구현 추가
		// ray_d = ray_d + (jitterX * 0.25f - 0.125f) * m_DX - (jitterY * 0.25f - 0.125f) * m_DY

		rp->d.x4 = _mm_sub_ps( rp->d.x4, ray_o_x4 );
		rp->d.y4 = _mm_sub_ps( rp->d.y4, ray_o_y4 );
		rp->d.z4 = _mm_sub_ps( rp->d.z4, ray_o_z4 );

		InitPacket2x2 ( 0 );	// RMask all clear

		is->addr4       = _mm_set1_epi32(pi);
		rp->Depth = 0;

		// ---------------------------------------------------------
		// Masking 기법을 이용하여 Coherence 를 맞추어 렌더링 시작
		// ---------------------------------------------------------
		unsigned int i, b;
		// coherence 체크 겸 ray dir 결정 (q = 8방향중하나)
		if (rp->IsCoherent()) {
			rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
			TracePacket2x2( 0 );
			Shading2x2( 0 );
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
				TracePacket2x2( 0 );
			}
			memset( &rm->mask4, 255, 16 );
			Shading2x2( 0 );
		}

		// ---------------------------
		// Copy color info to Memory
		// ---------------------------
		Color o_color[16];
		for ( j = 0; j < 4; j++ ) {
			o_color[j].rgba = _mm_mul_ps(is->color[j].rgba, color_weight);
		}

		m_ColorBufferCS.lock();
		for ( j = 0; j < 4; j++ ) {
			m_Dest[3*(is->addr[j])]   += o_color[j].r;
			m_Dest[3*(is->addr[j])+1] += o_color[j].g;
			m_Dest[3*(is->addr[j])+2] += o_color[j].b;
		}
		m_ColorBufferCS.unlock();
	}
}



// -----------------------------------------------------------
// SSERenderPipelineQ::GeneratePrimaryRay_inBlock
//		Block 안에서 Primary Ray 를 생성한다.
// -----------------------------------------------------------
void SSERenderPipeline::Render4x4_ADPSS_OnePass_2_1( int nThreadID )
{
	int xTileEnd = (m_Resolution.x) >> 2;
	int yTileEnd = (m_Resolution.y) >> 2;
	int tx, ty;
	int x, y, i;

	// start spawning rays
	_sse_4x4_rayitemData	*rayitemData;
	_sse_4x4_raypacket		*rp;
	_sse_4x4_raypacket		tpos;		// target position for ray casting (pixel center)
	_sse_4x4_raypacket		delta4;		// delta4 for next target position between packet4x4

	const union { unsigned int f[4]; __m128i v4; } addr_delta = { {4, 4, 4, 4} };

	union { __m128i addr4[4]; unsigned int addr[16]; };
	const __m128 ray_o_x4 = _mm_load1_ps( &m_Origin.x );
	const __m128 ray_o_y4 = _mm_load1_ps( &m_Origin.y );
	const __m128 ray_o_z4 = _mm_load1_ps( &m_Origin.z );
	const __m128 delta_x4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.x ));
	const __m128 delta_y4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.y ));
	const __m128 delta_z4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.z ));

	for ( y = 0; y < 4; y++ ) {
		delta4.d[y].x4 = delta_x4;
		delta4.d[y].y4 = delta_y4;
		delta4.d[y].z4 = delta_z4;
	}

	int yTileSize = int(yTileEnd * m_fThreadRcpCount);

	if (nThreadID == -1) {									// No thread
		ty = 0;
		yTileEnd;
	} else if (nThreadID == (m_iThreadCount - 1)) {			// This thread is last one.
		ty			= yTileSize * nThreadID;
		yTileEnd;
	} else {												// This thread is not last one.
		ty			= yTileSize * nThreadID;
		yTileEnd	= yTileSize * (nThreadID +1);
	}

	for ( ;ty < yTileEnd; ty++ ) {
		// -----------------------------------------------------------------------
		// Set tpos
		// -----------------------------------------------------------------------
		for(i = 0; i < 4; i++) {
			// tpos.d = m_LeftUp + m_DX * (float)iCastSeq16_x[i] - m_DY * (float)(ty * 4 + iCastSeq16_y[i]);
			tpos.d[i].x4 = _mm_add_ps(m_LeftUp4->x4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->x4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->x4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));
			tpos.d[i].y4 = _mm_add_ps(m_LeftUp4->y4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->y4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->y4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));
			tpos.d[i].z4 = _mm_add_ps(m_LeftUp4->z4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->z4, m_CastSeq4x4_x4[i]),
									  _mm_mul_ps(m_DY4->z4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty<<2)))));

			//is->addr[i] = iCastSeq16_x[i] + (m_Height - 1 - (ty * 4 + iCastSeq16_y[i])) * m_Width;
			addr4[i] = _mm_add_epi32(m_iCastSeq4x4_x4[i], _mm_sub_epi32(m_Tmp2_Addr4[i], _mm_set1_epi32(m_Width*ty*4)));
		}

		for ( tx = 0; tx < xTileEnd; tx++ ) {
			rayitemData = m_RayQ4x4->IdleQ_DeQueue();
			rp = rayitemData->rp;

			// -----------------------------------------------------------------------
			// Ray packet 을 셋팅 - 시작점(ray_o_x4, ray_o_y4, ray_o_z4) ~ 끝점(tpos)
			// -----------------------------------------------------------------------
			for ( i = 0; i < 4; i++ ) {
				rp->o[i].x4 = ray_o_x4;
				rp->o[i].y4 = ray_o_y4;
				rp->o[i].z4 = ray_o_z4;
				rp->d[i].x4 = _mm_sub_ps( tpos.d[i].x4, ray_o_x4 );
				rp->d[i].y4 = _mm_sub_ps( tpos.d[i].y4, ray_o_y4 );
				rp->d[i].z4 = _mm_sub_ps( tpos.d[i].z4, ray_o_z4 );

				rayitemData->addr4[i] = addr4[i];
			}
			rp->Depth = 0;

			m_RayQ4x4->TodoQ_EnQueue(rayitemData);

			// -----------------------------------------------------------------------
			// Render tile (tpos) 의 위치를 이동
			// -----------------------------------------------------------------------
			for ( i = 0; i < 4; i++ ) {
				tpos.d[i].x4 = _mm_add_ps( tpos.d[i].x4, delta4.d[i].x4 );
				tpos.d[i].y4 = _mm_add_ps( tpos.d[i].y4, delta4.d[i].y4 );
				tpos.d[i].z4 = _mm_add_ps( tpos.d[i].z4, delta4.d[i].z4 );
				addr4[i] = _mm_add_epi32( addr4[i], addr_delta.v4 );
			}
		}
	}
}

// -----------------------------------------------------------
// SSERenderPipeline::Render4x4_ADPSS_Detection
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Render4x4_ADPSS_OnePass_2_2( int nThreadID )
{
	//------------------------------------
	// Rendering Ray Packets
	//------------------------------------
	_sse_4x4_rayitemData	*rayitemData;
	_sse_4x4_raypacket		*rp	= &m_RayPk4x4[0];
	_sse_4x4_isect			*is	= &m_Isect4x4[0];
	_sse_4x4_raymask		*rm	= &m_RMask4x4[0];

	int j;

	while ( 1 ) {
		rayitemData	= m_RayQ4x4->TodoQ_DeQueue();
		if (rayitemData == NULL) break;
		*rp = *rayitemData->rp;

		for (j = 0; j < 4; j++) {
			is->addr4[j]  = rayitemData->addr4[j];
		}

		rp->Depth = 0;
		InitPacket4x4 ( 0 );	// RMask all clear

		// ---------------------------------------------------------
		// Masking 기법을 이용하여 Coherence 를 맞추어 렌더링 시작
		// ---------------------------------------------------------
		unsigned int i, b;
		// coherence 체크 겸 ray dir 결정 (q = 8방향중하나)
		if (rp->IsCoherent()) {
			rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
			TracePacket4x4( 0 );
			Shading4x4( 0 );
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
				TracePacket4x4( 0 );
			}
			memset( &rm->mask4, 255, 16 * 4 );
			Shading4x4( 0 );
		}

		// -----------------------------------------------------------------------
		// Copy color info to Memory
		// -----------------------------------------------------------------------
		//m_ColorBufferCS.lock();
		for ( j = 0; j < 16; j++ ) {
			m_Dest[3*(is->addr[j])]   = is->color[j].r;
			m_Dest[3*(is->addr[j])+1] = is->color[j].g;
			m_Dest[3*(is->addr[j])+2] = is->color[j].b;
		}
		//m_ColorBufferCS.unlock();
		m_RayQ4x4->IdleQ_EnQueue(rayitemData);
	}
}