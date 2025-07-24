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
void SSERenderPipeline::InitPacket2x2(int nIdx)
{
	COUNT_STATE( nIdx, RAY_COUNT, 1 );
	_sse_2x2_raypacket	*rp = &m_RayPk2x2[nIdx];
	_sse_2x2_isect		*is = &m_Isect2x2[nIdx];
	_sse_2x2_raymask	*rm	= &m_RMask2x2[nIdx];

	// Clear intersection information
	is->tacc4 = _mm_setzero_si128();
	memset( &is->color, 0,   64 );
	memset( &rm->mask,  255, 16 );

	// Normalize ray's direction vector
	const __m128 v1 = sse_rsqrt( _mm_add_ps( _mm_add_ps( _mm_mul_ps( rp->d.x4, rp->d.x4 ), 
					  _mm_mul_ps( rp->d.y4, rp->d.y4 ) ), _mm_mul_ps( rp->d.z4, rp->d.z4 ) ) );
	rp->d.x4 = _mm_mul_ps( rp->d.x4, v1 );
	rp->d.y4 = _mm_mul_ps( rp->d.y4, v1 );
	rp->d.z4 = _mm_mul_ps( rp->d.z4, v1 );

	// Pre-calculate dir_mask ray's direction vector
	rp->xmask =  _mm_movemask_ps( rp->d.x4 );
	rp->ymask =  _mm_movemask_ps( rp->d.y4 );
	rp->zmask =  _mm_movemask_ps( rp->d.z4 );
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::IsectPacket2x2( const KdTreeNode2 *node, int nIdx )
{
	int i;

	_sse_2x2_raypacket	*rp	= &m_RayPk2x2[nIdx];
	_sse_2x2_isect		*is	= &m_Isect2x2[nIdx];
	_sse_2x2_raymask	*rm	= &m_RMask2x2[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	COUNT_STATE( nIdx, VISITED_LEAF_NODE, 1 );
	if( nObjs == 0 )
	{COUNT_STATE( nIdx, EMPTY_NODE, 1 );}

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel2 &acc = m_Data->m_TriAccList[triID];

		union { __m128 Mask_Hit; __m128i iMask_Hit; };

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId)
		{
			COUNT_STATE( nIdx, MAILBOXED_TRIANGLE_COUNT, 1 );
			continue;
		}
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		COUNT_STATE( nIdx, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (acc.isTransparent || !m_bBackFaceCulling) {
			Mask_Hit = rm->mask4;
		} else {
			_sse_vec tri_N;
			tri_N.x4 = _mm_set1_ps(acc.N.x);
			tri_N.y4 = _mm_set1_ps(acc.N.y);
			tri_N.z4 = _mm_set1_ps(acc.N.z);
			Mask_Hit = _mm_and_ps(rm->mask4,
				_mm_cmpgt_ps(sse_vdot(rp->d, tri_N), _mm_setzero_ps()));
		}

		// -----------------------------------------------------------------------------
		// ==>> Plane test (t)
		// -----------------------------------------------------------------------------
		const unsigned int k	= acc.k;

		__m128 nd, f;

			nd	=_mm_add_ps(rp->d.v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->d.v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->d.v4[kv])));
			f	=_mm_sub_ps(_mm_set_ps1(acc.n_d),
					 _mm_add_ps(rp->o.v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->o.v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->o.v4[kv]))));
			f	=_mm_mul_ps(f,sse_inverse(nd));

			Mask_Hit	= _mm_and_ps(Mask_Hit,
				_mm_and_ps(_mm_cmpge_ps(is->dist4,f),	_mm_cmpgt_ps(f,_mm_set1_ps(EPSILON))));	// eps < f <= Hit4.dist

		if (_mm_movemask_ps(Mask_Hit)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> Triangle boundaries test (u, v)
		// -----------------------------------------------------------------------------
		__m128 hu, hv;
		__m128 lambda, mue;

			hu		= _mm_add_ps(rp->o.v4[ku], _mm_mul_ps(f,rp->d.v4[ku]));
			hv		= _mm_add_ps(rp->o.v4[kv], _mm_mul_ps(f,rp->d.v4[kv]));

			lambda	= _mm_add_ps(_mm_mul_ps(hu,_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv,_mm_set_ps1(acc.b_nv)));
			lambda	= _mm_add_ps(lambda,_mm_set_ps1(acc.b_d));
			mue		= _mm_add_ps(_mm_mul_ps(hu,_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv,_mm_set_ps1(acc.c_nv)));
			mue		= _mm_add_ps(mue,   _mm_set_ps1(acc.c_d));

			Mask_Hit = _mm_and_ps(Mask_Hit, _mm_cmpgt_ps(lambda,_mm_setzero_ps()));
			Mask_Hit = _mm_and_ps(Mask_Hit, _mm_cmpgt_ps(mue,   _mm_setzero_ps()));
			Mask_Hit = _mm_and_ps(Mask_Hit, _mm_cmple_ps(_mm_add_ps(lambda,mue), _mm_set1_ps(1)));

		if (_mm_movemask_ps(Mask_Hit)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
			is->u4		= sse_update(lambda, is->u4,		Mask_Hit);
			is->v4		= sse_update(mue,    is->v4,		Mask_Hit);
			is->dist4	= sse_update(f,      is->dist4,	Mask_Hit);
			is->tacc4	= _mm_or_si128( _mm_andnot_si128( iMask_Hit, is->tacc4 ),
										_mm_and_si128   ( iMask_Hit, tacc4 ) );
	}
}

// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::TracePacket2x2( int nIdx )
{
	int i;

	_sse_2x2_raypacket	*rp	= &m_RayPk2x2[nIdx];
	_sse_2x2_isect		*is	= &m_Isect2x2[nIdx];
	_sse_2x2_raymask	*rm	= &m_RMask2x2[nIdx];

	// ---------------------------------------------------------------------------
	// 유효한 Ray 없는 경우 return
	// ---------------------------------------------------------------------------
	if (_mm_movemask_ps(rm->mask4) == 0) return;

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist4	= sse_update(_mm_set1_ps(100000), is->dist4, rm->mask4);
	is->tacc4	= sse_update(_mm_setzero_si128(), is->tacc4, rm->imask4);

	// ray direction
	const unsigned int* ray_dir = &raydir[rp->RayWay][0][0];	// Get precomputed the traversal order (front/back)
																//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode2* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// Active Mask
	__m128 Mask_Active4 = rm->mask4;

	// ray reciprocal direction
	_sse_vec rcpRayDir4;
	for (i = 0; i < AXIS_SIZE; i++) {
		rcpRayDir4.v4[i] = sse_inverse(rp->d.v4[i]);
	}

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	__m128 t_near4, t_far_4;
	t_near4	= _mm_setzero_ps();				//	_mm_setzero_ps();
	t_far_4	= _mm_set1_ps(100000);			//	_mm_set_ps1(INFINITY);

	{
		__m128 scenebox_min_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.x);
		__m128 scenebox_min_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.y);
		__m128 scenebox_min_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.z);
		__m128 scenebox_max_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.x);
		__m128 scenebox_max_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.y);
		__m128 scenebox_max_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.z);

		__m128 l1, l2;
		l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_x, rp->o.x4), rcpRayDir4.x4);
		l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_x, rp->o.x4), rcpRayDir4.x4);
		t_near4 = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4 );
		t_far_4 = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4 );
		l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_y, rp->o.y4), rcpRayDir4.y4);
		l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_y, rp->o.y4), rcpRayDir4.y4);
		t_near4 = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4 );
		t_far_4 = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4 );
		l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_z, rp->o.z4), rcpRayDir4.z4);
		l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_z, rp->o.z4), rcpRayDir4.z4);
		t_near4 = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4 );
		t_far_4 = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4 );
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const __m128 node_split4 = _mm_set_ps1(SPLIT_POS(*node));
			const unsigned int dim	= SPLIT_AXIS(*node);
			KdTreeNode2 *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode2 *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			__m128 d;
			
				d		=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o.v4[dim]), rcpRayDir4.v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4, d), Mask_Active4));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4, d), Mask_Active4));

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
										t_far_4		= _mm_min_ps(t_far_4,d);
										Mask_Active4= _mm_and_ps(Mask_Active4, _mm_cmple_ps(t_near4, t_far_4));
										continue;
									}

					m_Stack2x2[stackIndex].t_far_4 = t_far_4;
					m_Stack2x2[stackIndex].t_near4 = _mm_max_ps(t_near4, d);
					t_far_4		= _mm_min_ps(t_far_4,d);
					Mask_Active4= _mm_and_ps(Mask_Active4, _mm_cmple_ps(t_near4, t_far_4));

				m_Stack2x2[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		IsectPacket2x2(node, nIdx);

		// Termination test
		if ((_mm_movemask_ps( 
			_mm_cmpgt_ps( is->dist4, t_far_4 )) == 0) || (stackIndex == 0)) break;

		COUNT_TREE_OPERATOR( nIdx, TRAVERSE_UP );
		// Stack pop
		--stackIndex;
		node		= m_Stack2x2[stackIndex].node;
		t_near4		= m_Stack2x2[stackIndex].t_near4;
		t_far_4		= m_Stack2x2[stackIndex].t_far_4;
		Mask_Active4 = _mm_cmple_ps(t_near4, t_far_4);
	}
}

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::InitPacket
//		Initialize a packet for tracing
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::InitShadowPacket2x2(void)
{
	COUNT_STATE( -1, RAY_COUNT, 1 );
	_sse_2x2_raypacket	*rp = &m_ShadowRayPk2x2[0];
	_sse_2x2_isect		*is = &m_ShadowIsect2x2[0];
	_sse_2x2_raymask	*rm	= &m_ShadowRMask2x2[0];

	// Clear intersection information
	is->tacc4 = _mm_setzero_si128();	// shadow rays : only tacc use

	// Normalize ray's direction vector
	const __m128 v1 = sse_rsqrt( _mm_add_ps( _mm_add_ps( _mm_mul_ps( rp->d.x4, rp->d.x4 ), 
					  _mm_mul_ps( rp->d.y4, rp->d.y4 ) ), _mm_mul_ps( rp->d.z4, rp->d.z4 ) ) );
	rp->d.x4 = _mm_mul_ps( rp->d.x4, v1 );
	rp->d.y4 = _mm_mul_ps( rp->d.y4, v1 );
	rp->d.z4 = _mm_mul_ps( rp->d.z4, v1 );

	// Pre-calculate dir_mask ray's direction vector
	rp->xmask =  _mm_movemask_ps( rp->d.x4 );
	rp->ymask =  _mm_movemask_ps( rp->d.y4 );
	rp->zmask =  _mm_movemask_ps( rp->d.z4 );
}


// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::IsectShadowPacket2x2( const KdTreeNode2 *node )
{
	int i;

	_sse_2x2_raypacket	*rp	= &m_ShadowRayPk2x2[0];
	_sse_2x2_isect		*is	= &m_ShadowIsect2x2[0];
	_sse_2x2_raymask	*rm	= &m_ShadowRMask2x2[0];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	COUNT_STATE( -1, VISITED_LEAF_NODE, 1 );
	if( nObjs == 0 )
	{COUNT_STATE( -1, EMPTY_NODE, 1 );}

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel2 &acc = m_Data->m_TriAccList[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId)
		{
			COUNT_STATE( -1, MAILBOXED_TRIANGLE_COUNT, 1 );
			continue;
		}
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		COUNT_STATE( -1, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// 투명한 물체는 투과
		// ---------------------------------------------------------------
		if (acc.isTransparent) continue;

		union { __m128 Mask_Hit; __m128i iMask_Hit; };
		Mask_Hit = rm->mask4;

		// -----------------------------------------------------------------------------
		// ==>> Plane test (t)
		// -----------------------------------------------------------------------------
		const unsigned int k	= acc.k;

		__m128 nd, f;

			nd	=_mm_add_ps(rp->d.v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->d.v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->d.v4[kv])));
			f	=_mm_sub_ps(_mm_set_ps1(acc.n_d),
					 _mm_add_ps(rp->o.v4[k], _mm_add_ps(_mm_mul_ps(_mm_set_ps1(acc.n_u),rp->o.v4[ku]),
								_mm_mul_ps(_mm_set_ps1(acc.n_v),rp->o.v4[kv]))));
			f	=_mm_mul_ps(f,sse_inverse(nd));

			Mask_Hit	= _mm_and_ps(Mask_Hit,
				_mm_and_ps(_mm_cmpge_ps(is->dist4,f),	_mm_cmpgt_ps(f,_mm_set1_ps(EPSILON))));	// eps < f <= Hit4.dist

		if (_mm_movemask_ps(Mask_Hit)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> Triangle boundaries test (u, v)
		// -----------------------------------------------------------------------------
		__m128 hu, hv;
		__m128 lambda, mue;

			hu		= _mm_add_ps(rp->o.v4[ku], _mm_mul_ps(f,rp->d.v4[ku]));
			hv		= _mm_add_ps(rp->o.v4[kv], _mm_mul_ps(f,rp->d.v4[kv]));

			lambda	= _mm_add_ps(_mm_mul_ps(hu,_mm_set_ps1(acc.b_nu)),	_mm_mul_ps(hv,_mm_set_ps1(acc.b_nv)));
			lambda	= _mm_add_ps(lambda,_mm_set_ps1(acc.b_d));
			mue		= _mm_add_ps(_mm_mul_ps(hu,_mm_set_ps1(acc.c_nu)),	_mm_mul_ps(hv,_mm_set_ps1(acc.c_nv)));
			mue		= _mm_add_ps(mue,   _mm_set_ps1(acc.c_d));

			Mask_Hit = _mm_and_ps(Mask_Hit, _mm_cmpgt_ps(lambda,_mm_setzero_ps()));
			Mask_Hit = _mm_and_ps(Mask_Hit, _mm_cmpgt_ps(mue,   _mm_setzero_ps()));
			Mask_Hit = _mm_and_ps(Mask_Hit, _mm_cmple_ps(_mm_add_ps(lambda,mue), _mm_set1_ps(1)));

		if (_mm_movemask_ps(Mask_Hit)==0) continue;

		// -----------------------------------------------------------------------------
		// ==>> hit4 updates ...
		// -----------------------------------------------------------------------------
		const __m128i tacc4 = _mm_set1_epi32( (unsigned int)(triID+1) );
			//is->u4		= sse_update(lambda, is->u4,		Mask_Hit);
			//is->v4		= sse_update(mue,    is->v4,		Mask_Hit);
			is->dist4	= sse_update(f,      is->dist4,	Mask_Hit);
			is->tacc4	= _mm_or_si128( _mm_andnot_si128( iMask_Hit, is->tacc4 ),
										_mm_and_si128   ( iMask_Hit, tacc4 ) );
	}
}


// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::TraceShadowPacket2x2( void )
{
	int i;

	_sse_2x2_raypacket	*rp	= &m_ShadowRayPk2x2[0];
	_sse_2x2_isect		*is	= &m_ShadowIsect2x2[0];
	_sse_2x2_raymask	*rm	= &m_ShadowRMask2x2[0];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist4	= sse_update(_mm_set1_ps(100000), is->dist4, rm->mask4);
	is->tacc4	= sse_update(_mm_setzero_si128(), is->tacc4, rm->imask4);

	// ray direction
	const unsigned int* ray_dir = &raydir[rp->RayWay][0][0];	// Get precomputed the traversal order (front/back)
																//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode2* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// Active Mask
	__m128 Mask_Active4 = rm->mask4;

	// ray reciprocal direction
	_sse_vec rcpRayDir4;
	for (i = 0; i < AXIS_SIZE; i++) {
		rcpRayDir4.v4[i] = sse_inverse(rp->d.v4[i]);
	}

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	__m128 t_near4, t_far_4;
	t_near4	= _mm_setzero_ps();				//	_mm_setzero_ps();
	t_far_4	= _mm_set1_ps(100000);			//	_mm_set_ps1(INFINITY);

	{
		__m128 scenebox_min_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.x);
		__m128 scenebox_min_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.y);
		__m128 scenebox_min_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Min.z);
		__m128 scenebox_max_x = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.x);
		__m128 scenebox_max_y = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.y);
		__m128 scenebox_max_z = _mm_set1_ps(m_Data->m_SceneBBox.m_Max.z);

		__m128 l1, l2;
		l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_x, rp->o.x4), rcpRayDir4.x4);
		l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_x, rp->o.x4), rcpRayDir4.x4);
		t_near4 = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4 );
		t_far_4 = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4 );
		l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_y, rp->o.y4), rcpRayDir4.y4);
		l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_y, rp->o.y4), rcpRayDir4.y4);
		t_near4 = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4 );
		t_far_4 = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4 );
		l1 = _mm_mul_ps(_mm_sub_ps(scenebox_min_z, rp->o.z4), rcpRayDir4.z4);
		l2 = _mm_mul_ps(_mm_sub_ps(scenebox_max_z, rp->o.z4), rcpRayDir4.z4);
		t_near4 = _mm_max_ps( _mm_min_ps( l1,l2 ), t_near4 );
		t_far_4 = _mm_min_ps( _mm_max_ps( l1,l2 ), t_far_4 );
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const __m128 node_split4 = _mm_set_ps1(SPLIT_POS(*node));
			const unsigned int dim	= SPLIT_AXIS(*node);
			KdTreeNode2 *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode2 *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			__m128 d;
			
				d		=  _mm_mul_ps(_mm_sub_ps(node_split4, rp->o.v4[dim]), rcpRayDir4.v4[dim]);
				d_near	|= _mm_movemask_ps(_mm_and_ps(_mm_cmple_ps(t_near4, d), Mask_Active4));
				d_far	|= _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(t_far_4, d), Mask_Active4));

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
										t_far_4		= _mm_min_ps(t_far_4,d);
										Mask_Active4= _mm_and_ps(Mask_Active4, _mm_cmple_ps(t_near4, t_far_4));
										continue;
									}

					m_Stack2x2[stackIndex].t_far_4 = t_far_4;
					m_Stack2x2[stackIndex].t_near4 = _mm_max_ps(t_near4, d);
					t_far_4		= _mm_min_ps(t_far_4,d);
					Mask_Active4= _mm_and_ps(Mask_Active4, _mm_cmple_ps(t_near4, t_far_4));

				m_Stack2x2[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		IsectShadowPacket2x2(node);

		// Termination test
		if ((_mm_movemask_ps( 
			_mm_cmpgt_ps( is->dist4, t_far_4 )) == 0) || (stackIndex == 0)) break;

		COUNT_TREE_OPERATOR( -1, TRAVERSE_UP );
		// Stack pop
		--stackIndex;
		node		= m_Stack2x2[stackIndex].node;
		t_near4		= m_Stack2x2[stackIndex].t_near4;
		t_far_4		= m_Stack2x2[stackIndex].t_far_4;
		Mask_Active4 = _mm_cmple_ps(t_near4, t_far_4);
	}
}

// ---------------------------------------------------------------------------
// checkVisibility
// ---------------------------------------------------------------------------
void SSERenderPipeline::checkVisibility2x2(const _sse_vec &objectPos, const GPoint* lightPos, const __m128 &shadingmask) {

	_sse_2x2_raypacket	*shadow_rp	= &m_ShadowRayPk2x2[0];
	_sse_2x2_isect		*shadow_is	= &m_ShadowIsect2x2[0];
	_sse_2x2_raymask	*shadow_rm	= &m_ShadowRMask2x2[0];

	_sse_vec lPos = sse_vset1(lightPos->x, lightPos->y, lightPos->z);

	shadow_rp->d = sse_vsub(lPos, objectPos);
	InitShadowPacket2x2();

	shadow_rp->o = sse_vadd(objectPos, sse_vmul(shadow_rp->d, sse_vset1(RAY_START_EPSILON)));

	unsigned int i, b;
	// coherence 체크 겸 ray dir 결정 (q = 8방향중하나)
	if (shadow_rp->IsCoherent()) {
		shadow_rp->RayWay = (shadow_rp->xmask & 1) + (shadow_rp->ymask & 2) + (shadow_rp->zmask & 4);
		shadow_rm->mask4 = shadingmask;
		TraceShadowPacket2x2();
	} else {
		// detect octants and render one by one
		bool b_quad[8] = { false, false, false, false, false, false, false, false };
		for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
			int q = ((shadow_rp->xmask & b)?1:0) + ((shadow_rp->ymask & b)?2:0) + ((shadow_rp->zmask & b)?4:0);
			b_quad[q] = true;
		}
		for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
			for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
				int q = ((shadow_rp->xmask & b)?1:0) + ((shadow_rp->ymask & b)?2:0) + ((shadow_rp->zmask & b)?4:0);
				if (q == cq) shadow_rm->mask[i] = 0xffffffff; else shadow_rm->mask[i] = 0;
			}
			shadow_rm->mask4 = _mm_and_ps(shadow_rm->mask4, shadingmask);
			shadow_rp->RayWay = cq;
			TraceShadowPacket2x2();
		}
	}
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void SSERenderPipeline::Shading2x2 (int nIdx) {

	int i, x;

	_sse_2x2_raypacket	*rp	= &m_RayPk2x2[nIdx];
	_sse_2x2_isect		*is	= &m_Isect2x2[nIdx];
	_sse_2x2_raymask	*rm	= &m_RMask2x2[nIdx];

	// --------------------------------------------------------------
	// 유효한 Ray 없는 경우 return
	// --------------------------------------------------------------
	union { __m128i iAvailMask4; __m128 fAvailMask4; };
	iAvailMask4 = _mm_and_si128(rm->imask4, _mm_cmpgt_epi32(is->tacc4, _mm_setzero_si128()));
	if (_mm_movemask_ps(fAvailMask4) == 0) return;

	_sse_vec hit_p;
	hit_p.x4 = _mm_add_ps(rp->o.x4, _mm_mul_ps(rp->d.x4, is->dist4));
	hit_p.y4 = _mm_add_ps(rp->o.y4, _mm_mul_ps(rp->d.y4, is->dist4));
	hit_p.z4 = _mm_add_ps(rp->o.z4, _mm_mul_ps(rp->d.z4, is->dist4));

	bool bIsEnableShadow       = m_bIsEnableShadow;
	bool bIsRunShadowChk       = false;
	bool bIsEnableLocalShading = m_bIsEnableLocalShading;
	bool bIsUseTexture         = m_bIsUseTexture;

	int n_refl = 0;
	int n_refr = 0;

	GColor global_ambient = m_Scene->getGlobalAmbient();
	_sse_vec global_ambt  = sse_vset1(global_ambient.r, global_ambient.g, global_ambient.b);

	_sse_vec	mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit;
	_sse_float	mat_fRough;
	_sse_float	mat_fRefl, mat_fRefr, mat_fRIdx;
	_sse_vec	mat_cTex;
	_sse_uint	obj_num;


#if 1
	__m128i _test_iTriID4 = _mm_set1_epi32(is->tacc[0]-1);
	union { __m128i iTemp4; __m128 fTemp4; };
	union { __m128i iTriID4; unsigned int iTriID[4]; __m128 fTriID4; float fTriID[4]; };

	int nCoherenceChk  = 0xf;
	iTriID4 = _mm_sub_epi32(is->tacc4, _mm_set1_epi32(1));
	iTemp4 = _mm_cmpeq_epi32(iTriID4, _test_iTriID4);
	nCoherenceChk &= _mm_movemask_ps(fTemp4);

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
		if ( bIsUseTexture ) {
			pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
		}

				mat_cAmbt.x4 = _mm_set1_ps(ambient.r);  mat_cDiff.x4 = _mm_set1_ps(diffuse.r);  mat_cSpec.x4 = _mm_set1_ps(specular.r);  mat_cEmit.x4 = _mm_set1_ps(emission.r);
				mat_cAmbt.y4 = _mm_set1_ps(ambient.g);  mat_cDiff.y4 = _mm_set1_ps(diffuse.g);  mat_cSpec.y4 = _mm_set1_ps(specular.g);  mat_cEmit.y4 = _mm_set1_ps(emission.g);
				mat_cAmbt.z4 = _mm_set1_ps(ambient.b);  mat_cDiff.z4 = _mm_set1_ps(diffuse.b);  mat_cSpec.z4 = _mm_set1_ps(specular.b);  mat_cEmit.z4 = _mm_set1_ps(emission.b);
				mat_fRough.v4 = _mm_set1_ps(0.7);//pMaterial->getRoughness());
				mat_fRefl.v4 = _mm_set1_ps(reflection);
				mat_fRefr.v4 = _mm_set1_ps(refraction);
				mat_fRIdx.v4 = _mm_set1_ps(refrIndex);
				obj_num.v4 = _mm_set1_epi32(object_num);

				__m128 beta  = is->u4;
				__m128 gamma = is->v4;
				__m128 alpha = _mm_sub_ps(_mm_sub_ps(_mm_set1_ps(1), beta), gamma);

				_sse_vec normal;
				m_Data->m_TriObjList[triID]->calBarycentricNormal( alpha, beta, gamma, &normal );
				is->n.x4 = normal.x4;
				is->n.y4 = normal.y4;
				is->n.z4 = normal.z4;

				if (pTexture && pTexture->isLoaded()) {
					__m128 uv[2];
					m_Data->m_TriObjList[triID]->calBarycentricUV( alpha, beta, gamma, uv );
					pTexture->getTexel4( uv, mat_cTex );
				} else {
					mat_cTex = mat_cDiff;
				}

		n_refr = (refraction>0)?4:0;
		n_refl = (reflection>0)?4:0;
	} else {	// triangle coherence : not case
#else
	{
#endif

	i = 3;
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

				mat_cAmbt.x[x] = ambient.r;  mat_cDiff.x[x] = diffuse.r;  mat_cSpec.x[x] = specular.r;  mat_cEmit.x[x] = emission.r;
				mat_cAmbt.y[x] = ambient.g;  mat_cDiff.y[x] = diffuse.g;  mat_cSpec.y[x] = specular.g;  mat_cEmit.y[x] = emission.g;
				mat_cAmbt.z[x] = ambient.b;  mat_cDiff.z[x] = diffuse.b;  mat_cSpec.z[x] = specular.b;  mat_cEmit.z[x] = emission.b;
				mat_fRough.f[x] = 0.7;//pMaterial->getRoughness();
				mat_fRefl.f[x] = reflection;
				mat_fRefr.f[x] = refraction;
				mat_fRIdx.f[x] = refrIndex;
				obj_num.f[x] = object_num;

				n_refl += ((reflection > 0)?1:0);
				n_refr += ((refraction > 0)?1:0);

				GVector N = m_Data->m_TriObjList[triID]->calBarycentricNormal(1-is->u[i]-is->v[i], is->u[i], is->v[i]);
				is->n.x[x] = N.x;
				is->n.y[x] = N.y;
				is->n.z[x] = N.z;

				// Get object color
				GColor texColor;
				if ( bIsUseTexture ) {
					GTexture  *pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
					if (pTexture && pTexture->isLoaded()) {
						GPoint point = m_Data->m_TriObjList[triID]->calBarycentricUV( 1-is->u[i]-is->v[i], is->u[i], is->v[i] );
						float u = point.x;
						float v = point.y;
						texColor = pTexture->getTexel( u, v );
					} else {
						texColor = diffuse;
					}
				} else {
					texColor = diffuse;
				}
				mat_cTex.x[x] = texColor.r;
				mat_cTex.y[x] = texColor.g;
				mat_cTex.z[x] = texColor.b;
		}
		i--;
	}

	} // triangle coherence : if end

	_sse_vec oColor;
	_sse_vec N, R, L;

	union { __m128 shadingmask;  __m128i ishadingmask; };			// Ray Hit(o) & RayMask(o)
	union { __m128 islightmask;  __m128i iislightmask; };			// Ray Hit object is current light?
	union { __m128 noshadowmask; __m128i inoshadowmask; };			// No shadow
	union { __m128 yeshadowmask; __m128i iyeshadowmask; };			// Yes shadow
	union { __m128 isisectmask;  __m128i iisisectmask; };			// ShadowRay Hit(o)
	union { __m128 noisectmask;  __m128i inoisectmask; };			// ShadowRay Hit(x)

	{
		ishadingmask = _mm_cmpgt_epi32(is->tacc4, _mm_setzero_si128());
		shadingmask  = _mm_and_ps(shadingmask, rm->mask4);
	}

	if ( bIsEnableLocalShading ) {
		{
			// Shading 에서, 투명한 물체일때, Normal 과 dir 의 dot 이 < 0 이라면 normal 을 뒤짚는다.
			__m128 mask = _mm_and_ps(
					_mm_cmpgt_ps(mat_fRefr.v4, _mm_setzero_ps()),
					_mm_cmpgt_ps(sse_vdot(is->n, rp->d), _mm_setzero_ps()));
			N = sse_vupdate(sse_vsigned(is->n), is->n, mask);
			R = sse_vnorm(sse_vadd(sse_vmul(_mm_mul_ps(_mm_set1_ps(-2),  sse_vdot(N, rp->d)), N), rp->d));

			// Background color
			oColor = sse_vset1(0.0f);

			// Ambient color
			oColor = sse_vupdate(sse_vmul(global_ambt, mat_cAmbt), oColor, shadingmask);

			// Emission color
			oColor = sse_vupdate(sse_vadd(oColor, mat_cEmit) ,oColor, shadingmask);
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
			iislightmask = _mm_cmpeq_epi32(obj_num.v4, _mm_set1_epi32(pLight->getObjectNumber()));
			oColor = sse_vupdate(sse_vadd(oColor, sse_vmul(lColor, sse_vset1(pLight->getIntensity()))), oColor, _mm_and_ps(islightmask, shadingmask));

			// 그림자 확인
			if ( bIsEnableShadow ) {
				_sse_2x2_raypacket	*shadow_rp;
				_sse_2x2_isect		*shadow_is;
				_sse_2x2_raymask	*shadow_rm;
				shadow_rp	= &m_ShadowRayPk2x2[0];
				shadow_is	= &m_ShadowIsect2x2[0];
				shadow_rm	= &m_ShadowRMask2x2[0];

				checkVisibility2x2(hit_p, &lightPos, shadingmask);

				{
					__m128 lDist = sse_vlength(sse_vsub(lPos, hit_p));

					// shadow 관련 visible 조건
					// 1) shadingmask           : 물체와 교점있는 것
					// 2) shadow_is->tacc4 > 0  : shadow ray 가 교차점이 뒤에 존재하거나 아니면 거리가 거의 가깝거나
					iisisectmask = _mm_cmpgt_epi32(shadow_is->tacc4, _mm_setzero_si128());
					inoisectmask = _mm_cmpeq_epi32(shadow_is->tacc4, _mm_setzero_si128());

					noshadowmask = _mm_or_ps(
							noisectmask,
							_mm_or_ps(
								_mm_cmplt_ps(_mm_abs_ps(_mm_sub_ps(lDist, shadow_is->dist4)), _mm_set1_ps(1.f*EPSILON)),
								_mm_cmpgt_ps(shadow_is->dist4, lDist)));
					yeshadowmask = _mm_andnot_ps(noshadowmask, _mm_andnot_ps(islightmask, shadingmask));
					noshadowmask = _mm_and_ps(noshadowmask, _mm_andnot_ps(islightmask, shadingmask));

					// Phong shading
					L = sse_vnorm(sse_vsub(lPos, hit_p));

					oColor = sse_vupdate(sse_vadd(oColor, sse_vadd(
						sse_vmul(sse_vmul(mat_cTex,  lColor), sse_vset1(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L, N)))),
						sse_vmul(sse_vmul(mat_cSpec, lColor), sse_vset1(sse_pow_Schlick(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L, R)), mat_fRough.v4)))
							)), oColor, noshadowmask);
					// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
					//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
				}
			} else {
				{
					noshadowmask = _mm_andnot_ps(islightmask, shadingmask);

					// Phong shading
					L = sse_vnorm(sse_vsub(lPos, hit_p));

					oColor = sse_vupdate(sse_vadd(oColor, sse_vadd(
						sse_vmul(sse_vmul(mat_cTex,  lColor), sse_vset1(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L, N)))),
						sse_vmul(sse_vmul(mat_cSpec, lColor), sse_vset1(sse_pow_Schlick(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L, R)), mat_fRough.v4)))
							)), oColor, noshadowmask);
					// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
					//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
				}
			}
		}
	} else {
		oColor = mat_cTex;
	}

	{
		if ( rp->Depth < m_iMaxReflectionDepth ) {
			oColor = sse_vupdate(
				sse_vmul(oColor, sse_vsub(sse_vsub(sse_vset1(1.0f), sse_vset1(mat_fRefl.v4)), sse_vset1(mat_fRefr.v4))),
				oColor, shadingmask);
		}
	}

	i = 3; for ( x = 3; x >= 0; x--, i-- ) {
		if (rm->mask[i])
		is->color[i] = GColor(oColor.x[x], oColor.y[x], oColor.z[x]);
	}

	if (rp->Depth < m_iMaxReflectionDepth) {
		// ---------------------------------------------------------------------------
		// reflection
		// ---------------------------------------------------------------------------
		if (n_refl > 0) {
			__m128 dot_i4;

			const int nNextIdx = nIdx+1;

			_sse_2x2_raypacket	*refl_rp	= &m_RayPk2x2[nNextIdx];
			_sse_2x2_isect		*refl_is	= &m_Isect2x2[nNextIdx];
			_sse_2x2_raymask	*refl_rm	= &m_RMask2x2[nNextIdx];

			dot_i4 = sse_vdot(rp->d, is->n);

			refl_rp->d.x4 = _mm_sub_ps(rp->d.x4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4, is->n.x4)));
			refl_rp->d.y4 = _mm_sub_ps(rp->d.y4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4, is->n.y4)));
			refl_rp->d.z4 = _mm_sub_ps(rp->d.z4, _mm_mul_ps(_mm_set1_ps(2), _mm_mul_ps(dot_i4, is->n.z4)));
			InitPacket2x2( nNextIdx );

			refl_rp->o.x4 = _mm_add_ps(hit_p.x4, _mm_mul_ps(refl_rp->d.x4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o.y4 = _mm_add_ps(hit_p.y4, _mm_mul_ps(refl_rp->d.y4, _mm_set1_ps(RAY_START_EPSILON)));
			refl_rp->o.z4 = _mm_add_ps(hit_p.z4, _mm_mul_ps(refl_rp->d.z4, _mm_set1_ps(RAY_START_EPSILON)));

			refl_rp->Depth = rp->Depth+1;

			_sse_2x2_raymask SecRay_mask4;
			SecRay_mask4.mask4 = _mm_and_ps( _mm_and_ps( rm->mask4, (__m128&)_mm_cmpgt_epi32(is->tacc4, _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefl.v4, _mm_setzero_ps()));

			unsigned int b;
			if (refl_rp->IsCoherent()) {
				refl_rp->RayWay = (refl_rp->xmask & 1) + (refl_rp->ymask & 2) + (refl_rp->zmask & 4);
				refl_rm->mask4 = SecRay_mask4.mask4;
				TracePacket2x2(nNextIdx);
				Shading2x2(nNextIdx);
				i = 3; for ( x = 3; x >= 0; x--, i-- ) {
					if (refl_is->tacc[i] && refl_rm->mask[i]) {
						is->color[i] += mat_fRefl.f[x] * refl_is->color[i] * GColor(mat_cTex.x[x], mat_cTex.y[x], mat_cTex.z[x]);
					}
				}
			} else {
				// detect octants and render one by one
				bool b_quad[8] = { false, false, false, false, false, false, false, false };
				for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
					int q = ((refl_rp->xmask & b)?1:0) + ((refl_rp->ymask & b)?2:0) + ((refl_rp->zmask & b)?4:0);
					b_quad[q] = true;
				}
				for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
					for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
						int q = ((refl_rp->xmask & b)?1:0) + ((refl_rp->ymask & b)?2:0) + ((refl_rp->zmask & b)?4:0);
						if (q == cq) refl_rm->mask[i] = 0xffffffff; else refl_rm->mask[i] = 0;
					}
					refl_rm->mask4 = _mm_and_ps( refl_rm->mask4, SecRay_mask4.mask4 );
					refl_rp->RayWay = cq;
					TracePacket2x2(nNextIdx);
				}
				refl_rm->mask4 = SecRay_mask4.mask4;
				Shading2x2(nNextIdx);
				i = 3; for ( x = 3; x >= 0; x--, i-- ) {
					if (refl_is->tacc[i] && refl_rm->mask[i]) {
						is->color[i] += mat_fRefl.f[x] * refl_is->color[i] * GColor(mat_cTex.x[x], mat_cTex.y[x], mat_cTex.z[x]);
					}
				}
			}
		}

		// ---------------------------------------------------------------------------
		// refraction
		// ---------------------------------------------------------------------------
		if (n_refr > 0) {
			__m128 dot_i4;
			__m128 dot_r4;
			__m128 n_div_nt4;

			const int nNextIdx = nIdx+1;

			_sse_2x2_raypacket	*refr_rp	= &m_RayPk2x2[nNextIdx];
			_sse_2x2_isect		*refr_is	= &m_Isect2x2[nNextIdx];
			_sse_2x2_raymask	*refr_rm	= &m_RMask2x2[nNextIdx];

			dot_i4 = sse_vdot(rp->d, is->n);

			__m128 mask;
			mask		 = _mm_cmplt_ps(dot_i4, _mm_setzero_ps());

			n_div_nt4 = _mm_or_ps ( _mm_and_ps(mask, fastxovery(_mm_set1_ps(AIR_INDEX), mat_fRIdx.v4)),
									_mm_andnot_ps(mask, fastxovery(mat_fRIdx.v4, _mm_set1_ps(AIR_INDEX))));

			dot_r4 = _mm_sqrt_ps(_mm_abs_ps(
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps( _mm_mul_ps(n_div_nt4, n_div_nt4), 
							_mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps(dot_i4,dot_i4))	))));

			mask = _mm_cmplt_ps(n_div_nt4, _mm_set1_ps(1));
			refr_rp->d.x4 = _mm_add_ps( _mm_mul_ps(n_div_nt4, _mm_sub_ps(rp->d.x4, _mm_mul_ps(is->n.x4, dot_i4))),  sse_update(_mm_neg_ps(_mm_mul_ps(is->n.x4, dot_r4)), _mm_mul_ps(is->n.x4, dot_r4), mask));
			refr_rp->d.y4 = _mm_add_ps( _mm_mul_ps(n_div_nt4, _mm_sub_ps(rp->d.y4, _mm_mul_ps(is->n.y4, dot_i4))),  sse_update(_mm_neg_ps(_mm_mul_ps(is->n.y4, dot_r4)), _mm_mul_ps(is->n.y4, dot_r4), mask));
			refr_rp->d.z4 = _mm_add_ps( _mm_mul_ps(n_div_nt4, _mm_sub_ps(rp->d.z4, _mm_mul_ps(is->n.z4, dot_i4))),  sse_update(_mm_neg_ps(_mm_mul_ps(is->n.z4, dot_r4)), _mm_mul_ps(is->n.z4, dot_r4), mask));
			InitPacket2x2( nNextIdx );

			refr_rp->o.x4 = _mm_add_ps(hit_p.x4, _mm_mul_ps(refr_rp->d.x4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o.y4 = _mm_add_ps(hit_p.y4, _mm_mul_ps(refr_rp->d.y4, _mm_set1_ps(RAY_START_EPSILON)));
			refr_rp->o.z4 = _mm_add_ps(hit_p.z4, _mm_mul_ps(refr_rp->d.z4, _mm_set1_ps(RAY_START_EPSILON)));

			refr_rp->Depth = rp->Depth+1;

			_sse_2x2_raymask SecRay_mask4;
			SecRay_mask4.mask4 = _mm_and_ps( _mm_and_ps( rm->mask4, (__m128&)_mm_cmpgt_epi32(is->tacc4, _mm_setzero_si128())), _mm_cmpgt_ps(mat_fRefr.v4, _mm_setzero_ps()));

			unsigned int b;
			if (refr_rp->IsCoherent()) {
				refr_rp->RayWay = (refr_rp->xmask & 1) + (refr_rp->ymask & 2) + (refr_rp->zmask & 4);
				refr_rm->mask4 = SecRay_mask4.mask4;
				TracePacket2x2(nNextIdx);
				Shading2x2(nNextIdx);
				i = 3; for ( x = 3; x >= 0; x--, i-- ) {
					if (refr_is->tacc[i] && refr_rm->mask[i]) {
						is->color[i] += mat_fRefr.f[x] * refr_is->color[i] * GColor(mat_cTex.x[x], mat_cTex.y[x], mat_cTex.z[x]);
					}
				}
			} else {
				// detect octants and render one by one
				bool b_quad[8] = { false, false, false, false, false, false, false, false };
				for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
					int q = ((refr_rp->xmask & b)?1:0) + ((refr_rp->ymask & b)?2:0) + ((refr_rp->zmask & b)?4:0);
					b_quad[q] = true;
				}
				for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
					for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
						int q = ((refr_rp->xmask & b)?1:0) + ((refr_rp->ymask & b)?2:0) + ((refr_rp->zmask & b)?4:0);
						if (q == cq) refr_rm->mask[i] = 0xffffffff; else refr_rm->mask[i] = 0;
					}
					refr_rm->mask4 = _mm_and_ps( refr_rm->mask4, SecRay_mask4.mask4 );
					refr_rp->RayWay = cq;
					TracePacket2x2(nNextIdx);
				}
				refr_rm->mask4 = SecRay_mask4.mask4;
				Shading2x2(nNextIdx);
				i = 3; for ( x = 3; x >= 0; x--, i-- ) {
					if (refr_is->tacc[i] && refr_rm->mask[i]) {
						is->color[i] += mat_fRefr.f[x] * refr_is->color[i] * GColor(mat_cTex.x[x], mat_cTex.y[x], mat_cTex.z[x]);
					}
				}
			}
		}
	}
}

// -----------------------------------------------------------
// SSERenderPipeline::RenderPacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::RenderPacket2x2( int nIdx )
{
	TracePacket2x2(nIdx);
	Shading2x2(nIdx);
}

// -----------------------------------------------------------
// SSERenderPipeline::RenderTiles
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Render2x2( int nJobID )
{
	int yTileStart = 0;
	int xTileEnd = (m_Resolution.x) >> 1;
	int yTileEnd = (m_Resolution.y) >> 1;
	int tx, ty;

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
	_sse_2x2_raypacket	*rp	= &m_RayPk2x2[0];
	_sse_2x2_isect		*is	= &m_Isect2x2[0];
	_sse_2x2_raymask	*rm	= &m_RMask2x2[0];
	_sse_2x2_raypacket	tpos;		// target position for ray casting (pixel center)
	_sse_2x2_raypacket	delta4;		// delta4 for next target position between packet4x4
	_sse_2x2_raypacket	jpos;		// jittered position for sampling

	const union { unsigned int f[4]; __m128i v4; } addr_delta = { 2, 2, 2, 2 };

	const __m128 ray_o_x4 = _mm_load1_ps( &m_Origin.x );
	const __m128 ray_o_y4 = _mm_load1_ps( &m_Origin.y );
	const __m128 ray_o_z4 = _mm_load1_ps( &m_Origin.z );
	const __m128 delta_x4 = _mm_mul_ps(_mm_set1_ps(2), _mm_load1_ps( &m_DX.x ));
	const __m128 delta_y4 = _mm_mul_ps(_mm_set1_ps(2), _mm_load1_ps( &m_DX.y ));
	const __m128 delta_z4 = _mm_mul_ps(_mm_set1_ps(2), _mm_load1_ps( &m_DX.z ));

	rp->o.x4    = ray_o_x4;	rp->o.y4    = ray_o_y4;	rp->o.z4    = ray_o_z4;
	delta4.d.x4 = delta_x4;	delta4.d.y4 = delta_y4;	delta4.d.z4 = delta_z4;

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
		{
			// tpos.d = m_LeftUp + m_DX * (float)iCastSeq2x2_x[i] - m_DY * (float)(ty * 2 + iCastSeq2x2_y[i]);
			tpos.d.x4 = _mm_add_ps(m_LeftUp4->x4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->x4, m_CastSeq2x2_x4[0]),
									  _mm_mul_ps(m_DY4->x4, _mm_add_ps(_mm_set1_ps(ty<<1),m_CastSeq2x2_y4[0]))));
			tpos.d.y4 = _mm_add_ps(m_LeftUp4->y4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->y4, m_CastSeq2x2_x4[0]),
									  _mm_mul_ps(m_DY4->y4, _mm_add_ps(_mm_set1_ps(ty<<1),m_CastSeq2x2_y4[0]))));
			tpos.d.z4 = _mm_add_ps(m_LeftUp4->z4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->z4, m_CastSeq2x2_x4[0]),
									  _mm_mul_ps(m_DY4->z4, _mm_add_ps(_mm_set1_ps(ty<<1),m_CastSeq2x2_y4[0]))));

			// is->addr[i] = iCastSeq2x2_x[i] + (m_Height - 1 - (ty * 2 + iCastSeq2x2_y[i])) * m_Width;
			// m_Tmp2_Addr = (m_Height - 1 - iCastSeq2x2_y[i]) * m_Width
			is->addr4 = _mm_add_epi32(m_iCastSeq2x2_x4[0], _mm_sub_epi32(m_Tmp2_Addr4[0], _mm_set1_epi32((m_Width*ty)<<1)));
		}

		for ( tx = 0; tx < xTileEnd; tx++ ) {
			unsigned int i, b;

			Color o_color[4];
//			for (int nSampX = 0; nSampX < m_SuperSampling.x; nSampX++) {
//			for (int nSampY = 0; nSampY < m_SuperSampling.y; nSampY++) {

				jpos = tpos;

				// -----------------------------------------------------------------------
				// Ray packet 을 셋팅 - 시작점(ray_o_x4, ray_o_y4, ray_o_z4) ~ 끝점(tpos)
				// -----------------------------------------------------------------------
				rp->d.x4 = _mm_sub_ps( jpos.d.x4, ray_o_x4 );
				rp->d.y4 = _mm_sub_ps( jpos.d.y4, ray_o_y4 );
				rp->d.z4 = _mm_sub_ps( jpos.d.z4, ray_o_z4 );

				rp->Depth = 0;
				InitPacket2x2( 0 );	// direction vector normalize 등

				// -----------------------------------------------------------------------
				// Coherence 체크 후 rendering
				// -----------------------------------------------------------------------
				if (rp->IsCoherent()) {
					rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
					TracePacket2x2( 0 );
					Shading2x2( 0 );
				} else {
					// detect octants and render one by one
					bool b_quad[8] = { false, false, false, false, false, false, false, false };
					for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
						int q = ((rp->xmask & b)?1:0) + ((rp->ymask & b)?2:0) + ((rp->zmask & b)?4:0);
						b_quad[q] = true;
					}
					for ( int cq = 0; cq < 8; cq++ ) if (b_quad[cq]) {
						for ( b = 1, i = 0; i < 4; i++, b <<= 1 ) {
							int q = ((rp->xmask & b)?1:0) + ((rp->ymask & b)?2:0) + ((rp->zmask & b)?4:0);
							if (q == cq) rm->mask[i] = 0xffffffff; else rm->mask[i] = 0;
						}
						rp->RayWay = cq;
						TracePacket2x2( 0 );
					}
					memset( &rm->mask4, 255, 16 );
					Shading2x2( 0 );
				}

				// -----------------------------------------------------------------------
				// Copy color info to Memory
				// -----------------------------------------------------------------------
				for ( i = 0; i < 4; i++ ) {
					o_color[i].rgba = _mm_add_ps(o_color[i].rgba, is->color[i].rgba);
				}

//			}}	// Loops of (nSampleX * nSampleY)

			for ( i = 0; i < 4; i++ ) {
				m_Dest[3*(is->addr[i])]   = o_color[i].r;// * fSampWeight;
				m_Dest[3*(is->addr[i])+1] = o_color[i].g;// * fSampWeight;
				m_Dest[3*(is->addr[i])+2] = o_color[i].b;// * fSampWeight;
			}

			// Render tile (tpos) 의 위치를 이동
			tpos.d.x4 = _mm_add_ps( tpos.d.x4, delta4.d.x4 );
			tpos.d.y4 = _mm_add_ps( tpos.d.y4, delta4.d.y4 );
			tpos.d.z4 = _mm_add_ps( tpos.d.z4, delta4.d.z4 );
			is->addr4 = _mm_add_epi32( is->addr4, addr_delta.v4 );
		}
	}
}


