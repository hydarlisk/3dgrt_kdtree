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
#include "SSERenderPipelineQ.h"

#include "GThreadManager.h"
#include "GThreadingOption.h"

#pragma warning ( disable : 4068 )
#pragma warning ( disable : 949 )

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------------------
// work
// ------------------------------------------------------------------------------------------------
void SSERenderPipelineQ::work( GThreadContext *pThreadContext ) 
{
	int nWorker;

	m_RenderWorkerCS.lock();
	nWorker = m_Worker++;
	m_RenderWorkerCS.unlock();

	Render4x4Q_onThread(nWorker);
}

// ------------------------------------------------------------------------------------------------
// stop
// ------------------------------------------------------------------------------------------------
void SSERenderPipelineQ::stop()
{
}

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
// SSERenderPipelineQ::InitPacket
//		Initialize a packet for tracing
// ------------------------------------------------------------------------------------------------
void SSERenderPipelineQ::InitPacket4x4Q(_sse_4x4_traceData *traceData)
{
	int i;
	_sse_4x4_raypacket	*rp = traceData->rp;
	_sse_4x4_isectQ		*is = traceData->is;
	_sse_4x4_raymask	*rm	= traceData->rm;

	// Clear initial color & mask
	memset( &is->color, 0,   256 );
	memset( &rm->mask,  255, 64 );

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
// SSERenderPipelineQ::IsectPacket4x4Q
//		Intersection check
// ------------------------------------------------------------------------------------------------
void SSERenderPipelineQ::IsectPacket4x4Q( const KdTreeNode *node, _sse_4x4_traceData *traceData )
{
	int i, j;

	_sse_4x4_raypacket	*rp = traceData->rp;
	_sse_4x4_isectQ		*is = traceData->is;
	_sse_4x4_raymask	*rm	= traceData->rm;

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		union { __m128 Mask_Hit[4]; __m128i iMask_Hit[4]; };

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		bool bContinue = false;
		m_MailboxCS.lock();
		if (acc.mbox == rp->RayId) { bContinue = true; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함
		m_MailboxCS.unlock();
		if (bContinue) continue;

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

static int test_tx, test_ty;
// -----------------------------------------------------------
// SSERenderPipelineQ::TracePacket4x4Q
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipelineQ::TracePacket4x4Q( _sse_4x4_traceData *traceData )
{
	int i, j;

	_sse_4x4_raypacket	*rp = traceData->rp;
	_sse_4x4_isectQ		*is = traceData->is;
	_sse_4x4_raymask	*rm	= traceData->rm;

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// near / far
	__m128 t_near4[4], t_far_4[4];
	t_near4[0] = t_near4[1]	= t_near4[2] = t_near4[3]	= _mm_setzero_ps();				//	_mm_setzero_ps();
	t_far_4[0] = t_far_4[1]	= t_far_4[2] = t_far_4[3]	= _mm_set1_ps(100000);			//	_mm_set_ps1(INFINITY);

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
	m_RayIdCS.lock();
	rp->RayId	= m_RayID;	m_RayID += 1;
	m_RayIdCS.unlock();

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
				if (d_near == 0)	continue;	// traverse the back  child
				node = FrontSideSon;
				if (d_far == 0)		continue;	// traverse the front child

									if (IS_LEAF(*BackSideSon) && OBJECT_SIZE(*BackSideSon) == 0) {
										node = FrontSideSon;
										for (j = 3; j >= 0; j--) {
											t_far_4[j]		= _mm_min_ps(t_far_4[j],d[j]);
											Mask_Active4[j]	= _mm_and_ps(Mask_Active4[j], _mm_cmple_ps(t_near4[j], t_far_4[j]));
										}
										continue;
									}


// ?????????????????????????????????????????????????????????????????
// ?????????????????????????????????????????????????????????????????
// m_Stack4x4 도 thread safe 하게 동기화 필요함... T.T
// 아직 미구현(수정) 차후 계획이 있으면 하자~~~~!!!
// ?????????????????????????????????????????????????????????????????
// ?????????????????????????????????????????????????????????????????


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
		IsectPacket4x4Q (node, traceData);

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


// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void SSERenderPipelineQ::shading4x4Q (_sse_4x4_traceData *traceData) {

	int i, x, y;

	_sse_4x4_raypacket	*rp = traceData->rp;
	_sse_4x4_isectQ		*is = traceData->is;
	_sse_4x4_raymask	*rm	= traceData->rm;

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

	_sse_4x4_raypacket	*shadow_rp	= &m_ShadowRayPk4x4[0];
	_sse_4x4_isectQ		*shadow_is	= &m_ShadowIsect4x4[0];
	_sse_4x4_raymask	*shadow_rm	= &m_ShadowRMask4x4[0];
	memcpy(shadow_rp->o, hit_p, sizeof(_sse_vec)*4);

	bool b_refl = false;
	bool b_refr = false;

	GColor global_ambient = m_Scene->getGlobalAmbient();
	_sse_vec global_ambt  = sse_vset1(global_ambient.r, global_ambient.g, global_ambient.b);

	_sse_vec	mat_cAmbt[4], mat_cDiff[4], mat_cSpec[4], mat_cEmit[4];
	_sse_float	mat_fRough[4];
	_sse_float	mat_fRefl[4], mat_fRefr[4], mat_fRIdx[4];
	_sse_vec	mat_cTex[4];
//	_sse_vec	mat_cRefl[4], mat_cRefr[4];

	_sse_uint	obj_num[4];

	i = 15;
	for ( y = 3; y >= 0; y-- ) {
	for ( x = 3; x >= 0; x-- ) {
		if (rm->mask[i] && is->tacc[i] != 0) {
				const int triID      = is->tacc[i]-1;
				GObject   *pObject   = m_Data->m_TriObjList[triID]->m_pObject;
				GMaterial *pMaterial = pObject->getMaterial();

				const GColor ambient  = pMaterial->getAmbient();
				const GColor diffuse  = pMaterial->getDiffuse();
				const GColor specular = pMaterial->getSpecular();
				const GColor emission = pMaterial->getEmission();

				mat_cAmbt[y].x[x] = ambient.r;  mat_cDiff[y].x[x] = diffuse.r;  mat_cSpec[y].x[x] = specular.r;  mat_cEmit[y].x[x] = emission.r;
				mat_cAmbt[y].y[x] = ambient.g;  mat_cDiff[y].y[x] = diffuse.g;  mat_cSpec[y].y[x] = specular.g;  mat_cEmit[y].y[x] = emission.g;
				mat_cAmbt[y].z[x] = ambient.b;  mat_cDiff[y].z[x] = diffuse.b;  mat_cSpec[y].z[x] = specular.b;  mat_cEmit[y].z[x] = emission.b;
				mat_fRough[y].f[x] = pMaterial->getRoughness();
				mat_fRefl[y].f[x] = pMaterial->getReflection();		if (mat_fRefl[y].f[x] > 0) b_refl = true;
				mat_fRefr[y].f[x] = pMaterial->getTransparency();	if (mat_fRefr[y].f[x] > 0) b_refr = true;
				mat_fRIdx[y].f[x] = pMaterial->getRefractionIndex();
				obj_num[y].f[x] = pObject->getObjectNumber();

				GVector N = m_Data->m_TriObjList[triID]->calBarycentricNormal(1-is->u[i]-is->v[i], is->u[i], is->v[i]);
				is->n[y].x[x] = N.x;
				is->n[y].y[x] = N.y;
				is->n[y].z[x] = N.z;

				// Get object color
				GColor texColor;
				if (m_Scene->isUseTexture()) {
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
				mat_cTex[y].x[x] = texColor.r;
				mat_cTex[y].y[x] = texColor.g;
				mat_cTex[y].z[x] = texColor.b;
		}
		i--;
	}
	}

	_sse_vec oColor[4];
	_sse_vec N[4], R[4], L[4];
	union { __m128 shadingmask[4]; __m128i ishadingmask[4]; };
	union { __m128 islightmask[4]; __m128i iislightmask[4]; };
	union { __m128 visiblemask[4]; __m128i ivisiblemask[4]; };
	union { __m128 isisectmask;    __m128i iisisectmask; };
	union { __m128 noisectmask;    __m128i inoisectmask; };

	for (i = 3; i >= 0; i--) {
		ishadingmask[i] = _mm_cmpgt_epi32(is->tacc4[i], _mm_setzero_si128());
		shadingmask[i]  = _mm_and_ps(shadingmask[i], rm->mask4[i]);

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
		if ( m_Scene->isEnableShadow() ) {
			checkVisibility4x4(&lightPos, shadingmask);

			for (i = 3; i >= 0; i--) {
				__m128 lDist = sse_vlength(sse_vsub(lPos, hit_p[i]));

				// shadow 관련 visible 조건
				// 1) shadingmask[i]           : 물체와 교점있는 것
				// 2) shadow_is->tacc4[i] > 0  : shadow ray 가 교차점이 뒤에 존재하거나 아니면 거리가 거의 가깝거나
				iisisectmask = _mm_cmpgt_epi32(shadow_is->tacc4[i], _mm_setzero_si128());
				inoisectmask = _mm_cmpeq_epi32(shadow_is->tacc4[i], _mm_setzero_si128());

				visiblemask[i] = _mm_or_ps(
						noisectmask,
						_mm_andnot_ps(noisectmask, _mm_or_ps(
							_mm_cmplt_ps(_mm_abs_ps(_mm_sub_ps(lDist, shadow_is->dist4[i])), _mm_set1_ps(1.f*EPSILON)),
							_mm_cmpgt_ps(shadow_is->dist4[i], lDist))));

				// Phong shading
				L[i] = sse_vnorm(sse_vsub(lPos, hit_p[i]));

				oColor[i] = sse_vupdate(sse_vadd(oColor[i], sse_vadd(
					sse_vmul(sse_vmul(mat_cTex[i],  lColor), sse_vset1(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], N[i])))),
					sse_vmul(sse_vmul(mat_cSpec[i], lColor), sse_vset1(sse_fast_pow(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], R[i])), mat_fRough[i].v4)))
						)), oColor[i], _mm_and_ps(visiblemask[i], _mm_andnot_ps(islightmask[i], shadingmask[i])));
				// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
				//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
			}
		} else {
			for (i = 3; i >= 0; i--) {
				// Phong shading
				L[i] = sse_vnorm(sse_vsub(lPos, hit_p[i]));

				oColor[i] = sse_vupdate(sse_vadd(oColor[i], sse_vadd(
					sse_vmul(sse_vmul(mat_cTex[i],  lColor), sse_vset1(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], N[i])))),
					sse_vmul(sse_vmul(mat_cSpec[i], lColor), sse_vset1(sse_fast_pow(_mm_max_ps(_mm_setzero_ps(), sse_vdot(L[i], R[i])), mat_fRough[i].v4)))
						)), oColor[i], _mm_andnot_ps(islightmask[i], shadingmask[i]));
				// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
				//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
			}
		}
	}

	for (i = 3; i >= 0; i--) {
		if ( rp->Depth < (m_Scene->getMaxReflectionDepth())) {
			oColor[i] = sse_vupdate(
				sse_vmul(oColor[i], sse_vsub(sse_vsub(sse_vset1(1.0f), sse_vset1(mat_fRefl[i].v4)), sse_vset1(mat_fRefr[i].v4))),
				oColor[i], shadingmask[i]);
		}
	}

	i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
		if (rm->mask[i])
		is->color[i] = GColor(oColor[y].x[x], oColor[y].y[x], oColor[y].z[x]);
	}

	if (rp->Depth < (m_Scene->getMaxReflectionDepth())) {
		// ---------------------------------------------------------------------------
		// reflection
		// ---------------------------------------------------------------------------
#if 0
		if (b_refl) {
			__m128 dot_i4[4];

			_sse_4x4_traceData	*refl_data	= m_TraceQ4x4->IdleQ_DeQueue();
			_sse_4x4_raypacket	*refl_rp	= refl_data->rp;
			_sse_4x4_isectQ		*refl_is	= refl_data->is;
			_sse_4x4_raymask	*refl_tm	= refl_data->tm;

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

			refl_rp->o[0].x4 = _mm_add_ps(hit_p[0].x4, _mm_mul_ps(refl_rp->d[0].x4, _mm_set1_ps(EPSILON)));
			refl_rp->o[0].y4 = _mm_add_ps(hit_p[0].y4, _mm_mul_ps(refl_rp->d[0].y4, _mm_set1_ps(EPSILON)));
			refl_rp->o[0].z4 = _mm_add_ps(hit_p[0].z4, _mm_mul_ps(refl_rp->d[0].z4, _mm_set1_ps(EPSILON)));
			refl_rp->o[1].x4 = _mm_add_ps(hit_p[1].x4, _mm_mul_ps(refl_rp->d[1].x4, _mm_set1_ps(EPSILON)));
			refl_rp->o[1].y4 = _mm_add_ps(hit_p[1].y4, _mm_mul_ps(refl_rp->d[1].y4, _mm_set1_ps(EPSILON)));
			refl_rp->o[1].z4 = _mm_add_ps(hit_p[1].z4, _mm_mul_ps(refl_rp->d[1].z4, _mm_set1_ps(EPSILON)));
			refl_rp->o[2].x4 = _mm_add_ps(hit_p[2].x4, _mm_mul_ps(refl_rp->d[2].x4, _mm_set1_ps(EPSILON)));
			refl_rp->o[2].y4 = _mm_add_ps(hit_p[2].y4, _mm_mul_ps(refl_rp->d[2].y4, _mm_set1_ps(EPSILON)));
			refl_rp->o[2].z4 = _mm_add_ps(hit_p[2].z4, _mm_mul_ps(refl_rp->d[2].z4, _mm_set1_ps(EPSILON)));
			refl_rp->o[3].x4 = _mm_add_ps(hit_p[3].x4, _mm_mul_ps(refl_rp->d[3].x4, _mm_set1_ps(EPSILON)));
			refl_rp->o[3].y4 = _mm_add_ps(hit_p[3].y4, _mm_mul_ps(refl_rp->d[3].y4, _mm_set1_ps(EPSILON)));
			refl_rp->o[3].z4 = _mm_add_ps(hit_p[3].z4, _mm_mul_ps(refl_rp->d[3].z4, _mm_set1_ps(EPSILON)));

			refl_rp->Depth = rp->Depth+1;

			InitPacket4x4Q( refl_data );

			refl_tm->mask4[0] = _mm_and_ps( rm->mask4[0], _mm_cmpgt_ps(mat_fRefl[0].v4, _mm_setzero_ps()));
			refl_tm->mask4[1] = _mm_and_ps( rm->mask4[1], _mm_cmpgt_ps(mat_fRefl[1].v4, _mm_setzero_ps()));
			refl_tm->mask4[2] = _mm_and_ps( rm->mask4[2], _mm_cmpgt_ps(mat_fRefl[2].v4, _mm_setzero_ps()));
			refl_tm->mask4[3] = _mm_and_ps( rm->mask4[3], _mm_cmpgt_ps(mat_fRefl[3].v4, _mm_setzero_ps()));

			refl_is->addr4[0] = is->addr4[0];
			refl_is->addr4[1] = is->addr4[1];
			refl_is->addr4[2] = is->addr4[2];
			refl_is->addr4[3] = is->addr4[3];

			mat_cRefl[0].x4 = _mm_mul_ps(mat_fRefl[0].v4, mat_cDiff[0].x4);
			mat_cRefl[0].y4 = _mm_mul_ps(mat_fRefl[0].v4, mat_cDiff[0].y4);
			mat_cRefl[0].z4 = _mm_mul_ps(mat_fRefl[0].v4, mat_cDiff[0].z4);
			mat_cRefl[1].x4 = _mm_mul_ps(mat_fRefl[1].v4, mat_cDiff[1].x4);
			mat_cRefl[1].y4 = _mm_mul_ps(mat_fRefl[1].v4, mat_cDiff[1].y4);
			mat_cRefl[1].z4 = _mm_mul_ps(mat_fRefl[1].v4, mat_cDiff[1].z4);
			mat_cRefl[2].x4 = _mm_mul_ps(mat_fRefl[2].v4, mat_cDiff[2].x4);
			mat_cRefl[2].y4 = _mm_mul_ps(mat_fRefl[2].v4, mat_cDiff[2].y4);
			mat_cRefl[2].z4 = _mm_mul_ps(mat_fRefl[2].v4, mat_cDiff[2].z4);
			mat_cRefl[3].x4 = _mm_mul_ps(mat_fRefl[3].v4, mat_cDiff[3].x4);
			mat_cRefl[3].y4 = _mm_mul_ps(mat_fRefl[3].v4, mat_cDiff[3].y4);
			mat_cRefl[3].z4 = _mm_mul_ps(mat_fRefl[3].v4, mat_cDiff[3].z4);

			i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
				refl_is->weight[i].rgba = _mm_mul_ps(is->weight[i].rgba, _mm_setr_ps(mat_cRefl[y].x[x], mat_cRefl[y].x[x], mat_cRefl[y].x[x], 1));
			}
			m_TraceQ4x4->TodoQ_EnQueue(refl_data);
		}
#endif
#if 0
		// ---------------------------------------------------------------------------
		// refraction
		// ---------------------------------------------------------------------------
		if (b_refr) {
			__m128 dot_i4[4];
			__m128 dot_r4[4];
			__m128 n_div_nt4[4];

			_sse_4x4_traceData	*refr_data	= m_TraceQ4x4->IdleQ_DeQueue();
			_sse_4x4_raypacket	*refr_rp	= refr_data->rp;
			_sse_4x4_isectQ		*refr_is	= refr_data->is;
			_sse_4x4_raymask	*refr_rm	= refr_data->rm;
			_sse_4x4_raymask	*refr_tm	= refr_data->tm;

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

			refr_rp->o[0].x4 = _mm_add_ps(hit_p[0].x4, _mm_mul_ps(refr_rp->d[0].x4, _mm_set1_ps(EPSILON)));
			refr_rp->o[0].y4 = _mm_add_ps(hit_p[0].y4, _mm_mul_ps(refr_rp->d[0].y4, _mm_set1_ps(EPSILON)));
			refr_rp->o[0].z4 = _mm_add_ps(hit_p[0].z4, _mm_mul_ps(refr_rp->d[0].z4, _mm_set1_ps(EPSILON)));
			refr_rp->o[1].x4 = _mm_add_ps(hit_p[1].x4, _mm_mul_ps(refr_rp->d[1].x4, _mm_set1_ps(EPSILON)));
			refr_rp->o[1].y4 = _mm_add_ps(hit_p[1].y4, _mm_mul_ps(refr_rp->d[1].y4, _mm_set1_ps(EPSILON)));
			refr_rp->o[1].z4 = _mm_add_ps(hit_p[1].z4, _mm_mul_ps(refr_rp->d[1].z4, _mm_set1_ps(EPSILON)));
			refr_rp->o[2].x4 = _mm_add_ps(hit_p[2].x4, _mm_mul_ps(refr_rp->d[2].x4, _mm_set1_ps(EPSILON)));
			refr_rp->o[2].y4 = _mm_add_ps(hit_p[2].y4, _mm_mul_ps(refr_rp->d[2].y4, _mm_set1_ps(EPSILON)));
			refr_rp->o[2].z4 = _mm_add_ps(hit_p[2].z4, _mm_mul_ps(refr_rp->d[2].z4, _mm_set1_ps(EPSILON)));
			refr_rp->o[3].x4 = _mm_add_ps(hit_p[3].x4, _mm_mul_ps(refr_rp->d[3].x4, _mm_set1_ps(EPSILON)));
			refr_rp->o[3].y4 = _mm_add_ps(hit_p[3].y4, _mm_mul_ps(refr_rp->d[3].y4, _mm_set1_ps(EPSILON)));
			refr_rp->o[3].z4 = _mm_add_ps(hit_p[3].z4, _mm_mul_ps(refr_rp->d[3].z4, _mm_set1_ps(EPSILON)));

			refr_rp->Depth = rp->Depth+1;

			InitPacket4x4Q( refr_data );

			refr_tm->mask4[0] = _mm_and_ps( rm->mask4[0], _mm_cmpgt_ps(mat_fRefr[0].v4, _mm_setzero_ps()));
			refr_tm->mask4[1] = _mm_and_ps( rm->mask4[1], _mm_cmpgt_ps(mat_fRefr[1].v4, _mm_setzero_ps()));
			refr_tm->mask4[2] = _mm_and_ps( rm->mask4[2], _mm_cmpgt_ps(mat_fRefr[2].v4, _mm_setzero_ps()));
			refr_tm->mask4[3] = _mm_and_ps( rm->mask4[3], _mm_cmpgt_ps(mat_fRefr[3].v4, _mm_setzero_ps()));

			refr_is->addr4[0] = is->addr4[0];
			refr_is->addr4[1] = is->addr4[1];
			refr_is->addr4[2] = is->addr4[2];
			refr_is->addr4[3] = is->addr4[3];

			mat_cRefr[0].x4 = _mm_mul_ps(mat_fRefr[0].v4, mat_cDiff[0].x4);
			mat_cRefr[0].y4 = _mm_mul_ps(mat_fRefr[0].v4, mat_cDiff[0].y4);
			mat_cRefr[0].z4 = _mm_mul_ps(mat_fRefr[0].v4, mat_cDiff[0].z4);
			mat_cRefr[1].x4 = _mm_mul_ps(mat_fRefr[1].v4, mat_cDiff[1].x4);
			mat_cRefr[1].y4 = _mm_mul_ps(mat_fRefr[1].v4, mat_cDiff[1].y4);
			mat_cRefr[1].z4 = _mm_mul_ps(mat_fRefr[1].v4, mat_cDiff[1].z4);
			mat_cRefr[2].x4 = _mm_mul_ps(mat_fRefr[2].v4, mat_cDiff[2].x4);
			mat_cRefr[2].y4 = _mm_mul_ps(mat_fRefr[2].v4, mat_cDiff[2].y4);
			mat_cRefr[2].z4 = _mm_mul_ps(mat_fRefr[2].v4, mat_cDiff[2].z4);
			mat_cRefr[3].x4 = _mm_mul_ps(mat_fRefr[3].v4, mat_cDiff[3].x4);
			mat_cRefr[3].y4 = _mm_mul_ps(mat_fRefr[3].v4, mat_cDiff[3].y4);
			mat_cRefr[3].z4 = _mm_mul_ps(mat_fRefr[3].v4, mat_cDiff[3].z4);

			i = 15; for ( y = 3; y >= 0; y-- ) for ( x = 3; x >= 0; x--, i-- ) {
				refr_is->weight[i].rgba = _mm_mul_ps(is->weight[i].rgba, _mm_setr_ps(mat_cRefr[y].x[x], mat_cRefr[y].x[x], mat_cRefr[y].x[x], 1));
			}
			m_TraceQ4x4->TodoQ_EnQueue(refr_data);
		}
#endif
	}
}

// -----------------------------------------------------------
// SSERenderPipelineQ::RenderPacket4x4Q
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipelineQ::RenderPacket4x4Q( _sse_4x4_traceData *traceData )
{
	TracePacket4x4Q(traceData);
	shading4x4Q(traceData);
}

// -----------------------------------------------------------
// SSERenderPipelineQ::RenderQueue
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipelineQ::RenderQueue4x4Q( void ) 
{
	//------------------------------------
	// Rendering Ray Packets
	//------------------------------------
	_sse_4x4_traceData	*traceData;
	_sse_4x4_raypacket	*rp;
	_sse_4x4_isectQ		*is;
	_sse_4x4_raymask	*rm;
	_sse_4x4_raymask	*tm;

	int j;

	while ( 1 ) {
		traceData	= m_TraceQ4x4->TodoQ_DeQueue();
		if (traceData == NULL) break;
		rp = traceData->rp;
		is = traceData->is;
		rm = traceData->rm;
		tm = traceData->tm;

		InitPacket4x4Q ( traceData );	// RMask all clear

		// ---------------------------------------------------------
		// Masking 기법을 이용하여 Coherence 를 맞추어 렌더링 시작
		// ---------------------------------------------------------
		unsigned int i, b;
		// coherence 체크 겸 ray dir 결정 (q = 8방향중하나)
		if (rp->IsCoherent()) {
			rp->RayWay = (rp->xmask & 1) + (rp->ymask & 2) + (rp->zmask & 4);
			rm->mask4[0] = tm->mask4[0];
			rm->mask4[1] = tm->mask4[1];
			rm->mask4[2] = tm->mask4[2];
			rm->mask4[3] = tm->mask4[3];
			RenderPacket4x4Q( traceData );
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
				rm->mask4[0] = _mm_and_ps(tm->mask4[0], rm->mask4[0]);
				rm->mask4[1] = _mm_and_ps(tm->mask4[1], rm->mask4[1]);
				rm->mask4[2] = _mm_and_ps(tm->mask4[2], rm->mask4[2]);
				rm->mask4[3] = _mm_and_ps(tm->mask4[3], rm->mask4[3]);
				rp->RayWay = cq;
				RenderPacket4x4Q( traceData );
			}
			memset( &rm->mask4, 255, 16 * 4 );
		}

		// -----------------------------------------------------------------------
		// Copy color info to Memory
		// -----------------------------------------------------------------------
		Color o_color[16];
		for ( j = 0; j < 16; j++ ) {
			o_color[j].rgba = _mm_mul_ps(is->color[j].rgba, is->weight[j].rgba);
		}

		m_ColorBufferCS.lock();
		for ( j = 0; j < 16; j++ ) {
			m_Dest[3*(is->addr[j])]   += o_color[j].r;
			m_Dest[3*(is->addr[j])+1] += o_color[j].g;
			m_Dest[3*(is->addr[j])+2] += o_color[j].b;
		}
		m_ColorBufferCS.unlock();
		m_TraceQ4x4->IdleQ_EnQueue(traceData);
	}
}

// -----------------------------------------------------------
// SSERenderPipelineQ::GeneratePrimaryRay_inBlock
//		Block 안에서 Primary Ray 를 생성한다.
// -----------------------------------------------------------
void SSERenderPipelineQ::GeneratePrimaryRay_inBlock4x4Q(int bx, int by) 
{
	int i;
	int tx, ty;

	_sse_4x4_traceData	*traceData;
	_sse_4x4_raypacket	*rp;
	_sse_4x4_isectQ		*is;
	_sse_4x4_raymask	*rm;
	_sse_4x4_raymask	*tm;
	_sse_4x4_raypacket	tpos;

	int xTileStart = ((bx) * m_BlockW) >> 2;
	int yTileStart = ((by) * m_BlockH) >> 2;
	int xTileEnd   = ((bx+1) * m_BlockW) >> 2;
	int yTileEnd   = ((by+1) * m_BlockH) >> 2;

	__m128 ox4 = _mm_load1_ps( &m_Origin.x );
	__m128 oy4 = _mm_load1_ps( &m_Origin.y );
	__m128 oz4 = _mm_load1_ps( &m_Origin.z );

	for (ty = yTileStart; ty < yTileEnd; ty++) {
	for (tx = xTileStart; tx < xTileEnd; tx++) {
	//for (ty = yTileStart+1; ty < yTileStart+3; ty++) {
	//for (tx = xTileStart+1; tx < xTileStart+3; tx++) {
		traceData = m_TraceQ4x4->IdleQ_DeQueue();
		rp = traceData->rp;
		is = traceData->is;
		rm = traceData->rm;
		tm = traceData->tm;

		for(i = 0; i < 4; i++) {
			rp->o[i]	= sse_vset1(m_Origin.x, m_Origin.y, m_Origin.z);

			// tpos.d = m_LeftUp + m_DX * (tx * 4 + x) - m_DY * (ty * 4 + y);
			tpos.d[i].x4 = _mm_add_ps(m_LeftUp4->x4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->x4, _mm_add_ps(m_CastSeq4x4_x4[i],_mm_set1_ps(tx*4.0f))),
									  _mm_mul_ps(m_DY4->x4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty*4.0f)))));
			tpos.d[i].y4 = _mm_add_ps(m_LeftUp4->y4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->y4, _mm_add_ps(m_CastSeq4x4_x4[i],_mm_set1_ps(tx*4.0f))),
									  _mm_mul_ps(m_DY4->y4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty*4.0f)))));
			tpos.d[i].z4 = _mm_add_ps(m_LeftUp4->z4, _mm_sub_ps(
									  _mm_mul_ps(m_DX4->z4, _mm_add_ps(m_CastSeq4x4_x4[i],_mm_set1_ps(tx*4.0f))),
									  _mm_mul_ps(m_DY4->z4, _mm_add_ps(m_CastSeq4x4_y4[i],_mm_set1_ps(ty*4.0f)))));

			//is->addr[i] = (tx * 4 + x) + (m_Height - 1 - (ty * 4 + y)) * m_Width;
			is->addr4[i] = _mm_add_epi32(_mm_add_epi32(_mm_set1_epi32(tx*4), m_iCastSeq4x4_x4[i]), _mm_sub_epi32(m_Tmp2_Addr4[i], _mm_set1_epi32(m_Width*ty*4)));

			rp->d[i].x4 = _mm_sub_ps( tpos.d[i].x4, ox4 );
			rp->d[i].y4 = _mm_sub_ps( tpos.d[i].y4, oy4 );
			rp->d[i].z4 = _mm_sub_ps( tpos.d[i].z4, oz4 );
		}

		// This settings are for primary rays 
		memset( &tm->mask4, 255, 16 * 4 );
		for (i = 0; i < 16; i++) {
			is->weight[i] = GColor(1,1,1);
		}
		rp->Depth = 0;
		m_TraceQ4x4->TodoQ_EnQueue(traceData);
	}}
}

// -----------------------------------------------------------
// SSERenderPipelineQ::Render4x4Q
//		4x4 ray packet 을 저장한 Queue 를 이용하여 렌더링
// -----------------------------------------------------------
void SSERenderPipelineQ::Render4x4Q_onThread( int nThreadID )
{
//	if (nThreadID == 1) return;

	int BlockYperThread = int(m_BlockY * m_fRcpThreadCount);
	int yBlockStart, yBlockEnd;

	if (nThreadID == (m_iThreadCount - 1)) {
		yBlockStart	= BlockYperThread * nThreadID;
		yBlockEnd	= m_BlockY;
	} else {
		yBlockStart	= BlockYperThread * nThreadID;
		yBlockEnd	= BlockYperThread * (nThreadID +1);
	}

	// For Testing
	//	GeneratePrimaryRay_inBlock4x4Q(18, 5);	// Primary Ray 를 Queue 에 저장
	//	RenderQueue4x4Q();						// Queue 에 생성된 Ray-packet 렌더링

	int bx, by;
	for (by = yBlockStart; by < yBlockEnd; by++) {
	for (bx = 0; bx < m_BlockX; bx++) {
		GeneratePrimaryRay_inBlock4x4Q(bx, by);	// Primary Ray 를 Queue 에 저장
		RenderQueue4x4Q();						// Queue 에 생성된 Ray-packet 렌더링
	}}
}

// -----------------------------------------------------------
// SSERenderPipelineQ::Render4x4Q
//		4x4 ray packet 을 저장한 Queue 를 이용하여 렌더링
// -----------------------------------------------------------
void SSERenderPipelineQ::Render4x4Q( void )
{
	if (m_iThreadCount > 1) {
		m_Worker = 0;
		GThreadManager::startThreadWork( this, m_iThreadCount );
		GThreadManager::waitThreadWork();
	} else {
		int bx, by;
		for (by = 0; by < m_BlockY; by++) {
		for (bx = 0; bx < m_BlockX; bx++) {
			GeneratePrimaryRay_inBlock4x4Q(bx, by);	// Primary Ray 를 Queue 에 저장
			RenderQueue4x4Q();						// Queue 에 생성된 Ray-packet 렌더링
		}}
	}
}