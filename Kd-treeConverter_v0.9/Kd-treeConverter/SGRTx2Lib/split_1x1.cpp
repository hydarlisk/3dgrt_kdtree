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
#include "GClassMacro.h"

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
void SSERenderPipeline::Split_InitPkt1x1(int nIdx)
{
	_sse_1x1_raypacket	*rp = &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is = &m_Isect1x1[nIdx];

	// Clear initial color
	memset( &is->color, 0,   16 );

	// Normalize ray's direction vector
	float v1 = 1.0f / sqrtf((rp->d.x * rp->d.x) + (rp->d.y * rp->d.y) + (rp->d.z * rp->d.z));
	rp->d.x *= v1;
	rp->d.y *= v1;
	rp->d.z *= v1;

	if (m_RunStatics == 1) {
		if (rp->Depth == 0)	{ G_PR++; }
		else				{ G_RR++; }
	}
}

void SSERenderPipeline::Split_InitPkt1x1_ShwRay(int temp)
{
	_sse_1x1_raypacket	*rp = &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is = &m_ShadowIsect1x1[0];

	// Clear initial color
	memset( &is->color, 0,   16 );

	// Normalize ray's direction vector
	float v1 = 1.0f / sqrtf((rp->d.x * rp->d.x) + (rp->d.y * rp->d.y) + (rp->d.z * rp->d.z));
	rp->d.x *= v1;
	rp->d.y *= v1;
	rp->d.z *= v1;

	if (m_RunStatics == 1) { G_SR++; }
}





// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
bool SSERenderPipeline::Split_Isect1x1_PlaneTest_PriRay( TriAccel &acc, int nIdx, float &f, int temp )
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const unsigned int k	= acc.k;

	float nd;
	nd = 1.0f / (rp->d.f[k]
			+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
	f  = acc.n_d - (rp->o.f[k]
			+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
	f = f * nd;

	if (!(is->dist >= f && f > EPSILON)) return false;	// eps < f <= Hit4.dist

	return true;
}

bool SSERenderPipeline::Split_Isect1x1_TriUVTest_PriRay( TriAccel &acc, int nIdx, float &f, float &lambda, float &mue, int temp )
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const unsigned int k	= acc.k;

	float hu, hv;
	hu = rp->o.f[ku] + f * rp->d.f[ku];
	hv = rp->o.f[kv] + f * rp->d.f[kv];

	lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
	mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

	if (lambda < 0.0f) return false;
	if (mue    < 0.0f) return false;
	if (lambda+mue > 1.0f) return false;

	return true;
}

bool SSERenderPipeline::Split_Isect1x1_PlaneTest_SecRay( TriAccel &acc, int nIdx, float &f )
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const unsigned int k	= acc.k;

	float nd;
	nd = 1.0f / (rp->d.f[k]
			+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
	f  = acc.n_d - (rp->o.f[k]
			+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
	f = f * nd;

	if (!(is->dist >= f && f > EPSILON)) return false;	// eps < f <= Hit4.dist

	return true;
}

bool SSERenderPipeline::Split_Isect1x1_TriUVTest_SecRay( TriAccel &acc, int nIdx, float &f, float &lambda, float &mue )
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const unsigned int k	= acc.k;

	float hu, hv;
	hu = rp->o.f[ku] + f * rp->d.f[ku];
	hv = rp->o.f[kv] + f * rp->d.f[kv];

	lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
	mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

	if (lambda < 0.0f) return false;
	if (mue    < 0.0f) return false;
	if (lambda+mue > 1.0f) return false;

	return true;
}


bool SSERenderPipeline::Split_Isect1x1_PlaneTest_ShwRay( TriAccel &acc, int nIdx, float &f )
{
	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	const unsigned int k	= acc.k;

	float nd;
	nd = 1.0f / (rp->d.f[k]
			+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
	f  = acc.n_d - (rp->o.f[k]
			+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
	f = f * nd;

	if (!(is->dist >= f && f > EPSILON)) return false;	// eps < f <= Hit4.dist

	return true;
}

bool SSERenderPipeline::Split_Isect1x1_TriUVTest_ShwRay( TriAccel &acc, int nIdx, float &f, float &lambda, float &mue )
{
	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	const unsigned int k	= acc.k;

	float hu, hv;
	hu = rp->o.f[ku] + f * rp->d.f[ku];
	hv = rp->o.f[kv] + f * rp->d.f[kv];

	lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
	mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

	if (lambda < 0.0f) return false;
	if (mue    < 0.0f) return false;
	if (lambda+mue > 1.0f) return false;

	return true;
}

void SSERenderPipeline::Split_Isect1x1__PriRay( const KdTreeNode *node, int nIdx, int temp )
{
	int i;

	I_PR++;
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { continue; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (!acc.isTransparent && m_bBackFaceCulling) {
			if (_GVEC_rINNDOT2(rp->d.f, acc.N) < 0) {
				continue;
			}
		}

		float f;
		if (!Split_Isect1x1_PlaneTest_PriRay(acc, nIdx, f, temp)) continue;

		float lambda, mue;
		if (!Split_Isect1x1_TriUVTest_PriRay(acc, nIdx, f, lambda, mue, temp)) continue;

		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}


void SSERenderPipeline::Split_Isect1x1__SecRay( const KdTreeNode *node, int nIdx )
{
	int i;

	I_RR++;
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { continue; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (!acc.isTransparent && m_bBackFaceCulling) {
			if (_GVEC_rINNDOT2(rp->d.f, acc.N) < 0) {
				continue;
			}
		}

		float f;
		if (!Split_Isect1x1_PlaneTest_SecRay(acc, nIdx, f)) continue;

		float lambda, mue;
		if (!Split_Isect1x1_TriUVTest_SecRay(acc, nIdx, f, lambda, mue)) continue;

		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}

void SSERenderPipeline::Split_Isect1x1__ShwRay( const KdTreeNode *node , int temp)
{
	int i;

	I_SR++;
	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayId) { continue; }
		else { acc.mbox = rp->RayId; }	// 무조건 isect 안되도 실행 되어야 함

		// ---------------------------------------------------------------
		// 투명한 물체는 투과
		// ---------------------------------------------------------------
		if (acc.isTransparent) continue;

		float f;
		if (!Split_Isect1x1_PlaneTest_ShwRay(acc, temp, f)) continue;

		float lambda, mue;
		if (!Split_Isect1x1_TriUVTest_ShwRay(acc, temp, f, lambda, mue)) continue;

		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::Split_Trace1x1__PriRay( unsigned int quad, int nIdx, int temp )
{
	T_PR++;
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist = 100000;
	is->tacc = 0;

	// ray direction
	const unsigned int* ray_dir = &raydir[quad][0][0];		// Get precomputed the traversal order (front/back)
															//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// ray reciprocal direction
	_sse_float rcpRayDir;
	_GVEC_vRCP(rcpRayDir, rp->d);

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	float t_near, t_far_;
	t_near = 0;
	t_far_ = 100000;

	{
		float l1, l2;
		l1 = (m_Data->m_SceneBBox.m_Min.x - rp->o.x) * rcpRayDir.x;
		l2 = (m_Data->m_SceneBBox.m_Max.x - rp->o.x) * rcpRayDir.x;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (m_Data->m_SceneBBox.m_Min.y - rp->o.y) * rcpRayDir.y;
		l2 = (m_Data->m_SceneBBox.m_Max.y - rp->o.y) * rcpRayDir.y;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (m_Data->m_SceneBBox.m_Min.z - rp->o.z) * rcpRayDir.z;
		l2 = (m_Data->m_SceneBBox.m_Max.z - rp->o.z) * rcpRayDir.z;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			float d;

				d = (node_split - rp->o.f[dim]) * rcpRayDir.f[dim];
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

				node = BackSideSon;
				if (d_near == 0)	continue;	// traverse the back  child
				node = FrontSideSon;
				if (d_far == 0)		continue;	// traverse the front child

									if (IS_LEAF(*FrontSideSon) && OBJECT_SIZE(*FrontSideSon) == 0) {
										node = BackSideSon;
										t_near	= d;
										continue;
									} else
									if (IS_LEAF(*BackSideSon) && OBJECT_SIZE(*BackSideSon) == 0) {
										node = FrontSideSon;
										t_far_	= d;
										continue;
									}

				// case:  near < d < far
				m_Stack1x1[stackIndex].t_far_ = t_far_;
				m_Stack1x1[stackIndex].t_near = d;
				t_far_ = d;

				m_Stack1x1[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		Split_Isect1x1__PriRay(node, nIdx, temp);

		// Terimination test
		if (is->dist <= t_far_ || stackIndex == 0) break;

		// Stack pop
		--stackIndex;
		node		= m_Stack1x1[stackIndex].node;
		t_near		= m_Stack1x1[stackIndex].t_near;
		t_far_		= m_Stack1x1[stackIndex].t_far_;
	}
}


void SSERenderPipeline::Split_Trace1x1__SecRay( unsigned int quad, int nIdx )
{
	T_RR++;
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist = 100000;
	is->tacc = 0;

	// ray direction
	const unsigned int* ray_dir = &raydir[quad][0][0];		// Get precomputed the traversal order (front/back)
															//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// ray reciprocal direction
	_sse_float rcpRayDir;
	_GVEC_vRCP(rcpRayDir, rp->d);

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	float t_near, t_far_;
	t_near = 0;
	t_far_ = 100000;

	{
		float l1, l2;
		l1 = (m_Data->m_SceneBBox.m_Min.x - rp->o.x) * rcpRayDir.x;
		l2 = (m_Data->m_SceneBBox.m_Max.x - rp->o.x) * rcpRayDir.x;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (m_Data->m_SceneBBox.m_Min.y - rp->o.y) * rcpRayDir.y;
		l2 = (m_Data->m_SceneBBox.m_Max.y - rp->o.y) * rcpRayDir.y;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (m_Data->m_SceneBBox.m_Min.z - rp->o.z) * rcpRayDir.z;
		l2 = (m_Data->m_SceneBBox.m_Max.z - rp->o.z) * rcpRayDir.z;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			float d;

				d = (node_split - rp->o.f[dim]) * rcpRayDir.f[dim];
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

				node = BackSideSon;
				if (d_near == 0)	continue;	// traverse the back  child
				node = FrontSideSon;
				if (d_far == 0)		continue;	// traverse the front child

									if (IS_LEAF(*FrontSideSon) && OBJECT_SIZE(*FrontSideSon) == 0) {
										node = BackSideSon;
										t_near	= d;
										continue;
									} else
									if (IS_LEAF(*BackSideSon) && OBJECT_SIZE(*BackSideSon) == 0) {
										node = FrontSideSon;
										t_far_	= d;
										continue;
									}

				// case:  near < d < far
				m_Stack1x1[stackIndex].t_far_ = t_far_;
				m_Stack1x1[stackIndex].t_near = d;
				t_far_ = d;

				m_Stack1x1[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		Split_Isect1x1__SecRay(node, nIdx);

		// Terimination test
		if (is->dist <= t_far_ || stackIndex == 0) break;

		// Stack pop
		--stackIndex;
		node		= m_Stack1x1[stackIndex].node;
		t_near		= m_Stack1x1[stackIndex].t_near;
		t_far_		= m_Stack1x1[stackIndex].t_far_;
	}
}

void SSERenderPipeline::Split_Trace1x1__ShwRay( unsigned int quad , int temp)
{
	T_SR++;
	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist = 100000;
	is->tacc = 0;

	// ray direction
	const unsigned int* ray_dir = &raydir[quad][0][0];		// Get precomputed the traversal order (front/back)
															//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode* node = &m_Data->m_pKDTreeNodes[0];

	unsigned int stackIndex = 0;

	// ray reciprocal direction
	_sse_float rcpRayDir;
	_GVEC_vRCP(rcpRayDir, rp->d);

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	float t_near, t_far_;
	t_near = 0;
	t_far_ = 100000;

	{
		float l1, l2;
		l1 = (m_Data->m_SceneBBox.m_Min.x - rp->o.x) * rcpRayDir.x;
		l2 = (m_Data->m_SceneBBox.m_Max.x - rp->o.x) * rcpRayDir.x;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (m_Data->m_SceneBBox.m_Min.y - rp->o.y) * rcpRayDir.y;
		l2 = (m_Data->m_SceneBBox.m_Max.y - rp->o.y) * rcpRayDir.y;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (m_Data->m_SceneBBox.m_Min.z - rp->o.z) * rcpRayDir.z;
		l2 = (m_Data->m_SceneBBox.m_Max.z - rp->o.z) * rcpRayDir.z;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			float d;

				d = (node_split - rp->o.f[dim]) * rcpRayDir.f[dim];
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

				node = BackSideSon;
				if (d_near == 0)	continue;	// traverse the back  child
				node = FrontSideSon;
				if (d_far == 0)		continue;	// traverse the front child

									if (IS_LEAF(*FrontSideSon) && OBJECT_SIZE(*FrontSideSon) == 0) {
										node = BackSideSon;
										t_near	= d;
										continue;
									} else
									if (IS_LEAF(*BackSideSon) && OBJECT_SIZE(*BackSideSon) == 0) {
										node = FrontSideSon;
										t_far_	= d;
										continue;
									}

				// case:  near < d < far
				m_Stack1x1[stackIndex].t_far_ = t_far_;
				m_Stack1x1[stackIndex].t_near = d;
				t_far_ = d;

				m_Stack1x1[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Isect check
		Split_Isect1x1__ShwRay(node, 0);

		// Terimination test
		if (is->dist <= t_far_ || stackIndex == 0) break;

		// Stack pop
		--stackIndex;
		node		= m_Stack1x1[stackIndex].node;
		t_near		= m_Stack1x1[stackIndex].t_near;
		t_far_		= m_Stack1x1[stackIndex].t_far_;
	}
}


// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void SSERenderPipeline::Split_Shading1x1 (int nIdx)
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	_sse_float	hit_p = cpu_fadd(rp->o, cpu_fmul(rp->d, is->dist));

	bool bIsEnableShadow       = m_bIsEnableShadow;
	bool bIsRunShadowChk       = false;
	bool bIsEnableLocalShading = m_bIsEnableLocalShading;
	bool bIsUseTexture         = m_bIsUseTexture;
	int  iMaxReflectionDepth   = m_iMaxReflectionDepth;

	bool b_refl = false;
	bool b_refr = false;

	GColor		global_ambient;

	GColor		mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit;
	float		mat_fRough;
	float		mat_fRefl, mat_fRefr, mat_fRIdx;
	GColor		mat_cTex;
	UINT		obj_num;
	bool		bLoadTexColor = !bIsUseTexture;

	if (is->tacc == 0) {
		is->color = GColor(0,0,0);		// Background color
		return;
	}

	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	// Setup
	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	Split_Shading1x1__Setup(nIdx, global_ambient, mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit,
		mat_fRough, mat_fRefl, mat_fRefr, mat_fRIdx, mat_cTex, obj_num, b_refl, b_refr);

	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	// Local Shading
	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	if ( bIsEnableLocalShading ) {
		Split_Shading1x1__LocalShading(nIdx, hit_p, global_ambient, mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit,
		mat_fRough, mat_fRefl, mat_fRefr, mat_fRIdx, mat_cTex, bLoadTexColor, obj_num);
	} else {
		is->color = mat_cTex;
	}

	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	// Secondary Ray Generation
	// *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
	if (rp->Depth < iMaxReflectionDepth) {
		//oColor = oColor * (1.0f - mat_fRefl - mat_fRefr);
		_GCOL_vMULF(is->color, is->color, (1.0f - mat_fRefl - mat_fRefr));

		Split_Shading1x1_RayGeneration_SecRay(nIdx, hit_p,
		mat_fRefl, mat_fRefr, mat_fRIdx, mat_cTex, bLoadTexColor, b_refl, b_refr);
	}
}

void SSERenderPipeline::Split_Shading1x1__Setup (const int nIdx, 
	GColor &global_ambient,
	GColor &mat_cAmbt, GColor &mat_cDiff, GColor &mat_cSpec, GColor &mat_cEmit,
	float &mat_fRough,	float &mat_fRefl, float &mat_fRefr, float &mat_fRIdx,
	GColor &mat_cTex,
	UINT &obj_num,
	bool &b_refl, bool &b_refr)
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	b_refl = false;
	b_refr = false;
	//bool b_tex = false;

	global_ambient = m_globalAmbient;


	const int triID   = is->tacc -1;
	const int       tri_idx    = m_Data->m_TriObjList[triID]->indexInObject;
	GPolygonObject	*pObject   = m_Data->m_TriObjList[triID]->m_pObject;
	GMaterial		*pMaterial = pObject->getMaterial();

	mat_cAmbt  = pMaterial->m_Ambient;
	mat_cDiff  = pMaterial->m_Diffuse;
	mat_cSpec  = pMaterial->m_Specular;
	mat_cEmit  = pMaterial->m_Emission;
	mat_fRough = 0.7;//pMaterial->getRoughness();
	mat_fRefl  = pMaterial->m_fReflection;	    if (mat_fRefl > 0) b_refl = true;
	mat_fRefr  = pMaterial->m_fTransparency;	if (mat_fRefr > 0) b_refr = true;
	mat_fRIdx  = pMaterial->m_fRefractionIndex;
	obj_num    = pObject->m_iObjectNumber;

	const GVector N = pObject->calBarycentricNormal(tri_idx, 1-is->u-is->v, is->u, is->v);
	is->n.x = N.x;
	is->n.y = N.y;
	is->n.z = N.z;

	// Get object color
	//GColor texColor;
	//if ( m_bIsUseTexture ) {
	//	GTexture  *pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
	//	if (pTexture && pTexture->isLoaded()) {
	//		GPoint point = pObject->calBarycentricUV(tri_idx, 1-is->u-is->v, is->u, is->v);
	//		float u = point.x;
	//		float v = point.y;
	//		texColor = pTexture->getTexel( u, v );
	//		b_tex = true;
	//	} else {
	//		texColor = mat_cDiff;
	//		b_tex = false;
	//	}
	//} else {
	//	texColor = mat_cDiff;
	//}
	//mat_cTex = texColor;
	//mat_cTex = getTexColor(nIdx);

	mat_cTex = mat_cDiff;
	

	// 통계치
	if (m_RunStatics == 1) {
		if (rp->Depth == 0) {
			if (b_refl || b_refr)	{ m_pf_Hit_SpecPnt_PR++; } else {
				m_pf_Hit_DiffPnt_PR++;
			}
			//if (b_tex)				{ m_pf_TexRef_PR++; }

			vTriID.push_back(triID);
		} else {
			if (b_refl || b_refr)	{ m_pf_Hit_SpecPnt_RR++; } else { m_pf_Hit_DiffPnt_RR++; }
			//if (b_tex)				{ m_pf_TexRef_RR++; }

		}
	}

}

inline GColor SSERenderPipeline::getTexColor (int nIdx)
{
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const int triID   = is->tacc -1;
	const int       tri_idx    = m_Data->m_TriObjList[triID]->indexInObject;
	GPolygonObject	*pObject   = m_Data->m_TriObjList[triID]->m_pObject;
	GMaterial		*pMaterial = pObject->getMaterial();

	bool b_tex;

	// Get object color
	GColor texColor;

	GTexture  *pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
	if (pTexture && pTexture->isLoaded()) {
		GPoint point = pObject->calBarycentricUV(tri_idx, 1-is->u-is->v, is->u, is->v);
		float u = point.x;
		float v = point.y;
		texColor = pTexture->getTexel( u, v );
		b_tex = true;
	} else {
		texColor = pMaterial->m_Diffuse;
		b_tex = false;
	}

	if (m_RunStatics == 1) {
		if (nIdx == 0) {
			if (b_tex)				{ m_pf_TexRef_PR++;  } 
		} else {
			if (b_tex)				{ m_pf_TexRef_RR++; }
		}
	}

	return texColor;
}

void SSERenderPipeline::Split_Shading1x1__LocalShading (const int nIdx, _sse_float &hit_p,
	GColor &global_ambient,
	GColor &mat_cAmbt, GColor &mat_cDiff, GColor &mat_cSpec, GColor &mat_cEmit,
	float &mat_fRough,	float &mat_fRefl, float &mat_fRefr, float &mat_fRIdx,
	GColor &mat_cTex, bool &bLoadTexColor,
	UINT &obj_num)
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	int _pf_shadow  = 0;
	int _pf_shading = 0;

	GColor oColor;
	GVector N, R, L;
	GVector rayD = GVector(rp->d.f);
	GPoint  hitP = GPoint(hit_p.f);

	_sse_1x1_raypacket	*shadow_rp;
	_sse_1x1_isect		*shadow_is;
	if ( m_bIsEnableShadow ) {
		shadow_rp	= &m_ShadowRayPk1x1[0];
		shadow_is	= &m_ShadowIsect1x1[0];
		shadow_rp->o = hit_p;
	}

	// Shading 에서, 투명한 물체일때, Normal 과 dir 의 dot 이 < 0 이라면 normal 을 뒤짚는다.
	// 확인 필요!!
	N = GVector(is->n.f);
	if (mat_fRefr > 0.0f && cpu_fdot(is->n, rp->d) > 0.0f) 	N = -N;
	const float fdot = _GVEC_rINNDOT(N,rayD) * -2;
	_GVEC_vMUL(R, N, fdot);
	_GVEC_vADD(R, R, rayD);
	_GVEC_vNORMAL(R); 

	// Background color
	oColor = GColor(0,0,0);

	if (is->tacc) {
		// Ambient color
		//oColor = global_ambient * mat_cAmbt;
		_GCOL_vMULC(oColor, global_ambient, mat_cAmbt);

		// Emission color
		//oColor = oColor + mat_cEmit;
		_GCOL_vADD(oColor, oColor, mat_cEmit);

		// Diffuse & Specular color
		const vector<GLight*>* pLightList = m_Scene->getLightList();
		for ( int lx = 0; lx < (int) pLightList->size(); ++lx ) {	GLight* pLight = (*pLightList)[ lx ];
			// Point Light 만 일단 지원
			if ( pLight->getLightType() != typePointLight || !pLight->isEnabled() )  continue;

			GColor   lightColor = pLight->getLightColor();
			GPoint   lightPos   = pLight->getPosition();

			// 광원 자기자신인 경우
			if (obj_num == pLight->m_iObjectNumber) {
				//oColor = oColor + lightColor * pLight->getIntensity();
				GColor _tcol_a;
				float fval = pLight->getIntensity();
				_GCOL_vMULF(_tcol_a, lightColor, fval);
				_GCOL_vADD(oColor, oColor, _tcol_a);
				continue;
			}

			// 그림자 확인
			if ( m_bIsEnableShadow ) {
				Split_Shading1x1_RayGeneration_ShwRay(&hitP, &lightPos);

				// Phong shading
				_GPNT_vSUB(L,lightPos,hitP);
				float lDist = _GVEC_vLENGTH(L);
				_GVEC_vDIV(L,L,lDist);

				// shadow 관련 visible 조건
				//		중간에 shadow ray 와 교점이 없거나
				//		shadow ray 가 교차점이 뒤에 존재하거나 아니면 거리가 거의 가깝거나
				if (shadow_is->tacc == 0 || fabsf(lDist - shadow_is->dist) < 1.f*EPSILON || shadow_is->dist > lDist) {
					//oColor += mat_cTex  * lightColor * max( 0.0f, _GVEC_rINNDOT(L,N) ) +
					//		    mat_cSpec * lightColor * pow( max( 0.0f, _GVEC_rINNDOT(R,L) ), mat_fRough);
					if(!bLoadTexColor) { mat_cTex = getTexColor(nIdx); bLoadTexColor = true; }
					GColor _tcol_a, _tcol_b;
					float fval = max( 0.0f, _GVEC_rINNDOT(L,N));
					_GCOL_vMULF(_tcol_a, lightColor, fval);
					_GCOL_vMULC(_tcol_a, _tcol_a, mat_cTex);
					fval = powf( max( 0.0f, _GVEC_rINNDOT(R,L) ), mat_fRough);
					_GCOL_vMULF(_tcol_b, lightColor, fval);
					_GCOL_vMULC(_tcol_b, _tcol_b, mat_cSpec);
					_GCOL_vADD(oColor, oColor, _tcol_a);
					_GCOL_vADD(oColor, oColor, _tcol_b);
					_pf_shading++;
				} else {
					if (m_Scene->IsTestFlag()) {
						if (shadow_is->tacc == is->tacc) {
							oColor.r = 1;
						}
					}

					_pf_shadow++;
				}
				// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
				//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
			} else {
				// Phong shading
				_GPNT_vSUB(L,lightPos,hitP);
				_GVEC_vNORMAL(L);

				//oColor += mat_cTex  * lightColor * max( 0.0f, _GVEC_rINNDOT(L,N) ) +
				//		    mat_cSpec * lightColor * pow( max( 0.0f, _GVEC_rINNDOT(R,L) ), mat_fRough);
				if(!bLoadTexColor) { mat_cTex = getTexColor(nIdx); bLoadTexColor = true; }
				GColor _tcol_a, _tcol_b;
				float fval = max( 0.0f, _GVEC_rINNDOT(L,N));
				_GCOL_vMULF(_tcol_a, lightColor, fval);
				_GCOL_vMULC(_tcol_a, _tcol_a, mat_cTex);
				fval = powf( max( 0.0f, _GVEC_rINNDOT(R,L) ), mat_fRough);
				_GCOL_vMULF(_tcol_b, lightColor, fval);
				_GCOL_vMULC(_tcol_b, _tcol_b, mat_cSpec);
				_GCOL_vADD(oColor, oColor, _tcol_a);
				_GCOL_vADD(oColor, oColor, _tcol_b);
				// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
				//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
				_pf_shading++;
			}
		}
	}

	if ( m_RunStatics == 1 ) {
		if ( _pf_shadow > 0 && rp->Depth == 0 ) {
			m_pf_Hit_ShwPnt_ALL++;
			m_pf_Hit_ShwCnt_PR+=_pf_shadow;
		}
		if (_pf_shading > 0 ) {
			if ( rp->Depth == 0 ) {
				m_pf_Hit_ShadPnt_PR++;
				m_pf_Hit_ShadCnt_PR+=_pf_shading;
			} else {
				m_pf_Hit_ShadPnt_RR++;
				m_pf_Hit_ShadCnt_RR+=_pf_shading;
			}
		}
	}

	is->color = oColor;
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shading - SecondaryRayGeneration
// ---------------------------------------------------------------------------
void SSERenderPipeline::Split_Shading1x1_RayGeneration_SecRay (const int nIdx, _sse_float &hit_p,
	float &mat_fRefl, float &mat_fRefr, float &mat_fRIdx,
	GColor &mat_cTex, bool &bLoadTexColor, bool &b_refl, bool &b_refr)
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	// ---------------------------------------------------------------------------
	// reflection
	// ---------------------------------------------------------------------------
	if (b_refl) {
		float dot_i;

		const int nNextIdx = nIdx+1;
		
		_sse_1x1_raypacket	*refl_rp	= &m_RayPk1x1[nNextIdx];
		_sse_1x1_isect		*refl_is	= &m_Isect1x1[nNextIdx];

		dot_i = cpu_fdot(rp->d, is->n);

		refl_rp->d = cpu_fsub(rp->d, cpu_fmul(2, cpu_fmul(dot_i, is->n)));
		Split_InitPkt1x1( nNextIdx );

		refl_rp->o = cpu_fadd(hit_p, cpu_fmul(refl_rp->d, RAY_START_EPSILON));
		refl_rp->Depth = rp->Depth+1;

		int q = (refl_rp->d.x < 0) + ((refl_rp->d.y < 0) << 1) + ((refl_rp->d.z < 0) << 2);
		Split_Trace1x1__SecRay(q, nNextIdx);
		Split_Shading1x1(nNextIdx);

		if (refl_is->tacc) {
			//is->color += mat_fRefl * refl_is->color * mat_cTex;
			if(!bLoadTexColor) { mat_cTex = getTexColor(nIdx); bLoadTexColor = true; }
			GColor _tcol_a;
			_GCOL_vMULF(_tcol_a, refl_is->color, mat_fRefl);
			_GCOL_vMULC(_tcol_a, _tcol_a, mat_cTex);
			_GCOL_vADD(is->color, is->color, _tcol_a);
		}
	}

	// ---------------------------------------------------------------------------
	// refraction
	// ---------------------------------------------------------------------------
	if (b_refr) {
		float dot_i, dot_r;
		float n_div_nt;

		const int nNextIdx = nIdx+1;

		_sse_1x1_raypacket	*refr_rp	= &m_RayPk1x1[nNextIdx];
		_sse_1x1_isect		*refr_is	= &m_Isect1x1[nNextIdx];

		dot_i = cpu_fdot(rp->d, is->n);

		if (dot_i < 0) {
			n_div_nt = AIR_INDEX / mat_fRIdx;
		} else {
			n_div_nt = mat_fRIdx / AIR_INDEX;
		}

		dot_r = sqrtf(fabsf(1.0f - n_div_nt * n_div_nt * (1 - dot_i * dot_i)));

		if(n_div_nt < 1.0) {
			refr_rp->d = cpu_fsub(cpu_fmul(n_div_nt, cpu_fsub(rp->d, cpu_fmul(is->n, dot_i))), cpu_fmul(is->n, dot_r));
		} else {
			refr_rp->d = cpu_fadd(cpu_fmul(n_div_nt, cpu_fsub(rp->d, cpu_fmul(is->n, dot_i))), cpu_fmul(is->n, dot_r));
		}
		Split_InitPkt1x1( nNextIdx );

		refr_rp->o = cpu_fadd(hit_p, cpu_fmul(refr_rp->d, RAY_START_EPSILON));
		refr_rp->Depth = rp->Depth+1;

		int q = (refr_rp->d.x < 0) + ((refr_rp->d.y < 0) << 1) + ((refr_rp->d.z < 0) << 2);
		Split_Trace1x1__SecRay(q, nNextIdx);
		Split_Shading1x1(nNextIdx);

		if (refr_is->tacc) {
			//is->color += mat_fRefr * refr_is->color * mat_cTex;
			if(!bLoadTexColor) { mat_cTex = getTexColor(nIdx); bLoadTexColor = true; }
			GColor _tcol_a;
			_GCOL_vMULF(_tcol_a, refr_is->color, mat_fRefr);
			_GCOL_vMULC(_tcol_a, _tcol_a, mat_cTex);
			_GCOL_vADD(is->color, is->color, _tcol_a);
		}
	}
}

void SSERenderPipeline::Split_Shading1x1_RayGeneration_ShwRay(const GPoint* objectPos, const GPoint* lightPos) {

	_sse_1x1_raypacket	*shadow_rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*shadow_is	= &m_ShadowIsect1x1[0];

	_sse_float oPos = cpu_fset1(objectPos->data);
	_sse_float lPos = cpu_fset1(lightPos->data);

	shadow_rp->d = cpu_fsub(lPos, oPos);
	Split_InitPkt1x1_ShwRay(0);
	shadow_rp->o = cpu_fadd(oPos, cpu_fmul(shadow_rp->d, RAY_START_EPSILON));

	// ray dir 결정 (q = 8방향중하나)
	int q = (shadow_rp->d.x < 0) + ((shadow_rp->d.y < 0) << 1) + ((shadow_rp->d.z < 0) << 2);
	Split_Trace1x1__ShwRay(q, 0);
}

// -----------------------------------------------------------
// SSERenderPipeline::RenderTiles
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Split_Render1x1__PriRay( int nJobID )
{
	int yTileStart = 0;
	int xTileEnd = m_Resolution.x;
	int yTileEnd = m_Resolution.y;
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
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[0];
	_sse_1x1_isect		*is	= &m_Isect1x1[0];
	_sse_1x1_raypacket	tpos;		// target position for ray casting (pixel center)
	_sse_1x1_raypacket	jpos;		// jittered position for sampling

	rp->o   = cpu_fset1(m_Origin.m_Vector);

	const float fSampSeq_x[5][16] = { { 0, }, { 0, }, { -0.25f, +0.25f, }, { -0.333f, 0, 0.333f, }, { -0.375f, -0.125f, 0.125f, 0.375f, }};
	const float fSampSeq_y[5][16] = { { 0, }, { 0, }, { -0.25f, +0.25f, }, { -0.333f, 0, 0.333f, }, { -0.375f, -0.125f, 0.125f, 0.375f, }};
	const float rcpSuperSampling_x = 1.0f / m_SuperSampling.x;
	const float rcpSuperSampling_y = 1.0f / m_SuperSampling.y;
	const float fXjitter = 0.5f * rcpSuperSampling_x;	// jitterRate = 0.5f
	const float fYjitter = 0.5f * rcpSuperSampling_y;	// jitterRate = 0.5f
	const float fSampWeight = rcpSuperSampling_x * rcpSuperSampling_y;

	GVector r;
	GVector LeftUp = GVector(m_LeftUp);
	GVector DX     = GVector(m_DX);
	GVector DY     = GVector(m_DY);
	GVector _tvec_a, _tvec_b;

	for ( ty = yTileStart; ty < yTileEnd; ty++ ) {
	for ( tx = 0; tx < xTileEnd;  tx++ ) {

		// -----------------------------------------------------------------------
		// tpos (Ray 를 쏠 방향지점) 계산
		// -----------------------------------------------------------------------
		// m_LeftUp : image screen 위쪽 왼편 모서리의 pixel 중심 으로 이미 셋팅 되어 있음
		//vector3 r = m_LeftUp + (m_DX * (float)tx) - (m_DY * (float)ty);
		_GVEC_vMUL(_tvec_a, DX, (float)tx);
		_GVEC_vMUL(_tvec_b, DY, (float)ty);
		_GVEC_vADD(r,LeftUp,_tvec_a);
		_GVEC_vSUB(r,r,_tvec_b);

		tpos.d.x = r.x;
		tpos.d.y = r.y;
		tpos.d.z = r.z;
		is->addr = tx + (m_Height - 1 - ty) * m_Width;

			GColor o_color;
			for (int nSampX = 0; nSampX < m_SuperSampling.x; nSampX++) {
			for (int nSampY = 0; nSampY < m_SuperSampling.y; nSampY++) {
				if (m_SuperSampling.x == 1) {
					jpos = tpos;
				} else {
					if ( m_Scene->isEnableJittering() ) {
						GVector samp_D;
						//samp_D  = m_DX * (fSampSeq_x[m_SuperSampling.x][nSampX] + ( (float)sgrtRadicalInverse(nSampX,3)*2.0f*fXjitter - fXjitter))
						//		  - m_DY * (fSampSeq_y[m_SuperSampling.y][nSampY] + ( (float)sgrtRadicalInverse(nSampY,5)*2.0f*fYjitter - fYjitter));
						_GVEC_vMUL(_tvec_a, DX, (fSampSeq_x[m_SuperSampling.x][nSampX] + ( (float)sgrtRadicalInverse(nSampX,3)*2.0f*fXjitter - fXjitter)));
						_GVEC_vMUL(_tvec_b, DY, (fSampSeq_y[m_SuperSampling.y][nSampY] + ( (float)sgrtRadicalInverse(nSampY,5)*2.0f*fYjitter - fYjitter)));
						_GVEC_vSUB(samp_D,_tvec_a,_tvec_b);
						jpos.d	= cpu_fadd(tpos.d, cpu_fset1(samp_D.m_Vector));
					} else {
						GVector samp_D;
						//samp_D  = m_DX * fSampSeq_x[m_SuperSampling.x][nSampX]
						//		  - m_DY * fSampSeq_y[m_SuperSampling.y][nSampY];
						_GVEC_vMUL(_tvec_a, DX, fSampSeq_x[m_SuperSampling.x][nSampX]);
						_GVEC_vMUL(_tvec_b, DY, fSampSeq_y[m_SuperSampling.y][nSampY]);
						_GVEC_vSUB(samp_D,_tvec_a,_tvec_b);
						jpos.d	= cpu_fadd(tpos.d, cpu_fset1(samp_D.m_Vector));
					}
				}

				// Ray 를 셋팅 - 시작점(rp->o) ~ 끝점(jpos)
				rp->d = cpu_fsub(jpos.d, rp->o);
				rp->Depth = 0;
				Split_InitPkt1x1( 0 );	// direction vector normalize 등

				// ray dir 결정 (q = 8방향중하나)
				int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
				Split_Trace1x1__PriRay(q, 0, 0);
				Split_Shading1x1(0);

				// -----------------------------------------------------------------------
				// Copy color info to Memory
				// -----------------------------------------------------------------------
				//o_color = o_color + is->color;
				_GCOL_vADD(o_color, o_color, is->color);
			}}	// Loops of (nSampleX * nSampleY)

			m_Dest[3*(is->addr)]   = o_color.r * fSampWeight;
			m_Dest[3*(is->addr)+1] = o_color.g * fSampWeight;
			m_Dest[3*(is->addr)+2] = o_color.b * fSampWeight;
	}
	}
}


