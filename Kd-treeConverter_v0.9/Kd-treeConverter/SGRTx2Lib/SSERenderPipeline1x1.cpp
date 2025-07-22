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

//#include <fstream>
//using namespace std;
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
void SSERenderPipeline::InitPacket1x1(int nIdx)
{
	COUNT_STATE( nIdx, RAY_COUNT, 1 );
	_sse_1x1_raypacket	*rp = &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is = &m_Isect1x1[nIdx];

	// Clear initial color
	COUNT_INSTRUCTION( nIdx, RAY_GENERATION, STORE, 1 );
	memset( &is->color, 0, 16);

	// Normalize ray's direction vector
	COUNT_INSTRUCTION( nIdx, RAY_GENERATION, LOAD, 9 );
	COUNT_INSTRUCTION( nIdx, RAY_GENERATION, MULTIPLY, 3 );
	COUNT_INSTRUCTION( nIdx, RAY_GENERATION, ADD, 3 );
	COUNT_INSTRUCTION( nIdx, RAY_GENERATION, SQRT, 1 );
	COUNT_INSTRUCTION( nIdx, RAY_GENERATION, DIVISION, 1 );
	COUNT_INSTRUCTION( nIdx, RAY_GENERATION, STORE, 3 );
	float v1 = 1.0f / sqrtf((rp->d.x * rp->d.x) + (rp->d.y * rp->d.y) + (rp->d.z * rp->d.z));
	rp->d.x *= v1;
	rp->d.y *= v1;
	rp->d.z *= v1;
}

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::InitPacket
//		Initialize a packet for tracing
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::InitShadowPacket1x1(void)
{
	COUNT_STATE( -1, RAY_COUNT, 1 );
	_sse_1x1_raypacket	*rp = &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is = &m_ShadowIsect1x1[0];

	// Clear initial color
	COUNT_INSTRUCTION( -1, RAY_GENERATION, STORE, 1 );
	memset( &is->color, 0,   16 );

	// Normalize ray's direction vector
	COUNT_INSTRUCTION( -1, RAY_GENERATION, LOAD, 9 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, MULTIPLY, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, ADD, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, SQRT, 1 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, DIVISION, 1 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, STORE, 3 );
	float v1 = 1.0f / sqrtf((rp->d.x * rp->d.x) + (rp->d.y * rp->d.y) + (rp->d.z * rp->d.z));
	rp->d.x *= v1;
	rp->d.y *= v1;
	rp->d.z *= v1;
}

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::IsectShadowPacket1x1( const KdTreeNode *node )
{
	int i;

	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, STORE, 2 );
	COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, SHIFT, 1 );
	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	COUNT_STATE( -1, VISITED_LEAF_NODE, 1 );
	if( nObjs == 0 )
	{COUNT_STATE( -1, EMPTY_NODE, 1 );}

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, ADD, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BRANCH, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, COMPARISON, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 2 );
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 2 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BRANCH, 1 );
		if (acc.mbox == rp->RayId)
		{
			COUNT_STATE( -1, MAILBOXED_TRIANGLE_COUNT, 1 );
			continue;
		}
		else { 
			COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 1 );
			COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, STORE, 1 );
			acc.mbox = rp->RayId;
		}	// 무조건 isect 안되도 실행 되어야 함

		COUNT_STATE( -1, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// 투명한 물체는 투과
		// ---------------------------------------------------------------
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BRANCH, 1 );
		if (acc.isTransparent) continue;

		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 1 );
		const unsigned int k	= acc.k;

		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 11 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, DIVISION, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, MULTIPLY, 5 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, ADD, 4 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, SUBTRACT, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, MOVE, 3 );
		float nd, f;
		nd = 1.0f / (rp->d.f[k]
				+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
		f  = acc.n_d - (rp->o.f[k]
				+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
		f = f * nd;

		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, COMPARISON, 2 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BRANCH, 2 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BOOLEAN_NEGATOR, 1 );
		if (!(is->dist >= f && f > EPSILON)) continue;	// eps < f <= Hit4.dist

		float hu, hv;
		float lambda, mue;
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, LOAD, 4 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, MULTIPLY, 2 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, ADD, 2 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, MOVE, 2 );
		hu = rp->o.f[ku] + f * rp->d.f[ku];
		hv = rp->o.f[kv] + f * rp->d.f[kv];

		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, ADD, 4 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, MULTIPLY, 4 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, MOVE, 2 );
		lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
		mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, COMPARISON, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BRANCH, 1 );
		if (lambda < 0.0f) continue;
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, COMPARISON, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BRANCH, 1 );
		if (mue    < 0.0f) continue;
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, COMPARISON, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, BRANCH, 1 );
		if (lambda+mue > 1.0f) continue;

		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, ADD, 1 );
		COUNT_INSTRUCTION( -1, INTERSECTION_CHECK, STORE, 4 );
		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}

// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::TraceShadowPacket1x1( unsigned int quad )
{
	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	COUNT_INSTRUCTION( -1, TRAVERSE, STORE, 2 );
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
	COUNT_INSTRUCTION( -1, TRAVERSE, LOAD, 4 );
	COUNT_INSTRUCTION( -1, TRAVERSE, DIVISION, 4 );
	cpu_inverse(rcpRayDir, rp->d);

	// ray id
	COUNT_INSTRUCTION( -1, TRAVERSE, STORE, 1 );
	COUNT_INSTRUCTION( -1, TRAVERSE, ADD, 1 );
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	float t_near, t_far_;
	t_near = 0;
	t_far_ = 100000;

	{
		float l1, l2;
		COUNT_INSTRUCTION( -1, TRAVERSE, SUBTRACT, 6 );
		COUNT_INSTRUCTION( -1, TRAVERSE, MULTIPLY, 6 );
		COUNT_INSTRUCTION( -1, TRAVERSE, LOAD, 12 );
		COUNT_INSTRUCTION( -1, TRAVERSE, MOVE, 6 );
		COUNT_INSTRUCTION( -1, TRAVERSE, MIN_OP, 6 );
		COUNT_INSTRUCTION( -1, TRAVERSE, MAX_OP, 6 );
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
			COUNT_INSTRUCTION( -1, TRAVERSE, BRANCH, 1 );
			COUNT_INSTRUCTION( -1, TRAVERSE, COMPARISON, 1 );
			COUNT_INSTRUCTION( -1, TRAVERSE, BRANCH, 1 );
			COUNT_INSTRUCTION( -1, TRAVERSE, BIT_AND, 2 );
			COUNT_INSTRUCTION( -1, TRAVERSE, LOAD, 2 );
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			COUNT_INSTRUCTION( -1, TRAVERSE, SHIFT, 4 );
			COUNT_INSTRUCTION( -1, TRAVERSE, ADD, 3 );
			COUNT_INSTRUCTION( -1, TRAVERSE, LOAD, 6 );
			COUNT_INSTRUCTION( -1, TRAVERSE, MOVE, 2 );
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			float d;

				COUNT_INSTRUCTION( -1, TRAVERSE, SUBTRACT, 1 );
				COUNT_INSTRUCTION( -1, TRAVERSE, MULTIPLY, 1 );
				COUNT_INSTRUCTION( -1, TRAVERSE, LOAD, 1 );
				COUNT_INSTRUCTION( -1, TRAVERSE, MOVE, 1 );
				d = (node_split - rp->o.f[dim]) * rcpRayDir.f[dim];
				COUNT_INSTRUCTION( -1, TRAVERSE, BIT_OR, 2 );
				COUNT_INSTRUCTION( -1, TRAVERSE, COMPARISON, 2 );
				COUNT_INSTRUCTION( -1, TRAVERSE, MOVE, 2 );
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

				COUNT_INSTRUCTION( -1, TRAVERSE, MOVE, 1 );
				node = BackSideSon;
				COUNT_INSTRUCTION( -1, TRAVERSE, BRANCH, 1 );
				COUNT_INSTRUCTION( -1, TRAVERSE, COMPARISON, 1 );
				if (d_near == 0)
				{
					COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the back  child
				}
				COUNT_INSTRUCTION( -1, TRAVERSE, MOVE, 1 );
				node = FrontSideSon;
				COUNT_INSTRUCTION( -1, TRAVERSE, BRANCH, 1 );
				COUNT_INSTRUCTION( -1, TRAVERSE, COMPARISON, 1 );
				if (d_far == 0)	
				{
					COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the front child
				}

				COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN );
				// case:  near < d < far
				COUNT_INSTRUCTION( -1, TRAVERSE, STORE, 3 );
				COUNT_INSTRUCTION( -1, TRAVERSE, MOVE, 1 );
				COUNT_INSTRUCTION( -1, TRAVERSE, ADD, 1 );
				m_Stack1x1[stackIndex].t_far_ = t_far_;
				m_Stack1x1[stackIndex].t_near = d;
				t_far_ = d;

				m_Stack1x1[stackIndex].node = BackSideSon;
				stackIndex++;
		}
		COUNT_INSTRUCTION( -1, TRAVERSE, BRANCH, 1 ); // while
		COUNT_INSTRUCTION( -1, TRAVERSE, COMPARISON, 1 );

		// Isect check
		IsectShadowPacket1x1(node);

		// Terimination test
		COUNT_INSTRUCTION( -1, TRAVERSE, BRANCH, 1 );
		COUNT_INSTRUCTION( -1, TRAVERSE, COMPARISON, 2 );
		COUNT_INSTRUCTION( -1, TRAVERSE, LOAD, 2 );
		COUNT_INSTRUCTION( -1, TRAVERSE, BOOLEAN_OR, 2 );
		if (is->dist <= t_far_ || stackIndex == 0) break;

		COUNT_TREE_OPERATOR( -1, TRAVERSE_UP );
		// Stack pop
		COUNT_INSTRUCTION( -1, TRAVERSE, SUBTRACT, 2 );
		--stackIndex;
		COUNT_INSTRUCTION( -1, TRAVERSE, LOAD, 3 );
		node		= m_Stack1x1[stackIndex].node;
		t_near		= m_Stack1x1[stackIndex].t_near;
		t_far_		= m_Stack1x1[stackIndex].t_far_;
	}
}

// ---------------------------------------------------------------------------
// checkVisibility
// ---------------------------------------------------------------------------
void SSERenderPipeline::checkVisibility1x1(const GPoint* objectPos, const GPoint* lightPos) {

	_sse_1x1_raypacket	*shadow_rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*shadow_is	= &m_ShadowIsect1x1[0];

	COUNT_INSTRUCTION( -1, RAY_GENERATION, LOAD, 3 );
	_sse_float oPos = cpu_fset1(objectPos->data);
	_sse_float lPos = cpu_fset1(lightPos->data);

	COUNT_INSTRUCTION( -1, RAY_GENERATION, LOAD, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, SUBTRACT, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, STORE, 3 );
	shadow_rp->d = cpu_fsub(lPos, shadow_rp->o);
	shadow_rp->d = cpu_fsub(lPos, oPos);

	COUNT_INSTRUCTION( -1, RAY_GENERATION, LOAD, 6 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, MULTIPLY, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, ADD, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, STORE, 3 );
	InitShadowPacket1x1();
	shadow_rp->o = cpu_fadd(oPos, cpu_fmul(shadow_rp->d, RAY_START_EPSILON));

	// ray dir 결정 (q = 8방향중하나)
	COUNT_INSTRUCTION( -1, RAY_GENERATION, LOAD, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, SHIFT, 2 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, COMPARISON, 3 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, ADD, 2 );
	COUNT_INSTRUCTION( -1, RAY_GENERATION, MOVE, 2 );
	int q = (shadow_rp->d.x < 0) + ((shadow_rp->d.y < 0) << 1) + ((shadow_rp->d.z < 0) << 2);
	TraceShadowPacket1x1(q);
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::IsectPacket1x1( const KdTreeNode *node, int nIdx )
{
	COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, FUNCTION_ENTER, 1 );
	int i;

	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, STORE, 2 );
	COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, SHIFT, 1 );
	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	COUNT_STATE( nIdx, VISITED_LEAF_NODE, 1 );
	if( nObjs == 0 )
	{COUNT_STATE( nIdx, EMPTY_NODE, 1 );}

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, ADD, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, BRANCH, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, LOAD, 2 );
		int     triID = m_Data->m_TriOffList[i];
		TriAccel &acc = m_Data->m_TriAccList[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_MAILBOX, FUNCTION_ENTER, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_MAILBOX, LOAD, 2 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_MAILBOX, BRANCH, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_MAILBOX, COMPARISON, 1 );
		if (acc.mbox == rp->RayId) { 
			COUNT_STATE( nIdx, MAILBOXED_TRIANGLE_COUNT, 1 );
			continue;
		}
		else { 
			COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_MAILBOX, LOAD, 1 );
			COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_MAILBOX, STORE, 1 );
			acc.mbox = rp->RayId;
		}	// 무조건 isect 안되도 실행 되어야 함

		COUNT_STATE( nIdx, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, LOAD, 6 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, DOT_PRODUCT, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, COMPARISON, 2 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, BRANCH, 1 );
		// ---------------------------------------------------------------
		if (!acc.isTransparent && m_bBackFaceCulling) {
			COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, BRANCH, 1 );
			if (vector3(rp->d.f).innerProduct(acc.N) < 0) {
				continue;
			}
		}

		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, LOAD, 1 );
		const unsigned int k	= acc.k;

		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, LOAD, 11 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, DIVISION, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, MULTIPLY, 5 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, ADD, 4 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, SUBTRACT, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, MOVE, 3 );
		float nd, f;
		nd = 1.0f / (rp->d.f[k]
				+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
		f  = acc.n_d - (rp->o.f[k]
				+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
		f = f * nd;

		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_DISTANCE, FUNCTION_ENTER, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_DISTANCE, LOAD, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_DISTANCE, COMPARISON, 2 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_DISTANCE, BRANCH, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_DISTANCE, BOOLEAN_NEGATOR, 1 );
		if (!(is->dist >= f && f > EPSILON)) continue;	// eps < f <= Hit4.dist

		float hu, hv;
		float lambda, mue;
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, LOAD, 4 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, MULTIPLY, 2 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, ADD, 2 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, MOVE, 2 );
		hu = rp->o.f[ku] + f * rp->d.f[ku];
		hv = rp->o.f[kv] + f * rp->d.f[kv];

		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, ADD, 4 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, MULTIPLY, 4 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK, MOVE, 2 );
		lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
		mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_U, FUNCTION_ENTER, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_U, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_U, BRANCH, 1 );
		if (lambda < 0.0f) continue;
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_V, FUNCTION_ENTER, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_V, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_V, BRANCH, 1 );
		if (mue    < 0.0f) continue;
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_U_V, FUNCTION_ENTER, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_U_V, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_U_V, BRANCH, 1 );
		if (lambda+mue > 1.0f) continue;

		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_LIVED, FUNCTION_ENTER, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_LIVED, ADD, 1 );
		COUNT_INSTRUCTION( nIdx, INTERSECTION_CHECK_LIVED, STORE, 4 );
		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}

// -----------------------------------------------------------
// SSERenderPipeline::TracePacket
//		Sends a 4x4 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::TracePacket1x1( unsigned int quad, int nIdx )
{
	COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, FUNCTION_ENTER, 1 );
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, STORE , 2 );
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
	COUNT_INSTRUCTION( -1, TRAVERSE_INITIALIZATION, LOAD, 4 );
	COUNT_INSTRUCTION( -1, TRAVERSE_INITIALIZATION, DIVISION, 4 );
	cpu_inverse(rcpRayDir, rp->d);

	// ray id
	COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, ADD, 1 );
	COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, STORE, 1 );
	rp->RayId	= m_RayID;	m_RayID += 1;

	// near / far
	float t_near, t_far_;
	COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, MOVE, 2 );
	t_near = 0;
	t_far_ = 100000;

	{
		float l1, l2;
		COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, SUBTRACT, 6 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, MULTIPLY, 6 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, LOAD, 12 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, MOVE, 12 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, MIN_OP, 6 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE_INITIALIZATION, MAX_OP, 6 );
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

	COUNT_INSTRUCTION( nIdx, TRAVERSE, FUNCTION_ENTER, 1 );

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			COUNT_INSTRUCTION( nIdx, TRAVERSE, BRANCH, 1 );
			COUNT_INSTRUCTION( nIdx, TRAVERSE, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, TRAVERSE, BIT_AND, 2 );
			COUNT_INSTRUCTION( nIdx, TRAVERSE, LOAD, 2 );
			COUNT_INSTRUCTION( nIdx, TRAVERSE, SHIFT, 4 );
			COUNT_INSTRUCTION( nIdx, TRAVERSE, ADD, 3 );
			COUNT_INSTRUCTION( nIdx, TRAVERSE, LOAD, 6 );
			COUNT_INSTRUCTION( nIdx, TRAVERSE, MOVE, 4 );
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			COUNT_INSTRUCTION( nIdx, TRAVERSE, MOVE, 2 );
			unsigned int d_near = 0, d_far = 0;
			float d;

				COUNT_INSTRUCTION( nIdx, TRAVERSE, SUBTRACT, 1 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE, MULTIPLY, 1 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE, LOAD, 1 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE, MOVE, 1 );
				d = (node_split - rp->o.f[dim]) * rcpRayDir.f[dim];
				COUNT_INSTRUCTION( nIdx, TRAVERSE, BIT_OR, 2 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE, COMPARISON, 2 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE, MOVE, 2 );
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

				COUNT_INSTRUCTION( nIdx, TRAVERSE_BACK, FUNCTION_ENTER, 1 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE_BACK, MOVE, 1 );
				node = BackSideSon;
				COUNT_INSTRUCTION( nIdx, TRAVERSE_BACK, COMPARISON, 1 );
				if (d_near == 0)
				{
					COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the back  child
				}
				COUNT_INSTRUCTION( nIdx, TRAVERSE_FRONT, FUNCTION_ENTER, 1 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE_FRONT, MOVE, 1 );
				node = FrontSideSon;
				COUNT_INSTRUCTION( nIdx, TRAVERSE_FRONT, COMPARISON, 1 );
				if (d_far == 0)	
				{
					COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN_WITHOUT_PUSH );
					continue;	// traverse the front child
				}

				COUNT_INSTRUCTION( nIdx, TRAVERSE_BOTH, FUNCTION_ENTER, 1 );
				COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN );
				COUNT_INSTRUCTION( nIdx, TRAVERSE_BOTH, STORE, 3 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE_BOTH, MOVE, 1 );
				COUNT_INSTRUCTION( nIdx, TRAVERSE_BOTH, ADD, 1 );
				// case:  near < d < far
				m_Stack1x1[stackIndex].t_far_ = t_far_;
				m_Stack1x1[stackIndex].t_near = d;
				t_far_ = d;

				m_Stack1x1[stackIndex].node = BackSideSon;
				stackIndex++;
		}
		COUNT_INSTRUCTION( nIdx, TRAVERSE, BRANCH, 1 ); // while 문 빠져나감
		COUNT_INSTRUCTION( nIdx, TRAVERSE, COMPARISON, 1 );

		// Isect check
		IsectPacket1x1(node, nIdx);
		SET_FUNCTION( TRAVERSE );

		// Terimination test
		COUNT_INSTRUCTION( nIdx, TRAVERSE, BOOLEAN_OR, 2 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE, COMPARISON, 2 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE, LOAD, 2 );
		if (is->dist <= t_far_ || stackIndex == 0) break;

		COUNT_TREE_OPERATOR( nIdx, TRAVERSE_UP );
		// Stack pop
		--stackIndex;
		COUNT_INSTRUCTION( nIdx, TRAVERSE, SUBTRACT, 2 );
		COUNT_INSTRUCTION( nIdx, TRAVERSE, LOAD, 3 );
		node		= m_Stack1x1[stackIndex].node;
		t_near		= m_Stack1x1[stackIndex].t_near;
		t_far_		= m_Stack1x1[stackIndex].t_far_;
	}
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void SSERenderPipeline::shading1x1 (int nIdx)
{
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, FUNCTION_ENTER, 1 );

	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 1+3+3 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, MULTIPLY, 3 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, ADD, 3 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, STORE, 3 );
	_sse_float	hit_p = cpu_fadd(rp->o, cpu_fmul(rp->d, is->dist));

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 5 );
	bool bIsEnableShadow       = m_bIsEnableShadow;
	bool bIsRunShadowChk       = false;
	bool bIsEnableLocalShading = m_bIsEnableLocalShading;
	bool bIsUseTexture         = m_bIsUseTexture;
	int  iMaxReflectionDepth   = m_iMaxReflectionDepth;

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, MOVE, 2 );
	bool b_refl = false;
	bool b_refr = false;

	GColor		global_ambient;

	GColor		mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit;
	float		mat_fRough;
	float		mat_fRefl, mat_fRefr, mat_fRIdx;
	GColor		mat_cTex;
	UINT		obj_num;

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, BRANCH, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, COMPARISON, 1 );
	if (is->tacc == 0) {
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, STORE, 4 );
		is->color = GColor(0,0,0);		// Background color
		return;
	}

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 4 );
	global_ambient = m_globalAmbient;
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, SUBTRACT, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 1 );

	const int triID      = is->tacc -1;
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 3 );
	const int       tri_idx    = m_Data->m_TriObjList[triID]->indexInObject;
	GPolygonObject	*pObject   = m_Data->m_TriObjList[triID]->m_pObject;
	GMaterial		*pMaterial = pObject->getMaterial();
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, STORE, 1 );
	is->ads.pri_oid = pObject->getObjectNumber();

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 20 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, COMPARISON, 2 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, BRANCH, 2 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, MOVE, 2 );
	mat_cAmbt  = pMaterial->m_Ambient;
	mat_cDiff  = pMaterial->m_Diffuse;
	mat_cSpec  = pMaterial->m_Specular;
	mat_cEmit  = pMaterial->m_Emission;
	mat_fRough = 0.7;//pMaterial->getRoughness();
	mat_fRefl  = pMaterial->m_fReflection;		if (mat_fRefl > 0) { COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, MOVE, 1 ); b_refl = true; }
	mat_fRefr  = pMaterial->m_fTransparency;	if (mat_fRefr > 0) { COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, MOVE, 1 ); b_refr = true; }
	mat_fRIdx  = pMaterial->m_fRefractionIndex;
	obj_num   = pObject->m_iObjectNumber;

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, BARYCENTRIC, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 4 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, SUBTRACT, 2 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, MOVE, 4 );
	GVector N = pObject->calBarycentricNormal(tri_idx, 1-is->u-is->v, is->u, is->v);
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, STORE, 3 );
	is->n.x = N.x;
	is->n.y = N.y;
	is->n.z = N.z;

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, STORE, 1 );
	is->ads.pri_oid = pObject->getObjectNumber();
	COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, FUNCTION_ENTER, 1 );
	// Get object color
	GColor texColor;
	COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, COMPARISON, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, BRANCH, 1 );
	if ( bIsUseTexture ) {
		COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, LOAD, 1 );
		GTexture  *pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
		COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, LOAD, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, BOOLEAN_AND, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, COMPARISON, 2 );
		COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, BRANCH, 1 );
		if (pTexture && pTexture->isLoaded()) {
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, STORE, 1 );
			is->ads.pri_texture = 1;
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, BARYCENTRIC, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, LOAD, 4 );
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, SUBTRACT, 2 );
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, MOVE, 4 );
			GPoint point = pObject->calBarycentricUV(tri_idx, 1-is->u-is->v, is->u, is->v);
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, MOVE, 2 );
			float u = point.x;
			float v = point.y;
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, MOVE, 4 );
			texColor = pTexture->getTexel( u, v );
		} else {
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, STORE, 1 );
			is->ads.pri_texture = 0;
			COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, MOVE, 4 );
			texColor = mat_cDiff;
		}
	} else {
		COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, STORE, 4 );
		texColor = mat_cDiff;
	}
	COUNT_INSTRUCTION( nIdx, SHADING_GET_TEXTURE, MOVE, 4 );
	mat_cTex = texColor;


	int shadowcount = 0;
	COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, FUNCTION_ENTER, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, COMPARISON, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, BRANCH, 1 );
	if ( bIsEnableLocalShading ) {
		GColor oColor;
		GVector R, L;
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, LOAD, 6 );
		GVector rayD = GVector(rp->d.f);
		GPoint  hitP = GPoint(hit_p.f);

		_sse_1x1_raypacket	*shadow_rp;
		_sse_1x1_isect		*shadow_is;
		COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, BRANCH, 1 );
		if ( bIsEnableShadow ) {
			shadow_rp	= &m_ShadowRayPk1x1[0];
			shadow_is	= &m_ShadowIsect1x1[0];
			COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, STORE, 4 );
			shadow_rp->o = hit_p;
		}

		// Shading 에서, 투명한 물체일때, Normal 과 dir 의 dot 이 < 0 이라면 normal 을 뒤짚는다.
		// 확인 필요!!
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, COMPARISON, 2 );
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, BRANCH, 1 );

		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, LOAD, 3+3 );
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, DOT_PRODUCT, 1 );
		if (mat_fRefr > 0.0f && cpu_fdot(is->n, rp->d) > 0.0f)
		{
			COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MULTIPLY, 4 );
			N = -N;
		}
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, DOT_PRODUCT, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MULTIPLY, 8 );
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, ADD, 4 );
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, NORMALIZATION, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MOVE, 4 );
		R = GVector(-2 * N.innerProduct(rayD) * N + rayD).normalize();
		
		// Background color
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MOVE, 4 );
		oColor = GColor(0,0,0);

		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, BRANCH, 1 );
		if (is->tacc) {
			// Ambient color
			COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MULTIPLY, 4 );
			COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MOVE, 4 );
			oColor = global_ambient * mat_cAmbt;

			// Emission color
			COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, ADD, 4 );
			COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MOVE, 4 );
			oColor = oColor + mat_cEmit;

			// Diffuse & Specular color
			COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, LOAD, 1 );
			const vector<GLight*>* pLightList = m_Scene->getLightList();
			for ( int lx = 0; lx < (int) pLightList->size(); ++lx ) {	GLight* pLight = (*pLightList)[ lx ];
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, COMPARISON, 1 );
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, BRANCH, 1 );
				// Point Light 만 일단 지원
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, COMPARISON, 2 );
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, LOAD, 2 );
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, BOOLEAN_OR, 1 );
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, BRANCH, 1 );
				if ( pLight->getLightType() != typePointLight || !pLight->isEnabled() )  continue;

				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, LOAD, 8 );
				GColor   lightColor = pLight->getLightColor();
				GPoint   lightPos   = pLight->getPosition();

				// 광원 자기자신인 경우
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, COMPARISON, 1 );
				COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, BRANCH, 1 );
				if (obj_num == pLight->getObjectNumber()) {
					COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, ADD, 4 );
					COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, LOAD, 1 );
					COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, MULTIPLY, 4 );
					oColor = oColor + lightColor * pLight->getIntensity();
					continue;
				}

				// 그림자 확인
				COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, COMPARISON, 1 );
				COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, BRANCH, 1 );
				if ( bIsEnableShadow ) {
					COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, FUNCTION_ENTER, 1 );
					checkVisibility1x1(&hitP, &lightPos);

					COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, SUBTRACT, 4 );
					COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, NORMALIZATION, 1 );
					float lDist = GVector(lightPos - hitP).length();

					// Phong shading
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, FUNCTION_ENTER, 1 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, SUBTRACT, 4 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, NORMALIZATION, 1 );
					L = GVector(lightPos - hitP).normalize();

					// shadow 관련 visible 조건
					//		중간에 shadow ray 와 교점이 없거나
					//		shadow ray 가 교차점이 뒤에 존재하거나 아니면 거리가 거의 가깝거나
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, SUBTRACT, 1 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, LOAD, 3 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, COMPARISON, 3 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, BOOLEAN_OR, 2 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, ABS, 2 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, BRANCH, 1 );
					if (shadow_is->tacc == 0 || fabsf(lDist - shadow_is->dist) < 1.f*EPSILON || shadow_is->dist > lDist) {
						COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, MAX_OP, 2 );
						COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, DOT_PRODUCT, 2 );
						COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, POWER, 1 );
						COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, MULTIPLY, 16 );
						COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, ADD, 4 );
						oColor += mat_cTex  * lightColor * max( 0.0f, L.innerProduct(N) ) +
								  mat_cSpec * lightColor * pow( max( 0.0f, R.innerProduct(L) ), mat_fRough);
					} else {
						COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, ADD, 1 );
						shadowcount++;
					}
					// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
					//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
				} else {
					// Phong shading
					//SET_FUNCTION( SHADING_PHONG_SHADING );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, FUNCTION_ENTER, 1 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, SUBTRACT, 4 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, NORMALIZATION, 1 );
					L = GVector(lightPos - hitP).normalize();

					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, MAX_OP, 2 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, DOT_PRODUCT, 2 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, POWER, 1 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, MULTIPLY, 16 );
					COUNT_INSTRUCTION( nIdx, SHADING_PHONG_SHADING, ADD, 4 );
					oColor += mat_cTex  * lightColor * max( 0.0f, L.innerProduct(N) ) +
							  mat_cSpec * lightColor * pow( max( 0.0f, R.innerProduct(L) ), mat_fRough);
					// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
					//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
				}
			}
		}
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, STORE, 4 );
		is->color = oColor;
	} else {
		COUNT_INSTRUCTION( nIdx, SHADING_LOCAL_SHADING, STORE, 4 );
		is->color = mat_cTex;
	}

	COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, COMPARISON, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_SHADOW, BRANCH, 1 );
	if (rp->Depth < 2) {
		is->ads.pri_shadow = shadowcount;
	}

	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, COMPARISON, 1 );
	COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, BRANCH, 1 );
	if (rp->Depth < iMaxReflectionDepth) {
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, LOAD, 4 );
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, STORE, 4 );
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, SUBTRACT, 2 );
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, MULTIPLY, 4 );
		is->color = is->color * (1.0f - mat_fRefl - mat_fRefr);

		// ---------------------------------------------------------------------------
		// reflection
		// ---------------------------------------------------------------------------
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, BRANCH, 1 );
		if (b_refl) {
			float dot_i;
			
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, ADD, 1 );
			const int nNextIdx = nIdx+1;

			_sse_1x1_raypacket	*refl_rp	= &m_RayPk1x1[nNextIdx];
			_sse_1x1_isect		*refl_is	= &m_Isect1x1[nNextIdx];

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, FUNCTION_ENTER, 1 );

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, DOT_PRODUCT, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 3+3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, MOVE, 3 );
			dot_i = cpu_fdot(rp->d, is->n);

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 3+3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, MULTIPLY, 3+3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, SUBTRACT, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, STORE, 3 );
			refl_rp->d = cpu_fsub(rp->d, cpu_fmul(2, cpu_fmul(dot_i, is->n)));
			InitPacket1x1( nNextIdx );

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, MULTIPLY, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, ADD, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, STORE, 3 );
			refl_rp->o = cpu_fadd(hit_p, cpu_fmul(refl_rp->d, RAY_START_EPSILON));
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, ADD, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, STORE, 1 );
			refl_rp->Depth = rp->Depth+1;

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, SHIFT, 2 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, COMPARISON, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, ADD, 2 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, MOVE, 2 );
			int q = (refl_rp->d.x < 0) + ((refl_rp->d.y < 0) << 1) + ((refl_rp->d.z < 0) << 2);
			RenderPacket1x1(q, nNextIdx);

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, BRANCH, 1 );
			if (refl_is->tacc) {
				COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, MULTIPLY, 8 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, ADD, 4 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 4 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, STORE, 4 );
				is->color += mat_fRefl * refl_is->color * mat_cTex;
			}

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, BRANCH, 1 );
			if (rp->Depth == 0) {
				COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 4 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, STORE, 4 );
				is->ads.sec_oid     = refl_is->ads.pri_oid;
				is->ads.sec_shadow  = refl_is->ads.pri_shadow;
				is->ads.sec_texture = refl_is->ads.pri_texture;
				is->ads.n2          = refl_is->n;
			}
		}

		// ---------------------------------------------------------------------------
		// refraction
		// ---------------------------------------------------------------------------
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, COMPARISON, 1 );
		COUNT_INSTRUCTION( nIdx, SHADING_INITIALIZATION, BRANCH, 1 );
		if (b_refr) {
			float dot_i, dot_r;
			float n_div_nt;

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, FUNCTION_ENTER, 1 );

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 1 );
			const int nNextIdx = nIdx+1;
			_sse_1x1_raypacket	*refr_rp	= &m_RayPk1x1[nNextIdx];
			_sse_1x1_isect		*refr_is	= &m_Isect1x1[nNextIdx];

			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, DOT_PRODUCT, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, LOAD, 3+3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFLECTION, MOVE, 3 );
			dot_i = cpu_fdot(rp->d, is->n);

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, BRANCH, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, DIVISION, 1 );
			if (dot_i < 0) {
				n_div_nt = AIR_INDEX / mat_fRIdx;
			} else {
				n_div_nt = mat_fRIdx / AIR_INDEX;
			}

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, MULTIPLY, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 2 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ABS, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, SQRT, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, MOVE, 1 );
			dot_r = sqrtf(fabsf(1.0f - n_div_nt * n_div_nt * (1 - dot_i * dot_i)));

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, BRANCH, 1 );
			if(n_div_nt < 1.0) {
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, LOAD, 3+3+3 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, MULTIPLY, 3+3+3 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, SUBTRACT, 3+3 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, STORE, 3 );
				refr_rp->d = cpu_fsub(cpu_fmul(n_div_nt, cpu_fsub(rp->d, cpu_fmul(is->n, dot_i))), cpu_fmul(is->n, dot_r));
			} else {
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, LOAD, 3+3+3 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, MULTIPLY, 3+3+3 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, SUBTRACT, 3 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 3 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, STORE, 3 );
				refr_rp->d = cpu_fadd(cpu_fmul(n_div_nt, cpu_fsub(rp->d, cpu_fmul(is->n, dot_i))), cpu_fmul(is->n, dot_r));
			}
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 1 );
			InitPacket1x1( nNextIdx );

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, LOAD, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, MULTIPLY, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, STORE, 3 );
			refr_rp->o = cpu_fadd(hit_p, cpu_fmul(refr_rp->d, RAY_START_EPSILON));
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, LOAD, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, STORE, 1 );
			refr_rp->Depth = rp->Depth+1;

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, LOAD, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, SHIFT, 2 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, COMPARISON, 3 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 2 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, MOVE, 2 );
			int q = (refr_rp->d.x < 0) + ((refr_rp->d.y < 0) << 1) + ((refr_rp->d.z < 0) << 2);
			RenderPacket1x1(q, nNextIdx);

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, BRANCH, 1 );
			if (refr_is->tacc) {
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, MULTIPLY, 8 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, ADD, 4 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, LOAD, 4 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, STORE, 4 );
				is->color += mat_fRefr * refr_is->color * mat_cTex;
			}

			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, COMPARISON, 1 );
			COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, BRANCH, 1 );
			if (rp->Depth == 0) {
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, LOAD, 4 );
				COUNT_INSTRUCTION( nIdx, SHADING_REFRACTION, STORE, 4 );
				is->ads.sec_oid     = refr_is->ads.pri_oid;
				is->ads.sec_shadow  = refr_is->ads.pri_shadow;
				is->ads.sec_texture = refr_is->ads.pri_texture;
				is->ads.n2          = refr_is->n;
			}
		}
	}
}

// -----------------------------------------------------------
// SSERenderPipeline::RenderPacket
//		Sends a 1x1 packet through the tree
// -----------------------------------------------------------
void SSERenderPipeline::RenderPacket1x1( unsigned int quad, int nIdx )
{
	TracePacket1x1(quad, nIdx);
	shading1x1(nIdx);
}

// -----------------------------------------------------------
// SSERenderPipeline::RenderTiles
//		Tile renderer
// -----------------------------------------------------------
void SSERenderPipeline::Render1x1( int nJobID )
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

	for ( ty = yTileStart; ty < yTileEnd; ty++ ) {
	for ( tx = 0; tx < xTileEnd;  tx++ ) {

		// -----------------------------------------------------------------------
		// tpos (Ray 를 쏠 방향지점) 계산
		// -----------------------------------------------------------------------
		// m_LeftUp : image screen 위쪽 왼편 모서리의 pixel 중심 으로 이미 셋팅 되어 있음
		COUNT_INSTRUCTION( 0, RAY_GENERATION, FUNCTION_ENTER, 1 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, LOAD, 8 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, MULTIPLY, 8 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, ADD, 4 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, SUBTRACT, 4 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, MOVE, 4 );
		vector3 r = m_LeftUp + (m_DX * (float)tx) - (m_DY * (float)ty);
		COUNT_INSTRUCTION( 0, RAY_GENERATION, LOAD, 3 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, STORE, 3 );
		tpos.d.x = r.x;
		tpos.d.y = r.y;
		tpos.d.z = r.z;
		COUNT_INSTRUCTION( 0, RAY_GENERATION, LOAD, 2 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, SUBTRACT, 2 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, ADD, 1 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, MULTIPLY, 1 );
		COUNT_INSTRUCTION( 0, RAY_GENERATION, STORE, 1 );
		is->addr = tx + (m_Height - 1 - ty) * m_Width;

			GColor o_color;
			for (int nSampX = 0; nSampX < m_SuperSampling.x; nSampX++) {
			for (int nSampY = 0; nSampY < m_SuperSampling.y; nSampY++) {
				if (m_SuperSampling.x == 1) {
					COUNT_INSTRUCTION( 0, RAY_GENERATION, MOVE, 4 );
					jpos = tpos;
				} else {
					if ( m_Scene->isEnableJittering() ) {
						GVector samp_D;
						samp_D  = m_DX * (fSampSeq_x[m_SuperSampling.x][nSampX] + ( (float)sgrtRadicalInverse(nSampX,3)*2.0f*fXjitter - fXjitter))
								- m_DY * (fSampSeq_y[m_SuperSampling.y][nSampY] + ( (float)sgrtRadicalInverse(nSampY,5)*2.0f*fYjitter - fYjitter));
						jpos.d	= cpu_fadd(tpos.d, cpu_fset1(samp_D.m_Vector));
					} else {
						GVector samp_D;
						samp_D  = m_DX * fSampSeq_x[m_SuperSampling.x][nSampX]
								- m_DY * fSampSeq_y[m_SuperSampling.y][nSampY];
						jpos.d	= cpu_fadd(tpos.d, cpu_fset1(samp_D.m_Vector));
					}
				}

				// Ray 를 셋팅 - 시작점(rp->o) ~ 끝점(jpos)
				COUNT_INSTRUCTION( 0, RAY_GENERATION, LOAD, 3 );
				COUNT_INSTRUCTION( 0, RAY_GENERATION, SUBTRACT, 3 );
				COUNT_INSTRUCTION( 0, RAY_GENERATION, STORE, 3 );
				rp->d = cpu_fsub(jpos.d, rp->o);

				COUNT_INSTRUCTION( 0, RAY_GENERATION, STORE, 1 );
				rp->Depth = 0;
				InitPacket1x1( 0 );	// direction vector normalize 등


				COUNT_INSTRUCTION( 0, RAY_GENERATION, LOAD, 3 );
				COUNT_INSTRUCTION( 0, RAY_GENERATION, SHIFT, 2 );
				COUNT_INSTRUCTION( 0, RAY_GENERATION, COMPARISON, 3 );
				COUNT_INSTRUCTION( 0, RAY_GENERATION, ADD, 2 );
				COUNT_INSTRUCTION( 0, RAY_GENERATION, MOVE, 2 );
				// ray dir 결정 (q = 8방향중하나)
				int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
				RenderPacket1x1(q, 0);

				// -----------------------------------------------------------------------
				COUNT_INSTRUCTION( 0, PIXEL_STORE, FUNCTION_ENTER, 1 );
				// Copy color info to Memory
				// -----------------------------------------------------------------------
				COUNT_INSTRUCTION( 0, PIXEL_STORE, LOAD, 4 );
				COUNT_INSTRUCTION( 0, PIXEL_STORE, ADD, 4 );
				o_color = o_color + is->color;
			}}	// Loops of (nSampleX * nSampleY)

			COUNT_INSTRUCTION( 0, PIXEL_STORE, MULTIPLY, 6 );
			COUNT_INSTRUCTION( 0, PIXEL_STORE, LOAD, 3 );
			COUNT_INSTRUCTION( 0, PIXEL_STORE, ADD, 2 );
			COUNT_INSTRUCTION( 0, PIXEL_STORE, STORE, 3 );
			m_Dest[3*(is->addr)]   = o_color.r * fSampWeight;
			m_Dest[3*(is->addr)+1] = o_color.g * fSampWeight;
			m_Dest[3*(is->addr)+2] = o_color.b * fSampWeight;
	}
	}
}

void SSERenderPipeline::Render1x1_ADPSS_OnePass( int nJobID )
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

	rp->o   = cpu_fset1(m_Origin.m_Vector);

	for ( ty = yTileStart; ty < yTileEnd; ty++ ) {
	for ( tx = 0; tx < xTileEnd;  tx++ ) {
		// Set tpos
		vector3 r = m_LeftUp + (m_DX * (float)tx) - (m_DY * (float)ty);
		tpos.d.x = r.x;
		tpos.d.y = r.y;
		tpos.d.z = r.z;
		is->addr = tx + (m_Height - 1 - ty) * m_Width;

		// -----------------------------------------------------------------------
		// Ray 를 셋팅 - 시작점(rp->o) ~ 끝점(tpos)
		// -----------------------------------------------------------------------
		rp->d = cpu_fsub(tpos.d, rp->o);
		rp->Depth = 0;
		InitPacket1x1( 0 );	// direction vector normalize 등

		// -----------------------------------------------------------------------
		// Ray dir 결정 (q = 8방향중하나) 후 rendering
		// -----------------------------------------------------------------------
		int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
		RenderPacket1x1(q, 0);

		// -----------------------------------------------------------------------
		// Copy color info to Memory
		// -----------------------------------------------------------------------
		m_Dest[3*(is->addr)]   = is->color.r;
		m_Dest[3*(is->addr)+1] = is->color.g;
		m_Dest[3*(is->addr)+2] = is->color.b;

		// -----------------------------------------------------------------------
		// Save informations for Adaptive SuperSampling
		// -----------------------------------------------------------------------
		m_ADPSS_data[is->addr].ray_d.x = tpos.d.x;
		m_ADPSS_data[is->addr].ray_d.y = tpos.d.y;
		m_ADPSS_data[is->addr].ray_d.z = tpos.d.z;

		m_ADPSS_data[is->addr].pri_oid = is->ads.pri_oid;			// Primary   hit Object ID
		m_ADPSS_data[is->addr].sec_oid = is->ads.sec_oid;			// Secondary hit Object ID
		m_ADPSS_data[is->addr].pri_shadow = is->ads.pri_shadow;	// Primary   hit Shadow
		m_ADPSS_data[is->addr].sec_shadow = is->ads.sec_shadow;	// Secondary hit Shadow

		m_ADPSS_data[is->addr].pri_normal.x = is->n.x;	// Primary   hit Object Normal
		m_ADPSS_data[is->addr].pri_normal.y = is->n.y;
		m_ADPSS_data[is->addr].pri_normal.z = is->n.z;

		m_ADPSS_data[is->addr].sec_normal.x = is->ads.n2.x;	// Secondary hit Object Normal
		m_ADPSS_data[is->addr].sec_normal.y = is->ads.n2.y;
		m_ADPSS_data[is->addr].sec_normal.z = is->ads.n2.z;

		m_ADPSS_data[is->addr].oColor.rgba = is->color.rgba;

		m_ADPSS_data[is->addr].pri_texture = is->ads.pri_texture;
		m_ADPSS_data[is->addr].sec_texture = is->ads.sec_texture;
	}
	}
}

void SSERenderPipeline::Render1x1_ADPSS_Detection( int nJobID )
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

	int adjPixAddr[4];
	int adjPixData[4];
	float fadjPixData[4];

	yPi = nPixelStart / wPixels;
	xPi = nPixelStart - yPi * wPixels;

	for (pi = nPixelStart; pi < nPixelEnd; pi++, xPi++) {
		if (xPi == wPixels)  { xPi = 0; yPi++; }

		int nSuperSample1x1_Count = 0;
		int nSuperSample1x1_Flag  = 0;

		// -----------------------------------------------------------------------------------------------
		// Pixel 의 difference value 생성
		// -----------------------------------------------------------------------------------------------
		float colordifference = 1.0f;
		if (1) {
			float xvalue = 0, yvalue = 0;

			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			float hx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
			float hy[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
			// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
				//  xSobelMask4        ySobelMask4
				//   -1  0  1			 1  2  1
				//   -2  0  2			 O  0  0
				//   -1  0  1			-1 -2 -1

			for ( int j = -1; j <= 1; ++j ) {
			for ( int i = -1; i <= 1; ++i ) {
				if ( xPi + i >= 0 && xPi + i < (int)wPixels && 
					 yPi + j >= 0 && yPi + j < (int)hPixels ) {
					int index = (yPi + j) * wPixels + (xPi + i);
					// GrayScale = 0.3 * red + 0.59 * green + 0.11 * blue
					float grayScale =   0.3f  * m_ADPSS_data[index].oColor.r +
										0.59f * m_ADPSS_data[index].oColor.g +
										0.11f * m_ADPSS_data[index].oColor.b;
					xvalue += hx[ j + 1 ][ i + 1 ] * grayScale;
					yvalue += hy[ j + 1 ][ i + 1 ] * grayScale;
				}
			}
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

			for (int i = 0; i < 4; i++) {
				adjPixAddr[i] = m_ADPSS_AdjPixAddr[sub_pi][i] + pi;
			}

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
				if (!(adjPixData[0] == adjPixData[1] && adjPixData[0] == adjPixData[2] && adjPixData[0] == adjPixData[3])) {
					checkRegion = true;
					if ( ( nCompareType & 1 ) == 1 && colordifference > fPrimaryOIDRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** primary normal */
				for (j = 0; j < 4; j++) {
					if (m_ADPSS_data[adjPixAddr[j]].pri_normal.x == 0 && m_ADPSS_data[adjPixAddr[j]].pri_normal.y == 0 &&
						m_ADPSS_data[adjPixAddr[j]].pri_normal.z == 0 && m_ADPSS_data[adjPixAddr[j]].pri_normal.w == 0) {
						m_ADPSS_data[adjPixAddr[j]].pri_normal.x = 1.0f; m_ADPSS_data[adjPixAddr[j]].pri_normal.y = 1.0f;
						m_ADPSS_data[adjPixAddr[j]].pri_normal.z = 1.0f; m_ADPSS_data[adjPixAddr[j]].pri_normal.w = 1.0f;
					}
					fadjPixData[j] = 
						m_ADPSS_data[adjPixAddr[j]].pri_normal.x * m_ADPSS_data[adjPixAddr[0]].pri_normal.x +
						m_ADPSS_data[adjPixAddr[j]].pri_normal.y * m_ADPSS_data[adjPixAddr[0]].pri_normal.y +
						m_ADPSS_data[adjPixAddr[j]].pri_normal.z * m_ADPSS_data[adjPixAddr[0]].pri_normal.z;
				}
				if (!(fadjPixData[0] > 0.5f && fadjPixData[1] > 0.5f && fadjPixData[2] > 0.5f && fadjPixData[3] > 0.5f)) { // 하나라도 0.5 보다 작은 경우
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
				if (!(adjPixData[0] == adjPixData[1] && adjPixData[0] == adjPixData[2] && adjPixData[0] == adjPixData[3])) {
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
				if (!(adjPixData[0] == adjPixData[1] && adjPixData[0] == adjPixData[2] && adjPixData[0] == adjPixData[3])) {
					checkRegion = true;
					if ( ( nCompareType & 16 ) == 16 && colordifference > fSecondaryOIDRegionColorThreshold) {
						flag = true;
						break;
					}
				}

				/** secondary normal */
				for (j = 0; j < 4; j++) {
					if (m_ADPSS_data[adjPixAddr[j]].sec_normal.x == 0 && m_ADPSS_data[adjPixAddr[j]].sec_normal.y == 0 &&
						m_ADPSS_data[adjPixAddr[j]].sec_normal.z == 0 && m_ADPSS_data[adjPixAddr[j]].sec_normal.w == 0) {
						m_ADPSS_data[adjPixAddr[j]].sec_normal.x = 1.0f; m_ADPSS_data[adjPixAddr[j]].sec_normal.y = 1.0f;
						m_ADPSS_data[adjPixAddr[j]].sec_normal.z = 1.0f; m_ADPSS_data[adjPixAddr[j]].sec_normal.w = 1.0f;
					}
					fadjPixData[j] = 
						m_ADPSS_data[adjPixAddr[j]].sec_normal.x * m_ADPSS_data[adjPixAddr[0]].sec_normal.x +
						m_ADPSS_data[adjPixAddr[j]].sec_normal.y * m_ADPSS_data[adjPixAddr[0]].sec_normal.y +
						m_ADPSS_data[adjPixAddr[j]].sec_normal.z * m_ADPSS_data[adjPixAddr[0]].sec_normal.z;
				}
				if (!(fadjPixData[0] > 0.5f && fadjPixData[1] > 0.5f && fadjPixData[2] > 0.5f && fadjPixData[3] > 0.5f)) { // 하나라도 0.5 보다 작은 경우
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
				if (!(adjPixData[0] == adjPixData[1] && adjPixData[0] == adjPixData[2] && adjPixData[0] == adjPixData[3])) {
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
				nSuperSample1x1_Flag |= 1 << sub_pi;
				nSuperSample1x1_Count++;
			}
		}	// sub_pi loop

		//----------------------------------------
		// 샘플링 되는 지역만 표시
		//----------------------------------------
		if (m_Scene->isEnableSamplingDebugInfo() == true) {
			if ( nSuperSample1x1_Count > 0 ) {
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
		if ( nSuperSample1x1_Count > 0 ) {
			float fSampWeight = 0.0625f; // 1/16
			for (sub_pi = 0; sub_pi < 4; sub_pi++) {
				if (nSuperSample1x1_Flag & (1 << sub_pi)) {
					// 여기서 super sampleing 함
					m_RayT1x1->Set_Item(pi, sub_pi);
				}
			}
		}

		float weight = 1 - 0.25f * nSuperSample1x1_Count;
		m_Dest[3*pi]   = m_ADPSS_data[pi].oColor.r * weight;
		m_Dest[3*pi+1] = m_ADPSS_data[pi].oColor.g * weight;
		m_Dest[3*pi+2] = m_ADPSS_data[pi].oColor.b * weight;

		// Primary hit obj 에 따른 이미지
		//m_Dest[3*pi] = 1.0f * m_ADPSS_data[pi].pri_oid / m_Scene->getObjectCount();
		//m_Dest[3*pi+1] = 0;
		//m_Dest[3*pi+2] = 0;
	}
}

static float fSampSeq_x[4][4] = { {-0.375f, -0.125f, -0.375f, -0.125f},
								  { 0.125f,  0.375f,  0.125f,  0.375f},
								  {-0.375f, -0.125f, -0.375f, -0.125f},
								  { 0.125f,  0.375f,  0.125f,  0.375f} };
static float fSampSeq_y[4][4] = { {-0.375f, -0.375f, -0.125f, -0.125f},
								  {-0.375f, -0.375f, -0.125f, -0.125f},
								  { 0.125f,  0.125f,  0.375f,  0.375f},
								  { 0.125f,  0.125f,  0.375f,  0.375f} };


// -----------------------------------------------------------
// SSERenderPipeline::Render1x1_ADPSS_Detection
// -----------------------------------------------------------
void SSERenderPipeline::Render1x1_ADPSS_TwoPass( int nJobID )
{
	const unsigned int wPixels = m_Resolution.x;
	const unsigned int hPixels = m_Resolution.y;

	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[0];
	_sse_1x1_isect		*is	= &m_Isect1x1[0];

	rp->o   = cpu_fset1(m_Origin.m_Vector);

	int i;

	while ( 1 ) {
		int pi, sub_pi;
		if (m_RayT1x1->Get_Item(pi, sub_pi) == 0) break;

		for (i = 0; i < 4; i++) {
			// ray_d = pixel_center + (m_DX * fSampSeq_x) - (m_DY * fSampSeq_y) 
			rp->d.x = m_ADPSS_data[pi].ray_d.x + (m_DX.x * fSampSeq_x[sub_pi][i]) - (m_DY.x * fSampSeq_y[sub_pi][i]);
			rp->d.y = m_ADPSS_data[pi].ray_d.y + (m_DX.y * fSampSeq_x[sub_pi][i]) - (m_DY.y * fSampSeq_y[sub_pi][i]);
			rp->d.z = m_ADPSS_data[pi].ray_d.z + (m_DX.z * fSampSeq_x[sub_pi][i]) - (m_DY.z * fSampSeq_y[sub_pi][i]);

			// jitter 시 아래 코드 구현 추가
			// ray_d = ray_d + (jitterX * 0.25f - 0.125f) * m_DX - (jitterY * 0.25f - 0.125f) * m_DY

			rp->d = cpu_fsub(rp->d, rp->o);

			InitPacket1x1 ( 0 );

			is->addr   = pi;
			is->weight = GColor(0.0625f, 0.0625f, 0.0625f);
			rp->Depth = 0;

			// ray dir 결정 (q = 8방향중하나)
			int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
			RenderPacket1x1(q, 0);

			// -----------------------------------------------------------------------
			// Copy color info to Memory
			// -----------------------------------------------------------------------
			GColor o_color;
			o_color = is->color * is->weight;

			m_ColorBufferCS.lock();
			m_Dest[3*(is->addr)]   += o_color.r;
			m_Dest[3*(is->addr)+1] += o_color.g;
			m_Dest[3*(is->addr)+2] += o_color.b;
			m_ColorBufferCS.unlock();
		}
	}
}