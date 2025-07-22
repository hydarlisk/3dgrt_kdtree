/**************************************************************
  File name: RayTraversal.cpp
  Version: 1.0
  Date: November 19, 2014
 **************************************************************/

#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "MyMathUtility.h"

#include "Kd-treeConverter.h"
#include "RayTraversal.h"
using namespace KDTConverter;

#pragma warning ( disable : 4068 )
#pragma warning ( disable : 949 )

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------

Ray g_Ray[MAX_RAY_DEPTH];
Hit g_Hit[MAX_RAY_DEPTH];
Ray g_ShadowRay[1];
Hit g_ShadowHit[1];
KdStack g_Stack[MAX_STACK_SIZE];


static KdTree *g_kd_tree = NULL;

unsigned int g_RayID;				// Ray ID
unsigned int g_RayDIR[8][3][2];		// Pointer address offset for ray direction


//AABB

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
// InitRay
// ------------------------------------------------------------------------------------------------
void InitRay(int nIdx, Ray *a_Ray)
{
	Ray	*rp = &g_Ray[nIdx];
	Hit *is = &g_Hit[nIdx];

	rp->o = a_Ray->o;
	rp->d = a_Ray->d;

	// Normalize ray's direction vector
	float v1 = 1.0f / sqrtf((rp->d.x * rp->d.x) + (rp->d.y * rp->d.y) + (rp->d.z * rp->d.z));
	rp->d.x *= v1;
	rp->d.y *= v1;
	rp->d.z *= v1;

	// Clear initial color
	//memset( &is->color, 0, sizeof(is->color) );
}

// ------------------------------------------------------------------------------------------------
// InitShadowRay
// ------------------------------------------------------------------------------------------------
void InitShadowRay(void)
{
	Ray *rp = &g_ShadowRay[0];
	Hit *is = &g_ShadowHit[0];

	// Normalize ray's direction vector
	float v1 = 1.0f / sqrtf((rp->d.x * rp->d.x) + (rp->d.y * rp->d.y) + (rp->d.z * rp->d.z));
	rp->d.x *= v1;
	rp->d.y *= v1;
	rp->d.z *= v1;
}

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------
// IsectShadowRay
// ------------------------------------------------------------------------------------------------
void IsectShadowRay( const KdTreeNode *node )
{
	int i;

	Ray	*rp	= &g_ShadowRay[0];
	Hit *is	= &g_ShadowHit[0];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = g_kd_tree->tri_offset_list[i];
		TriAccel &acc = g_kd_tree->tri_accel_list[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayID) continue;
		acc.mbox = rp->RayID;

		// ---------------------------------------------------------------
		// ������ ��ü�� ����
		// ---------------------------------------------------------------
		if (acc.isTransparent) continue;

		const unsigned int k	= acc.k;

		float nd, f;
		nd = 1.0f / (rp->df[k] + acc.n_u * rp->df[ku] + acc.n_v * rp->df[kv]);
		f  = acc.n_d - (rp->of[k] + acc.n_u * rp->of[ku] + acc.n_v * rp->of[kv]);
		f = f * nd;

		if (!(is->dist >= f && f > RAY_DIST_EPSILON)) continue;	// eps < f <= Hit4.dist

		float hu, hv;
		float lambda, mue;
		hu = rp->of[ku] + f * rp->df[ku];
		hv = rp->of[kv] + f * rp->df[kv];

		lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
		mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

		if (lambda < 0.0f) continue;
		if (mue    < 0.0f) continue;
		if (lambda+mue > 1.0f) continue;

		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}

// -----------------------------------------------------------
// TraceShadowRay
// -----------------------------------------------------------
void TraceShadowRay(void)
{
	Ray	*rp	= &g_ShadowRay[0];
	Hit	*is	= &g_ShadowHit[0];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist = FLT_MAX;
	is->tacc = 0;

	// ray direction
	int quad = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
	const unsigned int* ray_dir = &g_RayDIR[quad][0][0];		// Get precomputed the traversal order (front/back)
															//      as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	KdTreeNode* node = &g_kd_tree->tree[0];

	unsigned int stackIndex = 0;

	// ray reciprocal direction
	float rcpRayDir[3];
	rcpRayDir[0] = 1.0f / rp->d.x;
	rcpRayDir[1] = 1.0f / rp->d.y;
	rcpRayDir[2] = 1.0f / rp->d.z;

	// ray id
	rp->RayID	= g_RayID;	g_RayID += 1;

	// near / far
	float t_near, t_far_;
	t_near = 0;
	t_far_ = FLT_MAX;

	{
		float l1, l2;
		l1 = (g_kd_tree->AABB[0] - rp->o.x) * rcpRayDir[0];
		l2 = (g_kd_tree->AABB[1] - rp->o.x) * rcpRayDir[0];
		t_near = MyMAX( MyMIN( l1,l2 ), t_near );
		t_far_ = MyMIN( MyMAX( l1,l2 ), t_far_ );
		l1 = (g_kd_tree->AABB[2] - rp->o.y) * rcpRayDir[1];
		l2 = (g_kd_tree->AABB[3] - rp->o.y) * rcpRayDir[1];
		t_near = MyMAX( MyMIN( l1,l2 ), t_near );
		t_far_ = MyMIN( MyMAX( l1,l2 ), t_far_ );
		l1 = (g_kd_tree->AABB[4] - rp->o.z) * rcpRayDir[2];
		l2 = (g_kd_tree->AABB[5] - rp->o.z) * rcpRayDir[2];
		t_near = MyMAX( MyMIN( l1,l2 ), t_near );
		t_far_ = MyMIN( MyMAX( l1,l2 ), t_far_ );
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &g_kd_tree->tree[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &g_kd_tree->tree[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			float d;

				d = (node_split - rp->of[dim]) * rcpRayDir[dim];
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

				node = BackSideSon;
				if (d_near == 0)
				{
					continue;	// traverse the back  child
				}
				node = FrontSideSon;
				if (d_far == 0)	
				{
					continue;	// traverse the front child
				}

				// case:  near < d < far
				g_Stack[stackIndex].t_far_ = t_far_;
				g_Stack[stackIndex].t_near = d;
				t_far_ = d;

				g_Stack[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Intersection check
		IsectShadowRay(node);

		// Terimination test
		if (is->dist <= t_far_ || stackIndex == 0) break;

		// Stack pop
		--stackIndex;
		node		= g_Stack[stackIndex].node;
		t_near		= g_Stack[stackIndex].t_near;
		t_far_		= g_Stack[stackIndex].t_far_;
	}
}

// ---------------------------------------------------------------------------
// checkVisibility
// ---------------------------------------------------------------------------
void checkVisibility(const float* objectPos, const float* lightPos) {

	Ray *shadow_rp	= &g_ShadowRay[0];
	Hit *shadow_is	= &g_ShadowHit[0];

	shadow_rp->df[0] = lightPos[0] - objectPos[0];
	shadow_rp->df[1] = lightPos[1] - objectPos[1];
	shadow_rp->df[2] = lightPos[2] - objectPos[2];

	InitShadowRay();
	shadow_rp->of[0] = objectPos[0] + shadow_rp->df[0] * RAY_DIST_EPSILON;
	shadow_rp->of[1] = objectPos[1] + shadow_rp->df[1] * RAY_DIST_EPSILON;
	shadow_rp->of[2] = objectPos[2] + shadow_rp->df[2] * RAY_DIST_EPSILON;

	TraceShadowRay();
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// IsectRay
// ------------------------------------------------------------------------------------------------
void IsectRay( const KdTreeNode *node, int nIdx )
{
	int i;

	Ray *rp	= &g_Ray[nIdx];
	Hit *is	= &g_Hit[nIdx];

	const int baseOffset = OBJECTLIST_OFFSET(*node);
	const int nObjs		 = OBJECT_SIZE(*node);

	for (i = baseOffset; i < baseOffset+nObjs; i++) {
		int     triID = g_kd_tree->tri_offset_list[i];
		TriAccel &acc = g_kd_tree->tri_accel_list[triID];

		// ---------------------------------------------------------------
		// Mailbox
		// ---------------------------------------------------------------
		if (acc.mbox == rp->RayID) continue;
		acc.mbox = rp->RayID;

		// ---------------------------------------------------------------
		// Backface Culling : �������� �ʴ� ��ü�� �ش�
		// ---------------------------------------------------------------
		if (!acc.isTransparent && BACKFACE_CULLING) {
			if (fMyVecDotProduct(rp->df, acc.N) < 0) {
				continue;
			}
		}

		const unsigned int k	= acc.k;

		float nd, f;
		nd = 1.0f / (rp->df[k] + acc.n_u * rp->df[ku] + acc.n_v * rp->df[kv]);
		f  = acc.n_d - (rp->of[k] + acc.n_u * rp->of[ku] + acc.n_v * rp->of[kv]);
		f = f * nd;

		if (!(is->dist >= f && f > RAY_DIST_EPSILON)) continue;	// eps < f <= Hit4.dist

		float hu, hv;
		float lambda, mue;
		hu = rp->of[ku] + f * rp->df[ku];
		hv = rp->of[kv] + f * rp->df[kv];

		lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
		mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

		if (lambda < 0.0f) continue;
		if (mue    < 0.0f) continue;
		if (lambda+mue > 1.0f) continue;

		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
		is->material_ID = acc.material_ID;
	}
}

// -----------------------------------------------------------
// TraceRay
// -----------------------------------------------------------
void TraceRay(int nIdx, KdTree *a_kd_tree, Hit *a_is)
{
	Ray *rp	= &g_Ray[nIdx];
	Hit *is	= &g_Hit[nIdx];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist = FLT_MAX;
	is->tacc = 0;

	// ray direction
	int quad = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
	const unsigned int* ray_dir = &g_RayDIR[quad][0][0];		// Get precomputed the traversal order (front/back)
																// as offsets for a bundle of rays with equal directions
	// kdtree start node, stack index
	g_kd_tree = a_kd_tree;
	KdTreeNode* node = &g_kd_tree->tree[0];

	unsigned int stackIndex = 0;

	// ray reciprocal direction
	float rcpRayDir[3];
	rcpRayDir[0] = 1.0f / rp->d.x;
	rcpRayDir[1] = 1.0f / rp->d.y;
	rcpRayDir[2] = 1.0f / rp->d.z;

	// ray id
	rp->RayID	= g_RayID;	g_RayID += 1;

	// near / far
	float t_near, t_far_;
	t_near = 0;
	t_far_ = FLT_MAX;

	{
		float l1, l2;
		l1 = (g_kd_tree->AABB[0] - rp->o.x) * rcpRayDir[0];
		l2 = (g_kd_tree->AABB[1] - rp->o.x) * rcpRayDir[0];
		t_near = MyMAX( MyMIN( l1,l2 ), t_near );
		t_far_ = MyMIN( MyMAX( l1,l2 ), t_far_ );
		l1 = (g_kd_tree->AABB[2] - rp->o.y) * rcpRayDir[1];
		l2 = (g_kd_tree->AABB[3] - rp->o.y) * rcpRayDir[1];
		t_near = MyMAX( MyMIN( l1,l2 ), t_near );
		t_far_ = MyMIN( MyMAX( l1,l2 ), t_far_ );
		l1 = (g_kd_tree->AABB[4] - rp->o.z) * rcpRayDir[2];
		l2 = (g_kd_tree->AABB[5] - rp->o.z) * rcpRayDir[2];
		t_near = MyMAX( MyMIN( l1,l2 ), t_near );
		t_far_ = MyMIN( MyMAX( l1,l2 ), t_far_ );
	}

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &g_kd_tree->tree[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &g_kd_tree->tree[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			float d;

				d = (node_split - rp->of[dim]) * rcpRayDir[dim];
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

				node = BackSideSon;
				if (d_near == 0)
				{
					continue;	// traverse the back  child
				}
				node = FrontSideSon;
				if (d_far == 0)	
				{
					continue;	// traverse the front child
				}

				// case:  near < d < far
				g_Stack[stackIndex].t_far_ = t_far_;
				g_Stack[stackIndex].t_near = d;
				t_far_ = d;

				g_Stack[stackIndex].node = BackSideSon;
				stackIndex++;
		}

		// Intersection check
		IsectRay(node, nIdx);

		// Terimination test
		if (is->dist <= t_far_ || stackIndex == 0) break;

		// Stack pop
		--stackIndex;
		node		= g_Stack[stackIndex].node;
		t_near		= g_Stack[stackIndex].t_near;
		t_far_		= g_Stack[stackIndex].t_far_;
	}

	a_is->u = is->u;
	a_is->v = is->v;
	a_is->n = is->n;
	a_is->material_ID = is->material_ID;
	a_is->dist = is->dist;
	a_is->tacc = is->tacc;
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void Shading (int nIdx)
{
	Ray *rp	= &g_Ray[nIdx];
	Hit *is	= &g_Hit[nIdx];

	// Background color
	is->color[0] = is->color[1] = is->color[2] = 0.2f;

	// In case that ray hits nothing
	if (is->tacc == 0) return;

	float hit_p[3];
	hit_p[0] = rp->of[0] + rp->df[0] * is->dist;
	hit_p[1] = rp->of[1] + rp->df[1] * is->dist;
	hit_p[2] = rp->of[2] + rp->df[2] * is->dist;

	bool bIsEnableShadow       = true;
	bool bIsEnableLocalShading = true;
	bool bIsUseTexture         = false;
	int  iMaxReflectionDepth   = 1;

	// global ambient
	float global_ambient[3];
	global_ambient[0] = 1.0f;
	global_ambient[1] = 1.0f;
	global_ambient[2] = 1.0f;

	// ambient color
	float mat_cAmbt[3];
	mat_cAmbt[0] = 1.0f;
	mat_cAmbt[1] = 1.0f;
	mat_cAmbt[2] = 1.0f;

	// diffuse color
	float mat_cDiff[3];
	mat_cDiff[0] = 1.0f;
	mat_cDiff[1] = 1.0f;
	mat_cDiff[2] = 1.0f;
	
	// specular color
	float mat_cSpec[3];
	mat_cSpec[0] = 1.0f;
	mat_cSpec[1] = 1.0f;
	mat_cSpec[2] = 1.0f;

	// emit color
	float mat_cEmit[3];
	mat_cEmit[0] = 1.0f;
	mat_cEmit[1] = 1.0f;
	mat_cEmit[2] = 1.0f;

	// texture color
	float mat_cTex[3];
	mat_cTex[0] = 1.0f;
	mat_cTex[1] = 1.0f;
	mat_cTex[2] = 1.0f;

	// roughness
	float mat_fRough = 0.6f;

	// reflection ratio
	float mat_fRefl = 0.3f;
	
	// refraction ratio
	float mat_fRefr = 0.0f;
	
	// refraction index
	float mat_fRIdx = 0.7f;
	
	unsigned int obj_num;


#if 0
	const int triID      = is->tacc -1;
	const int       tri_idx    = m_Data->m_TriObjList[triID]->indexInObject;
	GPolygonObject	*pObject   = m_Data->m_TriObjList[triID]->m_pObject;
	GMaterial		*pMaterial = pObject->getMaterial();
	obj_num   = pObject->m_iObjectNumber;

	GVector N = pObject->calBarycentricNormal(tri_idx, 1-is->u-is->v, is->u, is->v);
	is->n.x = N.x;
	is->n.y = N.y;
	is->n.z = N.z;

	// Get object color
	if ( bIsUseTexture ) {
		mat_cDiff[0] = mat_cTex[0];
		mat_cDiff[1] = mat_cTex[1];
		mat_cDiff[2] = mat_cTex[2];
	}

	int shadowcount = 0;
	if ( bIsEnableLocalShading ) {
		float oColor[3];
		GVector R, L;
			GVector rayD = GVector(rp->d.f);
		GPoint  hitP = GPoint(hit_p.f);

		Ray *shadow_rp;
		Hit *shadow_is;
		if ( bIsEnableShadow ) {
			shadow_rp	= &g_ShadowRay[0];
			shadow_is	= &g_ShadowHit[0];
			shadow_rp->of[0] = hit_p[0];
			shadow_rp->of[1] = hit_p[1];
			shadow_rp->of[2] = hit_p[2];
		}

		// Shading ����, ������ ��ü�϶�, Normal �� dir �� dot �� < 0 �̶�� normal �� ��¤�´�.
		if (mat_fRefr > 0.0f && fMyVecDotProduct(is->nf, rp->df) > 0.0f) N = -N;

		R = GVector(-2 * N.innerProduct(rayD) * N + rayD).normalize();
		
		// Background color
		oColor[0] = oColor[1] = oColor[2] = 0.0f;		// Background color

		if (is->tacc) {
			// Ambient color
			oColor = global_ambient * mat_cAmbt;

			// Emission color
			oColor = oColor + mat_cEmit;

			// Diffuse & Specular color
			const vector<GLight*>* pLightList = m_Scene->getLightList();
			for ( int lx = 0; lx < (int) pLightList->size(); ++lx ) {	GLight* pLight = (*pLightList)[ lx ];
				// Point Light �� �ϴ� ����
				if ( pLight->getLightType() != typePointLight || !pLight->isEnabled() )  continue;

				GColor   lightColor = pLight->getLightColor();
				GPoint   lightPos   = pLight->getPosition();

				// ���� �ڱ��ڽ��� ���
				if (obj_num == pLight->getObjectNumber()) {
					oColor = oColor + lightColor * pLight->getIntensity();
					continue;
				}

				// �׸��� Ȯ��
				if ( bIsEnableShadow ) {
					checkVisibility(&hitP, &lightPos);

					float lDist = GVector(lightPos - hitP).length();

					// Phong shading
					L = GVector(lightPos - hitP).normalize();

					// shadow ���� visible ����
					//		�߰��� shadow ray �� ������ ���ų�
					//		shadow ray �� �������� �ڿ� �����ϰų� �ƴϸ� �Ÿ��� ���� �����ų�
					if (shadow_is->tacc == 0 || fabsf(lDist - shadow_is->dist) < 1.f*RAY_DIST_EPSILON || shadow_is->dist > lDist) {
						oColor += mat_cDiff * lightColor * max( 0.0f, L.innerProduct(N) ) +
								  mat_cSpec * lightColor * pow( max( 0.0f, R.innerProduct(L) ), mat_fRough);
					} else {
						shadowcount++;
					}
					// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
					//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
				} else {
					// Phong shading
					L = GVector(lightPos - hitP).normalize();

					oColor += mat_cDiff * lightColor * max( 0.0f, L.innerProduct(N) ) +
							  mat_cSpec * lightColor * pow( max( 0.0f, R.innerProduct(L) ), mat_fRough);
					// color += texColor * pLight->color * max( 0.0f, dot( L, N ) ) +
					//          specular * pLight->color * pow( max( 0.0f, dot( R, L ) ), roughness );
				}
			}
		}
		is->color[0] = oColor[0];
		is->color[1] = oColor[1];
		is->color[2] = oColor[2];
	} else {
		is->color[0] = mat_cDiff[0];
		is->color[1] = mat_cDiff[1];
		is->color[2] = mat_cDiff[2];
	}

	if (rp->Depth < iMaxReflectionDepth) {
		is->color[0] = is->color[0] * (1.0f - mat_fRefl - mat_fRefr);
		is->color[1] = is->color[1] * (1.0f - mat_fRefl - mat_fRefr);
		is->color[2] = is->color[2] * (1.0f - mat_fRefl - mat_fRefr);

		// ---------------------------------------------------------------------------
		// reflection
		// ---------------------------------------------------------------------------
		if (mat_fRefl > 0.0f) {
			float dot_i;
			
			const int nNextIdx = nIdx+1;
			Ray *refl_rp	= &g_Ray[nNextIdx];
			Hit *refl_is	= &g_Hit[nNextIdx];

			dot_i = fMyVecDotProduct(rp->df, is->nf);
			refl_rp->df[0] = rp->df[0] - 2.0f * dot_i * is->nf[0];
			refl_rp->df[1] = rp->df[1] - 2.0f * dot_i * is->nf[1];
			refl_rp->df[2] = rp->df[2] - 2.0f * dot_i * is->nf[2];

			InitRay( nNextIdx );
			refl_rp->of[0] = hit_p[0] + refl_rp->df[0] * RAY_DIST_EPSILON;
			refl_rp->of[1] = hit_p[1] + refl_rp->df[1] * RAY_DIST_EPSILON;
			refl_rp->of[2] = hit_p[2] + refl_rp->df[2] * RAY_DIST_EPSILON;
			refl_rp->Depth = rp->Depth+1;

			int q = (refl_rp->d.x < 0) + ((refl_rp->d.y < 0) << 1) + ((refl_rp->d.z < 0) << 2);
			TraceRay(q, nNextIdx);
			shading(nNextIdx);

			if (refl_is->tacc) {
				is->color[0] += mat_fRefl * refl_is->color[0] * mat_cTex[0];
				is->color[1] += mat_fRefl * refl_is->color[1] * mat_cTex[1];
				is->color[2] += mat_fRefl * refl_is->color[2] * mat_cTex[2];
			}
		}

		// ---------------------------------------------------------------------------
		// refraction
		// ---------------------------------------------------------------------------
		if (mat_fRefr > 0.0f) {
			float dot_i, dot_r;
			float n_div_nt;

			const int nNextIdx = nIdx+1;
			Ray *refr_rp	= &g_Ray[nNextIdx];
			Hit *refr_is	= &g_Hit[nNextIdx];

			dot_i = fMyVecDotProduct(rp->df, is->nf);
			if (dot_i < 0) {
				n_div_nt = AIR_INDEX / mat_fRIdx;
			} else {
				n_div_nt = mat_fRIdx / AIR_INDEX;
			}

			dot_r = sqrtf(fabsf(1.0f - n_div_nt * n_div_nt * (1 - dot_i * dot_i)));
			if(n_div_nt < 1.0) {
				refr_rp->df[0] = n_div_nt * (rp->df[0] - dot_i * is->nf[0]) - dot_r * is->nf[0];
				refr_rp->df[1] = n_div_nt * (rp->df[1] - dot_i * is->nf[1]) - dot_r * is->nf[1];
				refr_rp->df[2] = n_div_nt * (rp->df[2] - dot_i * is->nf[2]) - dot_r * is->nf[2];
			} else {
				refr_rp->df[0] = n_div_nt * (rp->df[0] - dot_i * is->nf[0]) + dot_r * is->nf[0];
				refr_rp->df[1] = n_div_nt * (rp->df[1] - dot_i * is->nf[1]) + dot_r * is->nf[1];
				refr_rp->df[2] = n_div_nt * (rp->df[2] - dot_i * is->nf[2]) + dot_r * is->nf[2];
			}
			InitRay( nNextIdx );
			refr_rp->of[0] = hit_p[0] + refr_rp->df[0] * RAY_DIST_EPSILON;
			refr_rp->of[1] = hit_p[1] + refr_rp->df[1] * RAY_DIST_EPSILON;
			refr_rp->of[2] = hit_p[2] + refr_rp->df[2] * RAY_DIST_EPSILON;
			refr_rp->Depth = rp->Depth+1;

			int q = (refr_rp->d.x < 0) + ((refr_rp->d.y < 0) << 1) + ((refr_rp->d.z < 0) << 2);
			TraceRay(q, nNextIdx);
			shading(nNextIdx);

			if (refr_is->tacc) {
				is->color[0] += mat_fRefr * refr_is->color[0] * mat_cTex[0];
				is->color[1] += mat_fRefr * refr_is->color[1] * mat_cTex[1];
				is->color[2] += mat_fRefr * refr_is->color[2] * mat_cTex[2];
			}
		}
	}
#endif
}

