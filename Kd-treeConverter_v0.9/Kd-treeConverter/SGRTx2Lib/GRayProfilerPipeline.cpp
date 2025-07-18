#include "GScene.h"
#include "GTexture.h"
#include "GTextureManager.h"
#include "GRenderSystem.h"

#include "GKDTreeStructure.h"

#include <stdio.h>
#include "SSE_math.h"
#include "SSERenderPipeline.h"

#include "GRayProfilerPipeline.h"

#include <fstream>

#include <gl/glew.h>

const int MAX_DEPTH = 10;
//! Empty Box (White)
//! Intersected Box (BLUE)
//! Leaf Step Box (RED)
float colors[MAX_DEPTH+3][4];

#define EMPTY_BOX -1
#define INTERSECTED_BOX -2
#define NODE_STEP_BOX -3

static const unsigned int modulo[] =  {0,1,2,0,1};
#define ku modulo[k+1]
#define kv modulo[k+2]

// 임시 레이
float ori[] = { 2.0f, 2.0f, 2.0f, 1.0f };
float dest[] = { 5.0f, 5.0f, 5.0f, 1.0f };

float RAY_COLOR[] = { 0.0f, 0.0f, 1.0f, 1.0f };

inline void RGBtoHSV( const float *RGB_Array, float *HSV_Array )
{
	float _max_ = max( max( RGB_Array[0], RGB_Array[1] ), RGB_Array[2] );
	float _min_ = min( min( RGB_Array[0], RGB_Array[1] ), RGB_Array[2] );

	// h
	if( _max_ == _min_ )
		HSV_Array[0] = 0.0f;
	else if( _max_ == RGB_Array[0] )
	{
		HSV_Array[0] = 60.0f * (RGB_Array[1]-RGB_Array[2])/(_max_-_min_) + 360.0f;
		while( HSV_Array[0] >= 360.0f )
			HSV_Array[0] -= 360.0f;
	}
	else if( _max_ == RGB_Array[1] )
		HSV_Array[0] = 60.0f * (RGB_Array[2]-RGB_Array[1])/(_max_-_min_) + 120.0f;
	else if( _max_ == RGB_Array[2] )
		HSV_Array[0] = 60.0f * (RGB_Array[0]-RGB_Array[1])/(_max_-_min_) + 240.0f;

	// s
	if( _max_ == _min_ )
		HSV_Array[1] = 0.0f;
	else
		HSV_Array[1] = 1.0f - _min_/_max_;

	// v
	HSV_Array[2] = _max_;
}

inline void HSVtoRGB( const float *HSV_Array, float *RGB_Array )
{
	unsigned int h_i = (unsigned int)(HSV_Array[0]/60.0f)%6;
	float f = HSV_Array[0]/60.0f - float((unsigned int)(HSV_Array[0]/60.0f));
	float p = HSV_Array[2] * (1.0f-HSV_Array[1]);
	float q = HSV_Array[2] * (1.0f-f*HSV_Array[1]);
	float t = HSV_Array[2] * (1.0f-(1.0f-f)*HSV_Array[1]);
	float v = HSV_Array[1];
	switch( h_i )
	{
	case 0:
		RGB_Array[0] = v;
		RGB_Array[1] = t;
		RGB_Array[2] = p;
		break;
	case 1:
		RGB_Array[0] = q;
		RGB_Array[1] = v;
		RGB_Array[2] = p;
		break;
	case 2:
		RGB_Array[0] = p;
		RGB_Array[1] = v;
		RGB_Array[2] = t;
		break;
	case 3:
		RGB_Array[0] = p;
		RGB_Array[1] = q;
		RGB_Array[2] = v;
		break;
	case 4:
		RGB_Array[0] = t;
		RGB_Array[1] = p;
		RGB_Array[2] = v;
		break;
	case 5:
		RGB_Array[0] = v;
		RGB_Array[1] = p;
		RGB_Array[2] = q;
		break;
	default:
		assert( false );
		break;
	}
}

float RED[] = { 1.0f, 0.0f, 0.0f, 1.0f };
float GREEN[] = { 0.0f, 1.0f, 0.0f, 1.0f };
float BLUE[] = { 0.0f, 0.0f, 1.0f, 1.0f };

float RGB_Begin[] = { 1.0f, 1.0f, 0.0f, 1.0f };
float RGB_End[] = { 0.0f, 1.0f, 1.0f, 1.0f };

GRayProfilerPipeline::GRayProfilerPipeline( GScene *pScene, SSESceneData *pSSESceneData )
	: SSERenderPipeline( pScene, pSSESceneData ), m_TestRayCount(0)
	 //SSESceneData* a_Data
{
	/*m_pSSESceneData = new SSESceneData(pScene);
	GError error = pScene->getKDTreeStructure()->makeSSERenderStructureInfo( m_pSSESceneData );
	ASSERT( error != errorNo );
	
	m_Stack1x1		= (_sse_1x1_kdstack*)	_aligned_malloc(nMaxTreeLevel	   *sizeof(_sse_1x1_kdstack), 16);
	m_RayPk1x1		= (_sse_1x1_raypacket*)	_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_1x1_raypacket), 16);
	m_Isect1x1		= (_sse_1x1_isect*)		_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_1x1_isect), 16);*/

	const float rep = 1.0f/float(MAX_DEPTH-1);

	//http://en.wikipedia.org/wiki/HSL_and_HSV#Conversion_from_HSV_to_RGB

	float HSV_Begin[3];
	float HSV_End[3];

	RGBtoHSV( RGB_Begin, HSV_Begin );
	RGBtoHSV( RGB_End, HSV_End );

	float HSV_Interpolated[3];
	float RGB_Interpolated[3];
	float t;

	for( unsigned int i = 0; i < MAX_DEPTH; i++ )
	{
		t = float(i)*rep;
		HSV_Interpolated[0] = HSV_Begin[0] * (1.0f-t) + HSV_End[0] * t;
		HSV_Interpolated[1] = HSV_Begin[1] * (1.0f-t) + HSV_End[1] * t;
		HSV_Interpolated[2] = HSV_Begin[2] * (1.0f-t) + HSV_End[2] * t;
		HSVtoRGB( HSV_Interpolated, RGB_Interpolated );
		/*RGB_Interpolated[0] = GREEN[0] * (1.0f-t) + RED[0] * t;
		RGB_Interpolated[1] = GREEN[1] * (1.0f-t) + RED[1] * t;
		RGB_Interpolated[2] = GREEN[2] * (1.0f-t) + RED[2] * t;*/

		colors[i][0] = RGB_Interpolated[0];
		colors[i][1] = RGB_Interpolated[1];
		colors[i][2] = RGB_Interpolated[2];
		colors[i][3] = 0.8f;

		/*colors[i][0] = 1.0f;
		colors[i][1] = 0.0f;
		colors[i][2] = 1.0f - float(i)*rep;
		colors[i][3] = 0.8f;*/
	}

	//! Empty Box (White)
	colors[MAX_DEPTH+0][0] = 1.0f;
	colors[MAX_DEPTH+0][1] = 1.0f;
	colors[MAX_DEPTH+0][2] = 1.0f;
	colors[MAX_DEPTH+0][3] = 0.8f;
	//! Intersected Box (BLUE)
	colors[MAX_DEPTH+1][0] = 0.0f;
	colors[MAX_DEPTH+1][1] = 0.0f;
	colors[MAX_DEPTH+1][2] = 1.0f;
	colors[MAX_DEPTH+1][3] = 0.8f;
	//! Leaf Step Box (RED)
	colors[MAX_DEPTH+2][0] = 1.0f;
	colors[MAX_DEPTH+2][1] = 0.0f;
	colors[MAX_DEPTH+2][2] = 0.0f;
	colors[MAX_DEPTH+2][3] = 0.8f;
}

GRayProfilerPipeline::~GRayProfilerPipeline()
{
}

void GRayProfilerPipeline::setTestRay( GRayProfilerOption *pOption )
{
	m_pOption = pOption;
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[0];
	m_TestRayCount = pOption->getTestRayCount();
	GRayTracerProfiler::SetCurrentRayProfiler( &pOption->m_TestRayProfiler );

	for( unsigned int i = 0; i < m_TestRayCount; i++ )
	{
		//rp->o = sse_fset1( ori );
		//rp->d = sse_fset1( dest );
		rp->o = sse_fset1( pOption->getTestRayOrigin().GetPointer() );
		rp->d = sse_fset1( pOption->getTestRayDirectionArray()[i].GetPointer() );
		//rp++;
	}
}

//! -1, Empty Box (White)
//! -2, Intersected Box (BLUE)
//! -3, Leaf Step Box (RED)
void DrawBB( GBoundingBox &bb, int depth = 0 )
{
	GVector max_ = bb.getMax();
	GVector min_ = bb.getMin();

	glLineWidth( 2.0f );
	glDisable( GL_LIGHTING );
	if( depth >= MAX_DEPTH )
		glColor4fv( colors[MAX_DEPTH-1] );
	else if( depth == EMPTY_BOX )
		glColor4fv( colors[MAX_DEPTH+0] );
	else if( depth == INTERSECTED_BOX )
		glColor4fv( colors[MAX_DEPTH+1] );
	else if( depth == NODE_STEP_BOX )
		glColor4fv( colors[MAX_DEPTH+2] );
	else
		glColor4fv( colors[depth] );

	glBegin( GL_LINE_LOOP );
		glVertex3f( max_.getElement(0), max_.getElement(1), max_.getElement(2) );
		glVertex3f( max_.getElement(0), min_.getElement(1), max_.getElement(2) );
		glVertex3f( max_.getElement(0), min_.getElement(1), min_.getElement(2) );
		glVertex3f( max_.getElement(0), max_.getElement(1), min_.getElement(2) );
	glEnd();

	glBegin( GL_LINE_LOOP );
		glVertex3f( min_.getElement(0), max_.getElement(1), max_.getElement(2) );
		glVertex3f( min_.getElement(0), min_.getElement(1), max_.getElement(2) );
		glVertex3f( min_.getElement(0), min_.getElement(1), min_.getElement(2) );
		glVertex3f( min_.getElement(0), max_.getElement(1), min_.getElement(2) );
	glEnd();

	glBegin( GL_LINES );
		glVertex3f( min_.getElement(0), max_.getElement(1), max_.getElement(2) );
		glVertex3f( max_.getElement(0), max_.getElement(1), max_.getElement(2) );

		glVertex3f( min_.getElement(0), min_.getElement(1), max_.getElement(2) );
		glVertex3f( max_.getElement(0), min_.getElement(1), max_.getElement(2) );

		glVertex3f( min_.getElement(0), min_.getElement(1), min_.getElement(2) );
		glVertex3f( max_.getElement(0), min_.getElement(1), min_.getElement(2) );

		glVertex3f( min_.getElement(0), max_.getElement(1), min_.getElement(2) );
		glVertex3f( max_.getElement(0), max_.getElement(1), min_.getElement(2) );
	glEnd();
	glEnable( GL_LIGHTING );
}

void GRayProfilerPipeline::TracePacket1x1( unsigned int quad, int nIdx )
{
	if( nIdx != 0 )
		return;

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
	cpu_inverse(rcpRayDir, rp->d);

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

	leaf_depth = 0;
	node_depth = 0;
	tree_depth = 0;
	empty_count = 0;
	push_count = 0;

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node); // 11 은 아니므로, 00, 01, 10 중에 하나
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;

			bool neg;
			if( ray_dir[dim << 1] == 0 )
				neg = false; // left(min) first
			else
				neg = true; // right(max) first

			float d;
				d = (node_split - rp->o.f[dim]) * rcpRayDir.f[dim];
				d_near |= (t_near <= d);
				d_far  |= (t_far_ >= d);

			COUNT_STATE( nIdx, INTERNAL_NODE, 1 );

			if( m_pOption->isInternalStepMode() && m_pOption->getCurrentStep() == node_depth )
				DrawBB( bb_stack[stackIndex], NODE_STEP_BOX );
			else if( m_pOption->isDrawInternalNodes() )
				DrawBB( bb_stack[stackIndex], tree_depth );

			node_depth++;

			// two boxes
			GVector temp;
			GBoundingBox minBox = bb_stack[stackIndex];
			temp = minBox.getMax();
			temp.GetPointer()[dim] = node_split;
			minBox.setMax( temp );

			GBoundingBox maxBox = bb_stack[stackIndex];
			temp = maxBox.getMin();
			temp.GetPointer()[dim] = node_split;
			maxBox.setMin( temp );

			node = BackSideSon;
			if (d_near == 0)
			{
				COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN_WITHOUT_PUSH );
				// traverse the back child
				if( !neg )
					bb_stack[stackIndex] = maxBox;
				else
					bb_stack[stackIndex] = minBox;
				continue;
			}
			node = FrontSideSon;
			if (d_far == 0)
			{
				COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN_WITHOUT_PUSH );
				// traverse the front child
				if( !neg )
					bb_stack[stackIndex] = minBox;
				else
					bb_stack[stackIndex] = maxBox;
				continue;
			}
				node = BackSideSon;
				if (d_near == 0)
				{
					// traverse the back child
					if( !neg )
						bb_stack[stackIndex] = maxBox;
					else
						bb_stack[stackIndex] = minBox;
					continue;
				}
				node = FrontSideSon;
				if (d_far == 0)
				{
					// traverse the front child
					if( !neg )
						bb_stack[stackIndex] = minBox;
					else
						bb_stack[stackIndex] = maxBox;
					continue;
				}

			COUNT_TREE_OPERATOR( nIdx, TRAVERSE_DOWN );
			// front 를 traverse, back 을 넣음
			// case:  near < d < far
			if( !neg )
			{
				bb_stack[stackIndex+1] = minBox; // traverse
				bb_stack[stackIndex] = maxBox; // 넣음
			}
			else
			{
				bb_stack[stackIndex+1] = maxBox; // traverse
				bb_stack[stackIndex] = minBox; // 넣음
			}

			m_Stack1x1[stackIndex].t_far_ = t_far_;
			m_Stack1x1[stackIndex].t_near = d;
			t_far_ = d;

			m_Stack1x1[stackIndex].node = BackSideSon;
			stackIndex++;

			tree_depth++;
			push_count++;
		}

		int leafTriangleCount = OBJECT_SIZE(*node);
		//accumulatedTriangleCount += leafTriangleCount;

		COUNT_STATE( nIdx, VISITED_LEAF_NODE, 1 );
		if( leafTriangleCount == 0 )
		{
			COUNT_STATE( nIdx, EMPTY_NODE, 1 );
		}
		// Isect check
		IsectPacket1x1(node, nIdx);

		if (is->dist <= t_far_ || stackIndex == 0 ) 
			break;

		if( m_pOption->isInternalStepMode() && m_pOption->getCurrentStep() == node_depth )
		{
			DrawBB( bb_stack[stackIndex], NODE_STEP_BOX );
			m_pOption->setTriangleCountInCurrentLeafNode( leafTriangleCount );
			if (is->dist <= t_far_ )
			{
				break;
			}
		}
		else if (is->dist <= t_far_ ) // Termination test
		{
			if( m_pOption->isDrawIntersectedNode() )
				DrawBB( bb_stack[stackIndex], INTERSECTED_BOX );
			break;
		}
		else if( m_pOption->isLeafStepMode() && m_pOption->getCurrentStep() == leaf_depth )
		{
			DrawBB( bb_stack[stackIndex], NODE_STEP_BOX );
			m_pOption->setTriangleCountInCurrentLeafNode( leafTriangleCount );
			if (is->dist <= t_far_ )
			{
				break;
			}
		}
		else if( m_pOption->isDrawLeafNodes() )
		{
			if( leafTriangleCount == 0 )
				DrawBB( bb_stack[stackIndex], EMPTY_BOX );
			else
				DrawBB( bb_stack[stackIndex], leaf_depth );
		}

		if( stackIndex == 0)
		{
			break;
		}

		if( tree_depth != 0 )
			tree_depth--;

		COUNT_TREE_OPERATOR( nIdx, TRAVERSE_UP );
		node_depth++;
		leaf_depth++;

		// Stack pop
		--stackIndex;
		node		= m_Stack1x1[stackIndex].node;
		t_near		= m_Stack1x1[stackIndex].t_near;
		t_far_		= m_Stack1x1[stackIndex].t_far_;
	}


	#if PROFILER_ON == ON
	m_pOption->setEmptyNodesCount( GRayTracerProfiler::GetCurrentRayProfiler()->GetStateCount( 0, EMPTY_NODE ) );
	m_pOption->setPushCount( GRayTracerProfiler::GetCurrentRayProfiler()->GetTreeOperatorCount( 0, TRAVERSE_DOWN ) );
	m_pOption->setPopCount( GRayTracerProfiler::GetCurrentRayProfiler()->GetTreeOperatorCount( 0, TRAVERSE_UP ) );
	if( m_pOption->isLeafStepMode() )
		m_pOption->setMaximumStep( leaf_depth );
	else
		m_pOption->setMaximumStep( node_depth );
	m_pOption->setTriangleCount( GRayTracerProfiler::GetCurrentRayProfiler()->GetStateCount( 0, TRIANGLE_COUNT ) );
	#endif
	if( m_pOption->isDrawTestRays() && !m_pOption->isSetRayToCamera() )
	{
		float dist = 100000.0f;
		if (is->dist <= t_far_ )
		//if (is->dist < dist )
			dist = is->dist;
		_sse_1x1_raypacket tpos;
		//tpos = sse_fset1( rp->d.f );
		tpos.d = rp->d;
		tpos.d = sse_fmul( tpos.d, dist );
		tpos.d = sse_fadd( rp->o, tpos.d );

		glDisable( GL_LIGHTING );
		glColor3fv( RAY_COLOR );
		glLineWidth( 2.0f );
		glBegin( GL_LINES );
			glVertex3fv( rp->o.f );
			glVertex3fv( tpos.d.f );
		glEnd();
		glEnable( GL_LIGHTING );
	}
}

void GRayProfilerPipeline::Render1x1( int nThreadID )
{
	// start spawning rays
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[0];
	_sse_1x1_isect		*is	= &m_Isect1x1[0];

	if( GRayTracerProfiler::IsProfilerOn() )
	{
		int xTileEnd = m_Resolution.x;
		int yTileEnd = m_Resolution.y;
		int tx, ty;

		// start spawning rays
		_sse_1x1_raypacket	tpos;		// target position for ray casting (pixel center)

		rp->o   = sse_fset1(m_Origin.m_Vector);

		const int yTileSize = int(m_Height * m_fThreadRcpCount);

		ty = 0;
		yTileEnd;

		for (; ty < yTileEnd; ty++) {
			for (tx = 0; tx < xTileEnd;  tx++) {

				// Set tpos
				vector3 r = m_LeftUp + (m_DX * (float)tx) - (m_DY * (float)ty);
				tpos.d.x = r.x;
				tpos.d.y = r.y;
				tpos.d.z = r.z;
				is->addr = tx + (m_Height - 1 - ty) * m_Width;

				// Ray 를 셋팅 - 시작점(rp->o) ~ 끝점(jpos)
				rp->d = sse_fsub(tpos.d, rp->o);
				rp->Depth = 0;
				m_RayID = 1;
				InitPacket1x1( 0 );	// direction vector normalize 등

				// ray dir 결정 (q = 8방향중하나)
				int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
				RenderPacket1x1(q, 0);
				//GRayTracerProfiler::AddData( GRayTracerProfiler::GetCurrentRayProfiler() );
			}
		}
		//GRayTracerProfiler::PrintData( "RayTracerProfile.dat" );
		GRayTracerProfiler::SetProfilerOn( false );
	}
	else
	{
		if( m_pOption->isDrawTestRays() && m_pOption->isSetRayToCamera() )
		{
			// 정면에 구 그리기
			_sse_1x1_raypacket tpos;
			//tpos = sse_fset1( rp->d.f );
			tpos.d = rp->d;
			tpos.d = sse_fmul( tpos.d, 5.0f );
			tpos.d = sse_fadd( rp->o, tpos.d );
			glDisable( GL_LIGHTING );
			glPushMatrix();
			glTranslatef( tpos.d.f[0], tpos.d.f[1], tpos.d.f[2] );
			GLUquadricObj* pObj = gluNewQuadric();
			gluQuadricNormals( pObj, GLU_SMOOTH );
			glColor3f( 1.0f, 0.0f, 0.0f );
			gluSphere( pObj, 0.01f, 20, 20 );
			gluDeleteQuadric( pObj );
			glPopMatrix();
			glEnable( GL_LIGHTING );
		}

		//GRayProfiler *pTestRayProfiler = &m_pOption->m_TestRayProfiler;
		if( m_pOption->isDrawKdTree() )
		{
			for( unsigned int i = 0; i < m_TestRayCount; i++ )
			{
				//pTestRayProfiler->ResetCounts();
				is->addr = 0;
				rp->Depth = 0;
				m_RayID = 1;
				InitPacket1x1( 0 );	// direction vector normalize 등

				// ray dir 결정 (q = 8방향중하나)
				int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);
				RenderPacket1x1(q, 0);
			}
		}
	}
}

// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void GRayProfilerPipeline::IsectPacket1x1( const KdTreeNode *node, int nIdx )
{
	GRayProfiler *pTestRayProfiler = &m_pOption->m_TestRayProfiler;

	int i;

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

		COUNT_STATE( nIdx, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// Backface Culling : 투명하지 않는 물체만 해당
		// ---------------------------------------------------------------
		if (!acc.isTransparent && m_Scene->isBackFaceCulling()) {
			if (vector3(rp->d.f).innerProduct(acc.N) < 0) {
				continue;
			}
		}

		const unsigned int k	= acc.k;

		float nd, f;
		nd = 1.0f / (rp->d.f[k]
			+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
		f  = acc.n_d - (rp->o.f[k]
			+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
		f = f * nd;

		if (!(is->dist >= f && f > EPSILON))
		{
			continue;	// eps < f <= Hit4.dist
		}

		float hu, hv;
		float lambda, mue;
		hu = rp->o.f[ku] + f * rp->d.f[ku];
		hv = rp->o.f[kv] + f * rp->d.f[kv];

		lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
		if (lambda < 0.0f)
		{
			continue;
		}

		mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;
		if (mue    < 0.0f)
		{
			continue;
		}

		if (lambda+mue > 1.0f)
		{
			continue;
		}

		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void GRayProfilerPipeline::shading1x1 (int nIdx) {

	GRayProfiler *pTestRayProfiler = &m_pOption->m_TestRayProfiler;

	return;
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	_sse_float	hit_p = sse_fadd(rp->o, sse_fmul(rp->d, is->dist));

	bool bIsEnableShadow       = m_Scene->isEnableShadow();
	bool bIsRunShadowChk       = false;
	bool bIsEnableLocalShading = m_Scene->isEnableLocalShading();
	bool bIsUseTexture         = (m_Scene->isUseTexture() & bIsEnableLocalShading);

	bool b_refl = false;
	bool b_refr = false;

	GColor		global_ambient = m_Scene->getGlobalAmbient();

	GColor		mat_ambt, mat_diff, mat_spec, mat_emit;
	float		mat_rough;
	float		mat_refl, mat_refr, mat_rIdx;
	GColor		mat_tex;
	UINT		obj_num;

	GVector N, R, L;
	GVector rayO = GVector(rp->o.f);
	GVector rayD = GVector(rp->d.f);
	GPoint  hitP = GPoint(hit_p.f);

	if (is->tacc == 0) {
		is->color = GColor(0,0,0);		// Background color
		return;
	}

	const int triID      = is->tacc -1;
	assert( triID >= 0 );
	GObject   *pObject   = m_Data->m_TriObjList[triID]->m_pObject;
	GMaterial *pMaterial = pObject->getMaterial();
	is->ads.pri_oid = pObject->getObjectNumber();

	mat_ambt  = pMaterial->getAmbient();
	mat_diff  = pMaterial->getDiffuse();
	mat_spec  = pMaterial->getSpecular();
	mat_emit  = pMaterial->getEmission();
	mat_rough = pMaterial->getRoughness();
	mat_refl  = pMaterial->getReflection();		if (mat_refl > 0) b_refl = true;
	mat_refr  = pMaterial->getTransparency();	if (mat_refr > 0) b_refr = true;
	mat_rIdx  = pMaterial->getRefractionIndex();
	obj_num   = pObject->getObjectNumber();

	N = m_Data->m_TriObjList[triID]->calBarycentricNormal(1-is->u-is->v, is->u, is->v);
	is->n.x = N.x;
	is->n.y = N.y;
	is->n.z = N.z;

	// Get object color
	GColor texColor;
	if ( bIsUseTexture ) {
		GTexture  *pTexture  = GTextureManager::getInstance()->getTexture( pObject->getTextureID() );
		if (pTexture && pTexture->isLoaded()) {
			is->ads.pri_texture = 1;
			GPoint point = m_Data->m_TriObjList[triID]->calBarycentricUV(1-is->u-is->v, is->u, is->v);
			float u = point.x;
			float v = point.y;
			texColor = pTexture->getTexel( u, v );
		} else {
			is->ads.pri_texture = 0;
			texColor = mat_diff;
		}
	} else {
		texColor = mat_diff;
	}
	mat_tex = texColor;

	GColor oColor;
	int shadowcount = 0;

	if ( bIsEnableLocalShading ) {
		_sse_1x1_raypacket	*shadow_rp;
		_sse_1x1_isect		*shadow_is;
		if ( bIsEnableShadow ) {
			shadow_rp	= &m_ShadowRayPk1x1[0];
			shadow_is	= &m_ShadowIsect1x1[0];
			shadow_rp->o = hit_p;
		}
		else

		// Shading 에서, 투명한 물체일때, Normal 과 dir 의 dot 이 < 0 이라면 normal 을 뒤짚는다.
		// 확인 필요!!
		if (mat_refr > 0.0f && sse_fdot(is->n, rp->d) > 0.0f) 
		{
			// neagtor
			N = -N;
		}
		R = GVector(-2 * N.innerProduct(rayD) * N + rayD).normalize();

		// Background color
		oColor = GColor(0,0,0);

		if (is->tacc) {
			// Ambient color
			//oColor = global_ambient * mat_ambt;

			// Emission color
			//oColor = oColor + mat_emit;

			// Diffuse & Specular color
			const vector<GLight*>* pLightList = m_Scene->getLightList();
			for ( int lx = 0; lx < (int) pLightList->size(); ++lx )
			{
				GLight* pLight = (*pLightList)[ lx ];

				if( !pLight->isEnabled() )
					continue;
				// Point Light 만 일단 지원
				if ( pLight->getLightType() != typePointLight )
				{
					continue;
				}

				GColor   lightColor = pLight->getLightColor();
				GPoint   lightPos   = pLight->getPosition();

				// 광원 자기자신인 경우
				if (obj_num == pLight->getObjectNumber()) {
					//oColor = oColor + lightColor * pLight->getIntensity();
					continue;
				}

				// 그림자 확인
				if (m_Scene->isEnableShadow()) {
					checkVisibility1x1(&hitP, &lightPos);

					float lDist = GVector(lightPos - hitP).length();

					// Phong shading
					L = GVector(lightPos - hitP).normalize();

					// shadow 관련 visible 조건
					//		중간에 shadow ray 와 교점이 없거나
					//		shadow ray 가 교차점이 뒤에 존재하거나 아니면 거리가 거의 가깝거나
					if (shadow_is->tacc == 0 || fabsf(lDist - shadow_is->dist) < 1.f*EPSILON || shadow_is->dist > lDist) {
						//oColor += mat_tex  * lightColor * max( 0.0f, L.innerProduct(N) ) +
						//	mat_spec * lightColor * pow( max( 0.0f, R.innerProduct(L) ), mat_rough);
					} else {
						shadowcount++;
					}
				} else {
					// Phong shading
					////COUNT_INSTRUCTION( MOVEMENT, 4 );
					L = GVector(lightPos - hitP).normalize();

					//oColor += mat_tex  * lightColor * max( 0.0f, L.innerProduct(N) ) +
					//	mat_spec * lightColor * pow( max( 0.0f, R.innerProduct(L) ), mat_rough);
				}
			}
		}
	} else {
		//oColor = mat_tex;
	}

	if ( rp->Depth < (m_Scene->getMaxReflectionDepth())) {
		//oColor = oColor * (1.0f - mat_refl - mat_refr);
	}

	if ( rp->Depth < 2 ) {
		is->ads.pri_shadow = shadowcount;
	}

	is->color = oColor;

	if (rp->Depth < (m_Scene->getMaxReflectionDepth())) {
		// ---------------------------------------------------------------------------
		// reflection
		// ---------------------------------------------------------------------------
		if (b_refl) {
			float dot_i;

			_sse_1x1_raypacket	*refl_rp	= &m_RayPk1x1[nIdx+1];
			_sse_1x1_isect		*refl_is	= &m_Isect1x1[nIdx+1];

			dot_i = sse_fdot(rp->d, is->n);

			refl_rp->d = sse_fsub(rp->d, sse_fmul(2, sse_fmul(dot_i, is->n)));
			refl_rp->o = sse_fadd(hit_p, sse_fmul(refl_rp->d, EPSILON));
			refl_rp->Depth = rp->Depth+1;

			InitPacket1x1( nIdx+1 );

			int q = (refl_rp->d.x < 0) + ((refl_rp->d.y < 0) << 1) + ((refl_rp->d.z < 0) << 2);
			RenderPacket1x1(q, nIdx+1);
			if (refl_is->tacc) {
				//is->color += mat_refl * refl_is->color * mat_tex;
			}

			if (rp->Depth == 0) {
				is->ads.sec_oid     = refl_is->ads.pri_oid;
				is->ads.sec_shadow  = refl_is->ads.pri_shadow;
				is->ads.sec_texture = refl_is->ads.pri_texture;
				is->ads.n2          = refl_is->n;
			}
		}

		// ---------------------------------------------------------------------------
		// refraction
		// ---------------------------------------------------------------------------
		if (b_refr) {
			float dot_i, dot_r;
			float n_div_nt;

			_sse_1x1_raypacket	*refr_rp	= &m_RayPk1x1[nIdx+1];
			_sse_1x1_isect		*refr_is	= &m_Isect1x1[nIdx+1];

			dot_i = sse_fdot(rp->d, is->n);
		
			if (dot_i < 0) {
				n_div_nt = AIR_INDEX / mat_rIdx;
			} else {
				n_div_nt = mat_rIdx / AIR_INDEX;
			}

			dot_r = sqrtf(fabsf(1.0f - n_div_nt * n_div_nt * (1 - dot_i * dot_i)));

			if(n_div_nt < 1.0) {
				refr_rp->d = sse_fsub(sse_fmul(n_div_nt, sse_fsub(rp->d, sse_fmul(is->n, dot_i))), sse_fmul(is->n, dot_r));
			} else {
				refr_rp->d = sse_fadd(sse_fmul(n_div_nt, sse_fsub(rp->d, sse_fmul(is->n, dot_i))), sse_fmul(is->n, dot_r));
			}
			refr_rp->o = sse_fadd(hit_p, sse_fmul(refr_rp->d, EPSILON));
			refr_rp->Depth = rp->Depth+1;

			InitPacket1x1( nIdx+1 );

			int q = (refr_rp->d.x < 0) + ((refr_rp->d.y < 0) << 1) + ((refr_rp->d.z < 0) << 2);
			RenderPacket1x1(q, nIdx+1);

			if (refr_is->tacc) {
				is->color += mat_refr * refr_is->color * mat_tex;
			}

			if (rp->Depth == 0) {
				is->ads.sec_oid     = refr_is->ads.pri_oid;
				is->ads.sec_shadow  = refr_is->ads.pri_shadow;
				is->ads.sec_texture = refr_is->ads.pri_texture;
				is->ads.n2          = refr_is->n;
			}
		}
	}
}

// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::IsectPacket
//		Intersection check
// ------------------------------------------------------------------------------------------------
void GRayProfilerPipeline::IsectShadowPacket1x1( const KdTreeNode *node )
{
	GRayProfiler *pTestRayProfiler = &m_pOption->m_TestRayProfiler;
	int i;

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
		if (acc.mbox == rp->RayId)
		{
			//COUNT_INSTRUCTION( BRANCH, 1 );
			continue;
		}
		else
		{
			//COUNT_INSTRUCTION( BRANCH, 1 );
			acc.mbox = rp->RayId;
		}	// 무조건 isect 안되도 실행 되어야 함

		COUNT_STATE( -1, TRIANGLE_COUNT, 1 );

		// ---------------------------------------------------------------
		// 투명한 물체는 투과
		// ---------------------------------------------------------------
		if (acc.isTransparent)
		{
			//COUNT_INSTRUCTION( BRANCH, 1 );
			continue;
		}

		const unsigned int k	= acc.k;

		float nd, f;
		nd = 1.0f / (rp->d.f[k]
		+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
		f  = acc.n_d - (rp->o.f[k]
		+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
		f = f * nd;

		if (!(is->dist >= f && f > EPSILON))
		{
			//COUNT_INSTRUCTION( BRANCH, 1 );
			continue;	// eps < f <= Hit4.dist
		}

		float hu, hv;
		float lambda, mue;
		hu = rp->o.f[ku] + f * rp->d.f[ku];
		hv = rp->o.f[kv] + f * rp->d.f[kv];

		lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
		mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;

		if (lambda < 0.0f)
		{
			//COUNT_INSTRUCTION( BRANCH, 1 );
			continue;
		}
		if (mue    < 0.0f)
		{
			//COUNT_INSTRUCTION( BRANCH, 1 );
			continue;
		}
		if (lambda+mue > 1.0f){
			//COUNT_INSTRUCTION( BRANCH, 1 );
			continue;
		}

		//COUNT_INSTRUCTION( ADD, 1 );
		is->u = lambda;
		is->v = mue;
		is->dist = f;
		is->tacc = triID+1;
	}
}

 //-----------------------------------------------------------
 //SSERenderPipeline::TracePacket
	//	Sends a 4x4 packet through the tree
 //-----------------------------------------------------------
void GRayProfilerPipeline::TraceShadowPacket1x1( unsigned int quad )
{
	GRayProfiler *pTestRayProfiler = &m_pOption->m_TestRayProfiler;

	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// near / far
	float t_near, t_far_;
	t_near = 0;
	t_far_ = 100000;

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
	//COUNT_INSTRUCTION( SSE_INVERSE, 1 );
	rcpRayDir.v4 = sse_inverse(rp->d.v4);
	unsigned int empty_count = 0;

	// ray id
	rp->RayId	= m_RayID;	m_RayID += 1;

	// ---------------------------------------------------------------------------
	// Traversal
	// ---------------------------------------------------------------------------
	while (1) {
		while (IS_LEAF(*node) == 0) {
			//COUNT_INSTRUCTION( COMPARISON, 1 );
			const float node_split = SPLIT_POS(*node);
			const unsigned int dim = SPLIT_AXIS(*node);
			KdTreeNode *FrontSideSon	= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[dim << 1])];
			KdTreeNode *BackSideSon		= &m_Data->m_pKDTreeNodes[(FIRST_CHILD_OFFSET(*node) + ray_dir[(dim << 1)+1])];

			unsigned int d_near = 0, d_far = 0;
			float d;

			COUNT_STATE( -1, INTERNAL_NODE, 1 );
			pTestRayProfiler->CountState( -1, INTERNAL_NODE );

			d = (node_split - rp->o.f[dim]) * rcpRayDir.f[dim];
			d_near |= (t_near <= d);
			d_far  |= (t_far_ >= d);

			node = BackSideSon;
			if (d_near == 0)
			{
				COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
				pTestRayProfiler->CountTreeOperator( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
				continue;	// traverse the back  child
			}
			node = FrontSideSon;
			if (d_far == 0)
			{
				COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
				pTestRayProfiler->CountTreeOperator( -1, TRAVERSE_DOWN_WITHOUT_PUSH );
				continue;	// traverse the front child
			}

			COUNT_TREE_OPERATOR( -1, TRAVERSE_DOWN );
			pTestRayProfiler->CountTreeOperator( -1, TRAVERSE_DOWN );
			// case:  near < d < far
			m_Stack1x1[stackIndex].t_far_ = t_far_;
			m_Stack1x1[stackIndex].t_near = d;
			t_far_ = d;

			m_Stack1x1[stackIndex].node = BackSideSon;
			stackIndex++;
		}
		//COUNT_INSTRUCTION( COMPARISON, 1 );

		COUNT_STATE( -1, VISITED_LEAF_NODE, 1 );
		pTestRayProfiler->CountState( -1, VISITED_LEAF_NODE );
		// Isect check
		IsectShadowPacket1x1(node);

		if( OBJECT_SIZE(*node) == 0 )
			pTestRayProfiler->CountState( -1, EMPTY_NODE );
		else
			pTestRayProfiler->CountState( -1, TRIANGLE_COUNT, OBJECT_SIZE(*node) );

		// Termination test
		if (is->dist <= t_far_ || stackIndex == 0) break;

		COUNT_TREE_OPERATOR( -1, TRAVERSE_UP );
		pTestRayProfiler->CountTreeOperator( -1, TRAVERSE_UP );
		// Stack pop
		--stackIndex;
		node		= m_Stack1x1[stackIndex].node;
		t_near		= m_Stack1x1[stackIndex].t_near;
		t_far_		= m_Stack1x1[stackIndex].t_far_;
	}
}