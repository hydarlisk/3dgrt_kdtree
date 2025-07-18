// -----------------------------------------------------------
// raytracer.cpp
// 2008 - oipini
// -----------------------------------------------------------
#include "GScene.h"
#include "GTexture.h"
#include "GTextureManager.h"
#include "GRenderSystem.h"
#include "GThreadingOption.h"

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
// SSERenderPipeline::SSERenderPipeline
//		Allocate memory & Initialize values
// ------------------------------------------------------------------------------------------------
SSERenderPipeline::SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data) : m_Scene(a_Scene), m_Data(a_Data)
{
	initialize();

	m_RayQ4x4 = NULL;
	m_RayT2x2 = NULL;
	m_RayT1x1 = NULL;

	m_ADPSS_data = NULL;
	m_ADPSS_AdjPixAddr4 = (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);			// m_ADPSS_AdjPixAddr4[4]
	m_Sobel_AdjPixAddr4 = (__m128i*) _aligned_malloc(2 * sizeof(__m128i), 16);			// m_Sobel_AdjPixAddr4[2]
	m_ADPSS_AdjPixAddr[0] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[1] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[2] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[3] = (int*) malloc(4 * sizeof(int));
}

SSERenderPipeline::SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data, Detect_ADPSS_measure* a_ADPSS_data,
									 SSERayQueue4x4* a_RayQ4x4, SSERayTable2x2* a_RayT2x2) : m_Scene(a_Scene), m_Data(a_Data)
{
	initialize();

	m_RayQ4x4 = a_RayQ4x4;
	m_RayT2x2 = a_RayT2x2;
	m_RayT1x1 = NULL;

	m_ADPSS_data = a_ADPSS_data;
	m_ADPSS_AdjPixAddr4 = (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);			// m_ADPSS_AdjPixAddr4[4]
	m_Sobel_AdjPixAddr4 = (__m128i*) _aligned_malloc(2 * sizeof(__m128i), 16);			// m_Sobel_AdjPixAddr4[2]
	m_ADPSS_AdjPixAddr[0] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[1] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[2] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[3] = (int*) malloc(4 * sizeof(int));
}

SSERenderPipeline::SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data, Detect_ADPSS_measure* a_ADPSS_data,
									 SSERayQueue4x4* a_RayQ4x4, SSERayTable2x2* a_RayT2x2, SSERayTable1x1* a_RayT1x1) : m_Scene(a_Scene), m_Data(a_Data)
{
	initialize();

	m_RayQ4x4 = a_RayQ4x4;
	m_RayT2x2 = a_RayT2x2;
	m_RayT1x1 = a_RayT1x1;

	m_ADPSS_data = a_ADPSS_data;
	m_ADPSS_AdjPixAddr4 = (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);			// m_ADPSS_AdjPixAddr4[4]
	m_Sobel_AdjPixAddr4 = (__m128i*) _aligned_malloc(2 * sizeof(__m128i), 16);			// m_Sobel_AdjPixAddr4[2]
	m_ADPSS_AdjPixAddr[0] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[1] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[2] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[3] = (int*) malloc(4 * sizeof(int));
}

SSERenderPipeline::SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data, Detect_ADPSS_measure* a_ADPSS_data,
									 SSERayTable1x1* a_RayT1x1) : m_Scene(a_Scene), m_Data(a_Data)
{
	initialize();

	m_RayQ4x4 = NULL;
	m_RayT2x2 = NULL;
	m_RayT1x1 = a_RayT1x1;

	m_ADPSS_data = a_ADPSS_data;
	m_ADPSS_AdjPixAddr4 = (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);			// m_ADPSS_AdjPixAddr4[4]
	m_Sobel_AdjPixAddr4 = (__m128i*) _aligned_malloc(2 * sizeof(__m128i), 16);			// m_Sobel_AdjPixAddr4[2]
	m_ADPSS_AdjPixAddr[0] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[1] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[2] = (int*) malloc(4 * sizeof(int));
	m_ADPSS_AdjPixAddr[3] = (int*) malloc(4 * sizeof(int));
}

SSERenderPipeline::~SSERenderPipeline()
{
	if(m_ADPSS_AdjPixAddr4) _aligned_free(m_ADPSS_AdjPixAddr4);
	if(m_Sobel_AdjPixAddr4) _aligned_free(m_Sobel_AdjPixAddr4);
	if(m_ADPSS_AdjPixAddr[0]) free(m_ADPSS_AdjPixAddr[0]);
	if(m_ADPSS_AdjPixAddr[1]) free(m_ADPSS_AdjPixAddr[1]);
	if(m_ADPSS_AdjPixAddr[2]) free(m_ADPSS_AdjPixAddr[2]);
	if(m_ADPSS_AdjPixAddr[3]) free(m_ADPSS_AdjPixAddr[3]);

	_aligned_free(m_Stack4x4);
	_aligned_free(m_RayPk4x4);
	_aligned_free(m_Isect4x4);
	_aligned_free(m_RMask4x4);
	_aligned_free(m_ShadowRayPk4x4);
	_aligned_free(m_ShadowIsect4x4);
	_aligned_free(m_ShadowRMask4x4);

	_aligned_free(m_FrustomRayPk2x2);

	_aligned_free(m_Stack2x2);
	_aligned_free(m_RayPk2x2);
	_aligned_free(m_Isect2x2);
	_aligned_free(m_RMask2x2);
	_aligned_free(m_ShadowRayPk2x2);
	_aligned_free(m_ShadowIsect2x2);
	_aligned_free(m_ShadowRMask2x2);

	_aligned_free(m_Stack1x1);
	_aligned_free(m_RayPk1x1);
	_aligned_free(m_Isect1x1);
	_aligned_free(m_ShadowRayPk1x1);
	_aligned_free(m_ShadowIsect1x1);

	_aligned_free(m_LeftUp4);

	_aligned_free(m_DX4);
	_aligned_free(m_DY4);

	_aligned_free(m_Tmp1_Addr4);
	_aligned_free(m_Tmp2_Addr4);

	_aligned_free(m_CastSeq2x2_x4);
	_aligned_free(m_CastSeq2x2_y4);
	_aligned_free(m_iCastSeq2x2_x4);
	_aligned_free(m_iCastSeq2x2_y4);
	_aligned_free(m_CastSeq4x4_x4);
	_aligned_free(m_CastSeq4x4_y4);
	_aligned_free(m_iCastSeq4x4_x4);
	_aligned_free(m_iCastSeq4x4_y4);
}


GError SSERenderPipeline::initialize( void )
{
	int nPrimtiveNum	= m_Data->m_TriObjCnt;
	int nMaxTreeLevel	= max(80, _Round2Int_(8 + 1.3f * _Log2Int_(float(nPrimtiveNum))));
	int nMaxTraceDepth	= max(MAX_TRACE_DEPTH, m_Scene->getMaxReflectionDepth());

	// Packet Size 1x1 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	m_Stack1x1		= (_sse_1x1_kdstack*)	_aligned_malloc(nMaxTreeLevel	   *sizeof(_sse_1x1_kdstack), 16);
	m_RayPk1x1		= (_sse_1x1_raypacket*)	_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_1x1_raypacket), 16);
	m_Isect1x1		= (_sse_1x1_isect*)		_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_1x1_isect), 16);
	m_ShadowRayPk1x1	= (_sse_1x1_raypacket*)	_aligned_malloc(1	* sizeof(_sse_1x1_raypacket), 16);
	m_ShadowIsect1x1	= (_sse_1x1_isect*)		_aligned_malloc(1	* sizeof(_sse_1x1_isect), 16);

	// Packet Size 2x2 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	m_RayPk2x2			= (_sse_2x2_raypacket*)	_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_2x2_raypacket), 128);
	m_Isect2x2			= (_sse_2x2_isect*)		_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_2x2_isect), 128);
	m_RMask2x2			= (_sse_2x2_raymask*)	_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_2x2_raymask), 16);
	m_Stack2x2			= (_sse_2x2_kdstack*)	_aligned_malloc(nMaxTreeLevel	   *sizeof(_sse_2x2_kdstack), 64);
	m_ShadowRayPk2x2	= (_sse_2x2_raypacket*)	_aligned_malloc(1	* sizeof(_sse_2x2_raypacket), 128);
	m_ShadowIsect2x2	= (_sse_2x2_isect*)		_aligned_malloc(1	* sizeof(_sse_2x2_isect), 128);
	m_ShadowRMask2x2	= (_sse_2x2_raymask*)	_aligned_malloc(1	* sizeof(_sse_2x2_raymask), 16);

	// Packet Size 4x4 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	m_Stack4x4		= (_sse_4x4_kdstack*)	_aligned_malloc(nMaxTreeLevel	   *sizeof(_sse_4x4_kdstack), 16);
	m_RayPk4x4		= (_sse_4x4_raypacket*)	_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_4x4_raypacket), 16);
	m_Isect4x4		= (_sse_4x4_isect*)		_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_4x4_isect), 16);
	m_RMask4x4		= (_sse_4x4_raymask*)	_aligned_malloc((nMaxTraceDepth+1) *sizeof(_sse_4x4_raymask), 16);
	m_ShadowRayPk4x4	= (_sse_4x4_raypacket*)	_aligned_malloc(1	* sizeof(_sse_4x4_raypacket), 16);
	m_ShadowIsect4x4	= (_sse_4x4_isect*)		_aligned_malloc(1	* sizeof(_sse_4x4_isect), 16);
	m_ShadowRMask4x4	= (_sse_4x4_raymask*)	_aligned_malloc(1	* sizeof(_sse_4x4_raymask), 16);

	m_FrustomRayPk2x2	= (_sse_2x2_rayfrustum*)	_aligned_malloc(1   * sizeof(_sse_2x2_rayfrustum), 16);

	// for Sampling =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

	m_LeftUp4 = (_sse_vec*) _aligned_malloc(1 * sizeof(_sse_vec), 16);

	m_DX4 = (_sse_vec*) _aligned_malloc(1 * sizeof(_sse_vec), 16);
	m_DY4 = (_sse_vec*) _aligned_malloc(1 * sizeof(_sse_vec), 16);

	m_Tmp1_Addr4 = (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);			// m_Tmp1_Addr4[4]
	m_Tmp2_Addr4 = (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);			// m_Tmp2_Addr4[4]

	m_CastSeq4x4_x4		= (__m128*) _aligned_malloc(4 * sizeof(__m128), 16);		// m_CastSeq4x4_x4[4]
	m_CastSeq4x4_y4		= (__m128*) _aligned_malloc(4 * sizeof(__m128), 16);		// m_CastSeq4x4_y4[4]
	m_iCastSeq4x4_x4	= (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);		// m_iCastSeq4x4_x4[4]
	m_iCastSeq4x4_y4	= (__m128i*) _aligned_malloc(4 * sizeof(__m128i), 16);		// m_iCastSeq4x4_y4[4]
	m_CastSeq4x4_x4[0] = _mm_setr_ps(0, 1, 0, 1);	m_iCastSeq4x4_x4[0] = _mm_setr_epi32(0, 1, 0, 1);
	m_CastSeq4x4_x4[1] = _mm_setr_ps(2, 3, 2, 3);	m_iCastSeq4x4_x4[1] = _mm_setr_epi32(2, 3, 2, 3);
	m_CastSeq4x4_x4[2] = _mm_setr_ps(0, 1, 0, 1);	m_iCastSeq4x4_x4[2] = _mm_setr_epi32(0, 1, 0, 1);
	m_CastSeq4x4_x4[3] = _mm_setr_ps(2, 3, 2, 3);	m_iCastSeq4x4_x4[3] = _mm_setr_epi32(2, 3, 2, 3);
	m_CastSeq4x4_y4[0] = _mm_setr_ps(0, 0, 1, 1);	m_iCastSeq4x4_y4[0] = _mm_setr_epi32(0, 0, 1, 1);
	m_CastSeq4x4_y4[1] = _mm_setr_ps(0, 0, 1, 1);	m_iCastSeq4x4_y4[1] = _mm_setr_epi32(0, 0, 1, 1);
	m_CastSeq4x4_y4[2] = _mm_setr_ps(2, 2, 3, 3);	m_iCastSeq4x4_y4[2] = _mm_setr_epi32(2, 2, 3, 3);
	m_CastSeq4x4_y4[3] = _mm_setr_ps(2, 2, 3, 3);	m_iCastSeq4x4_y4[3] = _mm_setr_epi32(2, 2, 3, 3);

	m_CastSeq2x2_x4	= (__m128*) _aligned_malloc(1 * sizeof(__m128), 16);		// m_CastSeq4x4_x4[4]
	m_CastSeq2x2_y4	= (__m128*) _aligned_malloc(1 * sizeof(__m128), 16);		// m_CastSeq4x4_y4[4]
	m_iCastSeq2x2_x4= (__m128i*) _aligned_malloc(1 * sizeof(__m128i), 16);		// m_iCastSeq4x4_x4[4]
	m_iCastSeq2x2_y4= (__m128i*) _aligned_malloc(1 * sizeof(__m128i), 16);		// m_iCastSeq4x4_y4[4]
	m_CastSeq2x2_x4[0] = _mm_setr_ps(0, 1, 0, 1);	m_iCastSeq2x2_x4[0] = _mm_setr_epi32(0, 1, 0, 1);
	m_CastSeq2x2_y4[0] = _mm_setr_ps(0, 0, 1, 1);	m_iCastSeq2x2_y4[0] = _mm_setr_epi32(0, 0, 1, 1);

	// ray packet id (mailbox)
	m_RayID = 1;

	// ray direction
	int i;
	for ( i = 0; i < 8; i++ ) {
		const unsigned int rdx = i & 1;
		const unsigned int rdy = (i >> 1) & 1;
		const unsigned int rdz = (i >> 2) & 1;
		raydir[i][0][0] = rdx, raydir[i][0][1] = rdx ^ 1;
		raydir[i][1][0] = rdy, raydir[i][1][1] = rdy ^ 1;
		raydir[i][2][0] = rdz, raydir[i][2][1] = rdz ^ 1;
	}
	return errorNo;
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------------------
// SSERenderPipeline::SetTarget
//		Sets the render target canvas
// ------------------------------------------------------------------------------------------------
void SSERenderPipeline::PrepareRender( int n_iThreadCount )
{
	m_iThreadCount		= n_iThreadCount;
	m_fThreadRcpCount	= 1.0f / n_iThreadCount;
	m_iThreadJobSize	= n_iThreadCount * THREADING_JITTER_SIZE;
	m_fThreadRcpJobSize	= 1.0f / m_iThreadJobSize;

	m_SuperSampling = m_Scene->getSuperSampling();

	m_Resolution  = m_Scene->getResolution();
	m_Width  = m_Resolution.x;
	m_Height = m_Resolution.y;

	int nBufferW = m_Scene->getImageBuffer()->getWidth();
	int nBufferH = m_Scene->getImageBuffer()->getHeight();
	m_Dest = m_Scene->getImageBuffer()->getBuffer();
	memset((void*)m_Dest, 0, nBufferW * nBufferH * 3 * 4);		// Width x Height x RGB x 4byte(float)

	GCamera* pCamera = m_Scene->getRenderCamera();
	float aspect = (float) m_Resolution.x / (float) m_Resolution.y;
	float cameraPlaneHeight = 2.0f * pCamera->getNear() * tanf( ( pCamera->getFovy() * 0.5f ) * G_TO_RADIAN );
	float cameraPlaneWidth = aspect * cameraPlaneHeight;

	m_DX =  pCamera->getUVec() * (cameraPlaneWidth  / (float)m_Resolution.x);
	m_DY =  pCamera->getVVec() * (cameraPlaneHeight / (float)m_Resolution.y);
	*m_DX4 = sse_vset1(m_DX.x, m_DX.y, m_DX.z);
	*m_DY4 = sse_vset1(m_DY.x, m_DY.y, m_DY.z);

	m_Origin = pCamera->getEye();
	m_LeftUp = pCamera->getEye() - ( pCamera->getNVec() * pCamera->getNear() ) +
					( pCamera->getVVec() * ( cameraPlaneHeight * 0.5f ) ) -
					( pCamera->getUVec() * ( cameraPlaneWidth * 0.5f ) );
	m_LeftUp = m_LeftUp + 0.5f * m_DX - 0.5f * m_DY;
	*m_LeftUp4 = sse_vset1(m_LeftUp.x, m_LeftUp.y, m_LeftUp.z);

	// Pre_calculate temperary values for pixel's address
	for (int i = 0; i < 4; i++) {
		// m_Tmp1_Addr =  m_Height - 1 - iCastSeq4x4_y[i];
		m_Tmp1_Addr4[i] = _mm_sub_epi32(_mm_set1_epi32(m_Height-1), m_iCastSeq4x4_y4[i]);

		// m_Tmp2_Addr = (m_Height - 1 - iCastSeq4x4_y[i]) * m_Width
		m_Tmp2_Addr4[i] = sse_imul(m_Tmp1_Addr4[i], _mm_set1_epi32(m_Width));
	}

	//  [0]		[1]		[2]		[3]		
	//  O O x	x O O	x x x	x x x	
	//  O t x	x t O	O t x	x t O	
	//  x x x	x x x	O O x	x O O	
	m_ADPSS_AdjPixAddr4[0] = _mm_setr_epi32(0, m_Width-1, m_Width, -1);
	m_ADPSS_AdjPixAddr4[1] = _mm_setr_epi32(0, m_Width, m_Width+1, 1);
	m_ADPSS_AdjPixAddr4[2] = _mm_setr_epi32(0, 0-1, 0-m_Width-1, 0-m_Width);
	m_ADPSS_AdjPixAddr4[3] = _mm_setr_epi32(0, 1, 0-m_Width, 0-m_Width+1);
	m_ADPSS_AdjPixAddr[0][0] = 0;	m_ADPSS_AdjPixAddr[0][1] = m_Width-1;	m_ADPSS_AdjPixAddr[0][2] = m_Width;		m_ADPSS_AdjPixAddr[0][3] = -1;
	m_ADPSS_AdjPixAddr[1][0] = 0;	m_ADPSS_AdjPixAddr[1][1] = m_Width;		m_ADPSS_AdjPixAddr[1][2] = m_Width+1;	m_ADPSS_AdjPixAddr[1][3] = 1;
	m_ADPSS_AdjPixAddr[2][0] = 0;	m_ADPSS_AdjPixAddr[2][1] = -1;			m_ADPSS_AdjPixAddr[2][2] = 0-m_Width-1;	m_ADPSS_AdjPixAddr[2][3] = 0-m_Width;
	m_ADPSS_AdjPixAddr[3][0] = 0;	m_ADPSS_AdjPixAddr[3][1] = 1;			m_ADPSS_AdjPixAddr[3][2] = 0-m_Width;	m_ADPSS_AdjPixAddr[3][3] = 0-m_Width+1;

	//  [0]		[1]	
	//  O O O	x x x
	//  O t x	x t O
	//  x x x	O O O
	m_Sobel_AdjPixAddr4[0] = _mm_setr_epi32(m_Width-1, m_Width, m_Width+1, -1);
	m_Sobel_AdjPixAddr4[1] = _mm_setr_epi32(1, 0-m_Width-1, 0-m_Width, 0-m_Width+1);

	m_bIsEnableShadow       = m_Scene->isEnableShadow();
	m_bIsEnableLocalShading = m_Scene->isEnableLocalShading();
	m_bIsUseTexture         = (m_Scene->isUseTexture() & m_bIsEnableLocalShading);
	m_iMaxReflectionDepth   = m_Scene->getMaxReflectionDepth();
	m_bBackFaceCulling      = m_Scene->isBackFaceCulling();
	m_globalAmbient         = m_Scene->getGlobalAmbient();

	// for profiling
	G_PR = G_RR = G_SR = 0;
	I_PR = I_RR = I_SR = 0;
	T_PR = T_RR = T_SR = 0;
	S_PR = S_RR = S_SR = 0;

	PriIsect_FtnCall_Count = 0;
	PriIsect_FC_Cull_Count = 0;
	PriIsect_TriChk_Count = 0;
}
