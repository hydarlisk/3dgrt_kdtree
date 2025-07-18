/**
 *	Cuda 로 Rendering 을 수행하기 위해서
 *	여러가지를 관리하는 class.
 *
 *	light, texture, shading, ray tracing, photon mapping
 *	등등.
 *	
 *	by graphicsian.
 */
#ifndef _SSE_RENDER_PIPELINEQ_H_
#define _SSE_RENDER_PIPELINEQ_H_

#include "GBase.h"
#include "SSERenderCommon.h"
#include "SSERayQueue.h"
#include "SSERenderData.h"

#include "GThreadWork.h"
#include "GThreadingOption.h"
#include "GCriticalSection.h"

#define AIR_INDEX	1
#define MAX_TRACE_DEPTH 23

class SSERenderPipelineQ : public GThreadWork
{
public:
	// thread work 를 위한 함수
	void work( GThreadContext *pThreadContext );
	void stop();
private:
	int m_Worker;
	GCriticalSection m_RenderWorkerCS;

public:
	SSERenderPipelineQ(Scene* a_Scene, SSESceneData* a_Data, Detect_ADPSS_measure* a_ADPSS_data = NULL);
	~SSERenderPipelineQ();

	GError initialize( void );

	// rendering 을 위한 함수
	void PrepareRender( int nMaxWorker );


	// Packet Size 1x1 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void InitPacket1x1  (int nIdx );
		void RenderPacket1x1( unsigned int quad, int nIdx );
		void TracePacket1x1 ( unsigned int quad, int nIdx );
		void IsectPacket1x1 ( const KdTreeNode *node, int nIdx );
		void InitShadowPacket1x1  ( void );
		void TraceShadowPacket1x1 ( unsigned int quad );
		void IsectShadowPacket1x1 ( const KdTreeNode *node );
		void Render1x1 ( int nThreadID = -1 );
		void checkVisibility1x1(const GPoint* lpos);
		__forceinline void shading1x1 (int nIdx);
	protected:
		_sse_1x1_kdstack	*m_Stack1x1;						// scene   dependent
		_sse_1x1_raypacket	*m_RayPk1x1;						// scene independent
		_sse_1x1_isect		*m_Isect1x1;						// scene independent
		_sse_1x1_raypacket	*m_ShadowRayPk1x1;					// scene independent
		_sse_1x1_isect		*m_ShadowIsect1x1;					// scene independent

	// Packet Size 4x4 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void InitPacket4x4  ( int nIdx );
		void RenderPacket4x4( int nIdx );
		void TracePacket4x4 ( int nIdx );
		void IsectPacket4x4 ( const KdTreeNode *node, int nIdx );
		void IsectPacket_P  ( const KdTreeNode *node, int nIdx );			// for pluecker? testing
		void InitShadowPacket4x4  ( void );
		void TraceShadowPacket4x4 ( void );
		void IsectShadowPacket4x4 ( const KdTreeNode *node );
		void Render4x4 ( int nThreadID = -1);
		void checkVisibility4x4(const GPoint* lpos, const __m128 shadingmask[]);
		__forceinline void shading4x4 (const int nIdx);
	protected:
		_sse_4x4_kdstack	*m_Stack4x4;						// scene   dependent
		_sse_4x4_raypacket	*m_RayPk4x4;						// scene independent
		_sse_4x4_isectQ		*m_Isect4x4;						// scene independent
		_sse_4x4_raymask	*m_RMask4x4;						// scene independent
		_sse_4x4_raymask	*m_TMask4x4;						// scene independent
		_sse_4x4_raypacket	*m_ShadowRayPk4x4;					// scene independent
		_sse_4x4_isectQ		*m_ShadowIsect4x4;					// scene independent
		_sse_4x4_raymask	*m_ShadowRMask4x4;					// scene independent
	public:
		SSETraceQueue4x4	*m_TraceQ4x4;

	// Packet Size 2x2 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void InitPacket2x2  ( const int nIdx );
		void RenderPacket2x2( const int nIdx );
		void TracePacket2x2 ( const int nIdx );
		void IsectPacket2x2 ( const KdTreeNode *node, const int nIdx );
		void InitShadowPacket2x2  ( void );
		void TraceShadowPacket2x2 ( void );
		void IsectShadowPacket2x2 ( const KdTreeNode *node );
		void Render2x2 ( int nThreadID = -1 );
		void checkVisibility2x2(const GPoint* lpos, const __m128 shadingmask);
		__forceinline void shading2x2 ( const int nIdx );
	protected:
		_sse_2x2_kdstack	*m_Stack2x2;						// scene   dependent
		_sse_2x2_raypacket	*m_RayPk2x2;						// scene independent
		_sse_2x2_isect		*m_Isect2x2;						// scene independent
		_sse_2x2_raymask	*m_RMask2x2;						// scene independent
		_sse_2x2_raypacket	*m_ShadowRayPk2x2;					// scene independent
		_sse_2x2_isect		*m_ShadowIsect2x2;					// scene independent
		_sse_2x2_raymask	*m_ShadowRMask2x2;					// scene independent
	public:
		SSETraceQueue2x2	*m_TraceQ2x2;

	// RayQueue 4x4 =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void Render4x4Q ( void );
		void Render4x4Q_onThread ( int nThreadID );
		void GeneratePrimaryRay_inBlock4x4Q( int tx, int ty );
		void RenderQueue4x4Q( void );
		void InitPacket4x4Q  ( _sse_4x4_traceData *traceData );
		void RenderPacket4x4Q( _sse_4x4_traceData *traceData );
		void TracePacket4x4Q ( _sse_4x4_traceData *traceData );
		void IsectPacket4x4Q ( const KdTreeNode *node, _sse_4x4_traceData *traceData );
		__forceinline void shading4x4Q ( _sse_4x4_traceData *traceData );
	protected:
		int m_BlockX, m_BlockY;								// tile Count
		int m_BlockW, m_BlockH;								// tile Size
		GCriticalSection m_RayIdCS;
		GCriticalSection m_MailboxCS;

	// Scene Infomation =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	protected:
		Pixel*				m_Dest;								// Output buffer for saving final color
		unsigned int		m_Width, m_Height;					// user    dependent	(Scene resolution)
		GDimension			m_Resolution;						// user    dependent	(Scene resolution)
		GDimension			m_SuperSampling;					// user    dependent	(Supersampling rate)
		vector3				m_Origin;							// user    dependent	(Camera pos)
		vector3				m_LeftUp;							// user    dependent	(Pos at LeftUp corner)
		vector3				m_DX, m_DY;							// user    dependent	(Delta between rays)

		_sse_vec			*m_LeftUp4;							// Pos at LeftUp corner for 4x4
		_sse_vec			*m_DX4, *m_DY4;						// Delta between rays for 4x4

		__m128		 *m_CastSeq2x2_x4,  *m_CastSeq2x2_y4;		// Sampling order for 2x2 packet (float)
		__m128i		*m_iCastSeq2x2_x4, *m_iCastSeq2x2_y4;		// Sampling order for 2x2 packet (int)
		__m128		 *m_CastSeq4x4_x4,  *m_CastSeq4x4_y4;		// Sampling order for 4x4 packet (float)
		__m128i		*m_iCastSeq4x4_x4, *m_iCastSeq4x4_y4;		// Sampling order for 4x4 packet (int)

		__m128i		*m_Tmp1_Addr4;								// Temperary value for Address stored pixel's color
		__m128i		*m_Tmp2_Addr4;								// Temperary value for Address stored pixel's color


	private:
		Scene*				m_Scene;

	// for traversal & intersection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	private:
		SSESceneData		*m_Data;							// scene data (Kd-tree/Primitives)
		unsigned int		m_RayID;							// Ray ID
		unsigned int		raydir[8][3][2];					// Pointer address offset for ray direction

	// for Super Sampling =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void Render4x4_ADPSS ( int nThreadID = -1);
		void Render4x4_ADPSS_OnePass   ( int nThreadID = -1);
		void Render4x4_ADPSS_Detection ( int nThreadID = -1);
		void Render4x4_ADPSS_TwoPass   ( int nThreadID = -1);
		void Render4x4_ADPSS_OnePass_2 ( int nThreadID = -1);
	private:
		Detect_ADPSS_measure *m_ADPSS_data;
	protected:
		__m128i				*m_ADPSS_AdjPixAddr4;
		SSERayQueue4x4		*m_RayQ4x4;

	// for Thread =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	protected:
		int					m_iThreadCount;
		float				m_fRcpThreadCount;
		GCriticalSection	m_ColorBufferCS;
};


#endif
