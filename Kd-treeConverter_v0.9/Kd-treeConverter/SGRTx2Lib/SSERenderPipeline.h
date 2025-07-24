/**
 *	Cuda 로 Rendering 을 수행하기 위해서
 *	여러가지를 관리하는 class.
 *
 *	light, texture, shading, ray tracing, photon mapping
 *	등등.
 *	
 *	by graphicsian.
 */
#ifndef _SSE_RENDER_PIPELINE_H_
#define _SSE_RENDER_PIPELINE_H_

#include "GBase.h"
#include "GDimension.h"
#include "SSERenderCommon.h"
#include "SSERayQueue.h"
#include "SSERenderData.h"

//#include "GThreadWork.h"
//#include "GThreadingOption.h"
#include "GCriticalSection.h"
#include "GClassMacro.h"

class GBVH_RayPacket;
class Ray;
#include"GGrid_RayPacket.h"


#define AIR_INDEX	1
#define MAX_TRACE_DEPTH 23


class SSERenderPipeline
{
public:
	SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data, Detect_ADPSS_measure* a_ADPSS_data,
		SSERayQueue4x4* a_RayQ4x4, SSERayTable2x2* a_RayT2x2);
	SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data, Detect_ADPSS_measure* a_ADPSS_data,
		SSERayTable1x1* a_RayT1x1);
	SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data, Detect_ADPSS_measure* a_ADPSS_data,
		SSERayQueue4x4* a_RayQ4x4, SSERayTable2x2* a_RayT2x2, SSERayTable1x1* a_RayT1x1);
	SSERenderPipeline(Scene* a_Scene, SSESceneData* a_Data);
	virtual ~SSERenderPipeline();

	virtual GError initialize( void );

	// rendering 을 위한 함수
	virtual void PrepareRender( int nMaxWorker );

	// Packet Size 1x1 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		virtual void InitPacket1x1  (int nIdx );
		virtual void RenderPacket1x1( unsigned int quad, int nIdx );
		virtual void TracePacket1x1 ( unsigned int quad, int nIdx );
		virtual void IsectPacket1x1 ( const KdTreeNode2 *node, int nIdx );
		virtual void InitShadowPacket1x1  ( void );
		virtual void TraceShadowPacket1x1 ( unsigned int quad );
		virtual void IsectShadowPacket1x1 ( const KdTreeNode2 *node );
		virtual void Render1x1 ( int nThreadID = -1 );
		virtual void checkVisibility1x1(const GPoint* opos, const GPoint* lpos);
		virtual void shading1x1 (int nIdx);
		//void TraceTestRayPacket1x1();
	protected:
	protected:
		_sse_1x1_kdstack	*m_Stack1x1;						// scene   dependent
		_sse_1x1_raypacket	*m_RayPk1x1;						// scene independent
		_sse_1x1_isect		*m_Isect1x1;						// scene independent
		_sse_1x1_raypacket	*m_ShadowRayPk1x1;					// scene independent
		_sse_1x1_isect		*m_ShadowIsect1x1;					// scene independent

	// Packet Size 2x2 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void InitPacket2x2  ( const int nIdx );
		void RenderPacket2x2( const int nIdx );
		void TracePacket2x2 ( const int nIdx );
		void IsectPacket2x2 ( const KdTreeNode2 *node, const int nIdx );
		void InitShadowPacket2x2  ( void );
		void TraceShadowPacket2x2 ( void );
		void IsectShadowPacket2x2 ( const KdTreeNode2 *node );
		void Render2x2 ( int nThreadID = -1 );
		void checkVisibility2x2(const _sse_vec &opos, const GPoint* lpos, const __m128 &shadingmask);
		void Shading2x2 ( const int nIdx );
	protected:
		/*virtual void FindLeafNode2x2( _sse_2x2_raypacket *rp, _sse_2x2_isect *is,
			float &t_near, float &t_far_, const unsigned int* ray_dir, KdTreeNode2* node, 
			unsigned int &stackIndex, 
			_sse_float &rcpRayDir );*/
	protected:
		_sse_2x2_kdstack	*m_Stack2x2;						// scene   dependent
		_sse_2x2_raypacket	*m_RayPk2x2;						// scene independent
		_sse_2x2_isect		*m_Isect2x2;						// scene independent
		_sse_2x2_raymask	*m_RMask2x2;						// scene independent
		_sse_2x2_raypacket	*m_ShadowRayPk2x2;					// scene independent
		_sse_2x2_isect		*m_ShadowIsect2x2;					// scene independent
		_sse_2x2_raymask	*m_ShadowRMask2x2;					// scene independent

	// Packet Size 4x4 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void InitPacket4x4  ( int nIdx );
		void RenderPacket4x4( int nIdx );
		void TracePacket4x4 ( int nIdx );
		void IsectPacket4x4 ( const KdTreeNode2 *node, int nIdx );
		void IsectPacket_P  ( const KdTreeNode2 *node, int nIdx );			// for pluecker? testing
		void InitShadowPacket4x4  ( void );
		void TraceShadowPacket4x4 ( void );
		void IsectShadowPacket4x4 ( const KdTreeNode2 *node );
		void Render4x4 ( int nThreadID = -1);
		void checkVisibility4x4(const _sse_vec opos[], const GPoint* lpos, const __m128 shadingmask[]);
		void Shading4x4 (const int nIdx);
	protected:
		/*virtual void FindLeafNode4x4( _sse_4x4_raypacket *rp, _sse_4x4_isect *is,
			float &t_near, float &t_far_, const unsigned int* ray_dir, KdTreeNode2* node, 
			unsigned int &stackIndex, 
			_sse_float &rcpRayDir );*/
	protected:
		_sse_4x4_kdstack	*m_Stack4x4;						// scene   dependent
		_sse_4x4_raypacket	*m_RayPk4x4;						// scene independent
		_sse_4x4_isect		*m_Isect4x4;						// scene independent
		_sse_4x4_raymask	*m_RMask4x4;						// scene independent
		_sse_4x4_raymask	*m_TMask4x4;						// scene independent
		_sse_4x4_raypacket	*m_ShadowRayPk4x4;					// scene independent
		_sse_4x4_isect		*m_ShadowIsect4x4;					// scene independent
		_sse_4x4_raymask	*m_ShadowRMask4x4;					// scene independent

	// Scene Information =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
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

		__m128i		*m_Tmp1_Addr4;								// Temporary value for Address stored pixel's color
		__m128i		*m_Tmp2_Addr4;								// Temporary value for Address stored pixel's color

		bool		m_bIsEnableShadow;
		bool		m_bIsEnableLocalShading;
		bool		m_bIsUseTexture;
		int			m_iMaxReflectionDepth;
		bool		m_bBackFaceCulling;
		GColor		m_globalAmbient;

	protected:
		Scene*				m_Scene;

	// for traversal & intersection =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	protected:
		SSESceneData		*m_Data;							// scene data (Kd-tree/Primitives)
		unsigned int		m_RayID;							// Ray ID
		unsigned int		raydir[8][3][2];					// Pointer address offset for ray direction

	// for Super Sampling =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	public:
		void Render4x4_ADPSS_OnePass   ( int nThreadID = -1);
		void Render4x4_ADPSS_Detection ( int nThreadID = -1);
		void Render4x4_ADPSS_TwoPass   ( int nThreadID = -1);
		void Render4x4_ADPSS_OnePass_2_1 ( int nThreadID = -1);
		void Render4x4_ADPSS_OnePass_2_2 ( int nThreadID = -1);
		void Render1x1_ADPSS_OnePass   ( int nThreadID = -1);
		void Render1x1_ADPSS_Detection ( int nThreadID = -1);
		void Render1x1_ADPSS_TwoPass   ( int nThreadID = -1);
	private:
		Detect_ADPSS_measure *m_ADPSS_data;
	protected:
		__m128i				*m_ADPSS_AdjPixAddr4;
		__m128i				*m_Sobel_AdjPixAddr4;

		SSERayQueue4x4		*m_RayQ4x4;
		SSERayTable2x2		*m_RayT2x2;

		int					*m_ADPSS_AdjPixAddr[4];			// for Cha	(1x1)
		SSERayTable1x1		*m_RayT1x1;						// for Cha	(1x1)

	// for Thread =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	protected:
		int					m_iThreadCount;
		int					m_iThreadJobSize;
		float				m_fThreadRcpJobSize;
		float				m_fThreadRcpCount;
		GCriticalSection	m_ColorBufferCS;

	//-------------------------------------------------------------------------------
	// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
	public:
		GTimer CullTime;
		double CullFreq;
		unsigned int CullCount;
		GTimer BBoxTime;
		double BBoxFreq;
		unsigned int BBoxCount;
		GBoundingBox m_bbox;
		unsigned int PriIsect_FtnCall_Count;
		unsigned int PriIsect_FC_Cull_Count;
		unsigned int PriIsect_TriChk_Count;
		_sse_2x2_rayfrustum	*m_FrustomRayPk2x2;
		__m128 m_FrustumRay_t_near4, m_FrustumRay_t_far_4;
	public:
		// 임시 poor code
	// ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
	//-------------------------------------------------------------------------------

	//-------------------------------------------------------------------------------
	// for VTUNE Testing ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
	public:
		__forceinline void Split_InitPkt4x4  ( int nIdx );
		__forceinline void Split_InitPkt4x4_ShwRay  ( void );

		void Split_Render4x4__PriRay ( int nThreadID = -1);

			// Trace -------------------------
			 void Split_Trace4x4__PriRay ( int nIdx, int temp );
			 void Split_Trace4x4__SecRay ( int nIdx );
			 void Split_Trace4x4__ShwRay ( int temp );

			// Isect -------------------------
			 void Split_FCull_Init   ( __m128 &term1, __m128 &term2, const int baseOffset, int nIdx );
			 bool Split_FCull_TriEdge( int &check_aperture, const TriAccel2 &acc, int negmaxabsdir );

			 void Split_Isect4x4__PriRay ( const KdTreeNode2 *node, int nIdx, int temp );
			 void Split_Isect4x4__SecRay ( const KdTreeNode2 *node, int nIdx );
			 void Split_Isect4x4__ShwRay ( const KdTreeNode2 *node, int temp );
			__forceinline bool Split_Isect4x4_PlaneTest_PriRay (TriAccel2 &acc1, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, int temp);
			__forceinline bool Split_Isect4x4_TriUVTest_PriRay (TriAccel2 &acc1, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, RDATA4 &lambda, RDATA4 &mue, int temp );
			__forceinline bool Split_Isect4x4_PlaneTest_SecRay (TriAccel2 &acc1, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f);
			__forceinline bool Split_Isect4x4_TriUVTest_SecRay (TriAccel2 &acc1, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, RDATA4 &lambda, RDATA4 &mue );
			__forceinline bool Split_Isect4x4_PlaneTest_ShwRay (TriAccel2 &acc1, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f);
			__forceinline bool Split_Isect4x4_TriUVTest_ShwRay (TriAccel2 &acc1, int nIdx, RMASK4 &Mask_Hit, RDATA4 &f, RDATA4 &lambda, RDATA4 &mue );

			// Shading -----------------------
			 void Split_Shading4x4 (const int nIdx);

			__forceinline void Split_Shading4x4__Setup (const int nIdx, _sse_vec *hit_p, _sse_vec &global_ambt, _sse_vec *mat_cAmbt, _sse_vec *mat_cDiff,
				_sse_vec *mat_cSpec, _sse_vec *mat_cEmit, _sse_float *mat_fRough, _sse_float *mat_fRefl, _sse_float *mat_fRefr, _sse_float *mat_fRIdx,
				_sse_vec *mat_cTex,	_sse_uint *obj_num, int &n_refl, int &n_refr);

			__forceinline void Split_Shading4x4__LocalShading 
				(const int nIdx, _sse_vec *hit_p, _sse_vec &global_ambt, _sse_vec *mat_cAmbt, _sse_vec *mat_cDiff,
				_sse_vec *mat_cSpec, _sse_vec *mat_cEmit, _sse_float *mat_fRough, _sse_float *mat_fRefl, _sse_float *mat_fRefr, _sse_float *mat_fRIdx,
				_sse_vec *mat_cTex,	_sse_uint *obj_num);

			__forceinline void Split_Shading4x4_RayGeneration_SecRay 
				(const int nIdx, _sse_vec *hit_p,
				_sse_float *mat_fRefl, _sse_float *mat_fRefr, _sse_float *mat_fRIdx,
				_sse_vec *mat_cTex,	int n_refl, int n_refr);

			__forceinline void Split_Shading4x4_RayGeneration_ShwRay(const _sse_vec opos[], const GPoint* lpos, const __m128 shadingmask[]);

	public:
		__forceinline void Split_InitPkt2x2  ( int nIdx );
		__forceinline void Split_InitPkt2x2_ShwRay  ( void );

		void Split_Render2x2__PriRay ( int nThreadID = -1);

			// Trace -------------------------
			void Split_Trace2x2__PriRay ( int nIdx, int temp );
			void Split_Trace2x2__SecRay ( int nIdx );
			void Split_Trace2x2__ShwRay ( int temp );

			// Isect -------------------------
			void Split_Isect2x2__PriRay ( const KdTreeNode2 *node, int nIdx, int temp );
			void Split_Isect2x2__SecRay ( const KdTreeNode2 *node, int nIdx );
			void Split_Isect2x2__ShwRay ( const KdTreeNode2 *node, int temp );
			bool Split_Isect2x2_PlaneTest_PriRay (TriAccel2 &acc1, int nIdx, RMASK &Mask_Hit, RDATA &f, int temp);
			bool Split_Isect2x2_TriUVTest_PriRay (TriAccel2 &acc1, int nIdx, RMASK &Mask_Hit, RDATA &f, RDATA &lambda, RDATA &mue, int temp );
			bool Split_Isect2x2_PlaneTest_SecRay (TriAccel2 &acc1, int nIdx, RMASK &Mask_Hit, RDATA &f);
			bool Split_Isect2x2_TriUVTest_SecRay (TriAccel2 &acc1, int nIdx, RMASK &Mask_Hit, RDATA &f, RDATA &lambda, RDATA &mue );
			bool Split_Isect2x2_PlaneTest_ShwRay (TriAccel2 &acc1, int nIdx, RMASK &Mask_Hit, RDATA &f);
			bool Split_Isect2x2_TriUVTest_ShwRay (TriAccel2 &acc1, int nIdx, RMASK &Mask_Hit, RDATA &f, RDATA &lambda, RDATA &mue );

			// Shading -----------------------
			void Split_Shading2x2 (const int nIdx);

			void Split_Shading2x2__Setup (const int nIdx, _sse_vec &hit_p, _sse_vec &global_ambt, _sse_vec &mat_cAmbt, _sse_vec &mat_cDiff,
				_sse_vec &mat_cSpec, _sse_vec &mat_cEmit, _sse_float &mat_fRough, _sse_float &mat_fRefl, _sse_float &mat_fRefr, _sse_float &mat_fRIdx,
				_sse_vec &mat_cTex,	_sse_uint &obj_num, int &n_refl, int &n_refr);

			void Split_Shading2x2__LocalShading 
				(const int nIdx, _sse_vec &hit_p, _sse_vec &global_ambt, _sse_vec &mat_cAmbt, _sse_vec &mat_cDiff,
				_sse_vec &mat_cSpec, _sse_vec &mat_cEmit, _sse_float &mat_fRough, _sse_float &mat_fRefl, _sse_float &mat_fRefr, _sse_float &mat_fRIdx,
				_sse_vec &mat_cTex,	_sse_uint &obj_num);

			void Split_Shading2x2_RayGeneration_SecRay 
				(const int nIdx, _sse_vec &hit_p,
				_sse_float &mat_fRefl, _sse_float &mat_fRefr, _sse_float &mat_fRIdx,
				_sse_vec &mat_cTex,	int n_refl, int n_refr);

			void Split_Shading2x2_RayGeneration_ShwRay(const _sse_vec &opos, const GPoint* lpos, const __m128 &shadingmask);

	public:
		__forceinline void Split_InitPkt1x1  (int nIdx );
		__forceinline void Split_InitPkt1x1_ShwRay  ( int temp );

		void Split_Render1x1__PriRay ( int nThreadID = -1 );

			// Trace -------------------------
			void Split_Trace1x1__PriRay ( unsigned int quad, int nIdx, int temp );
			void Split_Trace1x1__SecRay ( unsigned int quad, int nIdx );
			void Split_Trace1x1__ShwRay ( unsigned int quad , int temp);

			// Isect -------------------------
			void Split_Isect1x1__PriRay ( const KdTreeNode2 *node, int nIdx, int temp );
			void Split_Isect1x1__SecRay ( const KdTreeNode2 *node, int nIdx );
			void Split_Isect1x1__ShwRay ( const KdTreeNode2 *node, int temp);
			bool Split_Isect1x1_PlaneTest_PriRay(TriAccel2 &acc, int nIdx, float &f, int temp);
			bool Split_Isect1x1_TriUVTest_PriRay(TriAccel2 &acc, int nIdx, float &f, float &lambda, float &mue, int temp);
			bool Split_Isect1x1_PlaneTest_SecRay(TriAccel2 &acc, int nIdx, float &f);
			bool Split_Isect1x1_TriUVTest_SecRay(TriAccel2 &acc, int nIdx, float &f, float &lambda, float &mue);
			bool Split_Isect1x1_PlaneTest_ShwRay(TriAccel2 &acc, int nIdx, float &f);
			bool Split_Isect1x1_TriUVTest_ShwRay(TriAccel2 &acc, int nIdx, float &f, float &lambda, float &mue);

			// Shading -----------------------
			void Split_Shading1x1 (int nIdx);
			void Split_Shading1x1__Setup 
				(const int nIdx, GColor &global_ambt, GColor &mat_cAmbt, GColor &mat_cDiff,
				GColor &mat_cSpec, GColor &mat_cEmit, float &mat_fRough, float &mat_fRefl, float &mat_fRefr, float &mat_fRIdx,
				GColor &mat_cTex,	UINT &obj_num, bool &b_refl, bool &b_refr);
			void Split_Shading1x1__LocalShading 
				(const int nIdx, _sse_float &hit_p, GColor &global_ambt, GColor &mat_cAmbt, GColor &mat_cDiff,
				GColor &mat_cSpec, GColor &mat_cEmit, float &mat_fRough, float &mat_fRefl, float &mat_fRefr, float &mat_fRIdx,
				GColor &mat_cTex, bool &bLoadTexColor, UINT &obj_num);
			void Split_Shading1x1_RayGeneration_SecRay
				(const int nIdx, _sse_float &hit_p,
				float &mat_fRefl, float &mat_fRefr, float &mat_fRIdx,
				GColor &mat_cTex, bool &bLoadTexColor, bool &b_refl, bool &b_refr);
			void Split_Shading1x1_RayGeneration_ShwRay(const GPoint* opos, const GPoint* lpos);


			inline GColor getTexColor (int nIdx);

	public:
		// 통계치
		int m_RunStatics;

		unsigned int G_PR, G_RR, G_SR;								// Processed Ray count		: CMI-1
		unsigned int I_PR, I_RR, I_SR;								// Intersection  count
		unsigned int T_PR, T_RR, T_SR;								// Traversal     count
		unsigned int S_PR, S_RR, S_SR;								// Shading       count (x)

		unsigned int m_pf_Hit_DiffPnt_PR, m_pf_Hit_SpecPnt_PR;		// PRI-6 : diffuse / specualr 개수 (primary)
		unsigned int m_pf_Hit_DiffPnt_RR, m_pf_Hit_SpecPnt_RR;		// PRI-6 : diffuse / specualr 개수 (secondary)
		float        m_Area_Diff_PR, m_Area_Spec_PR;				// PRI-7 : diffuse / specualr 면적 (primary)

		unsigned int m_pf_TexRef_PR, m_pf_TexRef_RR;					// CMI-7 : 텍스쳐 access

		unsigned int m_pf_Hit_ShwPnt_ALL;								// PRI-9 : 그림자 지는 지점
		unsigned int m_pf_Hit_ShwCnt_PR;								//       : 그림자 처리 횟수 (Primary)
		unsigned int m_pf_Hit_ShwCnt_RR;								//       : 그림자 처리 횟수 (Secondary)

		unsigned int m_pf_Hit_ShadCnt_PR;								// CMI-6 : Shading 처리 (Primary)
		unsigned int m_pf_Hit_ShadCnt_RR;								// CMI-6 : Shading 처리 (Secondary)
		unsigned int m_pf_Hit_ShadPnt_PR;								// CMI-6 : Shading 지점 (Primary)
		unsigned int m_pf_Hit_ShadPnt_RR;								// CMI-6 : Shading 지점 (Secondary)

		std::vector<int> vTriID;

		unsigned int CMI_6, CMI_7;


	// for VTUNE Testing ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
	//-------------------------------------------------------------------------------

	//-------------------------------------------------------------------------------
	// BVH Spatial structure ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
	public:			
		void Split_Render1x1__PriRayBVH ( int nThreadID = -1 );
		void Split_Render1x1__PriRayBVH_PACKET ( int nThreadID = -1 );

		void Render1x1_BVHTraversal( int nThreadID = -1 );
		void Render1x1_BVHPacketTraversal( int nThreadID = -1 );
		void Render_SSE_BVHPacketTraversal( int nThreadID = -1 );

		bool BVH_Trace1x1(_sse_1x1_raypacket& ray, TMIntCandidate& can);
		bool BVH_Ranged_Traverse(GBVH_RayPacket& prays, TMIntCandidate* can);
		// ray - box intersection test
		bool testCollision(_sse_1x1_raypacket& ray, GBoundingBox& bbox, float* pfDist);
		bool testCollision(const Ray &ray, GBoundingBox& bbox, float* pfDist) const;
		bool testCollision(const Ray &ray, GBoundingBox& bbox ) const;
		bool RayBoxTest( Ray &ray, GBoundingBox& bbox ) const;
		bool intersect_ray_bbox( Ray *ray,GBoundingBox& bbox );
		bool intersect_ray_bbox( Ray &ray,GBoundingBox& bbox );
		bool intersect_bbox(const Ray& ray, GBoundingBox& bbox, float* pfDist) const;

		// BVH update after triangles move
		bool UpdateBBoxes( GBVHNode *node );

		// ray - tri intersection test
		bool testIntersection(TMIntCandidate* pCan, float* pT, const unsigned int* pFirstIndex, const _sse_1x1_raypacket& ray, float maxt);
		// first hit ray in packet
		int getFirstHit(GBVH_RayPacket& raypacket, GBoundingBox& aabb, const unsigned int &first);
		int getLastHit(GBVH_RayPacket& prays, GBoundingBox& aabb, unsigned int &first);

		// ray - tri intersection test
		bool Split_Isect1x1_PlaneTest_PriRay(TriAccel2 &acc, int nIdx, float &f, float t);
		bool BVH_Split_Isect1x1_PlaneTest_PriRay( _sse_1x1_raypacket& ray, TriAccel2 &acc, float &f, float t );
		bool BVH_Split_Isect1x1_TriUVTest_PriRay( _sse_1x1_raypacket& ray, TriAccel2 &acc, float &f, float &lambda, float &mue );

		bool temp_Split_Isect1x1_PlaneTest_ShwRay( TriAccel2 &acc, float &f );
		bool temp_Split_Isect1x1_TriUVTest_ShwRay( TriAccel2 &acc, float &f, float &lambda, float &mue );

		// Shading -----------------------
		void BVH_Shading1x1 (int nIdx);
			void BVH_Shading1x1__LocalShading 
				(const int nIdx, _sse_float &hit_p, GColor &global_ambt, GColor &mat_cAmbt, GColor &mat_cDiff,
				GColor &mat_cSpec, GColor &mat_cEmit, float &mat_fRough, float &mat_fRefl, float &mat_fRefr, float &mat_fRIdx,
				GColor &mat_cTex, bool &bLoadTexColor, UINT &obj_num);
		// shadow ray
		void BVH_Shading1x1_RayGeneration_ShwRay(const GPoint* opos, const GPoint* lpos);
			void Render1x1_BVH_ShwRayTrace( int nThreadID = -1 );
	

	// -----------------------------------------------------------------------------
	// SSE
	protected:
		_sse_4x4_raypacket_bvh	*m_RayPk4x4_bvh;						// scene independent
		_sse_4x4_isect		    *m_Isect4x4_bvh;						// scene independent

	// BVH Spatial structure ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.
	//-------------------------------------------------------------------------------

	//-------------------------------------------------------------------------------
	// Grid Spatial structure ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
	public:	
		void Grid_Rendering(void);
		bool grid_isect(int polyIdx, float tMin, GBoundingBox* cellBox, float& _t_far);
	//	bool grid_frustum_cull(GGridPacket localPacket, int triId);
	//	float grid_cull_hit(int rayFlag, TriAccel2 &acc);

	// Grid Spatial structure ~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~.~
	//-------------------------------------------------------------------------------
};


#endif
