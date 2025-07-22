#include "GScene.h"
#include "GTexture.h"
#include "GTextureManager.h"
#include "GRenderSystem.h"

#include <stdio.h>
#include "SSE_math.h"
#include "SSERenderPipeline.h"
#include "GClassMacro.h"

#include "GBVHStructure.h"
#include "GBVH_RayPacket.h"

// -----------------------------------------------------------------------
//  ++++++++++++++++++++++++ PACKET TRAVERSAL +++++++++++++++++++++++++++
// -----------------------------------------------------------------------
void 
SSERenderPipeline::Render_SSE_BVHPacketTraversal( int nThreadID )
{	
	int xTileEnd = (m_Resolution.x) >> 2;
	int yTileEnd = (m_Resolution.y) >> 2;
	int tx, ty;
	int i;

	// start spawning rays
	_sse_4x4_raypacket_bvh *rp = &m_RayPk4x4_bvh[0];
	_sse_4x4_isect		   *is = &m_Isect4x4[0];
	_sse_4x4_raypacket_bvh tpos;	// target position for ray casting (pixel center)
	_sse_4x4_raypacket_bvh delta4;		// delta4 for next target position between packet4x4
	_sse_4x4_raypacket_bvh	jpos;		// jittered position for sampling
	
	const __m128 ray_o_x4 = _mm_load1_ps( &m_Origin.x );
	const __m128 ray_o_y4 = _mm_load1_ps( &m_Origin.y );
	const __m128 ray_o_z4 = _mm_load1_ps( &m_Origin.z );
	const __m128 delta_x4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.x ));
	const __m128 delta_y4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.y ));
	const __m128 delta_z4 = _mm_mul_ps(_mm_set1_ps(4), _mm_load1_ps( &m_DX.z ));
	/*
	rp->o   = cpu_fset1(m_Origin.m_Vector);


	//GVector r;
	GVector LeftUp = GVector(m_LeftUp);
	GVector DX     = GVector(m_DX);
	GVector DY     = GVector(m_DY);
	GVector _tvec_a, _tvec_b;

	// ---- PACKET ----
	GBVH_RayPacket RayPacket;
	unsigned int temp_addr[PACKET_SIZE];	

	for (ty = 0; ty < yTileEnd; ty += PACKET_WIDTH) {
	for (tx = 0; tx < xTileEnd;  tx += PACKET_WIDTH) {

		for(int p=0; p<PACKET_SIZE; ++p)
		{
			RayPacket.t[p] = FLOAT_MAX;
		}
		

		// Make ray to PACKET
		unsigned int k=0;
		for(int height = ty; height < ty+PACKET_WIDTH; ++height)
		{
			for(int width = tx; width < tx+PACKET_WIDTH; ++width)
			{
				// -----------------------------------------------------------------------
				// tpos (Ray 瑜� �룧 諛⑺뼢吏��젏) 怨꾩궛
				// -----------------------------------------------------------------------
				// m_LeftUp : image screen �쐞履� �쇊�렪 紐⑥꽌由ъ쓽 pixel 以묒떖 �쑝濡� �씠誘� �뀑�똿 �릺�뼱 �엳�쓬
				vector3 r = m_LeftUp + (m_DX * (float)width) - (m_DY * (float)height);		

				tpos.d.x = r.x;
				tpos.d.y = r.y;
				tpos.d.z = r.z;
				//is->addr = tx + (m_Height - 1 - ty) * m_Width;
				temp_addr[k] = width + (m_Height - 1 - height) * m_Width;
				//m_Dest[3*( 448 + (168)*512 )  ] = 1.0f;
				//if( (m_Height -1 - height) == 168 && width == 446 )
				//	printf("aaa\n");

				jpos = tpos;

				// Ray 瑜� �뀑�똿 - �떆�옉�젏(rp->o) ~ �걹�젏(jpos)
				rp->d = cpu_fsub(jpos.d, rp->o);
				rp->Depth = 0;
				Split_InitPkt1x1( 0 );	// direction vector normalize �벑

				// ray dir 寃곗젙 (q = 8諛⑺뼢以묓븯�굹)
				//int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);

				RayPacket.dir[0][k] = rp->d.x;
				RayPacket.dir[1][k] = rp->d.y;
				RayPacket.dir[2][k] = rp->d.z;

				RayPacket.org[0][k] = rp->o.x;
				RayPacket.org[1][k] = rp->o.y;
				RayPacket.org[2][k] = rp->o.z;

				++k;
			}
		}		
		// RAY - FRUSTUM 怨꾩궛 �쐞�빐 肄붾꼫 �젅�씠��� �룊硫댁쓽 諛⑹젙�떇 怨꾩궛.
		RayPacket.pre_cal();

		//Split_Trace1x1__PriRay(q, 0, 0);
		
		// -----------------------------------
		TMIntCandidate can[PACKET_SIZE];
		BVH_Ranged_Traverse(RayPacket, can);	
		// -----------------------------------
		
		for(int p=0; p<PACKET_SIZE; ++p)
		{
			if(can[p].t != FLOAT_MAX)
			{
				is->tacc = *(can[p].pFirstIndex) + 1;
				is->u = can[p].vBaryP[0];
				is->v = can[p].vBaryP[1];
				is->dist = can[p].t;				
			}
			else
			{
				is->tacc = 0;				
			}
			
			GVector RayDir = RayPacket.getDirection(p);
			rp->d.x = RayDir.x;
			rp->d.y = RayDir.y;
			rp->d.z = RayDir.z;


			BVH_Shading1x1(0);

			//_GCOL_vADD(o_colors[p], o_colors[p], is->color);

			m_Dest[3*(temp_addr[p])  ] = is->color.r;
			m_Dest[3*(temp_addr[p])+1] = is->color.g;
			m_Dest[3*(temp_addr[p])+2] = is->color.b;

			is->color.r = 0.0f;
			is->color.g = 0.0f;
			is->color.b = 0.0f;
			//m_Dest[3*( 447 + (168)*512 )  ] = 1.0f;			
			//m_Dest[3*( 447 + (168)*512 )+1] = 0.0f;	
			//m_Dest[3*( 447 + (168)*512 )+2] = 0.0f;	
		}	
		
	}
	}
	//UpdateBBoxes(m_Data->m_pBVHNodes);
	*/
}