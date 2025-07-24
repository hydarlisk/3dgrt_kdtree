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

//--------------------------------------------------------------------------------
//For Grid traversal.
#include"GGridStructure.h" 
#include"GGrid_RayPacket.h"
#include"GGridConstants.h"

void SSERenderPipeline::Grid_Rendering(void){
	int xTileEnd = m_Resolution.x; int yTileEnd = m_Resolution.y; //해상도.
	int packetH = yTileEnd / packetSize; int packetW = xTileEnd / packetSize;

	//패킷 생성. 주 축 설정.
	GGridStructure *pGrid = m_Scene->getGridStructure();
	GGrid_RayPacket rayPacket, macroPacket;

	GTriangleWrapperList *pTriangleList = pGrid->get_tri_list(); 

	//----------------------------------------------------------------------
	//ray 생성.	
	GVector LeftUp = GVector(m_LeftUp);
	GVector DX     = GVector(m_DX);
	GVector DY     = GVector(m_DY);

	_sse_1x1_raypacket *rp = &m_RayPk1x1[0];
	_sse_1x1_isect *is = &m_Isect1x1[0];

	rp->o = cpu_fset1(m_Origin.m_Vector);
	GVector _rO, _rD;

	_rO.x = rp->o.x; _rO.y = rp->o.y; _rO.z = rp->o.z;
	for(int b = 0;b < packetH;b++){
	for(int a = 0;a < packetW;a++){		
		rayPacket.ray_generate(a, b, LeftUp, DX, DY, _rO);
		pGrid->grid_setup(rayPacket, _rO);//grid를 setup.
		
		int kS, kE, kVector = rayPacket.get_kVector();		

		rayPacket.set_packetId(packetW * b + a);
		rayPacket.set_packet(_rO, pGrid);

		kS = pGrid->get_kStart(); kE = pGrid->get_kEnd();

		if(kVector < 3){						
			rayPacket.camera_culling(kS, pGrid, _rO);
		for(int k = kS;k <= kE;k++){ 
			macroPacket = rayPacket;
			if(pGrid->macro_cell_traversal(kVector, k, macroPacket)){
				rayPacket = macroPacket; continue;
			}
			int uStart, uEnd, vStart, vEnd;
			pGrid->grid_packet_check(rayPacket, uStart, uEnd, vStart, vEnd);
			for(int v = vStart;v <= vEnd;v++){				
			for(int u = uStart;u <= uEnd;u++){
				int num = pGrid->get_cell_triN(kVector, u, v, k);				
				if(num != 0) pGrid->t_far_init(kVector, u, v, k);				
				
				for(int n = 0;n < num;n++){
					int polyIdx;
					float _t_far;
					pGrid->get_tri_id(kVector, u, v, k, n, polyIdx);

					//메일 박스 채크. 같으면 이미 채크한 것이므로 pass.
					if(rayPacket.get_packetId() == pTriangleList->getTriangleWrapper(polyIdx)->m_mailBoxId)	continue;							
					else pTriangleList->getTriangleWrapper(polyIdx)->m_mailBoxId = rayPacket.get_packetId();

					if(u == uStart||u == uEnd||v == vStart||v == vEnd) if(pGrid->frustum_culling(rayPacket, polyIdx, _rO)) continue;

					for(int pj = 0;pj < packetSize;pj++){
						int pjW = (m_Height - 1 - (b * packetSize + pj)) * m_Width;
					for(int pi = 0;pi < packetSize;pi++){								
						is->addr = (a * packetSize + pi) + pjW;									
						
						_rD = rayPacket.get_ray(pi, pj);
						rp->d.x = _rD.x; rp->d.y = _rD.y; rp->d.z = _rD.z; rp->Depth = 0;

						if(grid_isect(polyIdx, rayPacket.m_tmin[pj][pi], pGrid->get_cellBbox(), _t_far)){
							rayPacket.m_t_far[pj][pi] = _t_far;
							rayPacket.m_tmin[pj][pi] = is->dist;							
							rayPacket.m_hitPacketLv[pj][pi] = k;
							Split_Shading1x1(0);		

							m_Dest[3 * is->addr + 0] = is->color.r;
							m_Dest[3 * is->addr + 1] = is->color.g;
							m_Dest[3 * is->addr + 2] = is->color.b;
						} 		
					}}	
				}
			}}
			if(rayPacket.early_termination(k)) break;
		}}
		else{
			rayPacket.camera_culling(kS, pGrid, _rO);	
			pGrid->macroK = kS;
		for(int k = kS;k >= kE;k--){			
			macroPacket = rayPacket;
			if(pGrid->macro_cell_traversal(kVector, k, macroPacket)){
				rayPacket = macroPacket; continue;
			}
			int uStart, uEnd, vStart, vEnd;
			pGrid->grid_packet_check(rayPacket, uStart, uEnd, vStart, vEnd);
			for(int v = vStart;v <= vEnd;v++){				
			for(int u = uStart;u <= uEnd;u++){
				int num = pGrid->get_cell_triN(kVector, u, v, k);
				if(num != 0) pGrid->t_far_init(kVector, u, v, k);

				for(int n = 0;n < num;n++){
					int polyIdx; 
					float _t_far;
					pGrid->get_tri_id(kVector, u, v, k, n, polyIdx);

					//메일 박스 채크. 같으면 이미 채크한 것이므로 pass.
					if(rayPacket.get_packetId() == pTriangleList->getTriangleWrapper(polyIdx)->m_mailBoxId)	continue;							
					else pTriangleList->getTriangleWrapper(polyIdx)->m_mailBoxId = rayPacket.get_packetId();	

					if(u == uStart||u == uEnd||v == vStart||v == vEnd) if(pGrid->frustum_culling(rayPacket, polyIdx, _rO)) continue;

					for(int pj = 0;pj < packetSize;pj++){
						int pjW = (m_Height - 1 - (b * packetSize + pj)) * m_Width;
					for(int pi = 0;pi < packetSize;pi++){								
						is->addr = (a * packetSize + pi) + pjW;								

						_rD = rayPacket.get_ray(pi, pj);
						rp->d.x = _rD.x; rp->d.y = _rD.y; rp->d.z = _rD.z; rp->Depth = 0;

						if(grid_isect(polyIdx, rayPacket.m_tmin[pj][pi], pGrid->get_cellBbox(), _t_far)){			
							rayPacket.m_t_far[pj][pi] = _t_far;
							rayPacket.m_tmin[pj][pi] = is->dist;
							rayPacket.m_hitPacketLv[pj][pi] = k;
							Split_Shading1x1(0);		

							m_Dest[3 * is->addr + 0] = is->color.r;
							m_Dest[3 * is->addr + 1] = is->color.g;
							m_Dest[3 * is->addr + 2] = is->color.b;
						} 		
					}}	

				}	
			}}
			if(rayPacket.early_termination(k)) break;
		}}
	}}
}


bool SSERenderPipeline::grid_isect(int polyIdx, float tMin, GBoundingBox* cellBox, float& _t_far){
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[0];
	_sse_1x1_isect		*is	= &m_Isect1x1[0];
	TriAccel2 &acc = m_Data->m_TriAccList[polyIdx];

	if(!acc.isTransparent && m_bBackFaceCulling){
		if(vector3(rp->d.f).innerProduct(acc.N) < 0){
			return false;
		}
	}
		
	is->dist = tMin;
	
	float f;
	if (!Split_Isect1x1_PlaneTest_PriRay(acc, 0, f, 0)) return false;

	float lambda, mue;
	if (!Split_Isect1x1_TriUVTest_PriRay(acc, 0, f, lambda, mue, 0)) return false;	

	//hit점이 grid cell안에 있는가.
	_sse_float rcpRayDir;
	_GVEC_vRCP(rcpRayDir, rp->d);

	float t_near, t_far_;
	t_near = 0;
	t_far_ = 100000;

	{
		float l1, l2;
		l1 = (cellBox->m_Min.x - rp->o.x) * rcpRayDir.x;
		l2 = (cellBox->m_Max.x - rp->o.x) * rcpRayDir.x;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (cellBox->m_Min.y - rp->o.y) * rcpRayDir.y;
		l2 = (cellBox->m_Max.y - rp->o.y) * rcpRayDir.y;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
		l1 = (cellBox->m_Min.z - rp->o.z) * rcpRayDir.z;
		l2 = (cellBox->m_Max.z - rp->o.z) * rcpRayDir.z;
		t_near = max( min( l1,l2 ), t_near );
		t_far_ = min( max( l1,l2 ), t_far_ );
	}

	_t_far = t_far_;

//	if(!(t_near <= f && f <= t_far_)) return false;	

	is->u = lambda;
	is->v = mue;
	is->dist = f;
	is->tacc = polyIdx+1;

	return true;
}