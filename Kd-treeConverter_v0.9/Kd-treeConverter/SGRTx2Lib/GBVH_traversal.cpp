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


void 
SSERenderPipeline::Render1x1_BVHTraversal( int nThreadID )
{	
	int xTileEnd = m_Resolution.x;
	int yTileEnd = m_Resolution.y;
	int tx, ty;

	// start spawning rays
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[0];
	_sse_1x1_isect		*is	= &m_Isect1x1[0];
	_sse_1x1_raypacket	tpos;		// target position for ray casting (pixel center)
	_sse_1x1_raypacket	jpos;		// jittered position for sampling

	rp->o   = cpu_fset1(m_Origin.m_Vector);

	ty = 0;

	GVector r;
	GVector LeftUp = GVector(m_LeftUp);
	GVector DX     = GVector(m_DX);
	GVector DY     = GVector(m_DY);
	GVector _tvec_a, _tvec_b;

	for (; ty < yTileEnd; ty++) {
	for (tx = 0; tx < xTileEnd;  tx++) {

		// -----------------------------------------------------------------------
		// tpos (Ray 를 쏠 방향지점) 계산
		// -----------------------------------------------------------------------
		// m_LeftUp : image screen 위쪽 왼편 모서리의 pixel 중심 으로 이미 셋팅 되어 있음
		vector3 r = m_LeftUp + (m_DX * (float)tx) - (m_DY * (float)ty);		

		tpos.d.x = r.x;
		tpos.d.y = r.y;
		tpos.d.z = r.z;
		is->addr = tx + (m_Height - 1 - ty) * m_Width;

			GColor o_color(0.0f, 0.0f, 0.0f);
			jpos = tpos;

			// Ray 를 셋팅 - 시작점(rp->o) ~ 끝점(jpos)
			rp->d = cpu_fsub(jpos.d, rp->o);
			rp->Depth = 0;
			Split_InitPkt1x1( 0 );	// direction vector normalize 등

			//Split_Trace1x1__PriRay(q, 0, 0);

			TMIntCandidate can;			
			if( BVH_Trace1x1(*rp, can) )
			{				
				is->tacc = *(can.pFirstIndex) + 1;
				is->u = can.vBaryP[0];
				is->v = can.vBaryP[1];
				is->dist = can.t;
			}
			else
				is->tacc = 0;
			// -----------------------------------


			//Split_Shading1x1(0);
			BVH_Shading1x1(0);
/*
			if(is->addr == (447 + (168)*512) )
			{
				GLogManager::logging( LOG_INFO, "\n---------------------------------\n");
				GLogManager::logging( LOG_INFO, "tacc : %d\n", is->tacc);
				GLogManager::logging( LOG_INFO, "Ray dir : %f %f %f\n", rp->d.x, rp->d.y, rp->d.z);
			}


			if(is->addr == (447 + (168)*512) )
			{
				GLogManager::logging( LOG_INFO, "color : %f %f %f\n", is->color.r, is->color.g, is->color.b);
				GLogManager::logging( LOG_INFO, "dist : %f , u: %f, v: %f \n", is->dist, is->u, is->v);
			}
*/
			// -----------------------------------------------------------------------
			// Copy color info to Memory
			// -----------------------------------------------------------------------
			//o_color = o_color + is->color;
			_GCOL_vADD(o_color, o_color, is->color);
			
			m_Dest[3*(is->addr)]   = o_color.r;
			m_Dest[3*(is->addr)+1] = o_color.g;
			m_Dest[3*(is->addr)+2] = o_color.b;
	}
	}
}


// -----------------------------------------------------------------------
//  ++++++++++++++++++++++++ PACKET TRAVERSAL +++++++++++++++++++++++++++
// -----------------------------------------------------------------------
void 
SSERenderPipeline::Render1x1_BVHPacketTraversal( int nThreadID )
{	
	int xTileEnd = m_Resolution.x;
	int yTileEnd = m_Resolution.y;
	int tx, ty;

	// start spawning rays
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[0];
	_sse_1x1_isect		*is	= &m_Isect1x1[0];
	_sse_1x1_raypacket	tpos;		// target position for ray casting (pixel center)
	_sse_1x1_raypacket	jpos;		// jittered position for sampling

	rp->o   = cpu_fset1(m_Origin.m_Vector);


	bool a;
	int b = sizeof(a);

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
				// tpos (Ray 를 쏠 방향지점) 계산
				// -----------------------------------------------------------------------
				// m_LeftUp : image screen 위쪽 왼편 모서리의 pixel 중심 으로 이미 셋팅 되어 있음
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

				// Ray 를 셋팅 - 시작점(rp->o) ~ 끝점(jpos)
				rp->d = cpu_fsub(jpos.d, rp->o);
				rp->Depth = 0;
				Split_InitPkt1x1( 0 );	// direction vector normalize 등

				// ray dir 결정 (q = 8방향중하나)
				//int q = (rp->d.x < 0) + ((rp->d.y < 0) << 1) + ((rp->d.z < 0) << 2);

				RayPacket.dir[0][k] = rp->d.x;
				RayPacket.dir[1][k] = rp->d.y;
				RayPacket.dir[2][k] = rp->d.z;

				RayPacket.org[0][k] = rp->o.x;
				RayPacket.org[1][k] = rp->o.y;
				RayPacket.org[2][k] = rp->o.z;
/*
				// FOR RAY - AABB intersection
				// An Efficient and Robust Ray-Box Intersection Algorithm
				RayPacket.inv_direction_x[k] = 1.0f / rp->d.x;
				RayPacket.inv_direction_y[k] = 1.0f / rp->d.y;
				RayPacket.inv_direction_z[k] = 1.0f / rp->d.z;
				RayPacket.sign[0][k] = RayPacket.inv_direction_x[k] < 0;
				RayPacket.sign[1][k] = RayPacket.inv_direction_y[k] < 0;
				RayPacket.sign[2][k] = RayPacket.inv_direction_z[k] < 0;			
				// END FOR RAY - AABB intersection
*/				
				++k;
			}
		}		
		// RAY - FRUSTUM 계산 위해 코너 레이와 평면의 방정식 계산.
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

/*
			if(temp_addr[p] == (447 + (168)*512) )
			{
				GLogManager::logging( LOG_INFO, "\n---------------------------------\n");
				GLogManager::logging( LOG_INFO, "Ray dir : %f %f %f\n", RayDir.x, RayDir.y, RayDir.z);
			}
*/
			BVH_Shading1x1(0);
/*
			if(temp_addr[p] == (447 + (168)*512) )
			{
				GLogManager::logging( LOG_INFO, "color : %f %f %f\n", is->color.r, is->color.g, is->color.b);
				GLogManager::logging( LOG_INFO, "dist : %f , u: %f, v: %f \n", is->dist, is->u, is->v);
			}
*/
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
}

// -----------------------------------------------------------------------
//  ++++++++++++++++++++++++ SHADOW TRAVERSAL +++++++++++++++++++++++++++
// -----------------------------------------------------------------------
void 
SSERenderPipeline::Render1x1_BVH_ShwRayTrace( int nThreadID )
{	
	T_SR++;
	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	// ---------------------------------------------------------------------------
	// Initialize
	// ---------------------------------------------------------------------------
	// IsectData
	is->dist = FLOAT_MAX;
	is->tacc = 0;

	_sse_1x1_raypacket ray;
	ray.d = rp->d;
	ray.o = rp->o;
	
	// traverse from the root
	const GBVHNode* stackpNode[64];
	unsigned int nCurrDepth = 0;
	stackpNode[nCurrDepth++] = m_Data->m_pBVHNodes;

	while(nCurrDepth)
	{
		const GBVHNode* pNode = stackpNode[--nCurrDepth];
RECURSECHILD:
		float fDist;
		GBoundingBox bbox = pNode->getAABB();
		if( testCollision(ray, bbox, &fDist) && fDist < is->dist )
		{
			if( !pNode->isLeaf() ) 
			{
				// if not leaf node, traverse into child nodes
				const GBVHNode* pnodeTmp;
				pNode->getOrderedChilds( &pnodeTmp, &(stackpNode[nCurrDepth]), GVector(ray.d.f)/* GVector(ray.d.x, ray.d.y, ray.d.z)*/ );
				++nCurrDepth;
				pNode = pnodeTmp;

				goto RECURSECHILD;
			}
			else
			{
				// this is leaf node. check child poly intersection
				const SVPOLYINDEX& svpolyidx = pNode->getPolyIndex();
				const unsigned int nPoly = (unsigned int) svpolyidx.size();				
				//++++++++++++++++++++++++++++++++++++++++++++++++++++++++						
				for( unsigned int i=0; i<nPoly; ++i)
				{
					//++++++++++++++++++++++++++++++++++++++++++++++++++++++++											
					/*
					TMIntCandidate tmpcan;
					if( testIntersection(&tmpcan, 0, &svpolyidx[i], ray, can.t) )
						can = tmpcan;
						*/
					float f;
					TriAccel &acc = m_Data->m_TriAccList[svpolyidx[i]];
					
					if( !temp_Split_Isect1x1_PlaneTest_ShwRay( acc, f ) ) continue;				

					float lambda, mue;
					if (!temp_Split_Isect1x1_TriUVTest_ShwRay( acc, f, lambda, mue)) continue;					

					is->u = lambda;
					is->v = mue;
					is->dist = f;
					is->tacc = svpolyidx[i]+1;
					//++++++++++++++++++++++++++++++++++++++++++++++++++++++++
				}
			}
		}		
	}	
}


// ---------------------------------------------------
// TRAVERSE CODE
// ---------------------------------------------------
bool
SSERenderPipeline::BVH_Trace1x1(_sse_1x1_raypacket& ray, TMIntCandidate& can)
{
	//TMIntCandidate can;
	can.t = FLOAT_MAX;
	// traverse from the root
	const GBVHNode* stackpNode[64];
	unsigned int nCurrDepth = 0;
	stackpNode[nCurrDepth++] = m_Data->m_pBVHNodes;

	while(nCurrDepth)
	{
		const GBVHNode* pNode = stackpNode[--nCurrDepth];
RECURSECHILD:
		float fDist;
		GBoundingBox bbox = pNode->getAABB();
		if( testCollision(ray, bbox, &fDist) && fDist < can.t )
		{
			if( !pNode->isLeaf() ) 
			{
				// if not leaf node, traverse into child nodes
				const GBVHNode* pnodeTmp;
				pNode->getOrderedChilds( &pnodeTmp, &(stackpNode[nCurrDepth]), GVector(ray.d.x, ray.d.y, ray.d.z) );
				++nCurrDepth;
				pNode = pnodeTmp;

				goto RECURSECHILD;
			}
			else
			{
				// this is leaf node. check child poly intersection
				const SVPOLYINDEX& svpolyidx = pNode->getPolyIndex();
				const unsigned int nPoly = (unsigned int) svpolyidx.size();				
				//++++++++++++++++++++++++++++++++++++++++++++++++++++++++						
				for( unsigned int i=0; i<nPoly; ++i)
				{
					//++++++++++++++++++++++++++++++++++++++++++++++++++++++++											
					/*
					TMIntCandidate tmpcan;
					if( testIntersection(&tmpcan, 0, &svpolyidx[i], ray, can.t) )
						can = tmpcan;
						*/
					float f;
					TriAccel &acc = m_Data->m_TriAccList[svpolyidx[i]];
					//if( !temp_Split_Isect1x1_PlaneTest_PriRay(ray, acc, 0, f, can.t) ) continue;
					if( !Split_Isect1x1_PlaneTest_PriRay(acc, 0, f, can.t) ) continue;

					float lambda, mue;
					//if (!temp_Split_Isect1x1_TriUVTest_PriRay(ray, acc, 0, f, lambda, mue, 0)) continue;
					if (!Split_Isect1x1_TriUVTest_PriRay(acc, 0, f, lambda, mue, 0)) continue;

					can.vBaryP[0] = lambda;
					can.vBaryP[1] = mue;
					can.t = f;
					can.pFirstIndex = &svpolyidx[i];
					//++++++++++++++++++++++++++++++++++++++++++++++++++++++++
				}
			}
		}		
	}
	
	if(can.t == FLOAT_MAX)
		return false;

	return true;
}


bool
SSERenderPipeline::BVH_Ranged_Traverse(GBVH_RayPacket& prays, TMIntCandidate* can)
{
	for(int p=0; p<PACKET_SIZE; ++p)
	{
		can[p].t = FLOAT_MAX;		
	}

	// traverse from root node
	GBVHNode* stackpNode[64];
	unsigned int nCurrDepth = 0;
	unsigned int f_index = 0;
	stackpNode[nCurrDepth++] = m_Data->m_pBVHNodes;

	while(nCurrDepth)
	{
		GBVHNode* pNode = stackpNode[--nCurrDepth];				
RECURSECHILD:
		// getFirstHit 여기에 T값 비교하는 거 있어야 물체 앞 뒤 구분 가능!!!!		
		GBoundingBox bbox = pNode->getAABB();
		f_index = getFirstHit( prays, bbox, pNode->getFirstActive() );
		if(f_index < PACKET_SIZE) // hit
		{
			if( !pNode->isLeaf() )
			{
				GBVHNode* pnodeTmp;
				GVector rayDir = GVector(prays.dir[0][f_index], prays.dir[1][f_index], prays.dir[2][f_index]);				
				pNode->getOrderedChilds(&pnodeTmp, &stackpNode[nCurrDepth], rayDir);				
				stackpNode[nCurrDepth++]->setFirstActive(f_index); // 오른쪽 자식에 first index 저장.
				
				pNode = pnodeTmp;
				pNode->setFirstActive(f_index);                    // 왼쪽 자식에 first index 저장.

				goto RECURSECHILD;
			}
			else // leaf node
			{
				const SVPOLYINDEX& svpolyidx = pNode->getPolyIndex();
				//bbox = pNode->getGAABB();
				unsigned int e_index = getLastHit(prays, bbox, f_index);
				const unsigned int nPoly = (unsigned int) svpolyidx.size();
				for(unsigned int i=0; i<nPoly; ++i) // leaf 노드의 삼각형 수
				{
					for(unsigned int j=f_index; j<e_index; ++j)					
					{	
						_sse_1x1_raypacket ray;
						
						// get ray direction & origin
						prays.getRay(ray, j);
						
						float f;
						TriAccel &acc = m_Data->m_TriAccList[svpolyidx[i]];
						if( !BVH_Split_Isect1x1_PlaneTest_PriRay(ray, acc, f, can[j].t) ) continue;

						float lambda, mue;
						if (!BVH_Split_Isect1x1_TriUVTest_PriRay(ray, acc, f, lambda, mue)) continue;

						can[j].vBaryP[0] = lambda;
						can[j].vBaryP[1] = mue;
						can[j].t = f;
						can[j].pFirstIndex = &svpolyidx[i];													
					}
				}				
			}
		}
	}
	return true;
}


// ---------------------------------------------------
// BVH update after triangles move
// ---------------------------------------------------
bool
SSERenderPipeline::UpdateBBoxes( GBVHNode *node)
{	
	// traverse from root node
	GBVHNode* stackpNode[40];
	unsigned int nCurrDepth = 0;
	stackpNode[nCurrDepth++] = m_Data->m_pBVHNodes;	

	while(nCurrDepth)
	{
		GBVHNode* pNode = stackpNode[--nCurrDepth];

		if( !pNode->isLeaf() )  // inner node			
		{				
			if( pNode->getChildrenFlag() )
			{
				// 두 자식의 bbox를 비교하여 자신의 bbox를 update 함.
				pNode->in_nodeUpdateAABB( pNode->thisAABB() );
				pNode->flag = 1;
				// update한 자식의 flag는 다시 false로 setting
				pNode->setChildrenFlagZero();
				continue;
			}

			// 오른쪽 자식을 스택에 push하고 왼쪽 자식 push
			pNode->getChildforUpdate(&stackpNode[nCurrDepth+2], &stackpNode[nCurrDepth+1]);
			nCurrDepth += 3;			
		}
		else   // leaf node
		{
			pNode->flag = 1;
			// update bounding box.	
			// get updated coordinate.			
			GBoundingBox temp = pNode->getAABB();
			GVector tempMin, tempMax;
			tempMin = temp.getMin();
			tempMax = temp.getMax();
			tempMin.x -= 0.001f; tempMin.y -= 0.001f; tempMin.z -= 0.001f;
			tempMax.x += 0.001f; tempMax.y += 0.001f; tempMax.z += 0.001f;
			const GVector tmin = tempMin;
			const GVector tmax = tempMax;
			// triangles
			temp.tryupdateMin(tmin);
			temp.tryupdateMax(tmax);
			pNode->updateAABB(temp);
		}
	}
	
	m_Data->m_pBVHNodes->flag = false;

	return true;
}


// ---------------------------------------------------
// RAY-AABB intersection CODE
// ---------------------------------------------------
#define IR(x)	(*(reinterpret_cast<unsigned int long *>(&x)))

bool
SSERenderPipeline::testCollision(_sse_1x1_raypacket& ray, GBoundingBox& bbox, float* pfDist)
{
	bool bInside = true;
	GVector m_Min(bbox.m_Min);
	GVector m_Max(bbox.m_Max);

	GVector vMaxT(-1.0f, -1.0f, -1.0f);

	// Find candidate planes.
	// -----------------------------------------------------------
	if(ray.o.x < m_Min[0])														
	{																				
		/* Calculate T distances to candidate planes */								
		if(ray.d.x != 0.0f)	vMaxT[0] = (m_Min[0] - ray.o.x) / ray.d.x;	
		bInside		= false;														
	}																				
	else if(ray.o.x > m_Max[0])                                                  
	{                                                                               
		/* Calculate T distances to candidate planes */                             
		if(ray.d.x != 0.0f) vMaxT[0] = (m_Max[0] - ray.o.x) / ray.d.x;   
		bInside = false;                                                            
	}

	// -----------------------------------------------------------
	if(ray.o.y < m_Min[1])														
	{																				
		/* Calculate T distances to candidate planes */								
		if(ray.d.y != 0.0f)	vMaxT[1] = (m_Min[1] - ray.o.y) / ray.d.y;	
		bInside		= false;														
	}																				
	else if(ray.o.y > m_Max[1])                                                  
	{                                                                               
		/* Calculate T distances to candidate planes */                             
		if(ray.d.y != 0.0f) vMaxT[1] = (m_Max[1] - ray.o.y) / ray.d.y;   
		bInside = false;                                                            
	}

	// -----------------------------------------------------------
	if(ray.o.z < m_Min[2])														
	{																				
		/* Calculate T distances to candidate planes */								
		if(ray.d.z != 0.0f)	vMaxT[2] = (m_Min[2] - ray.o.z) / ray.d.z;	
		bInside		= false;														
	}																				
	else if(ray.o.z > m_Max[2])                                                  
	{                                                                               
		/* Calculate T distances to candidate planes */                             
		if(ray.d.z != 0.0f) vMaxT[2] = (m_Max[2] - ray.o.z) / ray.d.z;   
		bInside = false;                                                            
	}
	// -----------------------------------------------------------

	// Ray origin bInside bounding box
	if(bInside)
	{
		if(pfDist)
			*pfDist = 0.0f;
		return true;
	}
	// Get largest of the vMaxT's for final choice of intersection
	// - this version without FPU compares
	// - but branch prediction might suffer
	// - a bit faster on my Celeron, duno how it behaves on something like a P4
	unsigned int nWhichPlane;
	if(IR(vMaxT[0]) & 0x80000000)
	{
		// T[0] < 0
		if(IR(vMaxT[1]) & 0x80000000)
		{
			// T[0] < 0, T[1] < 0
			if(IR(vMaxT[2]) & 0x80000000)
				// T[0] < 0, T[1] < 0, T[2] < 0
				return false;
			else
				nWhichPlane = 2;
		}
		else if (IR(vMaxT[2]) & 0x80000000)
		{
			// T[0] < 0, T[1] > 0, T[2] < 0
			nWhichPlane = 1;
		}
		else
		{
			// T[0] < 0, T[1] > 0, T[2] > 0
			if( IR(vMaxT[2]) > IR(vMaxT[1]) )
			{
				nWhichPlane = 2;
			}
			else
			{
				nWhichPlane = 1;
			}
		}
	}
	else
	{
		// T[0] > 0
		if(IR(vMaxT[1]) & 0x80000000)
		{
			// T[0] > 0, T[1] < 0
			if(IR(vMaxT[2]) & 0x80000000) // T[0] > 0, T[1] < 0, T[2] < 0				
				nWhichPlane = 0;
			else                          // T[0] > 0, T[1] < 0, T[2] > 0
			{
				if(IR(vMaxT[2]) > IR(vMaxT[0]) )
					nWhichPlane = 2;
				else
					nWhichPlane = 0;
			}
		}
		else if (IR(vMaxT[2]) & 0x80000000)
		{
			// T[0] > 0, T[1] > 0, T[2] < 0
			if(IR(vMaxT[1]) > IR(vMaxT[0]))
				nWhichPlane = 1;
			else
				nWhichPlane = 0;
		}
		else
		{
			// T[0] > 0, T[1] > 0, T[2] > 0
			nWhichPlane = 0;
			if( IR(vMaxT[1]) > IR(vMaxT[0]) )  nWhichPlane = 1;
			if( IR(vMaxT[2]) > IR(vMaxT[nWhichPlane]) )  nWhichPlane = 2;
		}
	}
	// -----------------------------------------------------------
	if(nWhichPlane != 0)                                                            
	{                                                                               
		float tmp = ray.o.x + ray.d.x * vMaxT[nWhichPlane];                 
		if(tmp < m_Min[0] - FM_EPSILON || tmp > m_Max[0] + FM_EPSILON) 
			return false;  
	}

	// -----------------------------------------------------------
	if(nWhichPlane != 1)                                                            
	{                                                                               
		float tmp = ray.o.y + ray.d.y * vMaxT[nWhichPlane];                 
		if(tmp < m_Min[1] - FM_EPSILON || tmp > m_Max[1] + FM_EPSILON) 
			return false;  
	}

	// -----------------------------------------------------------
	if(nWhichPlane != 2)                                                            
	{                                                                               
		float tmp = ray.o.z + ray.d.z * vMaxT[nWhichPlane];                 
		if(tmp < m_Min[2] - FM_EPSILON || tmp > m_Max[2] + FM_EPSILON) 
			return false;  
	}
	// -----------------------------------------------------------


	if(pfDist)
		*pfDist = vMaxT[nWhichPlane];
	return true; // ray hits box
}

bool
SSERenderPipeline::testCollision(const Ray &ray, GBoundingBox& bbox, float* pfDist) const
{
	bool bInside = true;

	const GVector m_Min(bbox.m_Min);
	const GVector m_Max(bbox.m_Max);
	GVector vMaxT(-1.0f, -1.0f, -1.0f);

	// Find candidate planes.	
	if(ray.vPos.x < m_Min.x)														
	{																				
		// Calculate T distances to candidate planes 
		if(ray.vDir.x != 0.0f)	vMaxT.x = (m_Min.x - ray.vPos.x) / ray.vDir.x;	
		bInside		= false;														
	}																				
	else if(ray.vPos.x > m_Max.x)                                                  
	{                                                                               
		// Calculate T distances to candidate planes 
		if(ray.vDir.x != 0.0f) vMaxT.x = (m_Max.x - ray.vPos.x) / ray.vDir.x;   
		bInside = false;                                                            
	}
	if(ray.vPos.y < m_Min.y)														 
	{																				 
		// Calculate T distances to candidate planes 
		if(ray.vDir.y != 0.0f)	vMaxT.y = (m_Min.y - ray.vPos.y) / ray.vDir.y;	 
		bInside		= false;														 
	}																				 
	else if(ray.vPos.y > m_Max.y)                                                   
	{                                                                                
		// Calculate T distances to candidate planes 
		if(ray.vDir.y != 0.0f) vMaxT.y = (m_Max.y - ray.vPos.y) / ray.vDir.y;    
		bInside = false;                                                             
	}
	if(ray.vPos.z < m_Min.z)														 
	{																				 
		// Calculate T distances to candidate planes 
		if(ray.vDir.z != 0.0f)	vMaxT.z = (m_Min.z - ray.vPos.z) / ray.vDir.z;	 
		bInside		= false;														 
	}																				 
	else if(ray.vPos.z > m_Max.z)                                                   
	{                                                                                
		// Calculate T distances to candidate planes 
		if(ray.vDir.z != 0.0f) vMaxT.z = (m_Max.z - ray.vPos.z) / ray.vDir.z;    
		bInside = false;                                                             
	}

	// Ray origin bInside bounding box
	if(bInside)
	{
		if(pfDist)
			*pfDist = 0.0f;
		return true;
	}
	// Get largest of the vMaxT's for final choice of intersection
	// - this version without FPU compares
	// - but branch prediction might suffer
	// - a bit faster on my Celeron, duno how it behaves on something like a P4
	unsigned int nWhichPlane;
	if(IR(vMaxT[0]) & 0x80000000)
	{
		// T[0] < 0
		if(IR(vMaxT[1]) & 0x80000000)
		{
			// T[0] < 0, T[1] < 0
			if(IR(vMaxT[2]) & 0x80000000)
				// T[0] < 0, T[1] < 0, T[2] < 0
				return false;
			else
				nWhichPlane = 2;
		}
		else if (IR(vMaxT[2]) & 0x80000000)
		{
			// T[0] < 0, T[1] > 0, T[2] < 0
			nWhichPlane = 1;
		}
		else
		{
			// T[0] < 0, T[1] > 0, T[2] > 0
			if( IR(vMaxT[2]) > IR(vMaxT[1]) )
			{
				nWhichPlane = 2;
			}
			else
			{
				nWhichPlane = 1;
			}
		}
	}
	else
	{
		// T[0] > 0
		if(IR(vMaxT[1]) & 0x80000000)
		{
			// T[0] > 0, T[1] < 0
			if(IR(vMaxT[2]) & 0x80000000) // T[0] > 0, T[1] < 0, T[2] < 0				
				nWhichPlane = 0;
			else                          // T[0] > 0, T[1] < 0, T[2] > 0
			{
				if(IR(vMaxT[2]) > IR(vMaxT[0]) )
					nWhichPlane = 2;
				else
					nWhichPlane = 0;
			}
		}
		else if (IR(vMaxT[2]) & 0x80000000)
		{
			// T[0] > 0, T[1] > 0, T[2] < 0
			if(IR(vMaxT[1]) > IR(vMaxT[0]))
				nWhichPlane = 1;
			else
				nWhichPlane = 0;
		}
		else
		{
			// T[0] > 0, T[1] > 0, T[2] > 0
			nWhichPlane = 0;
			if( IR(vMaxT[1]) > IR(vMaxT[0]) )  nWhichPlane = 1;
			if( IR(vMaxT[2]) > IR(vMaxT[nWhichPlane]) )  nWhichPlane = 2;
		}
	}
	if(nWhichPlane != 0)                                                             
	{                                                                                
		float tmp = ray.vPos.x + ray.vDir.x * vMaxT[nWhichPlane];                  
		if(tmp < m_Min.x - FM_EPSILON || tmp > m_Max.x + FM_EPSILON) return false;   
	}
	if(nWhichPlane != 1)
	{	
		float tmp = ray.vPos.y + ray.vDir.y * vMaxT[nWhichPlane];                  
		if(tmp < m_Min.y - FM_EPSILON || tmp > m_Max.y + FM_EPSILON) return false;   
	}
	if(nWhichPlane != 2)
	{
		float tmp = ray.vPos.z + ray.vDir.z * vMaxT[nWhichPlane];                  
		if(tmp < m_Min.z - FM_EPSILON || tmp > m_Max.z + FM_EPSILON) return false;   
	}

	if(pfDist)
		*pfDist = vMaxT[nWhichPlane];
	return true; // ray hits box
}

bool
SSERenderPipeline::testCollision(const Ray &ray, GBoundingBox& bbox ) const
{
	bool bInside = true;

	const GVector m_Min(bbox.m_Min);
	const GVector m_Max(bbox.m_Max);
	GVector vMaxT(-1.0f, -1.0f, -1.0f);

	// Find candidate planes.	
	if(ray.vPos.x < m_Min.x)														
	{																				
		// Calculate T distances to candidate planes 
		if(ray.vDir.x != 0.0f)	vMaxT.x = (m_Min.x - ray.vPos.x) / ray.vDir.x;	
		bInside		= false;														
	}																				
	else if(ray.vPos.x > m_Max.x)                                                  
	{                                                                               
		// Calculate T distances to candidate planes 
		if(ray.vDir.x != 0.0f) vMaxT.x = (m_Max.x - ray.vPos.x) / ray.vDir.x;   
		bInside = false;                                                            
	}
	if(ray.vPos.y < m_Min.y)														 
	{																				 
		// Calculate T distances to candidate planes 
		if(ray.vDir.y != 0.0f)	vMaxT.y = (m_Min.y - ray.vPos.y) / ray.vDir.y;	 
		bInside		= false;														 
	}																				 
	else if(ray.vPos.y > m_Max.y)                                                   
	{                                                                                
		// Calculate T distances to candidate planes 
		if(ray.vDir.y != 0.0f) vMaxT.y = (m_Max.y - ray.vPos.y) / ray.vDir.y;    
		bInside = false;                                                             
	}
	if(ray.vPos.z < m_Min.z)														 
	{																				 
		// Calculate T distances to candidate planes 
		if(ray.vDir.z != 0.0f)	vMaxT.z = (m_Min.z - ray.vPos.z) / ray.vDir.z;	 
		bInside		= false;														 
	}																				 
	else if(ray.vPos.z > m_Max.z)                                                   
	{                                                                                
		// Calculate T distances to candidate planes 
		if(ray.vDir.z != 0.0f) vMaxT.z = (m_Max.z - ray.vPos.z) / ray.vDir.z;    
		bInside = false;                                                             
	}

	// Ray origin bInside bounding box
	if(bInside)
	{		
		return true;
	}
	// Get largest of the vMaxT's for final choice of intersection
	// - this version without FPU compares
	// - but branch prediction might suffer
	// - a bit faster on my Celeron, duno how it behaves on something like a P4
	unsigned int nWhichPlane;
	if(IR(vMaxT[0]) & 0x80000000)
	{
		// T[0] < 0
		if(IR(vMaxT[1]) & 0x80000000)
		{
			// T[0] < 0, T[1] < 0
			if(IR(vMaxT[2]) & 0x80000000)
				// T[0] < 0, T[1] < 0, T[2] < 0
				return false;
			else
				nWhichPlane = 2;
		}
		else if (IR(vMaxT[2]) & 0x80000000)
		{
			// T[0] < 0, T[1] > 0, T[2] < 0
			nWhichPlane = 1;
		}
		else
		{
			// T[0] < 0, T[1] > 0, T[2] > 0
			if( IR(vMaxT[2]) > IR(vMaxT[1]) )
			{
				nWhichPlane = 2;
			}
			else
			{
				nWhichPlane = 1;
			}
		}
	}
	else
	{
		// T[0] > 0
		if(IR(vMaxT[1]) & 0x80000000)
		{
			// T[0] > 0, T[1] < 0
			if(IR(vMaxT[2]) & 0x80000000) // T[0] > 0, T[1] < 0, T[2] < 0				
				nWhichPlane = 0;
			else                          // T[0] > 0, T[1] < 0, T[2] > 0
			{
				if(IR(vMaxT[2]) > IR(vMaxT[0]) )
					nWhichPlane = 2;
				else
					nWhichPlane = 0;
			}
		}
		else if (IR(vMaxT[2]) & 0x80000000)
		{
			// T[0] > 0, T[1] > 0, T[2] < 0
			if(IR(vMaxT[1]) > IR(vMaxT[0]))
				nWhichPlane = 1;
			else
				nWhichPlane = 0;
		}
		else
		{
			// T[0] > 0, T[1] > 0, T[2] > 0
			nWhichPlane = 0;
			if( IR(vMaxT[1]) > IR(vMaxT[0]) )  nWhichPlane = 1;
			if( IR(vMaxT[2]) > IR(vMaxT[nWhichPlane]) )  nWhichPlane = 2;
		}
	}
	if(nWhichPlane != 0)                                                             
	{                                                                                
		float tmp = ray.vPos.x + ray.vDir.x * vMaxT[nWhichPlane];                  
		if(tmp < m_Min.x - FM_EPSILON || tmp > m_Max.x + FM_EPSILON) return false;   
	}
	if(nWhichPlane != 1)
	{	
		float tmp = ray.vPos.y + ray.vDir.y * vMaxT[nWhichPlane];                  
		if(tmp < m_Min.y - FM_EPSILON || tmp > m_Max.y + FM_EPSILON) return false;   
	}
	if(nWhichPlane != 2)
	{
		float tmp = ray.vPos.z + ray.vDir.z * vMaxT[nWhichPlane];                  
		if(tmp < m_Min.z - FM_EPSILON || tmp > m_Max.z + FM_EPSILON) return false;   
	}
	
	return true; // ray hits box
}

bool
SSERenderPipeline::RayBoxTest(Ray &ray, GBoundingBox& bbox ) const
{
	float Tnear = -FLOAT_MAX;
	float Tfar  = FLOAT_MAX;
	float Xl = bbox.m_Min.x;
	float Xh = bbox.m_Max.x;

	float T1 = (Xl-ray.vPos.x) / ray.vDir.x;
	float T2 = (Xh-ray.vPos.x) / ray.vDir.x;

	if(T1 > T2) swap(T1, T2);
	if(T1 > Tnear) Tnear = T1;
	if(T2 < Tfar) Tfar = T2;

	if(Tnear > Tfar) return false;
	if(Tfar < 0) return false;

	// y에 대해서도
	float Yl = bbox.m_Min.y;
	float Yh = bbox.m_Max.y;

	T1 = (Yl-ray.vPos.y) / ray.vDir.y;
	T2 = (Yh-ray.vPos.y) / ray.vDir.y;

	if(T1 > T2) swap(T1, T2);
	if(T1 > Tnear) Tnear = T1;
	if(T2 < Tfar) Tfar = T2;

	if(Tnear > Tfar) return false;
	if( Tfar < 0) return false;
	
	// z에 대해서도
	float Zl = bbox.m_Min.z;
	float Zh = bbox.m_Max.z;

	T1 = (Zl-ray.vPos.z) / ray.vDir.z;
	T2 = (Zh-ray.vPos.z) / ray.vDir.z;

	if(T1 > T2) swap(T1, T2);
	if(T1 > Tnear) Tnear = T1;
	if(T2 < Tfar) Tfar = T2;

	if(Tnear > Tfar) return false;
	if(Tfar < 0) return false;	

	return true;	
}

bool
SSERenderPipeline::intersect_ray_bbox( Ray *ray,GBoundingBox& bbox )
{
	float t0, t1;
	float invraydir;
	float neart, fart;
	float tmp;
	const GVector *dir, *eye;
	float *bmin;
	float *bmax;  	

	bmin = bbox.m_Min.m_Vector;
	bmax = bbox.m_Max.m_Vector;

	dir = &(ray->vDir); eye = &(ray->vPos);

	t0 = 0.0f; t1 = (float)FLOAT_MAX;

	if (dir->x == 0.0 && dir->y == 0.0 && dir->z == 0.0) return 0;

	if (dir->x != 0.0) {
		invraydir = 1.0f / dir->x;

		neart = (bmin[0] - eye->x) * invraydir;
		fart  = (bmax[0] - eye->x) * invraydir;
		if (neart > fart) {
			tmp = neart; neart = fart; fart = tmp;
		}

		t0 = neart > t0 ? neart : t0;
		t1 = fart  < t1 ? fart  : t1;
		if (t0 > t1) return 0;
	}

	if (dir->y != 0.0) {
		invraydir = 1.0f / dir->y;

		neart = (bmin[1] - eye->y) * invraydir;
		fart  = (bmax[1] - eye->y) * invraydir;
		if (neart > fart) {
			tmp = neart; neart = fart; fart = tmp;
		}

		t0 = neart > t0 ? neart : t0;
		t1 = fart  < t1 ? fart  : t1;
		if (t0 > t1) return 0;
	}

	if (dir->z != 0.0) {
		invraydir = 1.0f / dir->z;

		neart = (bmin[2] - eye->z) * invraydir;
		fart  = (bmax[2] - eye->z) * invraydir;
		if (neart > fart) {
			tmp = neart; neart = fart; fart = tmp;
		}

		t0 = neart > t0 ? neart : t0;
		t1 = fart  < t1 ? fart  : t1;
		if (t0 > t1) return 0;
	}
	return 1;
}

bool
SSERenderPipeline::intersect_ray_bbox( Ray &ray,GBoundingBox& bbox )
{
	float t0, t1;
	float invraydir;
	float neart, fart;
	float tmp;

	float *bmin;
	float *bmax;  	

	bmin = bbox.m_Min.m_Vector;
	bmax = bbox.m_Max.m_Vector;

	t0 = 0.0f; t1 = FLOAT_MAX;

	if (ray.vDir.x == 0.0 && ray.vDir.y == 0.0 && ray.vDir.z == 0.0) return false;

	//if (ray.vDir.x != 0.0) {
		invraydir = 1.0f / ray.vDir.x;

		neart = (bmin[0] - ray.vPos.x) * invraydir;
		fart  = (bmax[0] - ray.vPos.x) * invraydir;
		if (neart > fart) {
			tmp = neart; neart = fart; fart = tmp;
		}

		t0 = neart > t0 ? neart : t0;
		t1 = fart  < t1 ? fart  : t1;
		if (t0 > t1) return false;
	//}


	//if (ray.vDir.y != 0.0) {
		invraydir = 1.0f / ray.vDir.y;

		neart = (bmin[1] - ray.vPos.y) * invraydir;
		fart  = (bmax[1] - ray.vPos.y) * invraydir;
		if (neart > fart) {
			tmp = neart; neart = fart; fart = tmp;
		}

		t0 = neart > t0 ? neart : t0;
		t1 = fart  < t1 ? fart  : t1;
		if (t0 > t1) return false;
	//}


	//if (ray.vDir.z != 0.0) {
		invraydir = 1.0f / ray.vDir.z;

		neart = (bmin[2] - ray.vPos.z) * invraydir;
		fart  = (bmax[2] - ray.vPos.z) * invraydir;
		if (neart > fart) {
			tmp = neart; neart = fart; fart = tmp;
		}

		t0 = neart > t0 ? neart : t0;
		t1 = fart  < t1 ? fart  : t1;
		if (t0 > t1) return false;
	//}
	
	return true;
}
//
// ---------------------------------------------------
// RAY - AABB INTERSECTION TEST
// ---------------------------------------------------

//An Efficient and Robust
//Ray-Box Intersection Algorithm

//http://delivery.acm.org/10.1145/1200000/1198748/a9-williams.pdf?key1=1198748&key2=4939435621&coll=GUIDE&dl=GUIDE&CFID=74847723&CFTOKEN=25985939
//

bool 
SSERenderPipeline::intersect_bbox(const Ray& ray, GBoundingBox& bbox, float* pfDist) const
{		
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	
	GVector bounds[2];
	bounds[0] = bbox.m_Min;;
	bounds[1] = bbox.m_Max;

	tmin = (bounds[ray.sign[0]].x - ray.vPos.x) * ray.inv_direction_x;
	tmax = (bounds[1-ray.sign[0]].x - ray.vPos.x) * ray.inv_direction_x;

	tymin = (bounds[ray.sign[1]].y - ray.vPos.y) * ray.inv_direction_y;
	tymax = (bounds[1-ray.sign[1]].y - ray.vPos.y) * ray.inv_direction_y;

	if ( (tmin > tymax) || (tymin > tmax) )
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bounds[ray.sign[2]].z - ray.vPos.z) * ray.inv_direction_z;
	tzmax = (bounds[1-ray.sign[2]].z - ray.vPos.z) * ray.inv_direction_z;

	if ( (tmin > tzmax) || (tzmin > tmax) )
		return false;
	if( tzmin > tmin )
		tmin = tzmin;
	//if( tzmax < tmax )
	//	tmax = tzmax;

	if(pfDist)
		*pfDist = tmin;

	return true;
}


// ---------------------------------------------------
// RAY - TRIANGLE INTERSECTION TEST
// ---------------------------------------------------
//! test if the ray intersects the triangle polygon specified. if the ray intersects, store its info in TMIntCandidate struct
/*!
	@param pCan
		CAN BE NULL! <br>
		pointer to the TMIntCandidate buffer, where the intersection info will be stored.
	@param pT
		pointer to the REAL buffer, where the distance from ray origin to the intersection point will be stored
	@param pFirstIndex
		pointer to the first index of the polygon.
	@param ray
		the ray to check intersection against.
	@return
        - @b true  : the ray intersect the polygon.
        - @b false : the ray DOES NOT intersect the polygon.
	@par Implementation Note:
		only accesses hot storage.
*/ 
// RAY_OBJECT INTERSECTION : TRIANGLE


bool 
SSERenderPipeline::testIntersection(TMIntCandidate* pCan, float* pT, const unsigned int* pFirstIndex, const _sse_1x1_raypacket& ray, float maxt)
{
	// intersection point in barycentric coordinates
	GVector vBaryP;

	// fetch position vectors
	const unsigned int* pIndex = pFirstIndex;
	
//////////////////
	GTriangleWrapper* TriData;
///////////////////
	GPoint p0, p1, p2;	
	TriData = m_Data->m_TriObjList[*pFirstIndex];
	TriData->getPoint(p0, p1, p2);

	GVector vEdge1 = p1 - p0;
	GVector vEdge2 = p2 - p0;

	// begin calculating determinant - also used to calculate U parameter
	GVector vDir(ray.d.x, ray.d.y, ray.d.z);
	GVector vP = vDir.outerProduct(vEdge2); // d x e2

	// if determinant is near zero, ray lies in plane of triangle
	float det = vEdge1.innerProduct(vP);           // (d x e2) dot e1

	//if(-FM_EPSILON < det && det < FM_EPSILON )
	//	return false;

	float inv_det = 1.0f / det;

	// calculate distance from vert0 to ray origin
	GVector vPos(ray.o.x, ray.o.y, ray.o.z);
	GVector vT = vPos - p0;          // s vector(o - p0)

	// calculate U parameter and test bounds
	vBaryP[0] = vT.innerProduct(vP) * inv_det;     //  s dot (d x e2) * inv_det
	if(vBaryP[0] < 0.0f || 1.0f < vBaryP[0])
		return false;

	// prepare to test V prarameter
	GVector vQ = vT.outerProduct(vEdge1);       // s x e1

	// calculate V parameter and test bounds
	vBaryP[1] = vDir.innerProduct(vQ) * inv_det; // d dot (s x e1) * inv_det
	if(vBaryP[1] < 0.0f || 1.0f < vBaryP[0] + vBaryP[1])
		return false;

	// calculate t, ray intersects triangle
	float t = vEdge2.innerProduct(vQ) * inv_det;   // e2 dot (s x e1) * inv_det

	if(t < LOCAL_EPSILON || maxt < t)
		return false;

	if(pT != 0)
		*pT = t;

	if(pCan != 0)
	{
		pCan->pFirstIndex = pFirstIndex;
		pCan->vBaryP = vBaryP;
		pCan->t = t;
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////////////
// RAY - TRI intersection test
//////////////////////////////////////////////////////////////////////////////////
static const unsigned int modulo[] =  {0,1,2,0,1};
#define ku modulo[k+1]
#define kv modulo[k+2]
bool SSERenderPipeline::Split_Isect1x1_PlaneTest_PriRay( TriAccel &acc, int nIdx, float &f, float t )
{
	_sse_1x1_raypacket	*rp	= &m_RayPk1x1[nIdx];
	//_sse_1x1_isect		*is	= &m_Isect1x1[nIdx];

	const unsigned int k	= acc.k;

	float nd;
	nd = 1.0f / (rp->d.f[k]
			+ acc.n_u * rp->d.f[ku] + acc.n_v * rp->d.f[kv]);
	f  = acc.n_d - (rp->o.f[k]
			+ acc.n_u * rp->o.f[ku] + acc.n_v * rp->o.f[kv]);
	f = f * nd;

	if (!(t >= f && f > EPSILON)) return false;	// eps < f <= Hit4.dist

	return true;
}

bool 
SSERenderPipeline::BVH_Split_Isect1x1_PlaneTest_PriRay( _sse_1x1_raypacket& ray, TriAccel &acc, float &f, float t )
{
	const unsigned int k	= acc.k;

	float nd;
	nd = 1.0f / (ray.d.f[k]
			+ acc.n_u * ray.d.f[ku] + acc.n_v * ray.d.f[kv]);
	f  = acc.n_d - (ray.o.f[k]
			+ acc.n_u * ray.o.f[ku] + acc.n_v * ray.o.f[kv]);
	f = f * nd;

	if (!(t >= f && f > EPSILON)) return false;	// eps < f <= Hit4.dist

	return true;
}

bool
SSERenderPipeline::BVH_Split_Isect1x1_TriUVTest_PriRay( _sse_1x1_raypacket& ray, TriAccel &acc, float &f, float &lambda, float &mue )
{
	const unsigned int k	= acc.k;

	float hu, hv;
	hu = ray.o.f[ku] + f * ray.d.f[ku];
	hv = ray.o.f[kv] + f * ray.d.f[kv];

	lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
	if (lambda < 0.0f) return false;

	mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;	
	if (mue    < 0.0f) return false;

	if (lambda+mue > 1.0f) return false;

	return true;
}

//////////////////////////////////////////////////////////////////////////////////
// SHADOWRAY - TRI intersection test
//////////////////////////////////////////////////////////////////////////////////
bool 
SSERenderPipeline::temp_Split_Isect1x1_PlaneTest_ShwRay( TriAccel &acc, float &f )
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

bool
SSERenderPipeline::temp_Split_Isect1x1_TriUVTest_ShwRay( TriAccel &acc, float &f, float &lambda, float &mue )
{
	_sse_1x1_raypacket	*rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*is	= &m_ShadowIsect1x1[0];

	const unsigned int k	= acc.k;

	float hu, hv;
	hu = rp->o.f[ku] + f * rp->d.f[ku];
	hv = rp->o.f[kv] + f * rp->d.f[kv];

	lambda = hu * acc.b_nu + hv * acc.b_nv + acc.b_d;
	if (lambda < 0.0f) return false;

	mue    = hu * acc.c_nu + hv * acc.c_nv + acc.c_d;	
	if (mue    < 0.0f) return false;

	if (lambda+mue > 1.0f) return false;

	return true;
}




int
SSERenderPipeline::getFirstHit(GBVH_RayPacket& prays, GBoundingBox& aabb, const unsigned int &first)
{	
	// ----------------------------------------------
	// !!! 먼저 EARLY HIT CHECK
	// First : Quick 'hit' test using 'first' ray.	
	Ray first_ray;
	prays.getRay(first_ray, first);	
	//float fDist = FLOAT_MAX;	
	//if( testCollision(first_ray, aabb, &fDist) )
	//if( testCollision(first_ray, aabb, &fDist) && fDist < prays.t[first])
	//if( intersect_bbox(first_ray, aabb, &fDist) && fDist < FLOAT_MAX )
	//	return first;
	//if( testCollision(first_ray, aabb ) )
	//	return first;
	if( intersect_ray_bbox(first_ray, aabb ) )
		return first;				
	// !!! EARLY HIT CHECK END ----------------------	
	

	// --------------------------------------------------------------------------------
	// FRUSTUM - AABB intersect check!!
	// --------------------------------------------------------------------------------	
	//if( prays.frustumAABBTest(aabb) )
	//	return PACKET_SIZE;
	if( prays.frustumAABBTest2(aabb) )
		return PACKET_SIZE;
	/*
	if( prays.frustumAABBTest2(aabb) )
		printf("false");

	if( prays.frustumAABBTest3(aabb) == false )
		return PACKET_SIZE;
	*/
	// --------------------------------------------------------------------------------
	// END FRUSTUM - AABB intersect check!!
	// --------------------------------------------------------------------------------

	// --------------------------------------------------------------------------------
	// AABB & FRUSTUM 교차 한다는 것.
	for(int i=first+1; i<PACKET_SIZE; ++i)
	{			
		prays.getRay(first_ray, i);
		//if( testCollision(first_ray, aabb, &fDist) && fDist < FLOAT_MAX )
		//if( testCollision(first_ray, aabb, &fDist) && fDist < prays.t[i] )

		if( testCollision(first_ray, aabb) )
			return i;			
		// 왜 이건 안빨라질까? ㅡㅡ;;;
		//if( intersect_ray_bbox(first_ray, aabb) )
		//	return i;
		
		/* // 이거로 하려면 inv, sign 계산해야 함.
		Ray temp_ray = prays.getRayMore( i );		
		if( intersect_bbox(temp_ray, aabb, &fDist) && fDist < prays.t[i] && fDist < FLOAT_MAX )
			return i;			
		*/
	}	

	return PACKET_SIZE;
}

int
SSERenderPipeline::getLastHit(GBVH_RayPacket& prays, GBoundingBox& aabb, unsigned int &first)
{
	float fDist;
	//_sse_1x1_raypacket r;
	//GVector temp;
	
	Ray temp_ray;	
	for(unsigned int last = PACKET_SIZE-1; last>first; --last)
	{		
		prays.getRay(temp_ray, last);
		if( testCollision(temp_ray, aabb, &fDist) && fDist < FLOAT_MAX )
			return (last+1);		
		/*
		temp = prays.getDirection( last );
		r.d.x = temp.x;
		r.d.y = temp.y;
		r.d.z = temp.z;
		temp = prays.getOrigin( last );
		r.o.x = temp.x;
		r.o.y = temp.y;
		r.o.z = temp.z;
		if( testCollision(r, aabb, &fDist) && fDist< FLOAT_MAX )
		//if( aabb.intersect_bbox(prays.getRayMore(last), &fDist) && fDist< FLOAT_MAX )
			return (last+1);
			*/
			
	}
	return (first+1);
}


// ---------------------------------------------------------------------------
// Generation Shadow Ray
// ---------------------------------------------------------------------------
void SSERenderPipeline::BVH_Shading1x1_RayGeneration_ShwRay(const GPoint* objectPos, const GPoint* lightPos) {

	_sse_1x1_raypacket	*shadow_rp	= &m_ShadowRayPk1x1[0];
	_sse_1x1_isect		*shadow_is	= &m_ShadowIsect1x1[0];

	_sse_float oPos = cpu_fset1(objectPos->data);
	_sse_float lPos = cpu_fset1(lightPos->data);

	shadow_rp->d = cpu_fsub(lPos, oPos);
	Split_InitPkt1x1_ShwRay(0);
	shadow_rp->o = cpu_fadd(oPos, cpu_fmul(shadow_rp->d, RAY_START_EPSILON));

	//Split_Trace1x1__ShwRay(q, 0);
	Render1x1_BVH_ShwRayTrace(0);
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------
void SSERenderPipeline::BVH_Shading1x1 (int nIdx)
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
		BVH_Shading1x1__LocalShading(nIdx, hit_p, global_ambient, mat_cAmbt, mat_cDiff, mat_cSpec, mat_cEmit,
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


// ---------------------------------------------------------------------------
// Local Shading
// ---------------------------------------------------------------------------
void SSERenderPipeline::BVH_Shading1x1__LocalShading (const int nIdx, _sse_float &hit_p,
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
				BVH_Shading1x1_RayGeneration_ShwRay(&hitP, &lightPos);

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
			} 
			else {
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