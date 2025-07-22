#include"GGridStructure.h"
#include"GGridIndex.h"
#include"GGridConstants.h"
#include"GClassMacro.h"
#include"GVector.h"
#include"GBase.h"
#include<math.h>

GGridStructure::GGridStructure(GScene* pScene){
	m_pScene = pScene;
}

GGridStructure::~GGridStructure(void){
}

GError GGridStructure::initialize(){	
	GTimer timer;
	timer.start();

	GLogManager::logging( LOG_INFO, "------------------------ GRID Spatial Structure -------------------" );
	GLogManager::logging( LOG_INFO, " -> GRID build started..." );	

	//씬 전체의 삼각형을 읽어 옴.
	m_pSceneTriangleList = m_pScene->createSceneTriangleList(m_gridBounding);
	m_iSceneTriangleCount = m_pSceneTriangleList->size();
	
	//bBox에 여분의 값을 줌.
	m_gridBounding.m_Min.x -= 0.05f; m_gridBounding.m_Min.y -= 0.05f; m_gridBounding.m_Min.z -= 0.05f;
	m_gridBounding.m_Max.x += 0.05f; m_gridBounding.m_Max.y += 0.05f; m_gridBounding.m_Max.z += 0.05f;
	
	GVector d;
	float V;
	float gridTempMinX, gridTempMaxX, gridTempMinY, gridTempMaxY, gridTempMinZ, gridTempMaxZ;

	//우선 grid resolution을 구한다.
	d.x = m_gridBounding.m_Max.x - m_gridBounding.m_Min.x;
	d.y = m_gridBounding.m_Max.y - m_gridBounding.m_Min.y;
	d.z = m_gridBounding.m_Max.z - m_gridBounding.m_Min.z;
	V = d.x * d.y * d.z;

	float temp = pow((lamda * m_iSceneTriangleCount) / V, 0.333333f); //pow(x, y) x에 y승.
	m_Nx = float_to_int(d.x * temp); m_Ny = float_to_int(d.y * temp); m_Nz = float_to_int(d.z * temp);			
	//바운더리 처리.
	if(m_Nx > 250) m_Nx = 250; if(m_Ny > 250) m_Ny = 250; if(m_Nz > 250) m_Nz = 250;

	//grid의 한 cell의 x, y, z축의 length를 구함. 
	//int로 캐스팅 후 연산 하므로 최적화 금지!!
	m_xLength = d.x / m_Nx; m_yLength = d.y / m_Ny; m_zLength = d.z / m_Nz;

	//계산량을 줄이기 위해 미리 계산해 둠.
	m_invXL = 1.0f / m_xLength; m_invYL = 1.0f / m_yLength; m_invZL = 1.0f / m_zLength;	

	//Grid 구성.
	for(int i = 0;i < m_iSceneTriangleCount;i++){		
		//모든 삼각형에 대해 삼각형의 바운딩을 구하고.
		GTriangleWrapper* poly = m_pSceneTriangleList->getTriangleWrapper(i);
		gridTempMinX = ((poly->m_BBox.m_Min.x - m_gridBounding.m_Min.x)) * m_invXL; 
		gridTempMaxX = ((poly->m_BBox.m_Max.x - m_gridBounding.m_Min.x)) * m_invXL;  
		gridTempMinY = ((poly->m_BBox.m_Min.y - m_gridBounding.m_Min.y)) * m_invYL; 
		gridTempMaxY = ((poly->m_BBox.m_Max.y - m_gridBounding.m_Min.y)) * m_invYL; 
		gridTempMinZ = ((poly->m_BBox.m_Min.z - m_gridBounding.m_Min.z)) * m_invZL; 
		gridTempMaxZ = ((poly->m_BBox.m_Max.z - m_gridBounding.m_Min.z)) * m_invZL; 

		//그 바운딩이 포함되는 cell에 삼각형의 아이디를 등록.
		for(int z = float_to_int(gridTempMinZ);z <= float_to_int(gridTempMaxZ);z++){
			for(int y = float_to_int(gridTempMinY);y <= float_to_int(gridTempMaxY);y++){
				for(int x = float_to_int(gridTempMinX);x <= float_to_int(gridTempMaxX);x++){					
					GGridIndex* index = new GGridIndex(i);
					m_gridCells[z][y][x].push_back(index);
				}
			}
		}
	}

	timer.end();

	GLogManager::logging( LOG_INFO, " -> GRID build end." );
	GLogManager::logging( LOG_INFO, " -> GRID Construction Time : %f sec", timer.getElapsedTime() );
	GLogManager::logging( LOG_INFO, "--------------------------------------------------------------------" );

	return errorNo;
}

GError GGridStructure::uninitialize(){
	for(int z = 0;z < 100;z++){
		for(int y = 0;y < 100;y++){
			for(int x = 0;x < 100;x++){					
				m_gridCells[z][y][x].clear();
			}
		}
	}

	return errorNo;
}

int GGridStructure::getTriangleCount(){
	return 0;
}

bool GGridStructure::loadStructureFromFile(const char *filename){
	return false;
}

bool GGridStructure::saveStructureToFile(const char *filename){
	return false;
}

int GGridStructure::float_to_int(float f){
	int i, *tmp;

	if(f >= 1.0f){
		tmp = (int *)&f;
		i = ((*tmp & 0x007fffff) | 0x00800000) >> (150 - ((*tmp & 0x7f800000) >> 23));		
	}
	else i = 0; 	

	return i;
}

void GGridStructure::grid_setup(GGrid_RayPacket rayPacket, GVector _rO){
	switch(rayPacket.get_kVector()){
		case PLUS_X:
			m_sliceL = m_xLength; m_uLength = m_invZL; m_vLength = m_invYL;
			m_Nu = m_Nz; m_Nv = m_Ny; m_Nk = m_Nx;
			m_kStart = 0; m_kEnd = m_Nx - 1;
			break;	
		case MINUS_X:
			m_sliceL = m_xLength; m_uLength = m_invZL; m_vLength = m_invYL;
			m_Nu = m_Nz; m_Nv = m_Ny; m_Nk = m_Nx;
			m_kStart = m_Nx - 1; m_kEnd = 0;		
			break;	
		case PLUS_Y:
			m_sliceL = m_yLength; m_uLength = m_invXL; m_vLength = m_invZL;
			m_Nu = m_Nx; m_Nv = m_Nz; m_Nk = m_Ny;
			m_kStart = 0; m_kEnd = m_Ny - 1;	
			break;	
		case MINUS_Y:
			m_sliceL = m_yLength; m_uLength = m_invXL; m_vLength = m_invZL;
			m_Nu = m_Nx; m_Nv = m_Nz; m_Nk = m_Ny;
			m_kStart = m_Ny - 1; m_kEnd = 0;	
			break;	
		case PLUS_Z:
			m_sliceL = m_zLength; m_uLength = m_invXL; m_vLength = m_invYL;
			m_Nu = m_Nx; m_Nv = m_Ny; m_Nk = m_Nz;
			m_kStart = 0; m_kEnd = m_Nz - 1;	
			break;	
		case MINUS_Z:
			m_sliceL = m_zLength; m_uLength = m_invXL; m_vLength = m_invYL;
			m_Nu = m_Nx; m_Nv = m_Ny; m_Nk = m_Nz;
			m_kStart = m_Nz - 1; m_kEnd = 0;	
			break;	
	}
}

bool GGridStructure::check_macro_cell_tri_num(int kVector, int muS, int muE, int mvS, int mvE, int mkS, int mkE){
	int xS, xE, yS, yE, zS, zE;

	switch(kVector){
	case PLUS_X: case MINUS_X:		
		zS = muS; zE = muE; 
		yS = mvS; yE = mvE; 
		xS = mkS; xE = mkE;
		break;		
	case PLUS_Y: case MINUS_Y:		
		zS = mvS; zE = mvE; 
		yS = mkS; yE = mkE;
		xS = muS; xE = muE; 
		break;
	case PLUS_Z: case MINUS_Z:		
		zS = mkS; zE = mkE; 
		yS = mvS; yE = mvE; 
		xS = muS; xE = muE; 	
		break;
	}

	int cellTriN = 0;

	for(int z = zS;z <= zE;z++){
		for(int y = yS;y <= yE;y++){
			for(int x = xS;x <= xE;x++){
				cellTriN += (int)m_gridCells[z][y][x].size();
			}
		}
	}

	if(cellTriN == 0) return true;

	return false;
}

bool GGridStructure::macro_cell_traversal(int kVector, int& k, GGrid_RayPacket& macroPacket){	
	int muS, muE, mvS, mvE, mkS, mkE;

	if(kVector < 3){
		if(k == macroK && k <= (m_Nk - macroCellSize)){
			macroK += macroCellSize;

			mkS = k; mkE = k + macroCellSize - 1;
			if(mkE > m_Nk - 1) mkE = m_Nk - 1;

			macro_packet_check(macroPacket, muS, muE, mvS, mvE);

			if(check_macro_cell_tri_num(kVector, muS, muE, mvS, mvE, mkS, mkE)){
				k += (macroCellSize - 1); 

				//점프 하기 때문에 뒤에 값을 넣어줌.
				m_lastuS1 = m_lastuS2; m_lastvS1 = m_lastvS2; m_lastuE1 = m_lastuE2; m_lastvE1 = m_lastvE2;
				//

				return true;
			}
		}
	}
	else{		
		if(k == macroK && k >= (macroCellSize - 1)){
			macroK -= macroCellSize;

			mkS = k - macroCellSize + 1; mkE = k;
			if(mkS < 0) mkS = 0;

			macro_packet_check(macroPacket, muS, muE, mvS, mvE);

			if(check_macro_cell_tri_num(kVector, muS, muE, mvS, mvE, mkS, mkE)){
				k -= (macroCellSize - 1); 

				//점프 하기 때문에 뒤에 값을 넣어줌.
				m_lastuS1 = m_lastuS2; m_lastvS1 = m_lastvS2; m_lastuE1 = m_lastuE2; m_lastvE1 = m_lastvE2;
				//

				return true;
			}
		}	
	}

	return false;
}

void GGridStructure::macro_packet_check(GGrid_RayPacket& localPacket, int& muS, int& muE, int& mvS, int& mvE){
	GGrid_RayPacket tempPacket = localPacket;	
	localPacket.packet_increase(macroCellSize);

	float uStart1, uEnd1, vStart1, vEnd1, uStart2, uEnd2, vStart2, vEnd2;
	float pMinU1 = tempPacket.get_minU(); float pMaxU1 = tempPacket.get_maxU(); float pMinV1 = tempPacket.get_minV(); float pMaxV1 = tempPacket.get_maxV();
	float pMinU2 = localPacket.get_minU(); float pMaxU2 = localPacket.get_maxU(); float pMinV2 = localPacket.get_minV(); float pMaxV2 = localPacket.get_maxV();
	float bBoxX0 = m_gridBounding.m_Min.x; float bBoxY0 = m_gridBounding.m_Min.y; float bBoxZ0 = m_gridBounding.m_Min.z;

	switch(localPacket.get_kVector()){
	case PLUS_X: case MINUS_X:
		uStart1 = (pMinU1 - bBoxZ0) * m_uLength; uEnd1 = (pMaxU1 - bBoxZ0) * m_uLength;
		vStart1 = (pMinV1 - bBoxY0) * m_vLength; vEnd1 = (pMaxV1 - bBoxY0) * m_vLength;

		uStart2 = (pMinU2 - bBoxZ0) * m_uLength; uEnd2 = (pMaxU2 - bBoxZ0) * m_uLength;
		vStart2 = (pMinV2 - bBoxY0) * m_vLength; vEnd2 = (pMaxV2 - bBoxY0) * m_vLength;
		break;
	case PLUS_Y: case MINUS_Y:
		uStart1 = (pMinU1 - bBoxX0) * m_uLength; uEnd1 = (pMaxU1 - bBoxX0) * m_uLength; 
		vStart1 = (pMinV1 - bBoxZ0) * m_vLength; vEnd1 = (pMaxV1 - bBoxZ0) * m_vLength; 

		uStart2 = (pMinU2 - bBoxX0) * m_uLength; uEnd2 = (pMaxU2 - bBoxX0) * m_uLength; 
		vStart2 = (pMinV2 - bBoxZ0) * m_vLength; vEnd2 = (pMaxV2 - bBoxZ0) * m_vLength; 
		break;
	case PLUS_Z: case MINUS_Z:
		uStart1 = (pMinU1 - bBoxX0) * m_uLength; uEnd1 = (pMaxU1 - bBoxX0) * m_uLength; 
		vStart1 = (pMinV1 - bBoxY0) * m_vLength; vEnd1 = (pMaxV1 - bBoxY0) * m_vLength; 

		uStart2 = (pMinU2 - bBoxX0) * m_uLength; uEnd2 = (pMaxU2 - bBoxX0) * m_uLength; 
		vStart2 = (pMinV2 - bBoxY0) * m_vLength; vEnd2 = (pMaxV2 - bBoxY0) * m_vLength; 
		break;
	}

	//저장해 놓기.
	m_lastuS1 = uStart1; m_lastvS1 = vStart1; m_lastuE1 = uEnd1; m_lastvE1 = vEnd1;
	m_lastuS2 = uStart2; m_lastvS2 = vStart2; m_lastuE2 = uEnd2; m_lastvE2 = vEnd2;
	//

	if(uStart1 > uStart2) uStart1 = uStart2;
	if(uEnd1 < uEnd2) uEnd1 = uEnd2;
	if(vStart1 > vStart2) vStart1 = vStart2;
	if(vEnd1 < vEnd2) vEnd1 = vEnd2;

	muS = float_to_int(uStart1); muE = float_to_int(uEnd1); mvS = float_to_int(vStart1); mvE = float_to_int(vEnd1);	

	if(muS < 0) muS = 0;
	if(mvS < 0) mvS = 0;

	if(muE > m_Nu - 1) muE = m_Nu - 1;
	if(mvE > m_Nv - 1) mvE = m_Nv - 1;	
}

void GGridStructure::grid_packet_check(GGrid_RayPacket& localPacket, int& uS, int& uE, int& vS, int& vE){
	localPacket.packet_increase(1);	

	float uStart1, uEnd1, vStart1, vEnd1, uStart2, uEnd2, vStart2, vEnd2;
	float pMinU2 = localPacket.get_minU(); float pMaxU2 = localPacket.get_maxU(); float pMinV2 = localPacket.get_minV(); float pMaxV2 = localPacket.get_maxV();
	float bBoxX0 = m_gridBounding.m_Min.x; float bBoxY0 = m_gridBounding.m_Min.y; float bBoxZ0 = m_gridBounding.m_Min.z;

	switch(localPacket.get_kVector()){
	case PLUS_X: case MINUS_X:
		uStart2 = (pMinU2 - bBoxZ0) * m_uLength; uEnd2 = (pMaxU2 - bBoxZ0) * m_uLength;
		vStart2 = (pMinV2 - bBoxY0) * m_vLength; vEnd2 = (pMaxV2 - bBoxY0) * m_vLength;
		break;
	case PLUS_Y: case MINUS_Y:
		uStart2 = (pMinU2 - bBoxX0) * m_uLength; uEnd2 = (pMaxU2 - bBoxX0) * m_uLength; 
		vStart2 = (pMinV2 - bBoxZ0) * m_vLength; vEnd2 = (pMaxV2 - bBoxZ0) * m_vLength; 
		break;
	case PLUS_Z: case MINUS_Z:
		uStart2 = (pMinU2 - bBoxX0) * m_uLength; uEnd2 = (pMaxU2 - bBoxX0) * m_uLength; 
		vStart2 = (pMinV2 - bBoxY0) * m_vLength; vEnd2 = (pMaxV2 - bBoxY0) * m_vLength; 
		break;
	}
	
	//
	uStart1 = m_lastuS1; vStart1 = m_lastvS1; uEnd1 = m_lastuE1; vEnd1 = m_lastvE1;
	m_lastuS1 = uStart2; m_lastvS1 = vStart2; m_lastuE1 = uEnd2; m_lastvE1 = vEnd2;
	//

	if(uStart1 > uStart2) uStart1 = uStart2;
	if(uEnd1 < uEnd2) uEnd1 = uEnd2;
	if(vStart1 > vStart2) vStart1 = vStart2;
	if(vEnd1 < vEnd2) vEnd1 = vEnd2;

	uS = float_to_int(uStart1); uE = float_to_int(uEnd1); vS = float_to_int(vStart1); vE = float_to_int(vEnd1);	

	if(uS < 0) uS = 0;
	if(vS < 0) vS = 0;

	if(uE > m_Nu - 1) uE = m_Nu - 1;
	if(vE > m_Nv - 1) vE = m_Nv - 1;	
}

bool GGridStructure::frustum_culling(GGrid_RayPacket localPacket, int triId, GVector rO){
	float alpha0, alpha1, alpha2, alpha3;
	int tag = packetSize - 1;

	alpha0 = culling_hit(0, triId, localPacket.get_ray(0, 0), rO);
	alpha1 = culling_hit(0, triId, localPacket.get_ray(0, tag), rO);
	alpha2 = culling_hit(0, triId, localPacket.get_ray(tag, 0), rO);
	alpha3 = culling_hit(0, triId, localPacket.get_ray(tag, tag), rO);

	if(alpha0 < 0.0f && alpha1 < 0.0f && alpha2 < 0.0f && alpha3 < 0.0f) return true;

	alpha0 = culling_hit(1, triId, localPacket.get_ray(0, 0), rO);
	alpha1 = culling_hit(1, triId, localPacket.get_ray(0, tag), rO);
	alpha2 = culling_hit(1, triId, localPacket.get_ray(tag, 0), rO);
	alpha3 = culling_hit(1, triId, localPacket.get_ray(tag, tag), rO);

	if(alpha0 < 0.0f && alpha1 < 0.0f && alpha2 < 0.0f && alpha3 < 0.0f) return true;

	alpha0 = culling_hit(2, triId, localPacket.get_ray(0, 0), rO);
	alpha1 = culling_hit(2, triId, localPacket.get_ray(0, tag), rO);
	alpha2 = culling_hit(2, triId, localPacket.get_ray(tag, 0), rO);
	alpha3 = culling_hit(2, triId, localPacket.get_ray(tag, tag), rO);

	if(alpha0 < 0.0f && alpha1 < 0.0f && alpha2 < 0.0f && alpha3 < 0.0f) return true;

	return false;
}

void GGridStructure::t_far_init(int kVector, int u, int v, int k){
	//m_cellBbox
	m_cellBbox.m_Min.x = m_gridBounding.m_Min.x;
	m_cellBbox.m_Min.y = m_gridBounding.m_Min.y;
	m_cellBbox.m_Min.z = m_gridBounding.m_Min.z;

	switch(kVector){
	case PLUS_X: case MINUS_X:		
		m_cellBbox.m_Min.x += k * m_xLength; m_cellBbox.m_Min.y += v * m_yLength; m_cellBbox.m_Min.z += u * m_zLength;
		break;		
	case PLUS_Y: case MINUS_Y:		
		m_cellBbox.m_Min.x += u * m_xLength; m_cellBbox.m_Min.y += k * m_yLength; m_cellBbox.m_Min.z += v * m_zLength;
		break;
	case PLUS_Z: case MINUS_Z:		
		m_cellBbox.m_Min.x += u * m_xLength; m_cellBbox.m_Min.y += v * m_yLength; m_cellBbox.m_Min.z += k * m_zLength;
		break;
	}

	m_cellBbox.m_Max.x = m_cellBbox.m_Min.x + m_xLength;
	m_cellBbox.m_Max.y = m_cellBbox.m_Min.y + m_yLength;
	m_cellBbox.m_Max.z = m_cellBbox.m_Min.z + m_zLength;
}

float GGridStructure::culling_hit(int index, int triId, GVector rD, GVector rO){
	float p0[3], p1[3], p2[3];
	GTriangleWrapper* poly = m_pSceneTriangleList->getTriangleWrapper(triId);

	switch(index){
	case 0:	
		p0[0] = poly->p0[0]; p1[0] = poly->p1[0]; p2[0] = poly->p2[0];
		p0[1] = poly->p0[1]; p1[1] = poly->p1[1]; p2[1] = poly->p2[1];
		p0[2] = poly->p0[2]; p1[2] = poly->p1[2]; p2[2] = poly->p2[2];		
	break;
	case 1:		
		p0[0] = poly->p1[0]; p1[0] = poly->p2[0]; p2[0] = poly->p0[0];
		p0[1] = poly->p1[1]; p1[1] = poly->p2[1]; p2[1] = poly->p0[1];
		p0[2] = poly->p1[2]; p1[2] = poly->p2[2]; p2[2] = poly->p0[2];		
	break;
	case 2:
		p0[0] = poly->p2[0]; p1[0] = poly->p0[0]; p2[0] = poly->p1[0];
		p0[1] = poly->p2[1]; p1[1] = poly->p0[1]; p2[1] = poly->p1[1];
		p0[2] = poly->p2[2]; p1[2] = poly->p0[2]; p2[2] = poly->p1[2];		
	break;
	}

	float a = p0[0] - p1[0], b = p0[0] - p2[0], c = rD.x, d = p0[0] - rO.x; 
	float e = p0[1] - p1[1], f = p0[1] - p2[1], g = rD.y, h = p0[1] - rO.y;
	float i = p0[2] - p1[2], j = p0[2] - p2[2], k = rD.z, l = p0[2] - rO.z;
		
	float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	float q = g * i - e * k, s = e * j - f * i;
	
	float inv_denom  = 1.0f / (a * m + b * q + c * s);
	
	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	float r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;
	
	return 1 - beta - gamma;
}

GError GGridStructure::makeSSERenderStructureInfo(SSESceneData *pSSEData){
	GError error;

	if ( m_pScene->getObjectCount() <= 0 || m_iSceneTriangleCount <= 0 )
		return errorUnknown;

	/**
	 *	Triangle Object List 세팅
	 */
	error = pSSEData->setTriangleObjectList( m_pSceneTriangleList );
	if ( error != errorNo )
		return error;

	/**
	 *	Triangle Accel List 세팅
	 */
	error = pSSEData->buildTriAccList_Barycentric();

	if ( error != errorNo )
		return error;

	return errorNo;
}