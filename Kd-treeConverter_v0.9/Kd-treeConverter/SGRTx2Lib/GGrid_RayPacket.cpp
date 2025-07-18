#include"GGridStructure.h"
#include"GGridConstants.h"
#include"GGrid_RayPacket.h"
#include"GVector.h"
#include<math.h>

GGrid_RayPacket::GGrid_RayPacket(void){
}
GGrid_RayPacket::~GGrid_RayPacket(void){

}

bool GGrid_RayPacket::early_termination(int lv){
//	bool check = true;

	for(int y = 0;y < packetSize;y++){
		for(int x = 0;x < packetSize;x++){
			//한번도 hit 안했거나.
			if(m_tmin[y][x] == kHugeValue || m_tmin[y][x] > m_t_far[y][x]) return false;// check = false;					

			if(m_kVector < 3){
				if(m_hitPacketLv[y][x] == -1 || m_hitPacketLv[y][x] >= lv) return false;// check = false;
			}
			else{
				if(m_hitPacketLv[y][x] == -1 || m_hitPacketLv[y][x] <= lv) return false;// check = false;					
			}			
		}
	}
	return true;
//	return check;
}

void GGrid_RayPacket::tmin_init(void){
	for(int y = 0;y < packetSize;y++){
		for(int x = 0;x < packetSize;x++){
			m_tmin[y][x] = kHugeValue;
			m_hitPacketLv[y][x] = -1;
		}
	}
}

void GGrid_RayPacket::set_majorAxis(GVector centerV){
	float xV = fabs(centerV.x); float yV = fabs(centerV.y); float zV = fabs(centerV.z);

	if(xV >= yV && xV >= zV){
		if(centerV.x >= 0) m_kVector = PLUS_X;
		else m_kVector = MINUS_X;
	}
	else if(yV >= xV && yV >= zV){
		if(centerV.y >= 0) m_kVector = PLUS_Y;
		else m_kVector = MINUS_Y;
	}
	else{
		if(centerV.z >= 0) m_kVector = PLUS_Z;
		else m_kVector = MINUS_Z;
	}
}

void GGrid_RayPacket::set_majorAxis(int pi, int pj, GVector LeftUp, GVector DX, GVector DY, GVector _rO){
	GVector centerV = LeftUp + (DX * (float)(pi * packetSize)) - (DY * (float)(pj * packetSize));

	centerV = centerV - _rO;
	centerV = centerV.normalize();

	//K vector를 찾는다.
	if(centerV.x >= centerV.y && centerV.x >= centerV.z && centerV.x > 0) m_kVector = PLUS_X;//+x	
	else if(centerV.y >= centerV.x && centerV.y >= centerV.z && centerV.y > 0) m_kVector = PLUS_Y;//+y	
	else if(centerV.z >= centerV.x && centerV.z >= centerV.y && centerV.z > 0) m_kVector = PLUS_Z;//+z	
	else if(centerV.x < centerV.y && centerV.x < centerV.z && centerV.x < 0) m_kVector = MINUS_X;//-x
	else if(centerV.y < centerV.x && centerV.y < centerV.z && centerV.y < 0) m_kVector = MINUS_Y;//-y
	else m_kVector = MINUS_Z;//-z
}

void GGrid_RayPacket::packet_increase(int size){
	m_rayPacket[0][0][0] += (m_rayIncrease[0][0][0] * size);
	m_rayPacket[0][0][1] += (m_rayIncrease[0][0][1] * size);
	m_rayPacket[0][1][0] += (m_rayIncrease[0][1][0] * size);
	m_rayPacket[0][1][1] += (m_rayIncrease[0][1][1] * size);
	m_rayPacket[1][0][0] += (m_rayIncrease[1][0][0] * size);
	m_rayPacket[1][0][1] += (m_rayIncrease[1][0][1] * size);
	m_rayPacket[1][1][0] += (m_rayIncrease[1][1][0] * size);
	m_rayPacket[1][1][1] += (m_rayIncrease[1][1][1] * size);
}

void GGrid_RayPacket::packet_decrease(int size){
	m_rayPacket[0][0][0] -= (m_rayIncrease[0][0][0] * size);
	m_rayPacket[0][0][1] -= (m_rayIncrease[0][0][1] * size);
	m_rayPacket[0][1][0] -= (m_rayIncrease[0][1][0] * size);
	m_rayPacket[0][1][1] -= (m_rayIncrease[0][1][1] * size);
	m_rayPacket[1][0][0] -= (m_rayIncrease[1][0][0] * size);
	m_rayPacket[1][0][1] -= (m_rayIncrease[1][0][1] * size);
	m_rayPacket[1][1][0] -= (m_rayIncrease[1][1][0] * size);
	m_rayPacket[1][1][1] -= (m_rayIncrease[1][1][1] * size);
}

void GGrid_RayPacket::ray_generate(int pi, int pj, GVector LeftUp, GVector DX, GVector DY, GVector _rO){
	//ray 생성.
	for(int b = 0;b < packetSize;b++){
		for(int a = 0;a < packetSize;a++){
			m_r[b][a] = LeftUp + (DX * (float)(pi * packetSize + a)) - (DY * (float)(pj * packetSize + b));
			m_r[b][a] = m_r[b][a] - _rO;
			m_r[b][a] = m_r[b][a].normalize();
		}
	}	
	set_majorAxis(m_r[0][0]);
}

void GGrid_RayPacket::set_packet(GVector _rO, GGridStructure *pGrid){
	float kS, kE, rLUTS, rLUTE, rRUTS, rRUTE, rLDTS, rLDTE, rRDTS, rRDTE;
	GBoundingBox* gridBounding = pGrid->getBBox();
	float sliceL = pGrid->get_sliceL();
	int kStart = pGrid->get_kStart();
	int kEnd = pGrid->get_kEnd();

	tmin_init();

	int tag = packetSize - 1;

	float rLUdx = m_r[0][0].x; float rLUdy = m_r[0][0].y; float rLUdz = m_r[0][0].z;
	float rRUdx = m_r[0][tag].x; float rRUdy = m_r[0][tag].y; float rRUdz = m_r[0][tag].z;
	float rLDdx = m_r[tag][0].x; float rLDdy = m_r[tag][0].y; float rLDdz = m_r[tag][0].z;
	float rRDdx = m_r[tag][tag].x; float rRDdy = m_r[tag][tag].y; float rRDdz = m_r[tag][tag].z;

	float packetE[2][2][2]; //[0]에 u, [1]에 v.

	switch(m_kVector){
	case PLUS_X: 
		kS = gridBounding->m_Min.x  - _rO.x;
		kE = gridBounding->m_Max.x  - _rO.x;				
		break;
	case MINUS_X:
		kS = gridBounding->m_Max.x  - _rO.x;
		kE = gridBounding->m_Min.x  - _rO.x;
		break;
	case PLUS_Y: 		
		kS = gridBounding->m_Min.y  - _rO.y;
		kE = gridBounding->m_Max.y  - _rO.y;						
		break;
	case MINUS_Y:
		kS = gridBounding->m_Max.y  - _rO.y;
		kE = gridBounding->m_Min.y  - _rO.y;
		break;
	case PLUS_Z: 
		kS = gridBounding->m_Min.z  - _rO.z;
		kE = gridBounding->m_Max.z  - _rO.z;						
		break;
	case MINUS_Z:
		kS = gridBounding->m_Max.z  - _rO.z;
		kE = gridBounding->m_Min.z  - _rO.z;
		break;
	}

	//패킷의 초기 u, v 구하기.
	switch(m_kVector){	
	case PLUS_X: case MINUS_X:
		rLUTS = kS / rLUdx; rLUTE = kE / rLUdx;
		rRUTS = kS / rRUdx; rRUTE = kE / rRUdx;
		rLDTS = kS / rLDdx; rLDTE = kE / rLDdx;
		rRDTS = kS / rRDdx; rRDTE = kE / rRDdx;

		m_rayPacket[0][0][0] = _rO.z + rLUTS * rLUdz; m_rayPacket[0][0][1] = _rO.y + rLUTS * rLUdy;
		m_rayPacket[0][1][0] = _rO.z + rRUTS * rRUdz; m_rayPacket[0][1][1] = _rO.y + rRUTS * rRUdy;
		m_rayPacket[1][0][0] = _rO.z + rLDTS * rLDdz; m_rayPacket[1][0][1] = _rO.y + rLDTS * rLDdy;
		m_rayPacket[1][1][0] = _rO.z + rRDTS * rRDdz; m_rayPacket[1][1][1] = _rO.y + rRDTS * rRDdy;

		packetE[0][0][0] = _rO.z + rLUTE * rLUdz; packetE[0][0][1] = _rO.y + rLUTE * rLUdy;
		packetE[0][1][0] = _rO.z + rRUTE * rRUdz; packetE[0][1][1] = _rO.y + rRUTE * rRUdy;
		packetE[1][0][0] = _rO.z + rLDTE * rLDdz; packetE[1][0][1] = _rO.y + rLDTE * rLDdy;
		packetE[1][1][0] = _rO.z + rRDTE * rRDdz; packetE[1][1][1] = _rO.y + rRDTE * rRDdy;

		break;			
	case PLUS_Y: case MINUS_Y:
		rLUTS = kS / rLUdy; rLUTE = kE / rLUdy;
		rRUTS = kS / rRUdy; rRUTE = kE / rRUdy;
		rLDTS = kS / rLDdy; rLDTE = kE / rLDdy;
		rRDTS = kS / rRDdy; rRDTE = kE / rRDdy;

		m_rayPacket[0][0][0] = _rO.x + rLUTS * rLUdx; m_rayPacket[0][0][1] = _rO.z + rLUTS * rLUdz;
		m_rayPacket[0][1][0] = _rO.x + rRUTS * rRUdx; m_rayPacket[0][1][1] = _rO.z + rRUTS * rRUdz;
		m_rayPacket[1][0][0] = _rO.x + rLDTS * rLDdx; m_rayPacket[1][0][1] = _rO.z + rLDTS * rLDdz;
		m_rayPacket[1][1][0] = _rO.x + rRDTS * rRDdx; m_rayPacket[1][1][1] = _rO.z + rRDTS * rRDdz;

		packetE[0][0][0] = _rO.x + rLUTE * rLUdx; packetE[0][0][1] = _rO.z + rLUTE * rLUdz;
		packetE[0][1][0] = _rO.x + rRUTE * rRUdx; packetE[0][1][1] = _rO.z + rRUTE * rRUdz;
		packetE[1][0][0] = _rO.x + rLDTE * rLDdx; packetE[1][0][1] = _rO.z + rLDTE * rLDdz;
		packetE[1][1][0] = _rO.x + rRDTE * rRDdx; packetE[1][1][1] = _rO.z + rRDTE * rRDdz;

		break;
	case PLUS_Z: case MINUS_Z:
		rLUTS = kS / rLUdz; rLUTE = kE / rLUdz;
		rRUTS = kS / rRUdz; rRUTE = kE / rRUdz;
		rLDTS = kS / rLDdz; rLDTE = kE / rLDdz;
		rRDTS = kS / rRDdz; rRDTE = kE / rRDdz;

		m_rayPacket[0][0][0] = _rO.x + rLUTS * rLUdx; m_rayPacket[0][0][1] = _rO.y + rLUTS * rLUdy;
		m_rayPacket[0][1][0] = _rO.x + rRUTS * rRUdx; m_rayPacket[0][1][1] = _rO.y + rRUTS * rRUdy;
		m_rayPacket[1][0][0] = _rO.x + rLDTS * rLDdx; m_rayPacket[1][0][1] = _rO.y + rLDTS * rLDdy;
		m_rayPacket[1][1][0] = _rO.x + rRDTS * rRDdx; m_rayPacket[1][1][1] = _rO.y + rRDTS * rRDdy;

		packetE[0][0][0] = _rO.x + rLUTE * rLUdx; packetE[0][0][1] = _rO.y + rLUTE * rLUdy;
		packetE[0][1][0] = _rO.x + rRUTE * rRUdx; packetE[0][1][1] = _rO.y + rRUTE * rRUdy;
		packetE[1][0][0] = _rO.x + rLDTE * rLDdx; packetE[1][0][1] = _rO.y + rLDTE * rLDdy;
		packetE[1][1][0] = _rO.x + rRDTE * rRDdx; packetE[1][1][1] = _rO.y + rRDTE * rRDdy;
		
		break;
	}	
	
	float temp;

	//패킷의 증가 값 구하기.
	if(m_kVector < 3){
		temp = 1.0f / (kEnd + 1);

		m_rayIncrease[0][0][0] = (packetE[0][0][0] - m_rayPacket[0][0][0]) * temp; 
		m_rayIncrease[0][0][1] = (packetE[0][0][1] - m_rayPacket[0][0][1]) * temp; 
		m_rayIncrease[0][1][0] = (packetE[0][1][0] - m_rayPacket[0][1][0]) * temp; 
		m_rayIncrease[0][1][1] = (packetE[0][1][1] - m_rayPacket[0][1][1]) * temp; 
		m_rayIncrease[1][0][0] = (packetE[1][0][0] - m_rayPacket[1][0][0]) * temp; 
		m_rayIncrease[1][0][1] = (packetE[1][0][1] - m_rayPacket[1][0][1]) * temp; 
		m_rayIncrease[1][1][0] = (packetE[1][1][0] - m_rayPacket[1][1][0]) * temp; 
		m_rayIncrease[1][1][1] = (packetE[1][1][1] - m_rayPacket[1][1][1]) * temp; 
	}
	else{
		temp = 1.0f / (kStart + 1);

		m_rayIncrease[0][0][0] = (packetE[0][0][0] - m_rayPacket[0][0][0]) * temp; 
		m_rayIncrease[0][0][1] = (packetE[0][0][1] - m_rayPacket[0][0][1]) * temp; 
		m_rayIncrease[0][1][0] = (packetE[0][1][0] - m_rayPacket[0][1][0]) * temp; 
		m_rayIncrease[0][1][1] = (packetE[0][1][1] - m_rayPacket[0][1][1]) * temp; 
		m_rayIncrease[1][0][0] = (packetE[1][0][0] - m_rayPacket[1][0][0]) * temp; 
		m_rayIncrease[1][0][1] = (packetE[1][0][1] - m_rayPacket[1][0][1]) * temp; 
		m_rayIncrease[1][1][0] = (packetE[1][1][0] - m_rayPacket[1][1][0]) * temp; 
		m_rayIncrease[1][1][1] = (packetE[1][1][1] - m_rayPacket[1][1][1]) * temp; 
	}
}

void GGrid_RayPacket::camera_culling(int& k, GGridStructure *pGrid, GVector _rO){
	GBoundingBox* gridBounding = pGrid->getBBox();
	int camP;//packet_increase(1);

	switch(m_kVector){
	case PLUS_X: 
		camP = pGrid->float_to_int((_rO.x - gridBounding->m_Min.x) * pGrid->get_invXL());
		if(camP <= k) return;
		do{
			k++; packet_increase(1);
		}while(camP > k);
		break;
	case MINUS_X:
		camP = pGrid->float_to_int((_rO.x - gridBounding->m_Min.x) * pGrid->get_invXL());		
		if(camP >= k) return;
		do{
			k--; packet_increase(1);
		}while(camP < k);
		break;
	case PLUS_Y: 		
		camP = pGrid->float_to_int((_rO.y - gridBounding->m_Min.y) * pGrid->get_invYL());
		if(camP <= k) return;
		do{
			k++; packet_increase(1);
		}while(camP > k);	
		break;
	case MINUS_Y:
		camP = pGrid->float_to_int((_rO.y - gridBounding->m_Min.y) * pGrid->get_invYL());
		if(camP >= k) return;
		do{
			k--; packet_increase(1);
		}while(camP < k);		
		break;
	case PLUS_Z: 
		camP = pGrid->float_to_int((_rO.z - gridBounding->m_Min.z) * pGrid->get_invZL());
		if(camP <= k) return;
		do{
			k++; packet_increase(1);
		}while(camP > k);	
		break;
	case MINUS_Z:
		camP = pGrid->float_to_int((_rO.z - gridBounding->m_Min.z) * pGrid->get_invZL());
		if(camP >= k) return;
		do{
			k--; packet_increase(1);
		}while(camP < k);		
		break;
	}
}

GGrid_RayPacket& GGrid_RayPacket::operator= (const GGrid_RayPacket& rhs){
	if (this == &rhs) return *this;

	for(int z = 0;z < 2;z++){
		for(int y = 0;y < 2;y++){
			for(int x = 0;x < 2;x++){
				m_rayIncrease[z][y][x] = rhs.m_rayIncrease[z][y][x]; 
				m_rayPacket[z][y][x] = rhs.m_rayPacket[z][y][x]; 
			}
		}
	}

	for(int j = 0;j < packetSize;j++){
		for(int i = 0;i < packetSize;i++){			
			m_hitPacketLv[j][i] = rhs.m_hitPacketLv[j][i];
			m_tmin[j][i] = rhs.m_tmin[j][i];	
			m_r[j][i] = rhs.m_r[j][i];
		}
	}

	m_kVector = rhs.m_kVector;
	m_packetId = rhs.m_packetId;

	return *this;
}