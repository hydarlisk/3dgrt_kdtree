#ifndef __GGRID_RAYPACKET__
#define __GGRID_RAYPACKET__

#include"GPoint.h"
#include"GVector.h"
#include"GGridConstants.h"
#include"GBoundingBox.h"

class GGridStructure;

class GGrid_RayPacket{
public:
	GGrid_RayPacket(void);
	~GGrid_RayPacket(void);

	void set_packetId(int id);
	void set_majorAxis(int pi, int pj, GVector LeftUp, GVector DX, GVector DY, GVector _rO);
	void set_majorAxis(GVector centerV);	
	int get_kVector(void);
	int get_packetId(void);
	void set_packet(GVector _rO, GGridStructure *pGrid);
	void packet_increase(int size);
	void packet_decrease(int size);	
	void ray_generate(int pi, int pj, GVector LeftUp, GVector DX, GVector DY, GVector _rO);
	float get_minU(void);
	float get_maxU(void);
	float get_minV(void);
	float get_maxV(void);
	GVector get_ray(int a, int b);
	GGrid_RayPacket& operator= (const GGrid_RayPacket& rhs);
	void tmin_init(void);
	bool early_termination(int lv);	
	void camera_culling(int& k, GGridStructure *pGrid, GVector _rO);	

	int m_hitPacketLv[packetSize][packetSize];
	float m_tmin[packetSize][packetSize];	
	float m_t_far[packetSize][packetSize];

private:
	GVector m_r[packetSize][packetSize];	
	float m_rayIncrease[2][2][2]; //[0]에 u, [1]에 v.
	float m_rayPacket[2][2][2]; //[0]에 u, [1]에 v.
	int m_kVector;//k vector.
	int m_packetId;
};

inline GVector GGrid_RayPacket::get_ray(int a, int b){
	return m_r[b][a];
}

inline int GGrid_RayPacket::get_packetId(void){
	return m_packetId;
}

inline void GGrid_RayPacket::set_packetId(int id){
	m_packetId = id;
}

inline int GGrid_RayPacket::get_kVector(void){
	return m_kVector;
}

inline float GGrid_RayPacket::get_minU(void){
	float temp;
	temp = m_rayPacket[0][0][0];
	if(temp > m_rayPacket[0][1][0]) temp = m_rayPacket[0][1][0];
	if(temp > m_rayPacket[1][0][0]) temp = m_rayPacket[1][0][0];
	if(temp > m_rayPacket[1][1][0]) temp = m_rayPacket[1][1][0];

	return temp;
}

inline float GGrid_RayPacket::get_maxU(void){
	float temp;
	temp = m_rayPacket[0][0][0];
	if(temp < m_rayPacket[0][1][0]) temp = m_rayPacket[0][1][0];
	if(temp < m_rayPacket[1][0][0]) temp = m_rayPacket[1][0][0];
	if(temp < m_rayPacket[1][1][0]) temp = m_rayPacket[1][1][0];

	return temp;
}

inline float GGrid_RayPacket::get_minV(void){
	float temp;
	temp = m_rayPacket[0][0][1];
	if(temp > m_rayPacket[0][1][1]) temp = m_rayPacket[0][1][1];
	if(temp > m_rayPacket[1][0][1]) temp = m_rayPacket[1][0][1];
	if(temp > m_rayPacket[1][1][1]) temp = m_rayPacket[1][1][1];

	return temp;
}

inline float GGrid_RayPacket::get_maxV(void){
	float temp;
	temp = m_rayPacket[0][0][1];
	if(temp < m_rayPacket[0][1][1]) temp = m_rayPacket[0][1][1];
	if(temp < m_rayPacket[1][0][1]) temp = m_rayPacket[1][0][1];
	if(temp < m_rayPacket[1][1][1]) temp = m_rayPacket[1][1][1];

	return temp;
}

#endif