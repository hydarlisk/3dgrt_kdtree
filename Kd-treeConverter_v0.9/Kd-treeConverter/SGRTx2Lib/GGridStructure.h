#ifndef __GRID__
#define __GRID__

//This class about grid structure. by wiseun.

#include<vector>
#include"GBase.h"
#include"GColor.h"
#include"GScene.h"
#include"GGridIndex.h"
#include"GGrid_RayPacket.h"
#include"GBoundingBox.h"
#include"GSpatialStructure.h"
#include"GTriangleWrapperList.h"
#include"SSERenderData.h"

using namespace std;

class GGridStructure : public GSpatialStructure
{
public:
	GGridStructure(GScene* pScene); //생성자.
	virtual ~GGridStructure(void); //소멸자.

	virtual GError makeSSERenderStructureInfo( SSESceneData *pSSEData );
	
	virtual GError initialize(void);
	virtual GError uninitialize(void);
	virtual int getTriangleCount(void);

	virtual bool loadStructureFromFile(const char *filename);
	virtual bool saveStructureToFile(const char *filename);
	
//	void macro_cell_init(void); //macro cell을 사용하기 전에 init하는 함수.

	GTriangleWrapperList* get_tri_list(void);
	GBoundingBox* getBBox(void);
	GBoundingBox* get_cellBbox(void);
	void grid_setup(GGrid_RayPacket rayPacket, GVector _rO);	
	void get_tri_id(int kVector, int u, int v, int k, int n, int& polyIdx);
	float get_sliceL(void);	
	int get_cell_triN(int kVector, int u, int v, int k);
	int get_kStart(void);
	int get_kEnd(void);	
	float get_invXL(void);
	float get_invYL(void);
	float get_invZL(void);

	int float_to_int(float f);
	
	bool check_macro_cell_tri_num(int kVector, int muS, int muE, int mvS, int mvE, int mkS, int mkE);
	void grid_packet_check(GGrid_RayPacket& localPacket, int& uS, int& uE, int& vS, int& vE);
	void macro_packet_check(GGrid_RayPacket& localPacket, int& muS, int& muE, int& mvS, int& mvE);
	bool macro_cell_traversal(int kVector, int& k, GGrid_RayPacket& macroPacket);

	bool frustum_culling(GGrid_RayPacket localPacket, int triId, GVector rO);

	float culling_hit(int index, int triId, GVector rD, GVector rO);
	void t_far_init(int kVector, int u, int v, int k);

	int macroK;
protected:
	GBoundingBox m_gridBounding; //바운딩 박스이고.
	vector<GGridIndex*> m_gridCells[250][250][250];//grid cell의 index. 각 셀에 삼각형의 index를 저장.	gridCells[z][y][x].
	GBoundingBox m_cellBbox;
	
	GScene *m_pScene; //씬정보 등을 보유하는 변수...인가

	int m_iSceneTriangleCount; //삼각형의 수 인 듯.
	GTriangleWrapperList *m_pSceneTriangleList; //삼각형의 리스트 겠지.

	int m_Nx, m_Ny, m_Nz; //grid cell의 수.	
	int m_Nu, m_Nv, m_Nk;
	int m_mNu, m_mNv, m_mNk;
	float m_xLength, m_yLength, m_zLength; //grid cell의 length.	
	float m_invXL, m_invYL, m_invZL; //연산을 위한 값. 한번만 계산하기 위해 여기에 저장.

	float m_sliceL, m_uLength, m_vLength;
	float m_lastuS1, m_lastvS1, m_lastuE1, m_lastvE1;
	float m_lastuS2, m_lastvS2, m_lastuE2, m_lastvE2;
	int m_kStart, m_kEnd;	
};

inline GTriangleWrapperList* GGridStructure::get_tri_list(void){
	return m_pSceneTriangleList;
}

inline void GGridStructure::get_tri_id(int kVector, int u, int v, int k, int n, int& polyIdx){
	//해당 순서의 삼각형의 object id와 triangle id를 return.
	switch(kVector){
	case PLUS_X: case MINUS_X:
		polyIdx = m_gridCells[u][v][k][n]->m_triangleId; 
		break;		
	case PLUS_Y: case MINUS_Y:
		polyIdx = m_gridCells[v][k][u][n]->m_triangleId; 
		break;
	case PLUS_Z: case MINUS_Z:
		polyIdx = m_gridCells[k][v][u][n]->m_triangleId; 
		break;
	}
}

inline float GGridStructure::get_invXL(void){
	return m_invXL;
}

inline float GGridStructure::get_invYL(void){
	return m_invYL;
}

inline float GGridStructure::get_invZL(void){
	return m_invZL;
}

inline int GGridStructure::get_cell_triN(int kVector, int u, int v, int k){
	//grid cell안의 삼각형읠 수를 return.
	switch(kVector){
	case PLUS_X: case MINUS_X:
		return (int)m_gridCells[u][v][k].size();
		break;
	case PLUS_Y: case MINUS_Y:
		return (int)m_gridCells[v][k][u].size();
		break;
	case PLUS_Z: case MINUS_Z:
		return (int)m_gridCells[k][v][u].size();
		break;
	}

	return -1;
}

inline GBoundingBox* GGridStructure::get_cellBbox(void){
	return &m_cellBbox;
}

inline GBoundingBox* GGridStructure::getBBox(void){
	return &m_gridBounding;
}

inline float GGridStructure::get_sliceL(void){
	return m_sliceL;
}

inline int GGridStructure::get_kStart(void){
	return m_kStart;
}

inline int GGridStructure::get_kEnd(void){
	return m_kEnd;
}

#endif