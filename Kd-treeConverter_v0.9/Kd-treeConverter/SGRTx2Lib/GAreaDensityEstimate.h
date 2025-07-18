#pragma once

#include "GBase.h"
#include "cudaPhotonMapping.h"
#include "GTriangleWrapperList.h"
#include <vector>
using namespace std;

class GAreaDensityEstimate
{
private:
	vector<cuPhoton*> m_AreaPhotonList;
	int m_iCurrentPhoton;
	int m_iOneIterateMaxPhoton;
	float m_fArea;

public:
	GAreaDensityEstimate( int oneIterateMaxPhoton );
	~GAreaDensityEstimate();

	void clear();

	/**
	 *	삼각형에 대해서 AreaPhoton 생성.
	 */
	void generateAreaPhoton( GTriangleWrapperList *pTriangleList, 
							 int *offset, int *depthInTri, float area );
	int makeAreaPhoton( 
				GPoint &p1, GPoint &p2, GPoint &p3,
				GVector &n0, GVector &n1, GVector &n2,
				float triArea, int totalRow, int startIndexInTri, int maxIndexTri );

	/**
	 *	현재 AreaPhoton 목록 리턴.
	 */
	const vector<cuPhoton*>* getAreaPhoton();

	inline float calArea( GPoint &p1, GPoint &p2, GPoint &p3 );

	inline void calPointInfo( int row, int col, int totalRow, 
				GPoint &p0, GPoint &p1, GPoint &p2,
				GVector &n0, GVector &n1, GVector &n2, 
				GPoint *p, GVector *n );

};