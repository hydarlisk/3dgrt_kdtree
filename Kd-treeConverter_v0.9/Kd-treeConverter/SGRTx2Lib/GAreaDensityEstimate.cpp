#include "GAreaDensityEstimate.h"
#include <math.h>

GAreaDensityEstimate::GAreaDensityEstimate( int oneIterateMaxPhoton )
{
	m_iOneIterateMaxPhoton = oneIterateMaxPhoton;
	m_iCurrentPhoton = 0;
	m_AreaPhotonList.reserve( oneIterateMaxPhoton + 5 );
}

GAreaDensityEstimate::~GAreaDensityEstimate()
{
	clear();
}

void GAreaDensityEstimate::clear()
{
	for ( int i = 0; i < (int) m_AreaPhotonList.size(); ++i ) {
		if ( m_AreaPhotonList[i] != NULL ) {
			delete m_AreaPhotonList[i];
		}
	}
	m_AreaPhotonList.clear();
}

float GAreaDensityEstimate::calArea( GPoint &p1, GPoint &p2, GPoint &p3 )
{
	GVector v1 = p2 - p1;
	GVector v2 = p3 - p1;

	return abs( v1.outerProduct( v2 ).length() ) / 2.0f;
}

void GAreaDensityEstimate::generateAreaPhoton( 
					GTriangleWrapperList *pTriangleList, 
					int *pTriOffset, int *pNumberInTriangle, float area )
{
	int processedTriangle = 0;

	GTimer timer;
	GTriangleWrapper *pTriObject = NULL;
	GPoint p0, p1, p2;
	GVector n0, n1, n2;
	float triArea, smallArea;
	int size = pTriangleList->size();
	int divCount = 0, index = 0, totalRow = 0;

	timer.start();

	m_fArea = area;
	m_iCurrentPhoton = 0;

	clear();

	/** 삼각형내의 각 삼각형 번호는 1번 부터 시작 */
	if ( *pNumberInTriangle == 0 ) 
		*pNumberInTriangle = 1;

	/** 
	 *	원하는 크기만큼 삼각형을 균일하게 서브디비전하고 삼각형의 중점에
	 *	photon 을 콕 찍는다.
	 */
	for ( int i = *pTriOffset; i < size; ++i ) {

		pTriObject = (*pTriangleList)[ i ];

		p0.setPoint( pTriObject->p0[0], pTriObject->p0[1], pTriObject->p0[2] );
		p1.setPoint( pTriObject->p1[0], pTriObject->p1[1], pTriObject->p1[2] );
		p2.setPoint( pTriObject->p2[0], pTriObject->p2[1], pTriObject->p2[2] );
		n0.setVector( pTriObject->n0[0], pTriObject->n0[1], pTriObject->n0[2] );
		n1.setVector( pTriObject->n1[0], pTriObject->n1[1], pTriObject->n1[2] );
		n2.setVector( pTriObject->n2[0], pTriObject->n2[1], pTriObject->n2[2] );

		triArea = pTriObject->calArea();

		/**
		 *	현재 몇개의 삼각형으로 나누어야 하는지와 total row.
		 */
		divCount = (int) pow( 4, ceil( log( triArea / area ) / log( 4.0 ) ) );
		totalRow = (int) sqrt( (float)divCount );
		smallArea = triArea / (float)divCount;

		/**
		 *	Normal 은 Shading Normal 을 계산해야 한다.
		 */
		*pNumberInTriangle = makeAreaPhoton( p0, p1, p2, n0, n1, n2,
			smallArea, totalRow, *pNumberInTriangle, divCount );

		/**
		 *	makeAreaPhoton 수행후 pOrderOffset == orderCount 이 되면, 삼각형 하나를 완전히
		 *	처리한 것이고, 그렇지 삼각형처리중 샘플링갯수가 제한개수를 넘어서서
		 *	다음번에 처리하려고 iteration 을 빠져나온것.
		 */
		if ( *pNumberInTriangle == divCount + 1 ) {
			(*pTriOffset)++;
			*pNumberInTriangle = 1;
		} else {
			break;
		}
	}

	timer.end();

	GLogManager::logging( LOG_DEBUG, ">> GENERATE AREA PHOTON START area photons = %d, time : %f", 
			m_iCurrentPhoton, timer.getElapsedTime() );
}


int GAreaDensityEstimate::makeAreaPhoton( 
				GPoint &p0, GPoint &p1, GPoint &p2, 
				GVector &n0, GVector &n1, GVector &n2,
				float triArea, int totalRow, int startIndexInTri, int maxIndexTri )
{
	int row = 0, col = 0;
	GPoint tp0, tp1, tp2;
	GVector tn0, tn1, tn2;
	float alpha = 0.0f, beta = 0.0f;

	while( startIndexInTri <= maxIndexTri && m_iCurrentPhoton <= m_iOneIterateMaxPhoton ) {
		
		row = (int) ceil( sqrt( (float)startIndexInTri ) );
		col = startIndexInTri - ( row - 1 ) * ( row - 1 );

		/**
		 *	삼각형이 한 row 에서 홀수 위치인지 짝수위치인지에 따라 photon 을
		 *	찍을 위치와 정보를 결정한다.
		 */
		if ( col % 2 == 1 ) {
			calPointInfo( row, ( col + 1 ) / 2, totalRow, p0, p1, p2, n0, n1, n2, &tp0, &tn0 );
			calPointInfo( row + 1, ( col + 1 ) / 2, totalRow, p0, p1, p2, n0, n1, n2, &tp1, &tn1 );
			calPointInfo( row + 1, ( col + 1 ) / 2 + 1, totalRow, p0, p1, p2, n0, n1, n2, &tp2, &tn2 );

		} else {
			calPointInfo( row, col / 2, totalRow, p0, p1, p2, n0, n1, n2, &tp0, &tn0 );
			calPointInfo( row, col / 2 + 1, totalRow, p0, p1, p2, n0, n1, n2, &tp1, &tn1 );
			calPointInfo( row + 1, col / 2 + 1, totalRow, p0, p1, p2, n0, n1, n2, &tp2, &tn2 );
		}

		cuPhoton *pPhoton = (cuPhoton*) malloc( sizeof( cuPhoton ) );
		memset( pPhoton, 0x00, sizeof( cuPhoton ) );

		// area photon 의 경우 power.x 에 area 정보 입력.
		pPhoton->power.x = triArea;

		// photon 의 정보는 삼각형의 중점.
		pPhoton->pos.x = ( tp0.x + tp1.x + tp2.x ) / 3.0f;
		pPhoton->pos.y = ( tp0.y + tp1.y + tp2.y ) / 3.0f;
		pPhoton->pos.z = ( tp0.z + tp1.z + tp2.z ) / 3.0f;
		
		pPhoton->normal.x = ( tn0.x + tn1.x + tn2.x ) / 3.0f;
		pPhoton->normal.y = ( tn0.y + tn1.y + tn2.y ) / 3.0f;
		pPhoton->normal.z = ( tn0.z + tn1.z + tn2.z ) / 3.0f;

		float nl = sqrt( pPhoton->normal.x * pPhoton->normal.x +
						pPhoton->normal.y * pPhoton->normal.y +
						pPhoton->normal.z * pPhoton->normal.z );

		pPhoton->normal.x = pPhoton->normal.x / nl;
		pPhoton->normal.y = pPhoton->normal.y / nl;
		pPhoton->normal.z = pPhoton->normal.z / nl;

		m_AreaPhotonList.push_back( pPhoton );

		m_iCurrentPhoton++;
		startIndexInTri++;
	}

	return startIndexInTri;
}

/**
 *	가상의 삼각형의 꼭지점 ( row, col ) 의 point, normal 정보를 계산한다.
 */
inline void GAreaDensityEstimate::calPointInfo( 
	int row, int col, int totalTriRow, 
	GPoint &p0, GPoint &p1, GPoint &p2,
	GVector &n0, GVector &n1, GVector &n2, 
	GPoint *p, GVector *n )
{
	int totalPointRow = totalTriRow + 1;
	int totalPointCol = row;

	GPoint point1 = ( p0 * (float)( totalPointRow - row ) + p1 * (float)( row - 1 ) ) / float( totalPointRow - 1 );
	GPoint point2 = ( p0 * (float)( totalPointRow - row ) + p2 * (float)( row - 1 ) ) / float( totalPointRow - 1  );

	GVector normal1 = ( n0 * (float)( totalPointRow - row ) + n1 * (float)( row - 1 ) ) / float( totalPointRow - 1 );
	GVector normal2 = ( n0 * (float)( totalPointRow - row ) + n2 * (float)( row - 1 ) ) / float( totalPointRow - 1 );

	if ( totalPointCol == 1 ) {
		(*p) = point1; (*n) = normal1;
	} else {
		(*p) = ( point1 * (float)( totalPointCol - col ) + point2 * (float)( col - 1 ) ) / float( totalPointCol - 1 );
		(*n) = ( normal1 * (float)( totalPointCol - col ) + normal2 * (float)( col - 1 ) ) / float( totalPointCol - 1 );
	}
}

/**
 *	현재 AreaPhoton 목록 리턴.
 */
const vector<cuPhoton*>* GAreaDensityEstimate::getAreaPhoton()
{
	return &m_AreaPhotonList;
}

