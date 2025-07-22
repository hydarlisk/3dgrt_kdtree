#ifndef _SGRT_GRID_BOX_H
#define _SGRT_GRID_BOX_H 

#include "GVector.h"

typedef struct _cell_info {
	int dataCount;
	int currentIndex;
	int offset;
} CellInfo;

/**
 *	Grid 형태로 데이터를 관리하는 클래스.
 *	
 *	by graphicsian.
 */
template<class T, class T2>
class GGridBox
{
public:
	T *m_pData;
	T2 *m_pData2;

	int m_iTotalCount;
	
	GVector m_BoundMin, m_BoundMax;

	/**
	 *	grid 의 각 cell 을 위한 정보.
	 */
	CellInfo *m_pCellInfo;

public:
	int m_iCellXCount;
	int m_iCellYCount;
	int m_iCellZCount;

	float m_fXUnitLength;
	float m_fYUnitLength;
	float m_fZUnitLength;

public:
	GGridBox( GBoundingBox bbox, float xunit, float yunit, float zunit ) {
	
		/** 수치적 오차가 있을수 있기 때문에 bounding box 를 약간 불린다. */
		m_BoundMin.x = bbox.m_Min.x - 0.001f;
		m_BoundMin.y = bbox.m_Min.y - 0.001f;
		m_BoundMin.z = bbox.m_Min.z - 0.001f;

		m_BoundMax.x = bbox.m_Max.x + 0.001f;
		m_BoundMax.y = bbox.m_Max.y + 0.001f;
		m_BoundMax.z = bbox.m_Max.z + 0.001f;

		m_fXUnitLength = xunit;
		m_fYUnitLength = yunit;
		m_fZUnitLength = zunit;
		m_iCellXCount = (int) ceil( ( m_BoundMax.x - m_BoundMin.x ) / m_fXUnitLength );
		m_iCellYCount = (int) ceil( ( m_BoundMax.y - m_BoundMin.y ) / m_fYUnitLength );
		m_iCellZCount = (int) ceil( ( m_BoundMax.z - m_BoundMin.z ) / m_fZUnitLength );

		m_iTotalCount = 0;
		m_pCellInfo = (CellInfo*) malloc( sizeof( CellInfo ) * m_iCellXCount * m_iCellYCount * m_iCellZCount );
		memset( m_pCellInfo, 0x00, sizeof( CellInfo ) * m_iCellXCount * m_iCellYCount * m_iCellZCount );

		m_pData = NULL;
		m_pData2 = NULL;
	}

	~GGridBox() {
		free( m_pCellInfo );
		if ( m_pData )
			free( m_pData );
		if ( m_pData2 )
			free( m_pData2 );
	}
	
	/**
	 *	Cell 에 들어갈 data 개수를 계산하기 위함.
	 */
	inline void counting( float x, float y, float z ) {
		
		int xIndex = (int) ( ( x - m_BoundMin.x ) / m_fXUnitLength );
		int yIndex = (int) ( ( y - m_BoundMin.y ) / m_fYUnitLength );
		int zIndex = (int) ( ( z - m_BoundMin.z ) / m_fZUnitLength );

		/**
		 *	grid 안에 들어간 좌표인지 체크.
		 */
		if ( xIndex >= 0 && xIndex < m_iCellXCount &&
			 yIndex >= 0 && yIndex < m_iCellYCount &&
			 zIndex >= 0 && zIndex < m_iCellZCount ) {
			CellInfo *pInfo = getCellInfo( xIndex, yIndex, zIndex );
			pInfo->dataCount++;
			m_iTotalCount++;
		} else {
			GLogManager::logging( LOG_ERROR, "error data position : ( %f, %f, %f )", x, y, z ); 
		}
	}

	/**
	 *	데이터를 Cell 에 추가.
	 */
	inline void insertData( float x, float y, float z, T *data, T2 *data2 ) {
		
		int xIndex = (int) ( ( x - m_BoundMin.x ) / m_fXUnitLength );
		int yIndex = (int) ( ( y - m_BoundMin.y ) / m_fYUnitLength );
		int zIndex = (int) ( ( z - m_BoundMin.z ) / m_fZUnitLength );

		/**
		 *	
		 */
		if ( xIndex >= 0 && xIndex < m_iCellXCount &&
			 yIndex >= 0 && yIndex < m_iCellYCount &&
			 zIndex >= 0 && zIndex < m_iCellZCount ) {
		
			CellInfo *pInfo = getCellInfo( xIndex, yIndex, zIndex );
			T *dest = m_pData + pInfo->offset + pInfo->currentIndex;
			memcpy( dest, data, sizeof( T ) );

			if ( data2 != NULL ) {
				T2 *dest2 = m_pData2 + pInfo->offset + pInfo->currentIndex;
				memcpy( dest2, data2, sizeof( T2 ) );
			}

			pInfo->currentIndex++;

		} else {
			GLogManager::logging( LOG_ERROR, "error data position : ( %f, %f, %f )", x, y, z ); 
		}
		
	}

	inline T* getData( CellInfo *pInfo ) {
		return m_pData + pInfo->offset;
	}

	inline T2* getData2( CellInfo *pInfo ) {
		return m_pData2 + pInfo->offset;
	}

	/**
	 *	각 Cell 에 들어갈 data 를 위한 전체 linear 한 memory 를 할당하고,
	 *	각 Cell 에 들어갈 data 가 memory 상의 어디에 존재해야 하는지
	 *	offset 을 계산한다.
	 */
	void allocate() {

		m_pData = (T*) malloc( sizeof( T ) * m_iTotalCount );
		memset( m_pData, 0x00, sizeof( T ) * m_iTotalCount );
		m_pData2 = (T2*) malloc( sizeof( T2 ) * m_iTotalCount );
		memset( m_pData2, 0x00, sizeof( T2 ) * m_iTotalCount );

		int offset = 0;
		CellInfo *pInfo = NULL;

		for ( int z = 0; z < m_iCellZCount; ++z ) {
			for ( int y = 0; y < m_iCellYCount; ++y ) {
				for ( int x = 0; x < m_iCellXCount; ++x ) {
					pInfo = getCellInfo( x, y, z );
					pInfo->offset = offset;
					offset += pInfo->dataCount;
				}
			}
		}
	}

	int calValidCellCount() {
		int count = 0;
		for ( int z = 0; z < m_iCellZCount; ++z ) {
			for ( int y = 0; y < m_iCellYCount; ++y ) {
				for ( int x = 0; x < m_iCellXCount; ++x ) {
					if ( getCellInfo( x, y, z )->dataCount > 0 ) {
						count++;
					}
				}
			}
		}
		return count;
	}

	void printInfo( const char* title ) {
		GLogManager::logging( LOG_DEBUG, "%s grid( %d, %d, %d ), totalDataCount=%d ", title,
			m_iCellXCount, m_iCellYCount, m_iCellZCount,
			m_iTotalCount );
	}

	inline bool isValidCell( int xIndex, int yIndex, int zIndex ) {
		return ( xIndex >= 0 && xIndex < m_iCellXCount && 
				 yIndex >= 0 && yIndex < m_iCellYCount &&
				 zIndex >= 0 && zIndex < m_iCellZCount );
	}

	inline CellInfo* getCellInfo( int xIndex, int yIndex, int zIndex ) {
		if( !isValidCell( xIndex, yIndex, zIndex ) )
			return NULL;
		return ( m_pCellInfo + zIndex * ( m_iCellXCount * m_iCellYCount ) +
		   yIndex * ( m_iCellXCount ) + xIndex );
	}
};

#endif
