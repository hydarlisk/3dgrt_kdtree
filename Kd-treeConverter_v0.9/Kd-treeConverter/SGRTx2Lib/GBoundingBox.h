#pragma once

//--------------------------------------------------------------------------//
//																			//
//	Bounding Box															//
//																			//
//--------------------------------------------------------------------------//
#include "GBase.h"
#include "GVector.h"

class GBoundingBox
{
public:
	GVector m_Min, m_Max;

public:
	GBoundingBox(void);
	GBoundingBox( const GVector &min, const GVector &max );
	~GBoundingBox(void);

	void operator= ( const GBoundingBox &box );

	void shrink( const GBoundingBox &box );
	bool fullyCover( const GBoundingBox &box ) const;

	/**
	 *	index 는 0 부터 7까지이고, 육면체의 각 vertex 를 계산해서
	 *	가져올 수 있다.
	 */
	GVector operator[] ( int index );
	void operator+= ( GBoundingBox &box );
	void operator+= ( GPoint &point );

	void setMin( const GVector &vmin );
	void setMax( const GVector &vmax );
	GVector getMin();
	GVector getMax();
	GVector getCenter();

	// -----------------------------------------------------------
	// 추가. FOR BVH. -ss
	const GVector size() const
	{
		return (m_Max - m_Min);
	}

	float calcSurfaceArea() const                       // 표면적
	{
		GVector vS = size();
		return (vS.x*vS.y + vS.y*vS.z + vS.z*vS.x) * 2.0f;
	}

	void tryupdateMin(const GVector& v)
	{
		m_Min.x = (v.x < m_Min.x) ? v.x : m_Min.x;
		m_Min.y = (v.y < m_Min.y) ? v.y : m_Min.y;
		m_Min.z = (v.z < m_Min.z) ? v.z : m_Min.z;
	}

	void tryupdateMax(const GVector& v)
	{
		m_Max.x = (v.x > m_Max.x) ? v.x : m_Max.x;
		m_Max.y = (v.y > m_Max.y) ? v.y : m_Max.y;
		m_Max.z = (v.z > m_Max.z) ? v.z : m_Max.z;
	}
	
	// FOR BVH. -ss
};
