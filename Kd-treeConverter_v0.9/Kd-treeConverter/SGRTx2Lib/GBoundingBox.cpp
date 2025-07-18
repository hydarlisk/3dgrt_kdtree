#include "GBoundingBox.h"
#include <math.h>

GBoundingBox::GBoundingBox(void)
{
}

GBoundingBox::GBoundingBox( const GVector &min, const GVector &max )
{
	m_Min = min;
	m_Max = max;
}

GBoundingBox::~GBoundingBox(void)
{
}

void GBoundingBox::operator= ( const GBoundingBox &box )
{
	m_Min = box.m_Min;
	m_Max = box.m_Max;
}

void GBoundingBox::operator+= ( GBoundingBox &box )
{
	m_Min.x = min( m_Min.x, box.getMin().x );
	m_Min.y = min( m_Min.y, box.getMin().y );
	m_Min.z = min( m_Min.z, box.getMin().z );
	m_Max.x = max( m_Max.x, box.getMax().x );
	m_Max.y = max( m_Max.y, box.getMax().y );
	m_Max.z = max( m_Max.z, box.getMax().z );
}

void GBoundingBox::operator+= ( GPoint &point )
{
	m_Min.x = min( m_Min.x, point.x );
	m_Min.y = min( m_Min.y, point.y );
	m_Min.z = min( m_Min.z, point.z );
	m_Max.x = max( m_Max.x, point.x );
	m_Max.y = max( m_Max.y, point.y );
	m_Max.z = max( m_Max.z, point.z );
}

GVector GBoundingBox::operator[] ( int index )
{
	switch( index ) {
		case 0:
			return m_Min;
		case 1:
			return GVector( m_Max.x, m_Min.y, m_Min.z );
		case 2:
			return GVector( m_Max.x, m_Max.y, m_Min.z );
		case 3:
			return GVector( m_Min.x, m_Max.y, m_Min.z );
		case 4:
			return GVector( m_Min.x, m_Min.y, m_Max.z );
		case 5:
			return GVector( m_Max.x, m_Min.y, m_Max.z );
		case 6:
			return GVector( m_Min.x, m_Max.y, m_Max.z );
		case 7:
			return m_Max;
	}

	return GVector();
}

bool GBoundingBox::fullyCover( const GBoundingBox &box ) const
{
	bool x = ( m_Max.x >= box.m_Max.x ) && ( m_Min.x <= box.m_Min.x );
	bool y = ( m_Max.y >= box.m_Max.y ) && ( m_Min.y <= box.m_Min.y );
	bool z = ( m_Max.z >= box.m_Max.z ) && ( m_Min.z <= box.m_Min.z );
	return ( x && y && z );
}
void GBoundingBox::shrink( const GBoundingBox &box )
{
	m_Max.x = min( m_Max.x, box.m_Max.x );
	m_Max.y = min( m_Max.y, box.m_Max.y );
	m_Max.z = min( m_Max.z, box.m_Max.z );
	m_Min.x = max( m_Min.x, box.m_Min.x );
	m_Min.y = max( m_Min.y, box.m_Min.y );
	m_Min.z = max( m_Min.z, box.m_Min.z );
}

void GBoundingBox::setMin( const GVector &vmin )
{
	m_Min = vmin;
}

void GBoundingBox::setMax( const GVector &vmax )
{
	m_Max = vmax;
}

GVector GBoundingBox::getMin()
{
	return m_Min;
}

GVector GBoundingBox::getMax()
{
	return m_Max;
}

GVector GBoundingBox::getCenter()
{
	return ( m_Max + m_Min ) * 0.5f;
}