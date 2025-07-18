#include "GDimension.h"

GDimension::GDimension(void)
{
	x = 0;
	y = 0;
}

GDimension::GDimension( int x, int y )
{
	this->x = x;
	this->y = y;
}

GDimension::~GDimension(void)
{
}

bool GDimension::operator== ( const GDimension &dim ) const
{
	return ( x == dim.x && y == dim.y );
}

bool GDimension::operator!= ( const GDimension &dim ) const
{
	return ( x != dim.x || y != dim.y );
}