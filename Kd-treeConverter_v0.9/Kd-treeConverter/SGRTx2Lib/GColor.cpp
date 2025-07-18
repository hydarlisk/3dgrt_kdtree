#include "GColor.h"

GColor::~GColor(void)
{
}

bool GColor::operator== ( const GColor& color )
{
	return ( r == color.r && g == color.g && b == color.b && a == color.a );
}

void GColor::operator+= ( const GColor& color )
{
	this->r += color.r;
	this->g += color.g;
	this->b += color.b;
	this->a += color.a;
}
	
void GColor::operator*= ( const GColor& color )
{
	this->r *= color.r;
	this->g *= color.g;
	this->b *= color.b;
	this->a *= color.a;
}

void GColor::operator-= ( const GColor& color )
{
	this->r -= color.r;
	this->g -= color.g;
	this->b -= color.b;
	this->a -= color.a;
}

GColor GColor::operator- ( const GColor& color )
{
	return GColor( r - color.r, g - color.g, b - color.b, a - color.a );
}

const float* GColor::getColor()
{
	return m_Color;
}

void GColor::setColor( const float color[] )
{
	m_Color[0] = color[0];
	m_Color[1] = color[1];
	m_Color[2] = color[2];
	m_Color[3] = color[3];
}

void GColor::setColor( float r, float g, float b, float a )
{
	m_Color[0] = r;
	m_Color[1] = g;
	m_Color[2] = b;
	m_Color[3] = a;
}
