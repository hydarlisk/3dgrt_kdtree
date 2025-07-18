#pragma once

#include "GBase.h"

class  GDimension
{
public:
	int x;
	int y;

public:
	GDimension(void);
	GDimension( int x, int y );
	~GDimension(void);

	bool operator== ( const GDimension &dim ) const;
	bool operator!= ( const GDimension &dim ) const;
};
