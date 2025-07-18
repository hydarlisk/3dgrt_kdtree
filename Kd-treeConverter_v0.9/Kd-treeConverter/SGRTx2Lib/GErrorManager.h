#pragma once

#include "GBase.h"

class  GErrorManager
{
public:
	GErrorManager(void);
	~GErrorManager(void);

	static const char* getGErrorString( GError error );
};
