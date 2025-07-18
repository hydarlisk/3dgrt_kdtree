#pragma once

#include "GBase.h"

class  GObjectNumberGenerator
{
private:
	static int g_iObjectNumberIndex;
	static int g_iSceneNumberIndex;

public:
	GObjectNumberGenerator(void);
	~GObjectNumberGenerator(void);

	static int generateObjectNumber();
	static int generateSceneNumber();

};
