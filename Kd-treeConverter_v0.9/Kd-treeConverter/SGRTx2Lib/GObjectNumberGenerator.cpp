#include "GObjectNumberGenerator.h"

int GObjectNumberGenerator::g_iObjectNumberIndex = 0;
int GObjectNumberGenerator::g_iSceneNumberIndex = 0;

GObjectNumberGenerator::GObjectNumberGenerator(void)
{
}

GObjectNumberGenerator::~GObjectNumberGenerator(void)
{
}

/**
 *	프로그램 상의 Scene 에 유일한 번호를 붙이기
 *	위한 번호를 생성한다.
 */
int GObjectNumberGenerator::generateSceneNumber()
{
	return ++g_iSceneNumberIndex;
}

/**
 *	프로그램 상의 모든 Object 에 유일한 번호를 붙이기
 *	위한 번호를 생성한다.
 */
int GObjectNumberGenerator::generateObjectNumber()
{
	return ++g_iObjectNumberIndex;
}
