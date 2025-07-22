#pragma once

/**
 *	Constructing KD-Tree 관련 옵션
 *
 *	by Hybrid.
 */
//#include "GRenderOption.h"

#include <string>
#include <map>

// Split Functions
enum SPLIT_FUNCTION {
	SAH_SPLIT_FUNCTION,
	EMPTY_SPLIT_FUNCTION, // ??
	SAH_WITH_EXTRA_COST_SPLIT_FUNCTION,
	VISIBILITY_SPLIT_FUNCTION
};

class GKDTreeOption// : public GRenderOption
{
public:
	GKDTreeOption();

	//! reset to default setting
	void ResetSetting();
	bool LoadFile( const char *filename );

	void setParameterInteger( std::string str, int value );
	int getParameterInteger( std::string str );
	bool isValidInteger( std::string str );

	void setParameterFloat( std::string str, float value );
	float getParameterFloat( std::string str );
	bool isValidFloat( std::string str );

	//! Not function pointer as parameter
	void setSplitFunction( SPLIT_FUNCTION SplitFunctionEnum );
	SPLIT_FUNCTION getSplitFunction();
	
private:
	
	//! 메모리의 최대 크기
	//! 렌더링 레이가 가지는 최대 크기 (생성시에 한쪽이 empty node 일 경우 이 depth 는 증가시키지 않는다.)

	//! Split Function Number
	SPLIT_FUNCTION m_SplitFunction;

	std::map<std::string, int> m_IntegerMap;
	std::map<std::string, float> m_FloatMap;
};