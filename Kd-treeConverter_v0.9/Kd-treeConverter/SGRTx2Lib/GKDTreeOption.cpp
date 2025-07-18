#include "GKDTreeOption.h"
#include <fstream>
using namespace std;

GKDTreeOption::GKDTreeOption()
	: m_SplitFunction(SAH_SPLIT_FUNCTION)
{
	ResetSetting();
}

void GKDTreeOption::ResetSetting()
{
	m_FloatMap.clear();
	m_IntegerMap.clear();

	//! default parameters
	setParameterFloat( "TraversalCost", 1.5f );
	setParameterFloat( "IntersectionCost", 20.0f );

	setParameterInteger( "MaximumTreeDepth", 1024 );
	setParameterInteger( "MaximumObjectCount", 4 );

	m_SplitFunction = SAH_SPLIT_FUNCTION;
}

bool GKDTreeOption::LoadFile( const char *filename )
{
	fstream fs;
	fs.open( filename );
	if( !fs )
	{
		return false;
	}
	string str;
	bool read_float = true;
	float f_value;
	int i_value;

	while( !fs )
	{
		fs >> str;
		if( str[0] == '[' )
		{
			if( str == "[integer]" )
				read_float = false;
			else
				read_float = true;
		}
		else
		{
			if( read_float )
			{
				fs >> f_value;
				setParameterFloat( str, f_value );
			}
			else
			{
				fs >> i_value;
				setParameterInteger( str, i_value );
			}
		}
	}
	return true;
}

void GKDTreeOption::setParameterInteger( std::string str, int value )
{
	m_IntegerMap[str] = value;
}

int GKDTreeOption::getParameterInteger( std::string str )
{
	return m_IntegerMap[str];
}

bool GKDTreeOption::isValidInteger( std::string str )
{
	std::map<std::string, int>::iterator it;
    it = m_IntegerMap.find( str );

	return it != m_IntegerMap.end();
}

void GKDTreeOption::setParameterFloat( std::string str, float value )
{
	m_FloatMap[str] = value;
}

float GKDTreeOption::getParameterFloat( std::string str )
{
	return m_FloatMap[str];
}

bool GKDTreeOption::isValidFloat( std::string str )
{
	std::map<std::string, float>::iterator it;
    it = m_FloatMap.find( str );

	return it != m_FloatMap.end();
}

void GKDTreeOption::setSplitFunction( SPLIT_FUNCTION SplitFunctionEnum )
{
	m_SplitFunction = SplitFunctionEnum;
}

SPLIT_FUNCTION GKDTreeOption::getSplitFunction()
{
	return m_SplitFunction;
}