#pragma once

#include "GBase.h"

/**
 *	여러가지 잡스러운 함수.
 *
 *	by graphicsian.
 */	
#include <vector>
using namespace std;

class  GUtil
{
public:
	GUtil(void);
	~GUtil(void);

	/**
	 *	filename 에서 확장자를 리턴한다. 리턴값은
	 *	szFileName 의 pointer 를 기준으로 상대적인 위치를
	 *	리턴하는 것이므로, 반드시 이 리턴값을 사용할때는 szFileName 이
	 *	남아 있어야 한다.
	 */
	static const char* getFileExt( const char* szFileName );

	/**
	 *	filename 중 parent path 를 구해서 리턴한다.
	 */
	static const char* getParentPath( const char* szFileName, char* result );

	/**
	 *	filename 중 file name 를 구해서 리턴한다.
	 */
	static const char* getFileName( const char* szFileName, char* result );
	
	/**
	 *	string 의 마지막에 cr/lf 를 null 문자로 바꾼다.
	 */
	static void removeCRLF( char *str );

	/**
	 *	data 에서 = 를 구분해서 key 와 value 를 리턴한다.
	 *	data 자체에서 = 를 0x00 으로 만들고 key 는 data 의 첫 포인터
	 *	value 에는 = 이후 포인터를 가리킨다. 따라서 data 의 값이 훼손됨을
	 *	주의하라.
	 */
	static void getKeyValue( char* data, char **key, char **value );

	/**
	 *	blank 로 구분되는 value 값의 list 를 만들어서 리턴한다.
	 *	getKeyValue과 같은 방식이며, data 정보는 훼손된다.
	 */
	static void getValueList( char *data, vector<char*> &valueList );
	static void getValueList( char *data, vector<char*> &valueList, const char* delimeter );

	/**
	 *	새로운 메모리를 생성해서 src 를 복사해서 리턴한다.
	 */
	static char* copyString( const char *src );

	/** 현재시간을 YYYY-MM-DD-HH-MM-SS 로 리턴 */
	static void getToday( char *dest );
};
