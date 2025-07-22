#pragma once

#include "GBase.h"
#include "GTriangleWrapper.h"
#include <vector>
using namespace std;

/**
 *	GTriangleWrapper 를 list 로 관리하는 클래스.
 *
 *	by graphicsian.
 */
class GTriangleWrapperList
{
private:
	vector<GTriangleWrapper*> m_List;

public:
	GTriangleWrapperList(void);
	~GTriangleWrapperList(void);

	GTriangleWrapper* operator[] ( int index );
	GTriangleWrapper* getTriangleWrapper ( int index );
	void reserve( int size );
	int size();
	void addTriangleWrapper( GTriangleWrapper* tri ); 
};
