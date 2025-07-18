#include ".\gtrianglewrapperlist.h"

GTriangleWrapperList::GTriangleWrapperList(void)
{
}

GTriangleWrapperList::~GTriangleWrapperList(void)
{
	for ( int i = 0; i < (int) m_List.size(); ++i ) {
		delete m_List[i];
	}
	m_List.clear();
}
	
void GTriangleWrapperList::reserve( int size )
{
	m_List.reserve( size );
}

void GTriangleWrapperList::addTriangleWrapper( GTriangleWrapper* tri )
{
	m_List.push_back( tri );
}

int GTriangleWrapperList::size()
{
	return (int)m_List.size();
}

GTriangleWrapper* GTriangleWrapperList::operator[] ( int index )
{
	return m_List[ index ];
}

GTriangleWrapper* GTriangleWrapperList::getTriangleWrapper ( int index )
{
	return m_List[ index ];
}
