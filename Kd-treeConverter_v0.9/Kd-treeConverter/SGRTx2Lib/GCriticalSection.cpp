#include "GCriticalSection.h"

GCriticalSection::GCriticalSection(void)
{
	::InitializeCriticalSection( &m_CriticalSection );
}

GCriticalSection::~GCriticalSection(void)
{
	::DeleteCriticalSection( &m_CriticalSection );
}

/**
 *  동기화 영역 시작
 *	lock 을 불렀다면 반드시 동기화 영역체크가
 *	다 끝난후에는 unlock 을 불러야 한다.
 */
void GCriticalSection::lock()
{
	::EnterCriticalSection( &m_CriticalSection );
}

/**
 *	동기화 영역 종료
 *	lock 을 불렀다면 반드시 동기화 영역체크가
 *	다 끝난후에는 unlock 을 불러야 한다.
 */
void GCriticalSection::unlock()
{
	::LeaveCriticalSection( &m_CriticalSection );
}
