#include "GEvent.h"

GEvent::GEvent(void)
{
	m_hEvent = ::CreateEvent( NULL, FALSE, FALSE, NULL );
}

GEvent::~GEvent(void)
{
	::CloseHandle( m_hEvent );
}

/**
 *	signal 가 발생할때까지 무한 대기
 *	signal 상태가 되면 함수를 빠져나오면서 현재상태를 
 *	unsignal 상태로 만듬.
 */
DWORD GEvent::wait()
{
	return ::WaitForSingleObject( m_hEvent, INFINITE );
}

/**
 *	signal 가 발생할때까지 time 시간만큼 대기
 *	signal 상태가 되면 함수를 빠져나오면서 현재상태를 
 *	unsignal 상태로 만듬.
 */
DWORD GEvent::wait( UINT time )
{
	return ::WaitForSingleObject( m_hEvent, time );
}

/**
 *	signal 를 발생시킨다.
 */
void GEvent::notify()
{
	::SetEvent( m_hEvent );
}

/**
 *	non-signal 상태로 만듬.
 */
void GEvent::reset()
{
	::ResetEvent( m_hEvent );
}