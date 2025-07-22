#pragma once
#define _SAL_VERSION 0
#include <windows.h>
#include <sal.h>
#include <winbase.h>

/**
 *	WIN32 Critical Section 를 Wrapping 한 클래스
 *	MFC 에는 CCriticalSection 이 있지만, 현재 MFC 를 사용하지 않으므로
 *	Wrapping 클래스를 만든다.
 *
 *	by poovi
 *	2007.12.17
 */

class GCriticalSection
{
private:
	CRITICAL_SECTION m_CriticalSection;

public:
	GCriticalSection(void);
	~GCriticalSection(void);

	void lock();
	void unlock();
};
