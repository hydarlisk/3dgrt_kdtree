#pragma once

#include <windows.h>
#include <process.h>

#include "GCriticalsection.h"
#include "GEvent.h"

/**
 *
 *	각각의 쓰레드 환경을 표현하는 클래스
 *
 *	by poovi
 *	2007.12.17
 *
 */
class GBaseThread
{
protected:
	/**
	 *	쓰레드 아이디
	 */
	UINT m_uiThreadID;

	/**
	 *	쓰레드 핸들
	 */
	HANDLE m_hThreadHandle;

	/**
	 *	종료코드
	 */
	DWORD m_dwExitCode;

private:
	/**
	 *	쓰레드 진입함수. 
	 *	이 함수안에서 아래의 run() 함수를 호출해준다.
	 */
	static unsigned _stdcall threadFunc( void* args );

protected:
	/**
	 *	실제로 각 서브 클래스에서 오버라이딩해서 사용할 함수
	 */
	virtual DWORD run() = 0;

public:
	/**
	 *	생성자
	 */
	GBaseThread();

	/**
	 *	소멸자
	 */
	virtual ~GBaseThread(void);

	/**
	 *	쓰레드를 시작한다.
	 */
	DWORD start();

	/**
	 *	쓰레드가 종료할때 까지 기다린다.
	 *	리턴값은 WaitForSingleObject 참고
	 */
	DWORD waitThread();
	DWORD waitThread( int time );

	/**
	 *	쓰레드 중지한다.
	 */
	void suspendThread();

	/**
	 *	쓰레드 재개
	 */
	void resumeThread();

	/**
	 *	쓰레드를 종료시킨다. 윈도우의 TerminateThread 를 사용하면
	 *	위험하기 때문에, 가급적이면 이 함수대신에
	 *	각 Thread 안에서 안전하게 쓰레드를 종료하는 방법을 사용하는게 좋다.
	 */
	void killThread( DWORD dwExitCode );

	/**
	 *	쓰레드가 종료한 후 리턴한 값
	 */	
	DWORD getExitCode();
};
