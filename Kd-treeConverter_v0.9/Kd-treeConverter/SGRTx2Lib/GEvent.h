#pragma once
#define _SAL_VERSION 0
#include <windows.h>
#include <sal.h>
#include <process.h>

/**
 * WIN32 Event 를 사용하기 쉽게 Wrapping 한 클래스
 * 
 *	by poovi
 * 2007.12.19
 */

class GEvent
{
private:
	HANDLE m_hEvent;

public:
	GEvent(void);
	~GEvent(void);

	/**
	 *	signal 상태가 발생할때까지 무한 대기.
	 *	signal 상태가 되면 함수를 빠져나오면서 현재상태를 
	 *	unsignal 상태로 만듬.
	 */
	DWORD wait();

	/**
	 *	signal 상태가 발생할때까지 time 시간만큼 대기 ( 밀리세컨드 )
	 *	signal 상태가 되면 함수를 빠져나오면서 현재상태를 
	 *	unsignal 상태로 만듬.
	 */
	DWORD wait( UINT time );

	/**
	 *	signal 상태로 변환
	 */
	void notify();

	/**
	 *	unsignal 상태로 강제로 변환.
	 */
	void reset();
};
