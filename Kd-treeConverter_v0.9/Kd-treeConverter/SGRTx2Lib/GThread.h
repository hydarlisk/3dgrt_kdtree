#pragma once

#include "GBasethread.h"
#include "GThreadcontext.h"
#include "GThreadwork.h"

/**
 *	프로그램 의 쓰레드를 의미하는 클래스
 *  이 쓰레드가 수행할 작업은 GThreadWork 로 구성해서 지정해 준다.
 *
 *	by poovi
 *	2007.12.17
 */
class GThread :	public GBaseThread
{
private:
	/**
	 *	현재 쓰레드의 Context
	 */
	GThreadContext *m_pThreadContext;

	/**
	 *	현재 쓰레드가 수행할 작업
	 */
	GThreadWork *m_pThreadWork;

	/**
	 *	쓰레드가 수행할 작업의 시작과 종료를
	 *	동기화하기 위한 Event 들
	 */
	GEvent m_WorkStartEvent, m_WorkEndEvent;
	
	/**
	 *	쓰레드 종료 여부.
	 */
	bool m_bStop;

public:
	GThread();
	~GThread(void);

	/**
	 *	쓰레드가 수행할 작업을 지정한다.
	 */
	void setWork( GThreadWork* pWork );

	/**
	 *	쓰레드에게 수행시킨 작업이 종료될때까지 기다린다.
	 *	(착각주의: 쓰레드가 종료하기를 기다리는게 아니라 쓰레드에게
	 *			   수행시킨 작업이다 !! )
	 *	쓰레드가 종료하기를 기다리는 함수는 부모클래스의 waitThread 이다.
	 */
	void waitWorkEnd();

	/**
	 *	쓰레드를 안전하게 종료시킨다.
	 */
	void stopThread();

	/**
	 *	쓰레드의 Work Number 를 지정한다.
	 */
	void setWorkNumber( UINT nWork );

	/**
	 *	쓰레드의 Work Number 를 반환한다.
	 */
	UINT getWorkNumber();

protected:
	/**
	 *	쓰레드 진입함수. 
	 *	쓰레드가 시작되면, 자동으로 호출된다.
	 */
	DWORD run();

};
