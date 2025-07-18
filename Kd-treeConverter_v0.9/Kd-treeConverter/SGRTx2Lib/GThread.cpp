#include "GThread.h"
#include "GThreadManager.h"

/**
 *	생성자
 */
GThread::GThread()
{
	m_pThreadWork = NULL;
	m_bStop = false;
}

/**
 *	소멸자
 */
GThread::~GThread(void)
{
}

/**
 *	실제로 Sampling 을 수행할 쓰레드 진입함수.
 *	이 함수가 완전 종료되면 쓰레드가 종료된다.
 *	 쓰레드를 종료하기 위해서는 stopThread 를 호출한다. 
 *	그러면 m_bStop 이 true 로 세팅되어 반복문을 빠져나오게 된다.
 *	가급적이면 강제로 시스템에서 쓰레드를 종료시키는 
 *	TerminateThread 는 사용하지 말고 stopThread 를 써서
 *	안전하게 쓰레드를 종료하는게 좋다.
 */
DWORD GThread::run()
{
	/**
	 *	현재 쓰레드와 관련된 ThreadContex 을 TLS 에 등록
	 *	반드시 이 부분은 run 함수 안에서 맨처음에 수행되어야 한다.
	 *	run 이 호출될 시점이 새로운 쓰레드가 생성되는 시점이다.
	 */
	m_pThreadContext = new GThreadContext( this->m_uiThreadID );
	GThreadManager::makeThreadContext( m_pThreadContext );

	/**
	 *	수행할 작업이 세팅되어 있으면, 수행하고
	 *	없으면 작업이 세팅될때까지 대기한다.
	 */
	while( !m_bStop ) {
		if ( m_pThreadWork ) {
			m_pThreadWork->work( m_pThreadContext );
			m_pThreadWork = NULL;
			m_WorkEndEvent.notify();
		} else {
			m_WorkEndEvent.notify();
			m_WorkStartEvent.wait();
		}
	}

	/**
	 *	현재 쓰레드와 관련된 ThreadContext 를 TLS 에서 제거
	 */
	GThreadManager::freeThreadContext();
	delete m_pThreadContext;

	return 0;
}

/**
 *	 쓰레드를 종료하기 위해서는 stopThread 를 호출한다. 
 *	그러면 m_bStop 이 true 로 세팅되어 반복문을 빠져나오게 된다.
 *	가급적이면 강제로 시스템에서 쓰레드를 종료시키는 
 *	TerminateThread 는 사용하지 말고 stopThread 를 써서
 *	안전하게 쓰레드를 종료하는게 좋다.
 */
void GThread::stopThread()
{
	if ( m_pThreadWork )
		m_pThreadWork->stop();

	m_bStop = true;
	m_WorkStartEvent.notify();
}

/**
 *	Thread 가 수행할 작업을 지정하고, 쓰레드를 깨워서
 *	수행하게 한다. m_WorkEndEvent 를 reset 시킨다.
 *	 작업수행이 완료되면 m_WorkEndEvent 가 set 된다.
 */
void GThread::setWork( GThreadWork *pWork )
{
	m_pThreadWork = pWork;
	m_WorkEndEvent.reset();
	m_WorkStartEvent.notify();
}

/**
 *	쓰레드에게 수행시킨 작업이 종료될때까지 기다린다.
 *	(착각주의: 쓰레드가 종료하기를 기다리는게 아니라 쓰레드에게
 *			   수행시킨 작업이다 !! )
 *	쓰레드가 종료하기를 기다리는 함수는 부모클래스의 waitThread 이다.
 */
void GThread::waitWorkEnd()
{
	m_WorkEndEvent.wait();
}

/**
 *	Thread 의 WorkNumber 를 지정한다.
 */
void GThread::setWorkNumber( UINT nWork )
{
	m_pThreadContext->setWorkNumber(nWork);
}

/**
 *	Thread 의 WorkNumber 를 반환한다.
 */
UINT GThread::getWorkNumber()
{
	return m_pThreadContext->getWorkNumber();
}