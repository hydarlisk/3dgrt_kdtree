#include "GBaseThread.h"

/**
 *	생성자
 */
GBaseThread::GBaseThread()
{
	m_dwExitCode = 0;
	m_uiThreadID = 0;

	/**
	 *	쓰레드를 생성한다. CREATE_SUSPENDED 를 옵션으로 주어서
	 *	생성후 바로 시작하지 않고, 나중에 따로 start() 를 호출해야
	 *	쓰레드가 동작하게 한다.
	 */
	m_hThreadHandle = (HANDLE) _beginthreadex( NULL, 0, threadFunc, this, 
											   CREATE_SUSPENDED, 
											   &m_uiThreadID );
}

/**
 *	소멸자
 */
GBaseThread::~GBaseThread(void)
{
	CloseHandle( m_hThreadHandle );
}

/**
 *	쓰레드를 시작한다.
 */
DWORD GBaseThread::start()
{
	return ResumeThread( m_hThreadHandle );
}

/**
 *	쓰레드 진입점. 각 서브 클래스에서 오버라이딩해서
 *	사용하면 된다.
 */
unsigned _stdcall GBaseThread::threadFunc( void* args )
{
	DWORD code = 0;
	GBaseThread* pThread = (GBaseThread*) args;
	_endthreadex( ( code = pThread->run() ) );
	return code;
}

/**
 *	쓰레드가 종료할때 까지 기다린다.
 *
 *	@param time time시간만큼만 기다린다 ( milisecond )
 */
DWORD GBaseThread::waitThread( int time )
{
	return WaitForSingleObject( m_hThreadHandle, time );
}

DWORD GBaseThread::waitThread()
{
	return WaitForSingleObject( m_hThreadHandle, INFINITE );
}

/**
 *	쓰레드 중지한다.
 */
void GBaseThread::suspendThread()
{
	::SuspendThread( m_hThreadHandle );
}

/**
 *	쓰레드 재개
 */
void GBaseThread::resumeThread()
{
	::ResumeThread( m_hThreadHandle );
}

/**
 *	쓰레드를 종료시킨다. 윈도우의 TerminateThread 를 사용하면
 *	위험하기 때문에, 가급적이면 이 함수대신에
 *	각 Thread 안에서 안전하게 쓰레드를 종료하는 방법을 사용하는게 좋다.
 *
 *	@param dwExitCode 쓰레드가 종료후 리턴할 값 지정
 */
void GBaseThread::killThread( DWORD dwExitCode )
{
	::TerminateThread( m_hThreadHandle, dwExitCode );
}

/**
 *	쓰레드가 종료한 후 리턴한 값
 */
DWORD GBaseThread::getExitCode()
{
	::GetExitCodeThread( m_hThreadHandle, &m_dwExitCode );
	return m_dwExitCode;
}

