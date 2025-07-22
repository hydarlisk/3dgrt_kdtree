#include "GThreadManager.h"
#include "GThread.h"

/**
 *	GThread Manager 가 사용할 전역변수들.
 */
DWORD GThreadManager::g_dwTlsIndex = 0;
GThreadContext *GThreadManager::g_pMainThreadContext = NULL;
vector<GThread*> GThreadManager::g_vecSgrtThreadPool;
int GThreadManager::m_iCurrentWorkers = 0;

/**
 *	생성자
 */
GThreadManager::GThreadManager(void)
{
}

/**
 *	소멸자
 */
GThreadManager::~GThreadManager(void)
{
}

/**
 *	프로그램에서 필요한 쓰레드들을 미리 생성해 둔다.
 *	 또한 생성될 쓰레드들의 ThreadContext 를 저장할 TLS 공간을 초기화 한다.
 *	그리고 디폴트로 이미 Main Thread 가 Process 에 존재하므로, 
 *	일관성을 맞추기 위해서 Main Thread 를 위한 ThreadContext 공간도 생성해둔다.
 */
bool GThreadManager::initThreadManager( int maxThread )
{
	/**
	 *	필요한 쓰레드를 생성하기 전에 반드시 이 부분이 먼저 수행되어야 한다.
	 */
	if ( ( g_dwTlsIndex = TlsAlloc() ) == TLS_OUT_OF_INDEXES ) {
	  printf( "ThreadManager init error ( fail tls alloc )\n" );
      return false;
	}
	
	printf( "ThreadManager init. \n" );
	
	/**
	 *	Main Thread 를 위한 ThreadContext 할당. 이 함수를 호출하는 쓰레드는
	 *	메인 쓰레드이다.
	 */
	g_pMainThreadContext = new GThreadContext( ::GetCurrentThreadId() );
	makeThreadContext( g_pMainThreadContext );

	/**
	 *	프로그램 에서 필요한 쓰레드를 미리 생성해 둔다.
	 */
	for ( int i = 0; i < maxThread; ++i ) {
		GThread *pThread = new GThread();
		pThread->start();
		g_vecSgrtThreadPool.push_back( pThread );
	}

	return true;
}

/**
 *	TLS 를 해제한다.
 *	Main Thread 를 위해서 생성해 놓은 공간도 제거한다.
 */
void GThreadManager::uninitThreadManager()
{
	/**
	 *	GThread 들에게 종료명령을 내리고, 각 thread 가 완전히
	 *	종료될때까지 대기한다.
	 */
	printf( "wait for all thread termination.\n" );
	for ( int i = 0; i < (int)g_vecSgrtThreadPool.size(); ++i ) {
		GThread *pThread = g_vecSgrtThreadPool[i];
		pThread->stopThread();
		pThread->waitThread();
		delete pThread;
	}
	printf( "All thread terminated.\n" );
	g_vecSgrtThreadPool.clear();

	/**
	 *	이 함수를 호출하는 쓰레드는	메인 쓰레드이므로 아래 freeThreadContext 는
	 *	호출하면 메인쓰레드의 ThreadContext 가 제거될 것이다.
	 */
	freeThreadContext();

	delete g_pMainThreadContext;

	/** 
	 *	이부분은 항상 가장 마지막에 호출되어야 한다. 
	 */
	if ( g_dwTlsIndex != 0 )
		::TlsFree( g_dwTlsIndex );

}

/**
 *	TLS 에서 현재 쓰레드와 관련된 GThreadContext 객체 포인터를
 *	리턴한다.( 이 함수를 호출한 쓰레드의 ThreadContext 리턴 )
 *	내부적으로 동기화 될것이므로 따로 동기화 하지 않아도 된다.
 */
GThreadContext* GThreadManager::getThreadContext()
{
	GThreadContext **pLocal = 
		(GThreadContext **)::TlsGetValue( g_dwTlsIndex );

	if ( pLocal != NULL ) {
		return (*pLocal);
	}
	return NULL;
}

/**
 *	현재 쓰레드의 GThreadContext 포인터를 
 *	TLS 에 저장하기 위해서 공간을 생성한다.
 *	( 이 함수를 호출한 쓰레드의 ThreadContext 등록 )
 */
bool GThreadManager::makeThreadContext( GThreadContext *pThread )
{
	GThreadContext **pLocal = 
		(GThreadContext **)::LocalAlloc( LPTR, sizeof( GThreadContext* ) );
	(*pLocal) = pThread;

	return ::TlsSetValue( g_dwTlsIndex, pLocal ) == TRUE;
}

/**
 *	현재 쓰레드를 위해서 할당한 Context 공간을 해제한다.
 *  ( 이 함수를 호출한 쓰레드의 ThreadContext 제거 )
 */
void GThreadManager::freeThreadContext()
{
	LPVOID lpVoid = ::TlsGetValue( g_dwTlsIndex );
	if ( lpVoid != 0 ) {
		::LocalFree( lpVoid );
	}
}

/**
 *	쓰레드로 구동시킬 작업들을 수행시킨다.
 *	이 함수는 이전에 구동시킨 쓰레드 작업들이 완료되었음을 가정한다.
 *	 따라서, 이 함수를 호출하기 전에 기존의 작업을 완벽히 종료시켜야 한다.
 *	이를 위해서 반드시 waitThreadWork 를 호출해서 쓰레드 작업이 다 끝날때까지 
 *	기다려야 한다. 즉 다음과 같은 수행순서를 가져야한다.
 *
 *	startTheadWork( irradianceMapWork, 5 );
 *	waitThreadWork();
 *  ...
 *	startTheadWork( samplingWork, 5 );
 *	waitThreadWork();
 *  ...
 *	startTheadWork( etcWork, 5 );
 *	waitThreadWork();
 *
 *
 *	@param pWork 쓰레드로 구동시킬 작업
 *	@param threadCount 쓰레드 갯수. ( threadPool 에 있는 쓰레드 갯수가 최대 )
 *
 *	@return 실제 구동된 쓰레드 갯수.
 */
int GThreadManager::startThreadWork( GThreadWork *pWork, int threadCount )
{
	/**
	 *	threadCount( 최대threadpool갯수 ) 만큼 
	 *	쓰레드 작업을 수행시킨다.
	 */
	m_iCurrentWorkers = 0;
	for ( int i = 0; i < threadCount && i < (int) g_vecSgrtThreadPool.size(); ++i ) {
		GThread *pThread = g_vecSgrtThreadPool[i];
		pThread->setWorkNumber(i);
		pThread->setWork( pWork );
		m_iCurrentWorkers++;
	}

	return m_iCurrentWorkers;
}

/**
 *	쓰레드들에게 수행시킨 모든 작업이 종료될때까지 기다린다.
 */
void GThreadManager::waitThreadWork()
{
	for ( int i = 0; i < m_iCurrentWorkers; ++i ) {
		GThread *pThread = g_vecSgrtThreadPool[i];
		pThread->waitWorkEnd();
	}
}

/**
 *	최대 쓰레드 갯수를 반환한다.
 */
int GThreadManager::getMaxThreadCount()
{
	return (int)g_vecSgrtThreadPool.size();
}

/**
 *	최대 쓰레드 갯수를 반환한다.
 */
int GThreadManager::getCurrentWorkers()
{
	return m_iCurrentWorkers;
}