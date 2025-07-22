#pragma once
#define _SAL_VERSION 0
#include <windows.h>
#include <sal.h>
#include "GThread.h"
#include <vector>

using namespace std;

/**
 *	프로그램 에서 필요한 Thread 를 관리하고,
 *	 각 Thread 별로 저장공간을 할당하기 위해서 TLS 를
 *	관리하는 클래스.
 *
 *	 반드시 프로그램시작시에 이 클래스내의 
 *	initThreadManager 함수를 호출해서 ThreadManager 를 초기화 하고
 *  uninitThreadManager 함수를 호출해서 자원을 해제해야 한다.
 *
 *	by poovi
 *
 */
class GThread;
class GThreadManager
{
private:
	/**
	 *	메인 쓰레드를 위한 디폴트 Thread Context
	 */
	static GThreadContext *g_pMainThreadContext;

	/**
	 *	TLS를 위한 Index
	 */
	static DWORD g_dwTlsIndex;

	/**
	 *	Thread Pool 에 있는 thread 들중 현재 작업을 수행하고
	 *	있는 thread 갯수.
	 */	
	static int m_iCurrentWorkers;

private:
	/**
	 *	프로그램 에서 사용할 쓰레드들
	 */
	static vector<GThread*> g_vecSgrtThreadPool;

public:
	GThreadManager(void);
	~GThreadManager(void);
	
	static bool initThreadManager( int maxThread );
	static void uninitThreadManager();

	static bool makeThreadContext( GThreadContext *pSgrtThread );
	static GThreadContext* getThreadContext();
	static void freeThreadContext();

	/**
	 *	쓰레드로 구동시킬 작업을 수행시킨다.
	 *	@param pWork 쓰레드로 구동시킬 작업
	 *	@param threadCount 쓰레드 갯수. ( threadPool 에 있는 쓰레드 갯수가 최대 )
	 */
	static int startThreadWork( GThreadWork *pWork, int threadCount );

	/**
	 *	현재 수행시킨 쓰레드의 모든 작업이 종료될때까지 기다린다.
	 */
	static void waitThreadWork();

	static int getMaxThreadCount();
	static int getCurrentWorkers();
};
