#pragma once

#include "GBase.h"
#include "GLogListener.h"
#include <vector>

#define LOG_FATAL		0
#define LOG_ERROR		1
#define LOG_WARNING		2
#define LOG_INFO		3
#define LOG_DEBUG		4

using namespace std;

/**
 *	Render 로깅 정보를 관리할 클래스.
 */
class  GLogManager
{
private:
	static int m_iProcessID;
	static vector<GLogListener*> m_LogListenerList;

public:
	GLogManager(void);
	virtual ~GLogManager(void);

	static void addLogListener( GLogListener* pListener );
	static void removeLogListener( GLogListener* pListener );

	static void setProcessID( int processID );
	static void logging( int level, const char* format, ... );
};
