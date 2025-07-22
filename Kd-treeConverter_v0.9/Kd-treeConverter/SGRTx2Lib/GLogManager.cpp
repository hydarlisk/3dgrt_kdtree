#include "GLogManager.h"
#include <stdio.h>

int GLogManager::m_iProcessID = 0;
vector<GLogListener*> GLogManager::m_LogListenerList;

GLogManager::GLogManager(void)
{
}

GLogManager::~GLogManager(void)
{
}

void GLogManager::setProcessID( int processID )
{
	m_iProcessID = processID;
}

void GLogManager::logging( int level, const char* format ... )
{
#ifndef _DEBUG
	if ( level == LOG_DEBUG )
		return;
#endif

	char buffer[1024];

	va_list ag;
	va_start( ag, format );
	vsprintf( buffer, format, ag );
	va_end( ag );

#ifdef _DEBUG
		fprintf( stdout, "[Process:%d] %s", m_iProcessID, buffer );
		fflush( stdout );
		OutputDebugString( buffer );
#else
		fprintf( stdout, "[Process:%d] %s", m_iProcessID, buffer );
		fflush( stdout );
#endif


	/** 
	 *	listener 들에게 log 를 보낸다. 
	 */
	for ( int i = 0; i < (int) m_LogListenerList.size(); ++i ) {
		m_LogListenerList[ i ]->printLog( buffer );
	}
}

void GLogManager::addLogListener( GLogListener* pListener )
{
	m_LogListenerList.push_back( pListener );
}

void GLogManager::removeLogListener( GLogListener* pListener )
{
	for ( int i = 0; i < (int) m_LogListenerList.size(); ++i ) {
		if ( m_LogListenerList[ i ] == pListener ) {
			m_LogListenerList.erase( m_LogListenerList.begin() + i );
			break;
		}
	}
}
