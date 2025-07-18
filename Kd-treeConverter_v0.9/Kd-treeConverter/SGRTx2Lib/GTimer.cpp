#include "GTimer.h"

#ifdef WIN32
	#include <windows.h>
#elif USE_LINUX
	#include <sys/time.h>
	#include <time.h>
	#include <unistd.h>
#endif

GTimer::GTimer()
{
#ifdef WIN32
	QueryPerformanceFrequency( (LARGE_INTEGER*)&freq );
	rcpFreq = 1.0f / freq;
#endif
}

GTimer::~GTimer()
{
}

void GTimer::start()
{
#ifdef WIN32
	QueryPerformanceCounter( (LARGE_INTEGER*)&startTime );
#elif USE_LINUX
	gettimeofday( &_time, &tz );
	startTime = (double)_time.tv_sec + (double)_time.tv_usec / ( 1000 * 1000 );	
#endif
}

void GTimer::end()
{
#ifdef WIN32
	QueryPerformanceCounter( (LARGE_INTEGER*)&endTime );
#elif USE_LINUX
	gettimeofday( &_time, &tz );
	endTime = (double)_time.tv_sec + (double)_time.tv_usec / ( 1000 * 1000 );	
#endif
}

float GTimer::getElapsedTime()
{
#ifdef WIN32
	return (float)( endTime - startTime ) * rcpFreq;
#elif USE_LINUX
	return endTime - startTime;
#endif
}

double GTimer::getElapsedFreq()
{
#ifdef WIN32
	return (double)( endTime - startTime );
#elif USE_LINUX
	return 1;
#endif
}
