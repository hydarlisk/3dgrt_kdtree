#ifndef G_TIMER_H_
#define G_TIMER_H_

#ifdef WIN32
	#include <windows.h>
#elif USE_LINUX
	#include <sys/time.h>
	#include <time.h>
	#include <unistd.h>
#endif

/**
 *	윈도우/리눅스 Time Check Routine
 *
 *	by graphicsian.
 */
class GTimer
{
private:
	#ifdef WIN32
		__int64 startTime, endTime, freq;
		float rcpFreq;
	#elif USE_LINUX
		struct timeval _time;
		struct timezone tz;
		double startTime, endTime;
	#endif
	
public:
	GTimer();
	virtual ~GTimer();
	
	void start();
	void end();
	float getElapsedTime();
	double getElapsedFreq();
};

#endif
