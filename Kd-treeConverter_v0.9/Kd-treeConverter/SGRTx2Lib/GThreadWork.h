#pragma once

#include "GThreadContext.h"

/**
 *	쓰레드가 수행할 작업을 의미하는 클래스
 *	이 클래스를 상속받아서 필요한 Work 클래스를 생성한다.
 *
 *	by poovi
 *	2007.12.19
 */
class GThreadWork
{
public:
	GThreadWork();
	virtual ~GThreadWork(void);
	
	/**
	 *	작업을 수행하라는 Event
	 */
	virtual void work( GThreadContext *pContext ) = 0;

	/** 
	 *	작업을 중지하라는 Event
	 */
	virtual void stop() = 0;
};
