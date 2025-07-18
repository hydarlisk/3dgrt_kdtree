#pragma once

#include <windows.h>

/**
 *	각 쓰레드별로 설정해야할 환경변수들.
 *	Main 프로그램의 쓰레드에도 적용.
 */
class GThreadContext
{
private:
	/**
	 *	현재 Thread ID 
	 */
	UINT m_uiThreadID;

	/**
	 *	쓰레드 Work Number.
	 */
	UINT m_uiWorkNumber;

public:

public:
	GThreadContext( UINT uiThreadID );
	~GThreadContext(void);

	/**
	 *	현재 쓰레드의 ID 를 리턴.
	 */
	UINT getThreadID();

	/**
	 *	쓰레드의 Work Number 를 지정한다.
	 */
	void setWorkNumber( UINT nWork );

	/**
	 *	쓰레드의 Work Number 를 반환한다.
	 */
	UINT getWorkNumber();

	//void setCalculateShadow(bool flag);
	//bool getCalculateShadow(void);

	//void setHideObject(bool flag);
	//bool getHideObject(void);
};

