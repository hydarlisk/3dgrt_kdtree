#pragma once

/**
 *	Log 출력 결과를 받을 클래스가 구현해야할
 *	Listener.
 *
 *	by graphicsian.
 */
class GLogListener
{
public:
	virtual void printLog( const char* str ) = 0;
};
