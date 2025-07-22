#include "GThreadContext.h"

GThreadContext::GThreadContext( UINT uiThreadID )
{
	/**
	 *	쓰레드 ID
	 */
	m_uiThreadID = uiThreadID;

//	m_bHideObject = GLOBAL_HIDE_OBJECT_SET;
}

GThreadContext::~GThreadContext(void)
{
}

UINT GThreadContext::getThreadID()
{
	return m_uiThreadID;
}

/**
 *	Thread 의 WorkNumber 를 지정한다.
 */
void GThreadContext::setWorkNumber( UINT nWork )
{
	m_uiWorkNumber = nWork;
}

/**
 *	Thread 의 WorkNumber 를 반환한다.
 */
UINT GThreadContext::getWorkNumber()
{
	return m_uiWorkNumber;
}

//void GThreadContext::setCalculateShadow(bool flag)
//{
//	m_bCalculateShadow = flag;
//}
//bool GThreadContext::getCalculateShadow(void)
//{
//	return m_bCalculateShadow;
//}
//
//void GThreadContext::setHideObject(bool flag)
//{
//	m_bHideObject = flag;
//}
//
//bool GThreadContext::getHideObject(void)
//{
//	return m_bHideObject;
//}