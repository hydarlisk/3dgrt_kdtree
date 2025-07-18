
#include "GScene.h"
#include "GRenderSystem.h"

#include "SSE_math.h"
#include "SSERayQueue.h"

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

SSETraceQueue4x4::SSETraceQueue4x4()
{
	initialize(100*1000);
}

SSETraceQueue4x4::SSETraceQueue4x4(int a_MaxSize)
{
	initialize(a_MaxSize);
}

void SSETraceQueue4x4::initialize(int a_MaxSize)
{
	m_MaxSize = a_MaxSize;

	m_RayQ4x4		= (_sse_4x4_raypacket*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_4x4_raypacket), 16);
	m_IsectQ4x4		= (_sse_4x4_isectQ*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_4x4_isectQ), 16);
	m_RMaskQ4x4		= (_sse_4x4_raymask*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_4x4_raymask), 16);
	m_TMaskQ4x4		= (_sse_4x4_raymask*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_4x4_raymask), 16);

	m_Item			= (_sse_4x4_traceData*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_4x4_traceData), 16);

	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].rp = &m_RayQ4x4[i];
		m_Item[i].is = &m_IsectQ4x4[i];
		m_Item[i].rm = &m_RMaskQ4x4[i];
		m_Item[i].tm = &m_TMaskQ4x4[i];
		m_Item[i].next = &m_Item[i+1];
		m_Item[i].id = i;
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;
}

SSETraceQueue4x4::~SSETraceQueue4x4(void)
{
	_aligned_free(m_RayQ4x4);
	_aligned_free(m_IsectQ4x4);
	_aligned_free(m_RMaskQ4x4);
	_aligned_free(m_TMaskQ4x4);
	_aligned_free(m_Item);
	m_RayQ4x4 = NULL;
	m_IsectQ4x4 = NULL;
	m_RMaskQ4x4 = NULL;
	m_TMaskQ4x4 = NULL;
	m_Item = NULL;
	m_MaxSize = 0;
}

void SSETraceQueue4x4::ClearIndex(void)
{
	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;
}

void SSETraceQueue4x4::IdleQ_EnQueue(_sse_4x4_traceData *item)
{
	m_IdleQ_CS.lock();
	if(m_IdleQ_rear) {
		m_IdleQ_rear->next = item;
	} else {
		m_IdleQ_front = item;
	}
	item->next = NULL;
	m_IdleQ_rear = item;
	m_IdleQ_CS.unlock();
}

_sse_4x4_traceData* SSETraceQueue4x4::IdleQ_DeQueue(void)
{
	_sse_4x4_traceData *item = NULL;
	m_IdleQ_CS.lock();
	if(m_IdleQ_front) {
		item = m_IdleQ_front;
		m_IdleQ_front = m_IdleQ_front->next;
	}
	if (m_IdleQ_front == NULL) {
		m_IdleQ_rear = NULL;
	}
	m_IdleQ_CS.unlock();
	return item;
}

void SSETraceQueue4x4::TodoQ_EnQueue(_sse_4x4_traceData *item)
{
	m_TodoQ_CS.lock();
	if(m_TodoQ_rear) {
		m_TodoQ_rear->next = item;
	} else {
		m_TodoQ_front = item;
	}
	item->next = NULL;
	m_TodoQ_rear = item;
	m_TodoQ_CS.unlock();
}

_sse_4x4_traceData* SSETraceQueue4x4::TodoQ_DeQueue(void)
{
	_sse_4x4_traceData *item = NULL;
	m_TodoQ_CS.lock();
	if(m_TodoQ_front) {
		item = m_TodoQ_front;
		m_TodoQ_front = m_TodoQ_front->next;
	}
	if (m_TodoQ_front == NULL) {
		m_TodoQ_rear = NULL;
	}
	m_TodoQ_CS.unlock();
	return item;
}

_sse_4x4_raypacket* SSETraceQueue4x4::GetRay(void)
{
	return m_RayQ4x4;
}

_sse_4x4_isectQ* SSETraceQueue4x4::GetIsect(void)
{
	return m_IsectQ4x4;
}

_sse_4x4_raymask* SSETraceQueue4x4::GetRMask(void)
{
	return m_RMaskQ4x4;
}

_sse_4x4_raymask* SSETraceQueue4x4::GetTMask(void)
{
	return m_TMaskQ4x4;
}


// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

SSETraceQueue2x2::SSETraceQueue2x2()
{
	initialize(100*1000);
}

SSETraceQueue2x2::SSETraceQueue2x2(int a_MaxSize)
{
	initialize(a_MaxSize);
}

void SSETraceQueue2x2::initialize(int a_MaxSize)
{
	m_MaxSize = a_MaxSize;

	m_RayList2x2		= (_sse_2x2_raypacket*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_2x2_raypacket), 16);
	m_IsectList2x2		= (_sse_2x2_isectQ*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_2x2_isectQ), 16);
	m_RMaskList2x2		= (_sse_2x2_raymask*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_2x2_raymask), 16);
	m_TMaskList2x2		= (_sse_2x2_raymask*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_2x2_raymask), 16);

	m_Item			= (_sse_2x2_traceData*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_2x2_traceData), 16);

	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].rp = &m_RayList2x2[i];
		m_Item[i].is = &m_IsectList2x2[i];
		m_Item[i].rm = &m_RMaskList2x2[i];
		m_Item[i].tm = &m_TMaskList2x2[i];
		m_Item[i].next = &m_Item[i+1];
		m_Item[i].id = i;
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;
}

SSETraceQueue2x2::~SSETraceQueue2x2(void)
{
	_aligned_free(m_RayList2x2);
	_aligned_free(m_IsectList2x2);
	_aligned_free(m_RMaskList2x2);
	_aligned_free(m_TMaskList2x2);
	_aligned_free(m_Item);
	m_RayList2x2 = NULL;
	m_IsectList2x2 = NULL;
	m_RMaskList2x2 = NULL;
	m_TMaskList2x2 = NULL;
	m_Item = NULL;
	m_MaxSize = 0;
}

void SSETraceQueue2x2::ClearIndex(void)
{
	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;
}

void SSETraceQueue2x2::IdleQ_EnQueue(_sse_2x2_traceData *item)
{
	m_IdleQ_CS.lock();
	if(m_IdleQ_rear) {
		m_IdleQ_rear->next = item;
	} else {
		m_IdleQ_front = item;
	}
	item->next = NULL;
	m_IdleQ_rear = item;
	m_IdleQ_CS.unlock();
}

_sse_2x2_traceData* SSETraceQueue2x2::IdleQ_DeQueue(void)
{
	_sse_2x2_traceData *item = NULL;
	m_IdleQ_CS.lock();
	if(m_IdleQ_front) {
		item = m_IdleQ_front;
		m_IdleQ_front = m_IdleQ_front->next;
	}
	if (m_IdleQ_front == NULL) {
		m_IdleQ_rear = NULL;
	}
	m_IdleQ_CS.unlock();
	return item;
}

void SSETraceQueue2x2::TodoQ_EnQueue(_sse_2x2_traceData *item)
{
	m_TodoQ_CS.lock();
	if(m_TodoQ_rear) {
		m_TodoQ_rear->next = item;
	} else {
		m_TodoQ_front = item;
	}
	item->next = NULL;
	m_TodoQ_rear = item;
	m_TodoQ_CS.unlock();
}

_sse_2x2_traceData* SSETraceQueue2x2::TodoQ_DeQueue(void)
{
	_sse_2x2_traceData *item = NULL;
	m_TodoQ_CS.lock();
	if(m_TodoQ_front) {
		item = m_TodoQ_front;
		m_TodoQ_front = m_TodoQ_front->next;
	}
	if (m_TodoQ_front == NULL) {
		m_TodoQ_rear = NULL;
	}
	m_TodoQ_CS.unlock();
	return item;
}

_sse_2x2_raypacket* SSETraceQueue2x2::GetRay(void)
{
	return m_RayList2x2;
}

_sse_2x2_isectQ* SSETraceQueue2x2::GetIsect(void)
{
	return m_IsectList2x2;
}

_sse_2x2_raymask* SSETraceQueue2x2::GetRMask(void)
{
	return m_RMaskList2x2;
}

_sse_2x2_raymask* SSETraceQueue2x2::GetTMask(void)
{
	return m_TMaskList2x2;
}


// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

SSERayQueue1x1::SSERayQueue1x1()
{
	initialize(100*1000);
}

SSERayQueue1x1::SSERayQueue1x1(int a_MaxSize)
{
	initialize(a_MaxSize);
}

void SSERayQueue1x1::initialize(int a_MaxSize)
{
	m_MaxSize = a_MaxSize;

	m_RayList		= (_sse_1x1_raypacket*)		_aligned_malloc((m_MaxSize+1) *sizeof(_sse_1x1_raypacket), 16);
	m_Item			= (_sse_1x1_rayitemData*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_1x1_rayitemData), 16);

	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].rp = &m_RayList[i];
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;

	m_TodoQ_count = 0;
	m_IdleQ_count = m_MaxSize;
}

SSERayQueue1x1::~SSERayQueue1x1(void)
{
	_aligned_free(m_RayList);
	_aligned_free(m_Item);
	m_RayList = NULL;
	m_Item = NULL;
	m_MaxSize = 0;
}

void SSERayQueue1x1::Clear(void)
{
	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;

	m_TodoQ_count = 0;
	m_IdleQ_count = m_MaxSize;
}

void SSERayQueue1x1::IdleQ_EnQueue(_sse_1x1_rayitemData *item)
{
	m_IdleQ_CS.lock();
	if(m_IdleQ_rear) {
		m_IdleQ_rear->next = item;
	} else {
		m_IdleQ_front = item;
	}
	item->next = NULL;
	m_IdleQ_rear = item;
#ifdef _DEBUG
	m_IdleQ_count++;
#endif
	m_IdleQ_CS.unlock();
}

_sse_1x1_rayitemData* SSERayQueue1x1::IdleQ_DeQueue(void)
{
	_sse_1x1_rayitemData *item = NULL;
	m_IdleQ_CS.lock();
	if(m_IdleQ_front) {
		item = m_IdleQ_front;
		m_IdleQ_front = m_IdleQ_front->next;
	}
	if (m_IdleQ_front == NULL) {
		m_IdleQ_rear = NULL;
	}
#ifdef _DEBUG
	m_IdleQ_count--;
#endif
	m_IdleQ_CS.unlock();
	return item;
}

void SSERayQueue1x1::TodoQ_EnQueue(_sse_1x1_rayitemData *item)
{
	m_TodoQ_CS.lock();
	if(m_TodoQ_rear) {
		m_TodoQ_rear->next = item;
	} else {
		m_TodoQ_front = item;
	}
	item->next = NULL;
	m_TodoQ_rear = item;
#ifdef _DEBUG
	m_TodoQ_count++;
#endif
	m_TodoQ_CS.unlock();
}

_sse_1x1_rayitemData* SSERayQueue1x1::TodoQ_DeQueue(void)
{
	_sse_1x1_rayitemData *item = NULL;
	m_TodoQ_CS.lock();
	if(m_TodoQ_front) {
		item = m_TodoQ_front;
		m_TodoQ_front = m_TodoQ_front->next;
	}
	if (m_TodoQ_front == NULL) {
		m_TodoQ_rear = NULL;
	}
#ifdef _DEBUG
	m_TodoQ_count--;
#endif
	m_TodoQ_CS.unlock();
	return item;
}

_sse_1x1_raypacket* SSERayQueue1x1::GetRay(void)
{
	return m_RayList;
}



// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

SSERayQueue2x2::SSERayQueue2x2()
{
	initialize(100*1000);
}

SSERayQueue2x2::SSERayQueue2x2(int a_MaxSize)
{
	initialize(a_MaxSize);
}

void SSERayQueue2x2::initialize(int a_MaxSize)
{
	m_MaxSize = a_MaxSize;

	m_RayList		= (_sse_2x2_raypacket*)		_aligned_malloc((m_MaxSize+1) *sizeof(_sse_2x2_raypacket), 16);
	m_Item			= (_sse_2x2_rayitemData*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_2x2_rayitemData), 16);

	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].rp = &m_RayList[i];
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;

	m_TodoQ_count = 0;
	m_IdleQ_count = m_MaxSize;
}

SSERayQueue2x2::~SSERayQueue2x2(void)
{
	_aligned_free(m_RayList);
	_aligned_free(m_Item);
	m_RayList = NULL;
	m_Item = NULL;
	m_MaxSize = 0;
}

void SSERayQueue2x2::Clear(void)
{
	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;

	m_TodoQ_count = 0;
	m_IdleQ_count = m_MaxSize;
}

void SSERayQueue2x2::IdleQ_EnQueue(_sse_2x2_rayitemData *item)
{
	m_IdleQ_CS.lock();
	if(m_IdleQ_rear) {
		m_IdleQ_rear->next = item;
	} else {
		m_IdleQ_front = item;
	}
	item->next = NULL;
	m_IdleQ_rear = item;
#ifdef _DEBUG
	m_IdleQ_count++;
#endif
	m_IdleQ_CS.unlock();
}

_sse_2x2_rayitemData* SSERayQueue2x2::IdleQ_DeQueue(void)
{
	_sse_2x2_rayitemData *item = NULL;
	m_IdleQ_CS.lock();
	if(m_IdleQ_front) {
		item = m_IdleQ_front;
		m_IdleQ_front = m_IdleQ_front->next;
	}
	if (m_IdleQ_front == NULL) {
		m_IdleQ_rear = NULL;
	}
#ifdef _DEBUG
	m_IdleQ_count--;
#endif
	m_IdleQ_CS.unlock();
	return item;
}

void SSERayQueue2x2::TodoQ_EnQueue(_sse_2x2_rayitemData *item)
{
	m_TodoQ_CS.lock();
	if(m_TodoQ_rear) {
		m_TodoQ_rear->next = item;
	} else {
		m_TodoQ_front = item;
	}
	item->next = NULL;
	m_TodoQ_rear = item;
#ifdef _DEBUG
	m_TodoQ_count++;
#endif
	m_TodoQ_CS.unlock();
}

_sse_2x2_rayitemData* SSERayQueue2x2::TodoQ_DeQueue(void)
{
	_sse_2x2_rayitemData *item = NULL;
	m_TodoQ_CS.lock();
	if(m_TodoQ_front) {
		item = m_TodoQ_front;
		m_TodoQ_front = m_TodoQ_front->next;
	}
	if (m_TodoQ_front == NULL) {
		m_TodoQ_rear = NULL;
	}
#ifdef _DEBUG
	m_TodoQ_count--;
#endif
	m_TodoQ_CS.unlock();
	return item;
}

_sse_2x2_raypacket* SSERayQueue2x2::GetRay(void)
{
	return m_RayList;
}



// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

SSERayQueue4x4::SSERayQueue4x4()
{
	initialize(100*1000);
}

SSERayQueue4x4::SSERayQueue4x4(int a_MaxSize)
{
	initialize(a_MaxSize);
}

void SSERayQueue4x4::initialize(int a_MaxSize)
{
	m_MaxSize = a_MaxSize;

	m_RayList	= (_sse_4x4_raypacket*)		_aligned_malloc((m_MaxSize+1) *sizeof(_sse_4x4_raypacket), 16);
	m_Item			= (_sse_4x4_rayitemData*)	_aligned_malloc((m_MaxSize+1) *sizeof(_sse_4x4_rayitemData), 16);

	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].rp = &m_RayList[i];
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;

	m_TodoQ_count = 0;
	m_IdleQ_count = m_MaxSize;
}

SSERayQueue4x4::~SSERayQueue4x4(void)
{
	_aligned_free(m_RayList);
	_aligned_free(m_Item);
	m_RayList = NULL;
	m_Item = NULL;
	m_MaxSize = 0;
}

void SSERayQueue4x4::Clear(void)
{
	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].next = &m_Item[i+1];
	}
	m_Item[m_MaxSize-1].next = NULL;
	m_IdleQ_front = &m_Item[0];
	m_IdleQ_rear  = &m_Item[m_MaxSize-1];
	m_TodoQ_front = NULL;
	m_TodoQ_rear  = NULL;

	m_TodoQ_count = 0;
	m_IdleQ_count = m_MaxSize;
}

void SSERayQueue4x4::IdleQ_EnQueue(_sse_4x4_rayitemData *item)
{
	m_IdleQ_CS.lock();
	if(m_IdleQ_rear) {
		m_IdleQ_rear->next = item;
	} else {
		m_IdleQ_front = item;
	}
	item->next = NULL;
	m_IdleQ_rear = item;
#ifdef _DEBUG
	m_IdleQ_count++;
#endif
	m_IdleQ_CS.unlock();
}

_sse_4x4_rayitemData* SSERayQueue4x4::IdleQ_DeQueue(void)
{
	_sse_4x4_rayitemData *item = NULL;
	m_IdleQ_CS.lock();
	if(m_IdleQ_front) {
		item = m_IdleQ_front;
		m_IdleQ_front = m_IdleQ_front->next;
	}
	if (m_IdleQ_front == NULL) {
		m_IdleQ_rear = NULL;
	}
#ifdef _DEBUG
	m_IdleQ_count--;
#endif
	m_IdleQ_CS.unlock();
	return item;
}

void SSERayQueue4x4::TodoQ_EnQueue(_sse_4x4_rayitemData *item)
{
	m_TodoQ_CS.lock();
	if(m_TodoQ_rear) {
		m_TodoQ_rear->next = item;
	} else {
		m_TodoQ_front = item;
	}
	item->next = NULL;
	m_TodoQ_rear = item;
#ifdef _DEBUG
	m_TodoQ_count++;
#endif
	m_TodoQ_CS.unlock();
}

_sse_4x4_rayitemData* SSERayQueue4x4::TodoQ_DeQueue(void)
{
	_sse_4x4_rayitemData *item = NULL;
	m_TodoQ_CS.lock();
	if(m_TodoQ_front) {
		item = m_TodoQ_front;
		m_TodoQ_front = m_TodoQ_front->next;
	}
	if (m_TodoQ_front == NULL) {
		m_TodoQ_rear = NULL;
	}
#ifdef _DEBUG
	m_TodoQ_count--;
#endif
	m_TodoQ_CS.unlock();
	return item;
}

_sse_4x4_raypacket* SSERayQueue4x4::GetRay(void)
{
	return m_RayList;
}

// ------------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------

SSERayTable2x2::SSERayTable2x2()
{
	initialize(100*1000);
}

SSERayTable2x2::SSERayTable2x2(int a_MaxSize)
{
	initialize(a_MaxSize);
}

void SSERayTable2x2::initialize(int a_MaxSize)
{
	m_MaxSize = a_MaxSize;

	m_Item			= (_tableitemData*)	_aligned_malloc((m_MaxSize+1) *sizeof(_tableitemData), 4);

	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].flag = 0;
	}
	m_CurrIdx = m_MaxIdx = 0;
}

SSERayTable2x2::~SSERayTable2x2(void)
{
	_aligned_free(m_Item);
	m_Item = NULL;
	m_MaxSize = 0;
}

void SSERayTable2x2::Clear(void)
{
	for (int i = 0; i < m_MaxSize; i++) {
		m_Item[i].flag = 0;
	}

	m_CurrIdx = m_MaxIdx = 0;
}

void SSERayTable2x2::Clear(int a_MaxIdx)
{
	for (int i = 0; i < a_MaxIdx; i++) {
		m_Item[i].flag = 0;
	}

	m_CurrIdx = 0;
	m_MaxIdx = a_MaxIdx;
}

int SSERayTable2x2::Set_Item(int pi, int sub_pi)
{
	m_Item_CS.lock();
	m_Item[(pi<<2)+sub_pi].flag = 1;
	m_Item_CS.unlock();
	return 0;
}

int SSERayTable2x2::Get_Item (int &a_pi, int &a_sub_pi)
{
	int nRet = 0;
	m_Item_CS.lock();
	if (m_CurrIdx >= m_MaxIdx) {
		nRet = 0;
	} else {
		for (; m_CurrIdx < m_MaxIdx; m_CurrIdx++) {
			if (m_Item[m_CurrIdx].flag == 1) {
				a_pi     = m_CurrIdx >> 2;
				a_sub_pi = m_CurrIdx - (a_pi<<2);
				nRet = 1; m_CurrIdx++;
				break;
			}
		}
	}
	m_Item_CS.unlock();
	return nRet;
}

_tableitemData* SSERayTable2x2::GetTable(void)
{
	return m_Item;
}