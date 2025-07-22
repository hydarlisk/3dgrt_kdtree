/**
 *	CPU rendering 시 Queue 를 사용할 때 사용되는 class
 *
 *	by oipini.
 */
#ifndef _SSE_RAY_QUEUE_H_
#define _SSE_RAY_QUEUE_H_

#include "GBase.h"
#include "SSERenderCommon.h"
#include "GCriticalSection.h"

class __declspec(align(16)) SSETraceQueue4x4
{
public:
	SSETraceQueue4x4();
	SSETraceQueue4x4(int a_MaxSize);
	~SSETraceQueue4x4();

	void initialize(int a_MaxSize);

	void ClearIndex(void);				// 전체 Queue 를 비움(인덱스만 초기화)
	_sse_4x4_raypacket*	GetRay(void);	// Queue array 포인터 반환
	_sse_4x4_isectQ*	GetIsect(void);	// Queue array 포인터 반환
	_sse_4x4_raymask*	GetRMask(void);	// Queue array 포인터 반환
	_sse_4x4_raymask*	GetTMask(void);	// Queue array 포인터 반환

	_sse_4x4_raypacket	*m_RayQ4x4;
	_sse_4x4_isectQ		*m_IsectQ4x4;
	_sse_4x4_raymask	*m_RMaskQ4x4;
	_sse_4x4_raymask	*m_TMaskQ4x4;

	int	m_MaxSize;
	_sse_4x4_traceData	*m_Item;

	// IdleQ
	void IdleQ_EnQueue(_sse_4x4_traceData *item);		// EnQueue
	_sse_4x4_traceData* IdleQ_DeQueue(void);			// DeQueue
	_sse_4x4_traceData	*m_IdleQ_front;					
	_sse_4x4_traceData	*m_IdleQ_rear;					
														
	// TodoQ											
	void TodoQ_EnQueue(_sse_4x4_traceData *item);		// EnQueue
	_sse_4x4_traceData* TodoQ_DeQueue(void);			// DeQueue
	_sse_4x4_traceData	*m_TodoQ_front;
	_sse_4x4_traceData	*m_TodoQ_rear;

private:
	GCriticalSection m_IdleQ_CS;
	GCriticalSection m_TodoQ_CS;
};


class __declspec(align(16)) SSETraceQueue2x2
{
public:
	SSETraceQueue2x2();
	SSETraceQueue2x2(int a_MaxSize);
	~SSETraceQueue2x2();

	void initialize(int a_MaxSize);

	void ClearIndex(void);				// 전체 Queue 를 비움(인덱스만 초기화)
	_sse_2x2_raypacket*	GetRay(void);	// Queue array 포인터 반환
	_sse_2x2_isectQ*	GetIsect(void);	// Queue array 포인터 반환
	_sse_2x2_raymask*	GetRMask(void);	// Queue array 포인터 반환
	_sse_2x2_raymask*	GetTMask(void);	// Queue array 포인터 반환

	// IdleQ
	void IdleQ_EnQueue(_sse_2x2_traceData *item);		// EnQueue
	_sse_2x2_traceData* IdleQ_DeQueue(void);			// DeQueue

	// TodoQ
	void TodoQ_EnQueue(_sse_2x2_traceData *item);		// EnQueue
	_sse_2x2_traceData* TodoQ_DeQueue(void);			// DeQueue

private:
	_sse_2x2_raypacket	*m_RayList2x2;
	_sse_2x2_isectQ		*m_IsectList2x2;
	_sse_2x2_raymask	*m_RMaskList2x2;
	_sse_2x2_raymask	*m_TMaskList2x2;

	int	m_MaxSize;
	_sse_2x2_traceData	*m_Item;

	_sse_2x2_traceData	*m_IdleQ_front;
	_sse_2x2_traceData	*m_IdleQ_rear;
	_sse_2x2_traceData	*m_TodoQ_front;
	_sse_2x2_traceData	*m_TodoQ_rear;

	GCriticalSection	m_IdleQ_CS;
	GCriticalSection	m_TodoQ_CS;
};

class __declspec(align(16)) SSERayQueue4x4
{
public:
	SSERayQueue4x4();
	SSERayQueue4x4(int a_MaxSize);
	~SSERayQueue4x4();

	void initialize(int a_MaxSize);

	void Clear(void);					// 전체 Queue 를 비움(인덱스만 초기화)
	_sse_4x4_raypacket*	GetRay(void);	// Queue array 포인터 반환

	// IdleQ
	void IdleQ_EnQueue(_sse_4x4_rayitemData *item);		// EnQueue
	_sse_4x4_rayitemData* IdleQ_DeQueue(void);			// DeQueue

	// TodoQ
	void TodoQ_EnQueue(_sse_4x4_rayitemData *item);		// EnQueue
	_sse_4x4_rayitemData* TodoQ_DeQueue(void);			// DeQueue

private:
	_sse_4x4_raypacket		*m_RayList;

	int	m_MaxSize;
	_sse_4x4_rayitemData	*m_Item;

	_sse_4x4_rayitemData	*m_IdleQ_front;
	_sse_4x4_rayitemData	*m_IdleQ_rear;
	_sse_4x4_rayitemData	*m_TodoQ_front;
	_sse_4x4_rayitemData	*m_TodoQ_rear;

	GCriticalSection	m_IdleQ_CS;
	GCriticalSection	m_TodoQ_CS;

	int m_TodoQ_count;
	int m_IdleQ_count;
};

class __declspec(align(16)) SSERayQueue2x2
{
public:
	SSERayQueue2x2();
	SSERayQueue2x2(int a_MaxSize);
	~SSERayQueue2x2();

	void initialize(int a_MaxSize);

	void Clear(void);					// 전체 Queue 를 비움(인덱스만 초기화)
	_sse_2x2_raypacket*	GetRay(void);	// Queue array 포인터 반환

	// IdleQ
	void IdleQ_EnQueue(_sse_2x2_rayitemData *item);		// EnQueue
	_sse_2x2_rayitemData* IdleQ_DeQueue(void);			// DeQueue

	// TodoQ
	void TodoQ_EnQueue(_sse_2x2_rayitemData *item);		// EnQueue
	_sse_2x2_rayitemData* TodoQ_DeQueue(void);			// DeQueue

private:
	_sse_2x2_raypacket		*m_RayList;

	int	m_MaxSize;
	_sse_2x2_rayitemData	*m_Item;

	_sse_2x2_rayitemData	*m_IdleQ_front;
	_sse_2x2_rayitemData	*m_IdleQ_rear;
	_sse_2x2_rayitemData	*m_TodoQ_front;
	_sse_2x2_rayitemData	*m_TodoQ_rear;

	GCriticalSection	m_IdleQ_CS;
	GCriticalSection	m_TodoQ_CS;

	int m_TodoQ_count;
	int m_IdleQ_count;
};

class __declspec(align(16)) SSERayQueue1x1
{
public:
	SSERayQueue1x1();
	SSERayQueue1x1(int a_MaxSize);
	~SSERayQueue1x1();

	void initialize(int a_MaxSize);

	void Clear(void);					// 전체 Queue 를 비움(인덱스만 초기화)
	_sse_1x1_raypacket*	GetRay(void);	// Queue array 포인터 반환

	// IdleQ
	void IdleQ_EnQueue(_sse_1x1_rayitemData *item);		// EnQueue
	_sse_1x1_rayitemData* IdleQ_DeQueue(void);			// DeQueue

	// TodoQ
	void TodoQ_EnQueue(_sse_1x1_rayitemData *item);		// EnQueue
	_sse_1x1_rayitemData* TodoQ_DeQueue(void);			// DeQueue

private:
	_sse_1x1_raypacket		*m_RayList;

	int	m_MaxSize;
	_sse_1x1_rayitemData	*m_Item;

	_sse_1x1_rayitemData	*m_IdleQ_front;
	_sse_1x1_rayitemData	*m_IdleQ_rear;
	_sse_1x1_rayitemData	*m_TodoQ_front;
	_sse_1x1_rayitemData	*m_TodoQ_rear;

	GCriticalSection	m_IdleQ_CS;
	GCriticalSection	m_TodoQ_CS;

	int m_TodoQ_count;
	int m_IdleQ_count;
};

class __declspec(align(16)) SSERayTable2x2
{
public:
	SSERayTable2x2();
	SSERayTable2x2(int a_MaxSize);
	~SSERayTable2x2();

	void initialize(int a_MaxSize);

	void Clear(void);						// 전체 Table 를 비움(인덱스만 초기화)
	void Clear(int a_MaxIdx);				// 전체 Table 를 비움(인덱스만 초기화)
	_tableitemData*			GetTable(void);	// Table array 포인터 반환
	void Make_LinkedList(void);

	// TodoQ
	int Set_Item (int pi, int sub_pi);				// EnQueue
	int Get_Item (int &a_pi, int &a_sub_pi);		// DeQueue


private:
	int	m_MaxSize;
	_tableitemData		*m_Item;
	int m_CurrIdx, m_MaxIdx;

	GCriticalSection	m_Item_CS;

	int m_Item_count;
};

typedef SSERayTable2x2  SSERayTable1x1;

#endif