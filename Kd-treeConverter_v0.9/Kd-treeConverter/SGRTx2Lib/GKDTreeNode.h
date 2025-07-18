#ifndef GKDTREE_NODE_H
#define GKDTREE_NODE_H

/**
 *	KD Tree 를 이용한 공간 구조체.
 *	KD Tree 생성. 탐색은 오상락군의 코드를 기부 받아
 *	수정함.
 *
 *	by graphicsian.
 */
#include "GBoundingBox.h"
#include "GTriangleWrapper.h"
#include <cuda_runtime.h>

typedef uint2 kdtreeNode;

typedef struct _triinfo_ {
	int offset;
	GBoundingBox boundingBox;				//	split 되었을때의 가상의 bounding box
	GTriangleWrapper *pTriangleWrapper;		
} TriangleInfo;

typedef struct _bound_edge_ {
	float t;
	enum { START, END } type;
	const TriangleInfo *triangleInfo;
	bool isPlanar;
} BoundEdge;


#if 1
// -----------------------------------------------------------
// kdtree node
// -----------------------------------------------------------
	// macros for extracting node information
	#define IS_LEAF(node)				(((node).x & 3) == 3)
	#define IS_ROPE(node)				(((node).x & 4) == 4)

	#define SPLIT_AXIS(node)			( (node).x & 3)
	#define FIRST_CHILD_OFFSET(node)	( (node).x >> 3)
	#define SECOND_CHILD_OFFSET(node)	(((node).x >> 3) + 1)

	#define SPLIT_POS(node)				(*(float *)&((node).y))
	#define OBJECT_SIZE(node)			( (node).x >> 3)

	#define OBJECTLIST_OFFSET(node)		( (node).y)
	#define ROPE_NODE_OFFSET(node)		( (node).y)
#endif


#endif
