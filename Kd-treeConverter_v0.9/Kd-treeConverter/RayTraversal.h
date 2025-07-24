#pragma once

#define BACKFACE_CULLING   false
#define MAX_RAY_DEPTH     16
#define MAX_STACK_SIZE    32

#define AIR_INDEX          1

#define RAY_DIST_EPSILON      0.00001f
#ifndef FLT_MAX
#define FLT_MAX  3.402823466e+38F        /* max value */
#endif

//using namespace KDTConstructor;
//using namespace KDTConverter;

typedef struct __declspec(align(16)) _Hit {
	float dist;				// hit distance
	int tacc;				// triangle index + 1 (0 if there is no hit)
	float u;				// u value of barycetric coordinate
	float v;				// v value of barycetric coordinate
	union {					// hit position
		float pf[3];
		struct {
			float x, y, z;
		} p;
	};
	unsigned int addr;		// pixel address
	union {					// plane normal
		float nf[3];
		struct {
			float x, y, z;
		} n;
	};
	int material_ID;		// material ID
	union {					// shading color
		float color[3];
		struct {
			float r, g, b;
		} c;
	};
	int pad;
} Hit;

typedef struct __declspec(align(16)) _KdStack {
	float t_far_;										// 4
	float t_near;										// 4
	KdTreeNode* node;									// 4
	unsigned int depth;									// 4
} KdStack;

// Macros for extracting node information
#define IS_LEAF(node)				(((node).x & 7) == 3)
#define SPLIT_AXIS(node)			( (node).x & 3)
#define FIRST_CHILD_OFFSET(node)	( (node).x >> 3)
#define SECOND_CHILD_OFFSET(node)	(((node).x >> 3) + 1)
#define SPLIT_POS(node)				(*(float *)&((node).y))
#define OBJECT_SIZE(node)			( (node).x >> 3)
#define OBJECTLIST_OFFSET(node)		( (node).y)

void InitRay(int nIdx, Ray *a_Ray);
void InitShadowRay(void);
void IsectShadowRay( const KdTreeNode *node );
void TraceShadowRay(void);
void checkVisibility(const float* objectPos, const float* lightPos);
void IsectRay( const KdTreeNode *node, int nIdx );
void TraceRay(int nIdx, KdTree *a_kd_tree, Hit *a_is);
void Shading (int nIdx);
void Render(int nJobID);

