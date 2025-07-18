
//**************************************
// Threading Option
//	0 : No Thread ( Original Code )
//	1 : 1 Thread
//	2 : 2 Thread
//	4 : 4 Thread
//	...
//	n : <---- must be less than MAX_THREADING
//**************************************
#define MAX_THREADING				16

#if !MAX_THREADING

#define SSE_RENDER_THREADING		0
#define LOCALSHADE_THREADING		0
#define CPU_FINALGATHER_THREADING	0

#else

#define SSE_RENDER_THREADING		16
#define LOCALSHADE_THREADING		4
#define CPU_FINALGATHER_THREADING	0

#endif

#define THREADING_JITTER_SIZE		8
