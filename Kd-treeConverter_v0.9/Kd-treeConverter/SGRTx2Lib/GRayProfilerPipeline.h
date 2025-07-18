#pragma once

#include "SSERenderPipeline.h"
#include "GRayProfilerOption.h"

class GRayProfilerPipeline : public SSERenderPipeline
{
public:
	GRayProfilerPipeline( GScene *pScene, SSESceneData *pSSESceneData );
	~GRayProfilerPipeline();

	void FindLeafNode1x1( _sse_1x1_raypacket *rp, _sse_1x1_isect *is, float &t_near, float &t_far_,
		const unsigned int* ray_dir, KdTreeNode* node, unsigned int &stackIndex, 
		_sse_float &rcpRayDir );
private:
	int leaf_depth;
	int node_depth;
	int tree_depth;
	int empty_count;
	int push_count;

public:
	void setTestRay( GRayProfilerOption *pOption );
	void Render1x1( int nThreadID );
	void TracePacket1x1 ( unsigned int quad, int nIdx );
	void IsectPacket1x1( const KdTreeNode *node, int nIdx );
	void shading1x1 (int nIdx);
	void TraceShadowPacket1x1( unsigned int quad );
	void IsectShadowPacket1x1( const KdTreeNode *node );

private:
	GBoundingBox bb_stack[100];
	GRayProfilerOption *m_pOption;
	unsigned int m_TestRayCount;
	SSESceneData *m_pSSESceneData;
};