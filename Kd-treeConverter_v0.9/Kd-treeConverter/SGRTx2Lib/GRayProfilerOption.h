
#include "GVector.h"

#pragma once

/**
 *	OpenGL 관련 옵션
 *  객체 복사 하면 안됨
 *	by Hybrid.
*/
#include "GRenderOption.h"
#include "GRayProfiler.h"
#include "GRayTracerProfiler.h"

class GRayProfilerOption : public GRenderOption
{
public:
	bool isTestRayMode()
		{ return m_bTestRayMode; }
	//! KD Tree 의 전체 혹은 일부를 그림
	bool isDrawKdTree()					// default : false
		{ return m_bDrawKdTree; }
	bool isDrawLeafNodes()				// default : true
		{ return m_bDrawLeafNodes; }
	bool isDrawInternalNodes()			// default : false
		{ return m_bDrawInternalNodes; }
	bool isDrawTestRays()				// default : true
		{ return m_bDrawTestRays; }
	bool isDrawScene()					// default : true
		{ return m_bDrawScene; }
	bool isDrawIntersectedNode()
		{ return m_bDrawIntersectedNode; }
	bool isSetRayToCamera()
		{ return m_bSetRayToCamera; }

	void setTestRayMode( bool flag = true )
		{ m_bTestRayMode = flag; }
	void setDrawKdTree( bool flag = true )
		{ m_bDrawKdTree = flag; }
	void setDrawLeafNodes( bool flag = true )
		{ m_bDrawLeafNodes = flag; }
	void setDrawInternalNodes( bool flag = true )
		{ m_bDrawInternalNodes = flag; }
	void setDrawTestRays( bool flag = true )
		{ m_bDrawTestRays = flag; }
	void setDrawScene( bool flag = true )
		{ m_bDrawScene = flag; }
	void setDrawIntersectedNode( bool flag = true )
		{ m_bDrawIntersectedNode = flag; }
	void setRayToCamera( bool flag = true )
		{ m_bSetRayToCamera = flag; }

//! Test Ray
	unsigned int getTestRayCount() { return m_TestRayCount; }
	void createTestRays( unsigned int count, GVector origin );
	GVector* getTestRayDirectionArray() { return m_pTestRayDirection; }

	void setTestRayOrigin( GVector origin );
	GVector getTestRayOrigin();

//! Leaf/Internal Step Mode
	void setLeafStepMode( bool flag = true )
	{
		if( flag == false )
			m_StepMode = 0;
		else
			m_StepMode = 1;
	}
	void setInternalStepMode( bool flag = true )
	{
		if( flag == false )
			m_StepMode = 0;
		else
			m_StepMode = 2;
	}
	bool isLeafStepMode()
	{
		return m_StepMode==1;
	}
	bool isInternalStepMode()
	{
		return m_StepMode==2;
	}

	//! return false, if no further leaf/internal node exists
	bool increaseStep();
	//! return false, if no nearer leaf/internal node exists
	bool decreaseStep();
	//! 0 step means draw all leaf/internal nodes
	unsigned int getCurrentStep() { return m_CurrentStep; }

	void setMaximumStep( unsigned int maximum );
	unsigned int getMaximumStep() { return m_MaximumStep; }

	void setEmptyNodesCount( unsigned int count ) { m_EmptyNodesCount = count; }
	unsigned int getEmptyNodesCount() { return m_EmptyNodesCount; }
	void setPushCount( unsigned int count ) { m_PushCount = count; }
	unsigned int getPushCount() { return m_PushCount; }
	void setPopCount( unsigned int count ) { m_PopCount = count; }
	unsigned int getPopCount() { return m_PopCount; }

//! Information
	void setTriangleCountInCurrentLeafNode( unsigned int count )
		{ m_TriangleCountInCurrentLeafNode = count; }
	void setTriangleCount( unsigned int count )
		{ m_TriangleCoun = count; }
	unsigned int getTriangleCountInCurrentLeafNode() { return m_TriangleCountInCurrentLeafNode; }
	unsigned int getTriangleCount() { return m_TriangleCoun; }

	//GRayTracerProfiler m_RayTracerProfiler;
	GRayProfiler m_TestRayProfiler;

private:
	bool m_bTestRayMode;
	bool m_bSetRayToCamera;
	bool m_bDrawScene;
	bool m_bDrawKdTree;
	bool m_bDrawTestRays;
	bool m_bDrawLeafNodes;
	bool m_bDrawInternalNodes;
	bool m_bDrawIntersectedNode;
	unsigned int m_TestRayCount;
	GVector m_TestRayOrigin;
	GVector* m_pTestRayDirection;
	int m_StepMode;
	unsigned int m_CurrentStep;
	unsigned int m_TriangleCountInCurrentLeafNode;
	unsigned int m_TriangleCoun;
	unsigned int m_MaximumStep;
	unsigned int m_EmptyNodesCount;
	unsigned int m_PushCount;
	unsigned int m_PopCount;

public:
	GRayProfilerOption(void);
	~GRayProfilerOption(void);
};
