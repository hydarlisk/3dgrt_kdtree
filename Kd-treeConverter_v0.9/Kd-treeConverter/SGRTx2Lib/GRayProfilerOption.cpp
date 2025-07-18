#include "GRayProfilerOption.h"

GRayProfilerOption::GRayProfilerOption()
	: m_bTestRayMode(false), m_bDrawKdTree(false), m_bDrawTestRays(true), m_TestRayCount(0), m_pTestRayDirection(NULL), m_bDrawLeafNodes(true),
	m_bDrawInternalNodes(false), m_bDrawScene(true), m_bDrawIntersectedNode(true), m_bSetRayToCamera(true), m_StepMode(0), m_CurrentStep(0), m_MaximumStep(0), 
	m_TriangleCountInCurrentLeafNode(0), m_TriangleCoun(0), m_EmptyNodesCount(0), m_PushCount(0), m_PopCount(0)
{
	
}

GRayProfilerOption::~GRayProfilerOption()
{
	if( m_pTestRayDirection != NULL )
	{
		//delete m_pTestRayDirection;
		m_pTestRayDirection = NULL;
	}
}

void GRayProfilerOption::createTestRays( unsigned int count, GVector origin )
{
	if( m_pTestRayDirection != NULL )
	{
		delete m_pTestRayDirection;
		m_pTestRayDirection = NULL;
	}

	m_TestRayCount = count;
	m_pTestRayDirection = new GVector[count];
	m_TestRayOrigin = origin;
}

void GRayProfilerOption::setTestRayOrigin( GVector origin )
{
	m_TestRayOrigin = origin;
}

GVector GRayProfilerOption::getTestRayOrigin()
{
	return m_TestRayOrigin;
}

//! return false, if no further leaf node exists
bool GRayProfilerOption::increaseStep()
{
	if( m_CurrentStep >= getMaximumStep() )
	{
		m_CurrentStep = getMaximumStep();
		return false;
	}
	m_CurrentStep++;
	return true;
}

//! return false, if no nearer leaf node exists
bool GRayProfilerOption::decreaseStep()
{
	if( m_CurrentStep == 0 )
		return false;
	m_CurrentStep--;
	if( m_CurrentStep >= getMaximumStep() )
		m_CurrentStep = getMaximumStep();
	return true;
}

void GRayProfilerOption::setMaximumStep( unsigned int maximum )
{
	m_MaximumStep = maximum; 
	if( m_CurrentStep >= m_MaximumStep )
		m_CurrentStep = m_MaximumStep;
}