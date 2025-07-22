#include "GRayTracerProfiler.h"

#include <assert.h>
#include <math.h>
#include <fstream>
#include <vector>
using namespace std;
#define _SAL_VERSION 0
#include <windows.h>
#include <sal.h>

namespace GRayTracerProfiler
{
	bool m_Running = false;

	unsigned int m_Width = 0;
	unsigned int m_Height = 0;

	int *m_TraverseUpCount[3];
	int *m_TraverseDownWithPushCount[3];
	int *m_TraverseDownWithoutPushCount[3];
	int *m_VisitedLeafCount[3];
	int *m_MailboxedTriangleCount[3];
	int *m_IntersectionCheckedTriangleCount[3];
	int *m_EmptyNodesCount[3];
	int *m_InternalNodeCount[3];

	GRayProfiler* m_pRayProfiler;

	int m_RayCount[3];
	char m_ProfileDataFileName[255];

	int m_CurrentDepth = 0;
	FUNCTION m_CurrentFunction;

	float m_PrintData[15*3];
	bool m_InstructionFlag = false;

	void ClearMemory();

	int m_PrimaryInstructionCount[NUM_OF_FUNCTIONS][NUM_OF_INSTRUCTIONS];
	ResultData m_ResultData;
};

int g_RayMaxSize[3];
//int g_MaxRaySize = 0;

void GRayTracerProfiler::Initialization( int ScreenWidth, int ScreenHeight, int LightCount, int MaxRayDepth )
{
	//g_MaxRaySize = 0;
	m_Width = 0;
	m_Height = 0;

	m_pRayProfiler = NULL;

	for( int i = 0; i < 3; i++ )
	{
		m_RayCount[i] = 0;
		g_RayMaxSize[i] = 0;
		m_TraverseUpCount[i] = NULL;
		m_TraverseDownWithPushCount[i] = NULL;
		m_TraverseDownWithoutPushCount[i] = NULL;
		m_VisitedLeafCount[i] = NULL;
		m_MailboxedTriangleCount[i] = NULL;
		m_IntersectionCheckedTriangleCount[i] = NULL;
		m_EmptyNodesCount[i] = NULL;
		m_InternalNodeCount[i] = NULL;
	}

	m_Width = ScreenWidth;
	m_Height = ScreenHeight;

	for( int i = 0; i < NUM_OF_FUNCTIONS; i++ )
		for( int j = 0; j < NUM_OF_INSTRUCTIONS; j++ )
			m_PrimaryInstructionCount[i][j] = 0;

	for( unsigned int i = 0; i < 3; i++ )
	{
		unsigned int size = m_Width * m_Height;
		switch( i )
		{
			// secondary ray
		case 1:
			size *= MaxRayDepth;
			break;
			// shadow ray
		case 2:
			size *= (1 + MaxRayDepth) * LightCount;
			break;
		}
		g_RayMaxSize[i] = size;
		m_TraverseUpCount[i] = new int[size];
		m_TraverseDownWithPushCount[i] = new int[size];
		m_TraverseDownWithoutPushCount[i] = new int[size];
		m_VisitedLeafCount[i] = new int[size];
		m_MailboxedTriangleCount[i] = new int[size];
		m_IntersectionCheckedTriangleCount[i] = new int[size];
		m_EmptyNodesCount[i] = new int[size];
		m_InternalNodeCount[i] = new int[size];

		for( unsigned int j = 0; j < size; j++ )
		{
			m_TraverseUpCount[i][j] = 0;
			m_TraverseDownWithPushCount[i][j] = 0;
			m_TraverseDownWithoutPushCount[i][j] = 0;
			m_VisitedLeafCount[i][j] = 0;
			m_MailboxedTriangleCount[i][j] = 0;
			m_IntersectionCheckedTriangleCount[i][j] = 0;
			m_EmptyNodesCount[i][j] = 0;
			m_InternalNodeCount[i][j] = 0;
		}
	}
}

void GRayTracerProfiler::Finalize()
{
	// Initialize
	m_Width = 0;
	m_Height = 0;

	m_pRayProfiler = NULL;

	ClearMemory();

	for( int i = 0; i < 3; i++ )
	{
		m_RayCount[i] = 0;
		g_RayMaxSize[i] = 0;
		m_TraverseUpCount[i] = NULL;
		m_TraverseDownWithPushCount[i] = NULL;
		m_TraverseDownWithoutPushCount[i] = NULL;
		m_VisitedLeafCount[i] = NULL;
		m_MailboxedTriangleCount[i] = NULL;
		m_IntersectionCheckedTriangleCount[i] = NULL;
		m_EmptyNodesCount[i] = NULL;
		m_InternalNodeCount[i] = NULL;
	}
}

unsigned int GRayTracerProfiler::GetRayCount()
{
	return m_RayCount[0];
}

unsigned int GRayTracerProfiler::GetSecondaryRayCount()
{
	return m_RayCount[1];
}

unsigned int GRayTracerProfiler::GetShadowRayCount()
{
	return m_RayCount[2];
}

bool GRayTracerProfiler::IsInitialized()
{
	return (m_Width!=0 && m_Height!=0);
}

void GRayTracerProfiler::SetProfilerOn( bool flag )
{
	m_Running = IsInitialized() && flag;
}

bool GRayTracerProfiler::IsProfilerOn()
{
	return m_Running;
}

void GRayTracerProfiler::SetProfileDataFileName( const char *filename )
{
	strcpy( m_ProfileDataFileName, filename );
}

const char* GRayTracerProfiler::GetProfileDataFileName()
{
	return m_ProfileDataFileName;
}

ResultData* GRayTracerProfiler::GetResultData()
{
	return &m_ResultData;
}

void GRayTracerProfiler::SetCurrentRayProfiler( GRayProfiler* pRayProfiler )
{
	m_pRayProfiler = pRayProfiler;
}

GRayProfiler* GRayTracerProfiler::GetCurrentRayProfiler()
{
	return m_pRayProfiler;
}

template<typename T>
void SafeDelete( T **Memory )
{
	if( *Memory != NULL )
	{
		delete *Memory;
		*Memory = NULL;
	}
}

void GRayTracerProfiler::ClearMemory()
{
	for( unsigned int i = 0; i < 3; i++ )
	{
		SafeDelete( &m_TraverseUpCount[i] );
		SafeDelete( &m_TraverseDownWithPushCount[i] );
		SafeDelete( &m_TraverseDownWithoutPushCount[i] );
		SafeDelete( &m_VisitedLeafCount[i] );
		SafeDelete( &m_MailboxedTriangleCount[i] );
		SafeDelete( &m_IntersectionCheckedTriangleCount[i] );
		SafeDelete( &m_EmptyNodesCount[i] );
		SafeDelete( &m_InternalNodeCount[i] );
	}
}

void GRayTracerProfiler::SetPrimaryRay()
{
	m_CurrentDepth = 0;
}

void GRayTracerProfiler::SetShadowRay()
{
	m_CurrentDepth = -1;
}

void GRayTracerProfiler::SetSecondaryRay( int depth )
{
	assert( depth > 0 );
	//m_CurrentDepth = depth;
	m_CurrentDepth = 1;
}

void GRayTracerProfiler::SetFunction( FUNCTION Function )
{
	m_CurrentFunction = Function;
}

void GRayTracerProfiler::CountInstruction( int Depth, FUNCTION Function, INSTRUCTION instruction, int count )
{
	m_CurrentDepth = Depth;
	m_CurrentFunction = Function;

	if( !IsProfilerOn() || !m_InstructionFlag )
		return;

	if( m_CurrentDepth != 0 )
		return;

	int *pArray = m_PrimaryInstructionCount[m_CurrentFunction];

	if( instruction == DOT_PRODUCT )
	{
		// x*x+y*y+z*z
		pArray[MULTIPLY] += 3 * count;
		pArray[ADD] += 2 * count;
	}
	else if( instruction == CROSS_PRODUCT )
	{
		// x_i = e_{ijk}*x_j*x_k
		pArray[MULTIPLY] += 2 * 3 * count;
		pArray[SUBTRACT] += 3 * count;
	}
	else if( instruction == NORMALIZATION )
	{
		pArray[MULTIPLY] += 3 * count;
		pArray[ADD] += 2 * count;
		pArray[SQRT] += 1 * count;
		pArray[DIVISION] += 4 * count;
	}
	else if( instruction == BARYCENTRIC )
	{
		pArray[ADD] += 8 * count;
		pArray[MULTIPLY] += 9 * count;
		//pArray[MOVEMENT] += 9 * count;

		CountInstruction( Depth, Function, NORMALIZATION, count );
	}
	else if( instruction == SSE_LOAD )
	{
		pArray[LOAD] += 4 * count;
	}
	else if( instruction == SSE_MOVE )
	{
		pArray[MOVE] += 4 * count;
	}
	else if( instruction == SSE_STORE )
	{
		pArray[STORE] += 4 * count;
	}
	else if( instruction == SSE_SET )
	{
		pArray[MOVE] += 3 * count;
	}
	else if( instruction == SSE_ADD )
	{
		pArray[ADD] += 3 * count;
	}
	else if( instruction == SSE_SUBTRACT )
	{
		pArray[SUBTRACT] += 3 * count;
	}
	else if( instruction == SSE_MULTIPLY )
	{
		pArray[MULTIPLY] += 3 * count;
	}
	m_PrimaryInstructionCount[m_CurrentFunction][instruction] += count;
}

void GRayTracerProfiler::CountTreeOperator( int Depth, TREE_OPERATOR TreeOperator )
{
	if( !IsProfilerOn() )
		return;
	m_CurrentDepth = Depth;
	int index = m_CurrentDepth;
	if( index == -1 )
		index = 2;
	else if( index > 0 )
		index = 1;

	int last = m_RayCount[index]-1;
	assert( last >= 0 );
	if( last < 0 )
	{
		MessageBox( NULL, "last < 0", "Error", MB_OK );
		exit( -1 );
	}
		//MessageBox( NULL, "Last is negative", "Error", MB_OK );
	assert( last < g_RayMaxSize[index] );
	if( last >= g_RayMaxSize[index] )
	{
		MessageBox( NULL, "last >= g_RayMaxSize[index]", "Error", MB_OK );
		exit( -1 );
	}

	if( TreeOperator == TRAVERSE_UP )
		m_TraverseUpCount[index][last] += 1;
	else if( TreeOperator == TRAVERSE_DOWN )
		m_TraverseDownWithPushCount[index][last] += 1;
	else if( TreeOperator == TRAVERSE_DOWN_WITHOUT_PUSH )
		m_TraverseDownWithoutPushCount[index][last] += 1;
}

void GRayTracerProfiler::CountState( int Depth, STATE State, int Count )
{
	if( !IsProfilerOn() )
		return;
	m_CurrentDepth = Depth;
	int index = m_CurrentDepth;
	if( index == -1 )
		index = 2;
	else if( index > 0 )
		index = 1;

	/*if( index == 0 )
	{
		m_pRayProfiler->CountState( State, Count );
	}*/

	if( State == RAY_COUNT )
	{
		int newIndex = m_RayCount[index];
		//assert( newIndex < g_MaxRaySize );
		assert( newIndex < g_RayMaxSize[index] );
		if( newIndex >= g_RayMaxSize[index] )
		{
			MessageBox( NULL, "newIndex >= g_RayMaxSize[index]", "Error", MB_OK );
			exit( -1 );
		}
		m_RayCount[index]++;
	}
	else
	{
		int last = m_RayCount[index]-1;
		assert( last >= 0 );
		if( last < 0 )
		{
			MessageBox( NULL, "last < 0", "Error", MB_OK );
			exit(-1);
		}
		//assert( last < g_MaxRaySize );
		assert( last < g_RayMaxSize[index] );
		if( last >= g_RayMaxSize[index] )
		{
			MessageBox( NULL, "last >= g_RayMaxSize[index]", "Error", MB_OK );
			exit(-1);
		}

		if( State == VISITED_LEAF_NODE )
			m_VisitedLeafCount[index][last] += Count;
		else if( State == INTERNAL_NODE )
			m_InternalNodeCount[index][last] += Count;
		else if( State == EMPTY_NODE )
			m_EmptyNodesCount[index][last] += Count;
		else if( State == MAILBOXED_TRIANGLE_COUNT )
			m_MailboxedTriangleCount[index][last] += Count;
		else if( State == TRIANGLE_COUNT )
			m_IntersectionCheckedTriangleCount[index][last] += Count;
	}
}

void GRayTracerProfiler::PrintData( const char *filename2 )
{
	/*assert( m_Width * m_Height == GetRayCount() );
	if( m_Width * m_Height != GetRayCount() )
	{
		MessageBox( NULL, "m_Width * m_Height != GetRayCount()", "Error", MB_OK );
		exit( -1 );
	}*/

	unsigned int TraverseUpTotal[3];
	unsigned int TraverseDownWithPushTotal[3];
	unsigned int TraverseDownWithoutPushTotal[3];
	unsigned int VisitedLeafTotal[3];
	unsigned int MailboxOffedTrianglesTotal[3];
	unsigned int IntersectionCheckedTrianglesTotal[3];
	unsigned int EmptyNodesTotal[3];

	float TraverseUpAvg[3];
	float TraverseDownWithPushAvg[3];
	float TraverseDownWithoutPushAvg[3];
	float VisitedLeafAvg[3];
	float MailboxOffedTrianglesAvg[3];
	float IntersectionCheckedTrianglesAvg[3];
	float EmptyNodesAvg[3];

	float TraverseUpStdDv[3];
	float TraverseDownWithPushStdDv[3];
	float TraverseDownWithoutPushStdDv[3];
	float VisitedLeafStdDv[3];
	float MailboxOffedTrianglesStdDv[3];
	float IntersectionCheckedTrianglesStdDv[3];
	float EmptyNodesStdDv[3];

	for( int j = 0; j < 3; j++ )
	{
		TraverseUpTotal[j] = 0;
		TraverseDownWithPushTotal[j] = 0;
		TraverseDownWithoutPushTotal[j] = 0;
		VisitedLeafTotal[j] = 0;
		MailboxOffedTrianglesTotal[j] = 0;
		IntersectionCheckedTrianglesTotal[j] = 0;
		EmptyNodesTotal[j] = 0;

		TraverseUpAvg[j] = 0.0f;
		TraverseDownWithPushAvg[j] = 0.0f;
		TraverseDownWithoutPushAvg[j] = 0.0f;
		VisitedLeafAvg[j] = 0.0f;
		MailboxOffedTrianglesAvg[j] = 0;
		IntersectionCheckedTrianglesAvg[j] = 0.0f;
		EmptyNodesAvg[j] = 0.0f;

		TraverseUpStdDv[j] = 0.0f;
		TraverseDownWithPushStdDv[j] = 0.0f;
		TraverseDownWithoutPushStdDv[j] = 0.0f;
		VisitedLeafStdDv[j] = 0.0f;
		MailboxOffedTrianglesStdDv[j] = 0;
		IntersectionCheckedTrianglesStdDv[j] = 0.0f;
		EmptyNodesStdDv[j] = 0.0f;

		for( int i = 0; i < m_RayCount[j]; i++ )
		{
			TraverseUpTotal[j] += m_TraverseUpCount[j][i];
			TraverseDownWithPushTotal[j] += m_TraverseDownWithPushCount[j][i];
			TraverseDownWithoutPushTotal[j] += m_TraverseDownWithoutPushCount[j][i];
			VisitedLeafTotal[j] += m_VisitedLeafCount[j][i];
			MailboxOffedTrianglesTotal[j] += m_MailboxedTriangleCount[j][i] + m_IntersectionCheckedTriangleCount[j][i];
			IntersectionCheckedTrianglesTotal[j] += m_IntersectionCheckedTriangleCount[j][i];
			EmptyNodesTotal[j] += m_EmptyNodesCount[j][i];
		}
		if( m_RayCount[j] != 0 )
		{
			TraverseUpAvg[j] = float(TraverseUpTotal[j])/float(m_RayCount[j]);
			TraverseDownWithPushAvg[j] = float(TraverseDownWithPushTotal[j])/float(m_RayCount[j]);
			TraverseDownWithoutPushAvg[j] = float(TraverseDownWithoutPushTotal[j])/float(m_RayCount[j]);
			VisitedLeafAvg[j] = float(VisitedLeafTotal[j])/float(m_RayCount[j]);
			MailboxOffedTrianglesAvg[j] += float(MailboxOffedTrianglesTotal[j])/float(m_RayCount[j]);
			IntersectionCheckedTrianglesAvg[j] = float(IntersectionCheckedTrianglesTotal[j])/float(m_RayCount[j]);
			EmptyNodesAvg[j] = float(EmptyNodesTotal[j])/float(m_RayCount[j]);
			for( int i = 0; i < m_RayCount[j]; i++ )
			{
				TraverseUpStdDv[j] += (TraverseUpAvg[j] - m_TraverseUpCount[j][i])*(TraverseUpAvg[j] - m_TraverseUpCount[j][i]);
				TraverseDownWithPushStdDv[j] += (TraverseDownWithPushAvg[j] - m_TraverseDownWithPushCount[j][i])*(TraverseDownWithPushAvg[j] - m_TraverseDownWithPushCount[j][i]);
				TraverseDownWithoutPushStdDv[j] += (TraverseDownWithoutPushAvg[j] - m_TraverseDownWithoutPushCount[j][i])*(TraverseDownWithoutPushAvg[j] - m_TraverseDownWithoutPushCount[j][i]);
				VisitedLeafStdDv[j] += (VisitedLeafAvg[j] - m_VisitedLeafCount[j][i])*(VisitedLeafAvg[j] - m_VisitedLeafCount[j][i]);
				MailboxOffedTrianglesStdDv[j] += (MailboxOffedTrianglesAvg[j] - m_MailboxedTriangleCount[j][i]-m_IntersectionCheckedTriangleCount[j][i])*(MailboxOffedTrianglesAvg[j] - m_MailboxedTriangleCount[j][i]-m_IntersectionCheckedTriangleCount[j][i]);
				IntersectionCheckedTrianglesStdDv[j] += (IntersectionCheckedTrianglesAvg[j] - m_IntersectionCheckedTriangleCount[j][i])*(IntersectionCheckedTrianglesAvg[j] - m_IntersectionCheckedTriangleCount[j][i]);
				EmptyNodesStdDv[j] += (EmptyNodesAvg[j] - m_EmptyNodesCount[j][i])*(EmptyNodesAvg[j] - m_EmptyNodesCount[j][i]);
			}
			TraverseUpStdDv[j] = sqrt(float(TraverseUpStdDv[j])/float(m_RayCount[j]));
			TraverseDownWithPushStdDv[j] = sqrt(float(TraverseDownWithPushStdDv[j])/float(m_RayCount[j]));
			TraverseDownWithoutPushStdDv[j] = sqrt(float(TraverseDownWithoutPushStdDv[j])/float(m_RayCount[j]));
			MailboxOffedTrianglesStdDv[j] = sqrt(float(MailboxOffedTrianglesStdDv[j])/float(m_RayCount[j]));
			VisitedLeafStdDv[j] = sqrt(float(VisitedLeafStdDv[j])/float(m_RayCount[j]));
			IntersectionCheckedTrianglesStdDv[j] = sqrt(float(IntersectionCheckedTrianglesStdDv[j])/float(m_RayCount[j]));
			EmptyNodesStdDv[j] = sqrt(float(EmptyNodesStdDv[j])/float(m_RayCount[j]));
		}
	}
	
	fstream fs;
	fs.open( m_ProfileDataFileName, ios_base::out );

	for( unsigned int j = 0; j < 3; j++ )
	{
		if( j == 0 )
		{
			fs << "Primary Ray" << endl;
		}
		else if( j == 1 )
		{
			fs << "Secondary Ray (Reflection & Refraction)" << endl;
		}
		else
		{
			fs << "Shadow Ray" << endl;
		}

		fs << "# of Rays : " << m_RayCount[j] << endl;

		fs << "Total # of Traverse Up : " << TraverseUpTotal[j] << endl;
		fs << "Average # of Traverse Up : " << TraverseUpAvg[j] << endl;
		fs << "Standard Deviation of Traverse Up : " << TraverseUpStdDv[j] << endl;

		fs << "Total # of Traverse Down with Push : " << TraverseDownWithPushTotal[j] << endl;
		fs << "Average # of Traverse Down with Push : " << TraverseDownWithPushAvg[j] << endl;
		fs << "Standard Deviation of Traverse Down with Push : " << TraverseDownWithPushStdDv[j] << endl;

		fs << "Total # of Traverse Down without Push : " << TraverseDownWithoutPushTotal[j] << endl;
		fs << "Average # of Traverse Down without Push : " << TraverseDownWithoutPushAvg[j] << endl;
		fs << "Standard Deviation of Traverse Down without Push : " << TraverseDownWithoutPushStdDv[j] << endl;

		fs << "Total # of Visited Leaf Nodes : " << VisitedLeafTotal[j] << endl;
		fs << "Average # of Visited Leaf Nodes : " << VisitedLeafAvg[j] << endl;
		fs << "Standard Deviation of Visited Leaf Nodes : " << VisitedLeafStdDv[j] << endl;

		fs << "Total # of Intersection-checked Triangles : " << IntersectionCheckedTrianglesTotal[j] << endl;
		fs << "Average # of Intersection-checked Triangles : " << IntersectionCheckedTrianglesAvg[j] << endl;
		fs << "Standard Deviation of Intersection-checked Triangles : " << IntersectionCheckedTrianglesStdDv[j] << endl;

		fs << "Total # of Intersect-checked Triangles(Mailbox-offed) : " << MailboxOffedTrianglesTotal[j] << endl;
		fs << "Average # of Intersect-checked Triangles(Mailbox-offed) : " << MailboxOffedTrianglesAvg[j] << endl;
		fs << "Standard Deviation of Intersect-checked Triangles(Mailbox-offed) : " << MailboxOffedTrianglesStdDv[j] << endl;

		fs << "Total # of Visited Empty Nodes : " << EmptyNodesTotal[j] << endl;
		fs << "Average # of Visited Empty Nodes : " << EmptyNodesAvg[j] << endl;
		fs << "Standard Deviation of Visited Empty Nodes : " << EmptyNodesStdDv[j] << endl;

		if( j != 2 )
			fs << endl;
	}

	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.RayCount[j] = m_RayCount[j];

	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.TraverseUpAvg[j] = TraverseUpAvg[j];
	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.TraverseUpStdDv[j] = TraverseUpStdDv[j];

	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.TraverseDownWithoutPushAvg[j] = TraverseDownWithoutPushAvg[j];
	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.TraverseDownWithoutPushStdDv[j] = TraverseDownWithoutPushStdDv[j];

	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.TraverseDownWithPushAvg[j] = TraverseDownWithPushAvg[j];
	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.TraverseDownWithPushStdDv[j] = TraverseDownWithPushStdDv[j];

	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.VisitedLeafAvg[j] = VisitedLeafAvg[j];
	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.VisitedLeafStdDv[j] = VisitedLeafStdDv[j];

	for( unsigned int j = 0; j < 3; j++ )
	{
		m_ResultData.IntersectionCheckedTrianglesAvg[j] = IntersectionCheckedTrianglesAvg[j];
		m_ResultData.MailboxOffedTrianglesAvg[j] = MailboxOffedTrianglesAvg[j];
	}
	for( unsigned int j = 0; j < 3; j++ )
	{
		m_ResultData.IntersectionCheckedTrianglesStdDv[j] = IntersectionCheckedTrianglesStdDv[j];
		m_ResultData.MailboxOffedTrianglesStdDv[j] = MailboxOffedTrianglesStdDv[j];
	}

	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.EmptyNodesAvg[j] = EmptyNodesAvg[j];
	for( unsigned int j = 0; j < 3; j++ )
		m_ResultData.EmptyNodesStdDv[j] = EmptyNodesStdDv[j];

	fs.close();

	if( m_InstructionFlag )
	{
		fstream instructionResultFile( "instruction_result.txt", ios_base::out );
		if( !instructionResultFile == false )
		{

			for( int j = 0; j < NUM_OF_INSTRUCTIONS; j++ )
			{
				for( int i = 0; i < NUM_OF_FUNCTIONS; i++ )
					instructionResultFile << m_PrimaryInstructionCount[i][j] << '\t';
				instructionResultFile << endl;
			}
		}
		instructionResultFile.close();
		m_InstructionFlag = false;
	}
}

//float* GRayTracerProfiler::GetPrintData()
//{
//	return m_PrintData;
//}

void GRayTracerProfiler::SetInstructionCount( bool flag )
{
	m_InstructionFlag = flag;
}