#include "GRayProfiler.h"
#include <assert.h>

//__class_code_content;
//__func__;

GRayProfiler::GRayProfiler()
	: m_bRayProfileMode(false)
{
	ResetCounts();
}

void GRayProfiler::ResetCounts()
{
	ResetInstructionCounts();
	ResetTreeOperatorCounts();
	ResetStateCounts();
}

void GRayProfiler::ResetInstructionCounts()
{
	for( unsigned int i = 0; i < NUM_OF_INSTRUCTIONS; i++ )
	{
		for( unsigned int j = 0; j < MAX_TREE_DEPTH; j++ )
			m_InstructionCounts[j][i] = 0;
		m_ShadowInstructionCounts[i] = 0;
	}
}

void GRayProfiler::CountInstruction( int depth, INSTRUCTION instruction, unsigned int count )
{
	unsigned int *pArray;
	if( depth != 0 )
		return;

	return;
	
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

		CountInstruction( depth, NORMALIZATION, count );
	}
	//else if( instruction == GET_TEXEL )
	//{
	//	pArray[SUBTRACT] += 4 * count;
	//	//pArray[MOVEMENT] += 14 * count;
	//	pArray[COMPARISON] += 6 * count;
	//	pArray[ADD] += 8 * count;
	//	pArray[MULTIPLY] += 11 * count;
	//	pArray[DIVISION] += 1 * count;

	//	// COLOR_SAMPLES = 4
	//	/*for(i=0; i<COLOR_SAMPLES; i++) {
	//		tex0[i] = m_pTextureData[tcoord0+i]	* rTexR;
	//		tex1[i] = m_pTextureData[tcoord1+i]	* rTexR;
	//		tex2[i] = m_pTextureData[tcoord2+i]	* rTexR;
	//		tex3[i] = m_pTextureData[tcoord3+i]	* rTexR;
	//	}*/
	//	//pArray[MOVEMENT] += 4 * 4 * count;
	//	pArray[MULTIPLY] += 4 * 4 * count;
	//	pArray[ADD] += 4 * 4 * count;

	//	//pArray[MOVEMENT] += 17 * count;
	//	pArray[SUBTRACT] += 5 * count;
	//	pArray[ADD] += 3 * count;
	//	pArray[MULTIPLY] += 6 * count;
	//}

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
	pArray[instruction] += count;
}

unsigned int GRayProfiler::GetInstructionCount( int depth, INSTRUCTION instruction )
{
	if( depth != 0 )
		return 0;
	return m_InstructionCounts[depth][instruction];
}

void GRayProfiler::ResetTreeOperatorCounts()
{
	for( unsigned int i = 0; i < NUM_OF_OPERTORS; i++ )
	{
		for( unsigned int j = 0; j < MAX_TREE_DEPTH; j++ )
			m_OperatorCounts[j][i] = 0;
		m_ShadowOperatorCounts[i] = 0;
	}
}

void GRayProfiler::CountTreeOperator( int depth, TREE_OPERATOR tree_operator, unsigned int count )
{
	if( depth != 0 )
		return;
	m_OperatorCounts[depth][tree_operator] += count;
}

unsigned int GRayProfiler::GetTreeOperatorCount( int depth, TREE_OPERATOR tree_operator )
{
	if( depth != 0 )
		return 0;
	return m_OperatorCounts[depth][tree_operator];
}

void GRayProfiler::ResetStateCounts()
{
	for( unsigned int i = 0; i < NUM_OF_STATES; i++ )
	{
		for( unsigned int j = 0; j < MAX_TREE_DEPTH; j++ )
			m_StateCounts[j][i] = 0;
		m_ShadowStateCounts[i] = 0;
	}
}

void GRayProfiler::CountState( int depth, STATE state, unsigned int count  )
{
	assert( state != NODES_COUNT );
	if( depth != 0 )
		return;
	m_StateCounts[depth][state] += count;
}

unsigned int GRayProfiler::GetStateCount( int depth, STATE state )
{
	if( depth != 0 )
		return 0;
	if( state == NODES_COUNT )
		return m_StateCounts[depth][INTERNAL_NODE] + m_StateCounts[depth][VISITED_LEAF_NODE];

	return m_StateCounts[depth][state];
}

bool GRayProfiler::NoInstructions( int depth )
{
	if( depth != 0 )
		return true;
	bool Nodata = true;
	
	for( unsigned int i = 0; i < NUM_OF_INSTRUCTIONS; i++ )
	{
		if( m_InstructionCounts[depth][i] != 0 )
		{
			Nodata = false;
			break;
		}
	}

	return Nodata;
}