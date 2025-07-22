#pragma once

#include "GlobalOption.h"

enum INSTRUCTION { FUNCTION_ENTER, 
LOAD, MOVE, STORE, BRANCH, 
ADD, SUBTRACT, MULTIPLY, DIVISION, SQRT, COMPARISON, 
NEGATION, //!< arithmetic negation
BOOLEAN_AND, BOOLEAN_OR, BOOLEAN_XOR, BOOLEAN_NEGATOR, 
BIT_AND, BIT_OR, BIT_XOR, SHIFT,
POWER, ABS, 
// Special
DOT_PRODUCT, CROSS_PRODUCT, NORMALIZATION, BARYCENTRIC, //GET_TEXEL, 
MIN_OP, MAX_OP, 
// SSE
SSE_LOAD, SSE_MOVE, SSE_STORE, SSE_SET, 
SSE_ADD, SSE_SUBTRACT, SSE_MULTIPLY, 
SSE_DOT_PRODUCT, SSE_INVERSE, 
NUM_OF_INSTRUCTIONS };

enum TREE_OPERATOR {
	TRAVERSE_DOWN,				//! # of traverse down operations
	TRAVERSE_DOWN_WITHOUT_PUSH, //! # of traverse down without push operations
	TRAVERSE_UP,				//! # of traverse up (pop) operations
	NUM_OF_OPERTORS
};

enum RAY_TYPE {
	PRIMARY_RAY,					//! TracePacket1x1(), IsectPacket1x1(), shading1x1()
	SHADOW_RAY,					//! TraceShadowPacket1x1(), IsectShadowPacket1x1(), shading1x1()
	REFLECTION_RAY,					//! with in TracePacket1x1()
	REFRACTION_RAY,					//! with in TracePacket1x1()
	NUM_OF_RAY_TYPES
};

enum FUNCTION {
	RAY_GENERATION, 
	PIXEL_STORE, // SEPERATE1
	TRAVERSE_INITIALIZATION, 
	TRAVERSE, 
	TRAVERSE_FRONT, 
	TRAVERSE_BACK, 
	TRAVERSE_BOTH, 
	SEPERATE2, 
	INTERSECTION_CHECK,			//! IsectPacket1x1(), IsectShadowPacket1x1()
	INTERSECTION_CHECK_MAILBOX, 
	INTERSECTION_CHECK_DISTANCE, 
	INTERSECTION_CHECK_U, 
	INTERSECTION_CHECK_V, 
	INTERSECTION_CHECK_U_V, 
	INTERSECTION_CHECK_LIVED, 
	SEPERATE3, 
	SHADING_INITIALIZATION, 
	SHADING_LOCAL_SHADING, 
	SHADING_SHADOW,
	SHADING_GET_TEXTURE, 
	SHADING_PHONG_SHADING, 
	SHADING_REFLECTION, 
	SHADING_REFRACTION, 
	NUM_OF_FUNCTIONS
};

enum STATE {
	NODES_COUNT,				//! # of visited nodes
	EMPTY_NODE,					//! # of visited empty nodes
	INTERNAL_NODE,				//! # of visited internal nodes
	VISITED_LEAF_NODE,			//! # of visited leaf nodes
	MAILBOXED_TRIANGLE_COUNT,	//! # of Mailboxed-triangles
	TRIANGLE_COUNT,				//! # of intersection-checked triangles
	//SHADING_INITIALIZATION, 
	//TEXTURE_ACCESS, 

	RAY_COUNT, 

	NUM_OF_STATES
};


#define MAX_TREE_DEPTH 100

/*! \class GRayProfiler
 * \brief Ray 하나에 대한 Profiler
 * 
 * @author Hybrid
*/
class GRayProfiler
{
public:
	GRayProfiler();

	void ResetCounts();

	void ResetInstructionCounts();
	void CountInstruction( int depth, INSTRUCTION instruction, unsigned int count = 1 );
	unsigned int GetInstructionCount( int depth, INSTRUCTION instruction );

	bool NoInstructions( int depth );

	void ResetTreeOperatorCounts();
	void CountTreeOperator( int depth, TREE_OPERATOR tree_operator, unsigned int count = 1 );
	unsigned int GetTreeOperatorCount( int depth, TREE_OPERATOR tree_operator );

	void ResetStateCounts();
	//! state parameter value must not be NODES_COUNT.
	void CountState( int depth, STATE state, unsigned int count = 1 );
	unsigned int GetStateCount( int depth, STATE state );

	void setRayProfileMode( bool flag = true ) { m_bRayProfileMode = flag; }
	bool isRayProfileMode() { return m_bRayProfileMode; }

	//SetMode?

private:
	bool m_bRayProfileMode;

	// for Primary and Secondary Ray
	unsigned int m_InstructionCounts[MAX_TREE_DEPTH][NUM_OF_INSTRUCTIONS];
	unsigned int m_OperatorCounts[MAX_TREE_DEPTH][NUM_OF_OPERTORS];
	unsigned int m_StateCounts[MAX_TREE_DEPTH][NUM_OF_STATES];

	// for Shadow Ray
	unsigned int m_ShadowInstructionCounts[NUM_OF_INSTRUCTIONS];
	unsigned int m_ShadowOperatorCounts[NUM_OF_OPERTORS];
	unsigned int m_ShadowStateCounts[NUM_OF_STATES];
};