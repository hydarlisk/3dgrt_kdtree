#pragma once

#include "GRayProfiler.h"

/*! GRayProfiler
* \brief Ray 하나에 대한 Profiler
* 
* @author Hybrid
*/

struct ResultData
{
	int RayCount[3];
	float TraverseUpAvg[3];
	float TraverseUpStdDv[3];
	float TraverseDownWithoutPushAvg[3];
	float TraverseDownWithoutPushStdDv[3];
	float TraverseDownWithPushAvg[3];
	float TraverseDownWithPushStdDv[3];
	float VisitedLeafAvg[3];
	float VisitedLeafStdDv[3];
	float IntersectionCheckedTrianglesAvg[3];
	float IntersectionCheckedTrianglesStdDv[3];
	float MailboxOffedTrianglesAvg[3];
	float MailboxOffedTrianglesStdDv[3];
	float EmptyNodesAvg[3];
	float EmptyNodesStdDv[3];
};

/*! \namespace GRayTracerProfiler
* \brief Image 를 이루는 Rays 에 대한 Profiler
* 
* @author Hybrid
*/
namespace GRayTracerProfiler
{
	void count_secondary();
	void count_shadow();
	void PrintTest();

	void Initialization( int ScreenWidth, int ScreenHeight, int LightCount, int MaxRayDepth );
	void Finalize();

	//! 1x1 sampling only
	unsigned int GetRayCount();
	unsigned int GetSecondaryRayCount();
	unsigned int GetShadowRayCount();

	//void ResetCounts();

	void SetCurrentRayProfiler( GRayProfiler* pRayProfiler );
	GRayProfiler* GetCurrentRayProfiler();

	//void AddData( GRayProfiler *pRayProfiler );

	void PrintData( const char *filename );

	void SetProfilerOn( bool flag = true );
	bool IsProfilerOn();
	void SetProfileDataFileName( const char *filename );
	const char* GetProfileDataFileName();
	//float* GetPrintData();

	ResultData* GetResultData();

	void SetPrimaryRay();
	void SetShadowRay();
	void SetSecondaryRay( int depth ); //!< depth > 0

	void SetFunction( FUNCTION Function );
	void CountInstruction( int Depth, FUNCTION Function, INSTRUCTION Instruction, int Count );
	void CountTreeOperator( int Depth, TREE_OPERATOR TreeOperator );
	void CountState( int Depth, STATE State, int Count );
	bool IsInitialized();

	void SetInstructionCount( bool flag = true );

	#if PROFILER_ON == ON
	#define SET_CURRENT_RAY( depth ) \
		{if(depth==0) \
		{GRayTracerProfiler::SetPrimaryRay();} \
		else \
		{GRayTracerProfiler::SetSecondaryRay(depth);}}
	#define SET_PRIMARY_RAY() \
		GRayTracerProfiler::SetPrimaryRay();
	#define SET_SECONDARY_RAY( depth ) \
		GRayTracerProfiler::SetSecondaryRay( depth );
	#define SET_SHADOW_RAY() \
		GRayTracerProfiler::SetShadowRay();
	#define	SET_FUNCTION( Function ) \
		GRayTracerProfiler::SetFunction( Function );
	#define COUNT_INSTRUCTION( Depth, Function, Instruction, Count ) \
		GRayTracerProfiler::CountInstruction( Depth, Function, Instruction, Count ); // Depth, Function, 
	#define COUNT_TREE_OPERATOR( Depth, TreeOperator ) \
		GRayTracerProfiler::CountTreeOperator( Depth, TreeOperator );
	#define COUNT_STATE( Depth, State, Count ) \
		GRayTracerProfiler::CountState( Depth, State, Count );
	#define RUN_PROFILER() \
		GRayTracerProfiler::SetProfilerOn();
	#define STOP_PROFILER() \
		GRayTracerProfiler::SetProfilerOn( false );
	#else
	#define SET_CURRENT_RAY( depth )
	#define SET_PRIMARY_RAY()
	#define SET_SECONDARY_RAY( depth )
	#define SET_SHADOW_RAY()
	#define	SET_FUNCTION( Function )
	#define COUNT_INSTRUCTION( Depth, Function, Instruction, Count )
	#define COUNT_TREE_OPERATOR( Depth, TreeOperator )
	#define COUNT_STATE( Depth, State, Count )
	#define RUN_PROFILER()
	#define STOP_PROFILER()
	#endif
};