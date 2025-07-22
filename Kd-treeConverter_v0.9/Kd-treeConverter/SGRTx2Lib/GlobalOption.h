#pragma once

//! 요건 아직 만드는 중
/**
 * Global Parameter.
 * Global Parameter 는 로딩할 때 parameters.ini 에서 얻어온 값을 사용한다.
 * 대소문자는 가리지 않으며, parameter 값은 성능을 위해서 별도의 변수로 얻어온 후 사용하는 것을 추천함.
**/
void setGlobalParameterFloat( const char *parameter_name, float value );
void setGlobalParameterInteger( const char *parameter_name, int value );

float getGlobalParameterFloat( const char *parameter_name );
int getGlobalParameterInteger( const char *parameter_name );

#define ON  0
#define OFF 1

/**
 * Profiler On/Off.
 * 성능을 위해서는 반드시 OFF 로 해야함.
**/
#define PROFILER_ON OFF


/**
 * Frustum Culling On/Off.
 * kD-tree 의 Leaf node 의 구성방법이나 SSE용 RayPacket Trav/Isec 방법이 결정됨.
**/
#define FRUSTUM_CULLING OFF

/**
 * SAH 를 Function Pointer 로 할지를 결정.
 * OFF 일 경우 기본 SAH 방법을 사용함. 
 * 성능을 위해서는 반드시 OFF 로 해야함.
**/
#define SPLIT_FUNCTION_POINTER OFF

//! eg. 사용예
enum { ARITHMETIC, HARMONIC };

/**
 * VISIBILITY 를 계산하는 방법.
**/
#define VISIBILITY_METHOD HARMONIC

//! 실제 구현 코드 내에서 사용 예
// ******************************************************************
// ************* #if 와 함께 #elif 로 비교해주는 것이 좋음 **********
// ******************************************************************
#if VISIBILITY_METHOD == ARITHMETIC
// Arithmetic Code
/*
float getArithmeticVisibility()  
{
}
*/
#elif VISIBILITY_METHOD == HARMONIC 
// Harmonic Code
/*
float getHarmonicisibility()
{
}
*/
#endif