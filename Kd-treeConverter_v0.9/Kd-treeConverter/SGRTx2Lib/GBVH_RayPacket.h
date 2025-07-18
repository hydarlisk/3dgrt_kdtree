#ifndef __GBVH_RayPacket__
#define __GBVH_RayPacket__

#include "GScene.h"
//#include "GTexture.h"
//#include "GTextureManager.h"
#include "GRenderSystem.h"

#include <stdio.h>
#include "SSE_math.h"
//#include "SSERenderPipeline.h"
#include "SSERenderCommon.h"
#include "GClassMacro.h"
#include "GRAY.h"
//#include "GBVHStructure.h"

#define PACKET_WIDTH 8
#define PACKET_SIZE (PACKET_WIDTH * PACKET_WIDTH)

class GBVH_RayPacket
{	
public:
	GBVH_RayPacket()
	{ 	};
	~GBVH_RayPacket()
	{   };

	GVector getDirection(int which) const
	{		       
		return GVector(dir[0][which], dir[1][which], dir[2][which]) ;		
	}
	GVector getOrigin(int which) const
	{
		return GVector(org[0][which], org[1][which], org[2][which]) ;									
	}
	_sse_1x1_raypacket getRay(int which) const;	

	Ray getRay2(int which) const;	
	void getRay(_sse_1x1_raypacket& tmp, int which) const;
	void getRay(Ray& ray, int which) const;	

	Ray getRayMore(int which) const;
/*
	void setOrigin(const ss_Point3& origin);
	void setup_eye_packet(int i, int j, ss_Window& win, ss_Camera& camera);
*/
	int frustumAABBTest(GBoundingBox& aabb);          // FRUSTUM - AABB intersect check!!

	void pre_cal(void);
	int  frustumAABBTest2(GBoundingBox& aabb);         // FRUSTUM - AABB intersect check!!
	bool frustumAABBTest3(GBoundingBox& aabb);         // FRUSTUM - AABB intersect check!!

	float org    [4][PACKET_SIZE];
    float dir    [4][PACKET_SIZE];	
    float inv_dir[4][PACKET_SIZE];  // 1/dir
	float t         [PACKET_SIZE];  // distance

	// FOR RAY - AABB 
	// An Efficient and Robust Ray-Box Intersection Algorithm
	// 현재 안 쓰이고 있음.
	int sign[3][PACKET_SIZE];
	float inv_direction_x[PACKET_SIZE];
	float inv_direction_y[PACKET_SIZE];
	float inv_direction_z[PACKET_SIZE];

	// FOR FRUSTUM - AABB 
	// Large Ray Packets for Real-time Whitted Ray Tracing
	GVector d[4]; // corner ray direction
	GVector n[4]; // corner ray normal
	GVector o;    // corner ray origin
	float b[4]; // 평면 거리.

};

#endif