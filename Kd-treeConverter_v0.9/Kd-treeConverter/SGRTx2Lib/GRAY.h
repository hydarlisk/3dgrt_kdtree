
#ifndef __GRAY_h__
#define __GRAY_h__

#include "GVector.h"

class Ray 
{
public:
	GVector vPos, vDir;

	Ray() {    };
	Ray(const GVector& vP, const GVector& vD)
		:vPos(vP), vDir(vD)
	{     };

	// BVH¿¡¼­ 
	// FOR RAY - AABB 
	// An Efficient and Robust Ray-Box Intersection Algorithm
	int sign[3];
	float inv_direction_x;
	float inv_direction_y;
	float inv_direction_z;
};

#endif
