#include "GBVH_RayPacket.h"


/*
void
ss_RayPacket::setOrigin(const ss_Point3& origin)
{
	for(int i=0; i<PACKET_SIZE; ++i)
	{
		org[0][i] = origin.x;
		org[1][i] = origin.y;
		org[2][i] = origin.z;
	}
}

void
ss_RayPacket::setup_eye_packet(int i, int j, ss_Window& win, ss_Camera& camera)
{
	
	float tmp[3], leng;
	unsigned int k=0;	
	for(int height = j; height < j+PACKET_WIDTH; ++height)
	{
		for(int width = i; width < i+PACKET_WIDTH; ++width)
		{
			tmp[0] = win.a.x + (width+0.5f)*win.winWw*win.b.x/win.winW + (height+0.5f)*win.winWh*win.c.x/win.winH - camera.eye.x;
			tmp[1] = win.a.y + (width+0.5f)*win.winWw*win.b.y/win.winW + (height+0.5f)*win.winWh*win.c.y/win.winH - camera.eye.y;
			tmp[2] = win.a.z + (width+0.5f)*win.winWw*win.b.z/win.winW + (height+0.5f)*win.winWh*win.c.z/win.winH - camera.eye.z;

			leng = 1.0f / sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);
			dir[0][k] = tmp[0] * leng;
			dir[1][k] = tmp[1] * leng;
			dir[2][k] = tmp[2] * leng;

			// FOR RAY - AABB intersection
			// An Efficient and Robust Ray-Box Intersection Algorithm
			inv_direction_x[k] = 1.0f / dir[0][k];
			inv_direction_y[k] = 1.0f / dir[1][k];
			inv_direction_z[k] = 1.0f / dir[2][k];
			sign[0][k] = inv_direction_x[k] < 0;
			sign[1][k] = inv_direction_y[k] < 0;
			sign[2][k] = inv_direction_z[k] < 0;			
			// END FOR RAY - AABB intersection

			++k;
		}
	}
	
	// FOR FRUSTUM - AABB 
	// Large Ray Packets for Real-time Whitted Ray Tracing
	d[0] = FVector3(dir[0][0],                           dir[1][0],                           dir[2][0]); // left up
	d[1] = FVector3(dir[0][PACKET_SIZE - PACKET_WIDTH],  dir[1][PACKET_SIZE - PACKET_WIDTH],  dir[2][PACKET_SIZE - PACKET_WIDTH]); // left down
	d[2] = FVector3(dir[0][PACKET_SIZE-1],               dir[1][PACKET_SIZE-1],               dir[2][PACKET_SIZE-1]); // right down
	d[3] = FVector3(dir[0][PACKET_WIDTH-1],              dir[1][PACKET_WIDTH-1],              dir[2][PACKET_WIDTH-1]); // right up
	
	o = FVector3(org[0][0], org[1][0], org[2][0]);

	n[0] = d[0].crossp(d[1]);         // corner ray normal
	n[1] = d[1].crossp(d[2]);
	n[2] = d[2].crossp(d[3]);
	n[3] = d[3].crossp(d[0]);
	
	b[0] = (o.dotp(n[0]));            // 평면 거리.
	b[1] = (o.dotp(n[1]));
	b[2] = (o.dotp(n[2]));
	b[3] = (o.dotp(n[3]));
	// END FOR FRUSTUM - AABB intersection
}
*/

_sse_1x1_raypacket
GBVH_RayPacket::getRay(int which) const
{
	_sse_1x1_raypacket tmp;
	tmp.d.x = dir[0][which];
	tmp.d.y = dir[1][which];
	tmp.d.z = dir[2][which];

	tmp.o.x = org[0][which];
	tmp.o.y = org[1][which];
	tmp.o.z = org[2][which];
	return tmp;
}
// get ray direction & origin
void
GBVH_RayPacket::getRay(_sse_1x1_raypacket& tmp, int which) const
{
	tmp.d.x = dir[0][which];
	tmp.d.y = dir[1][which];
	tmp.d.z = dir[2][which];

	tmp.o.x = org[0][which];
	tmp.o.y = org[1][which];
	tmp.o.z = org[2][which];
}

void
GBVH_RayPacket::getRay(Ray& r, int which) const
{
	/*
	r.vPos.x = org[0][which];
	r.vPos.y = org[1][which];
	r.vPos.z = org[2][which];	

	r.vDir.x = dir[0][which];
	r.vDir.y = dir[1][which];
	r.vDir.z = dir[2][which];
	*/
	r.vPos = GVector(org[0][which], org[1][which], org[2][which]);
	r.vDir = GVector(dir[0][which], dir[1][which], dir[2][which]);
}


Ray
GBVH_RayPacket::getRay2(int which) const
{
	return Ray( GVector(org[0][which], org[1][which], org[2][which]),
			    GVector(dir[0][which], dir[1][which], dir[2][which]) );
}


Ray 
GBVH_RayPacket::getRayMore(int which) const
{
	Ray r;
	
	r.vPos = GVector( org[0][which], org[1][which], org[2][which] );
	r.vDir = GVector( dir[0][which], dir[1][which], dir[2][which] );
	
	r.inv_direction_x = inv_direction_x[which];
	r.inv_direction_y = inv_direction_y[which];
	r.inv_direction_z = inv_direction_z[which];
	r.sign[0] = sign[0][which];
	r.sign[1] = sign[1][which];
	r.sign[2] = sign[2][which];	
	
	return r;
}


// --------------------------------------------------------------------------------
// FRUSTUM - AABB intersect check!!
// --------------------------------------------------------------------------------
// !! Large Ray Packets for Real-time Whitted Ray Tracing 여기에 있는 방법을 그냥 내가 짠거.
// 4개의 corner ray로 이루어진
// 1. frustum의 각 평면의 방정식을 찾는다. -> n[i], b[i] : normal과 offset.
// n vector와 b로 평면이 정의됨.
int
GBVH_RayPacket::frustumAABBTest(GBoundingBox& aabb)
{
	GVector p[8]; // AABB의 8 꼭지점.
	
	p[0] = aabb.m_Min;
	p[1] = GVector(aabb.m_Min.x, aabb.m_Min.y, aabb.m_Max.z);
	p[2] = GVector(aabb.m_Max.x, aabb.m_Min.y, aabb.m_Max.z);
	p[3] = GVector(aabb.m_Max.x, aabb.m_Min.y, aabb.m_Min.z);

	p[4] = GVector(aabb.m_Min.x, aabb.m_Max.y, aabb.m_Min.z);
	p[5] = GVector(aabb.m_Min.x, aabb.m_Max.y, aabb.m_Max.z);
	p[6] = aabb.m_Max;
	p[7] = GVector(aabb.m_Max.x, aabb.m_Max.y, aabb.m_Min.z);
	
	
	// http://www.everfall.com/paste/id.php?e7nyaefyeins
	// 여기에 있는 방법.
	for(int i=0; i<4; ++i)
	{
		bool inside = false;

		for(int j=0; j<8; ++j)
		{
			if(p[j].innerProduct(n[i]) - b[i] < 0)
			{
				inside = true;
				break;
			}
		}
		if(!inside)
			return PACKET_SIZE;
	}
	return 0;	
}

int
GBVH_RayPacket::frustumAABBTest2(GBoundingBox& aabb)
{
	// http://groups.google.com/group/comp.graphics.algorithms/browse_thread/thread/33a96fc960ba3ea5
	int    ret = 0; 
	GVector vmin, vmax; 
	for(int i = 0; i < 4; ++i) 
	{ 
		// X axis 
		if(n[i].x > 0) 
		{ 
			vmin.x = aabb.m_Min.x; 
			vmax.x = aabb.m_Max.x; 
		} 
		else 
		{ 
			vmin.x = aabb.m_Max.x; 
			vmax.x = aabb.m_Min.x; 
		} 
		// Y axis 
		if(n[i].y > 0) 
		{ 
			vmin.y = aabb.m_Min.y; 
			vmax.y = aabb.m_Max.y; 
		} 
		else 
		{ 
			vmin.y = aabb.m_Max.y; 
			vmax.y = aabb.m_Min.y; 
		} 
		// Z axis 
		if(n[i].z > 0) 
		{ 
			vmin.z = aabb.m_Min.z; 
			vmax.z = aabb.m_Max.z; 
		} 
		else 
		{ 
			vmin.z = aabb.m_Max.z; 
			vmax.z = aabb.m_Min.z; 
		} 
		if( vmin.innerProduct (n[i]) - b[i] > 0) 
			return PACKET_SIZE; 
		if( vmax.innerProduct (n[i]) - b[i] >= 0) 
			ret = 0; 
	} 
	return ret;
	/*
	int    ret = 0; 
	GVector vmin, vmax; 
	for(int i = 0; i < 4; ++i) 
	{ 		
		// X axis 
		n[i].x > 0 ? (vmin.x = aabb.m_Min.x, vmax.x = aabb.m_Max.x) : (vmin.x = aabb.m_Max.x, vmax.x = aabb.m_Min.x) ;

		// Y axis 
		n[i].y > 0 ? (vmin.y = aabb.m_Min.y, vmax.y = aabb.m_Max.y) : (vmin.y = aabb.m_Max.y, vmax.y = aabb.m_Min.y) ;
				
		// Z axis 
		n[i].z > 0 ? (vmin.z = aabb.m_Min.z, vmax.z = aabb.m_Max.z) : (vmin.z = aabb.m_Max.z, vmax.z = aabb.m_Min.z) ;
		

		if( vmin.innerProduct (n[i]) - b[i] > 0) 
			return PACKET_SIZE; 
		if( vmax.innerProduct (n[i]) - b[i] >= 0) 
			ret = 0; 
	} 
	return ret;
	*/
}

// http://www.ce.chalmers.se/~uffe/vfc.pdf
// 이 논문에 있는 방법.
// outside면 return false
// 겹치면 return true.
// 아직 수정중!!!
bool
GBVH_RayPacket::frustumAABBTest3(GBoundingBox& aabb)
{
	bool intersect = false;
	GVector vn, vp;
	for(int i=0; i<4; ++i)
	{
		if(n[i].x > 0)
		{
			vn = aabb.m_Max;
			vp = aabb.m_Min;
		}
		else
		{
			vn = aabb.m_Min;
			vp = aabb.m_Max;
		}
		if( vn.innerProduct(n[i]) + b[i] > 0)
			return false;		
		
		if( vp.innerProduct(n[i]) + b[i] > 0)
			intersect = true;			
	}
	if(intersect)
		return true;
	else
		return true;
}



void
GBVH_RayPacket::pre_cal(void)
{	
	// FOR FRUSTUM - AABB 
	// Large Ray Packets for Real-time Whitted Ray Tracing
	d[0] = GVector(dir[0][0],                           dir[1][0],                           dir[2][0]); // left up
	d[1] = GVector(dir[0][PACKET_SIZE - PACKET_WIDTH],  dir[1][PACKET_SIZE - PACKET_WIDTH],  dir[2][PACKET_SIZE - PACKET_WIDTH]); // left down
	d[2] = GVector(dir[0][PACKET_SIZE-1],               dir[1][PACKET_SIZE-1],               dir[2][PACKET_SIZE-1]); // right down
	d[3] = GVector(dir[0][PACKET_WIDTH-1],              dir[1][PACKET_WIDTH-1],              dir[2][PACKET_WIDTH-1]); // right up
	
	o = GVector(org[0][0], org[1][0], org[2][0]);

	n[0] = d[0].outerProduct(d[1]);         // corner ray normal
	n[1] = d[1].outerProduct(d[2]);
	n[2] = d[2].outerProduct(d[3]);
	n[3] = d[3].outerProduct(d[0]);
	
	b[0] = (o.innerProduct(n[0]));            // 평면 거리.
	b[1] = (o.innerProduct(n[1]));
	b[2] = (o.innerProduct(n[2]));
	b[3] = (o.innerProduct(n[3]));
	// END FOR FRUSTUM - AABB intersection
}



