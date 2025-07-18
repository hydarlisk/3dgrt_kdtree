
#pragma once

#include <math.h>

#define _GVEC_vADD(e,a,b)  { e.x=a.x+b.x; e.y=a.y+b.y; e.z=a.z+b.z; e.w=a.w+b.w; }
#define _GVEC_vSUB(e,a,b)  { e.x=a.x-b.x; e.y=a.y-b.y; e.z=a.z-b.z; e.w=a.w-b.w; }
#define _GVEC_vMUL(e,a,v)  { e.x=a.x*v; e.y=a.y*v; e.z=a.z*v; e.w=a.w*v; }
#define _GVEC_vDIV(e,a,v)  { float invf = 1.f/v; e.x=a.x*invf; e.y=a.y*invf; e.z=a.z*invf; e.w=a.w*invf; }
#define _GVEC_rINNDOT(a,b)   (a.x*b.x + a.y*b.y + a.z*b.z)
#define _GVEC_rINNDOT2(a,b)  (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
#define _GVEC_vNORMAL(a)  { float length = sqrtf( a.x*a.x + a.y*a.y + a.z*a.z ); if ( length == 0.0f ) { a.x = 0.0f; a.y = 0.0f; a.z = 0.0f; } \
	float invLength = 1 / length; a.x = a.x * invLength; a.y = a.y * invLength; a.z = a.z * invLength; }
#define _GVEC_vLENGTH(a)    (sqrt( a.x*a.x + a.y*a.y + a.z*a.z ))
#define _GVEC_vRCP(e,a)    { e.x=1.f/a.x; e.y=1.f/a.y; e.z=1.f/a.z; e.w=1.f/a.w; }


#define _GPNT_vADD(e,a,b)  { e.x=a.x+b.x; e.y=a.y+b.y; e.z=a.z+b.z; }
#define _GPNT_vSUB(e,a,b)  { e.x=a.x-b.x; e.y=a.y-b.y; e.z=a.z-b.z; }
#define _GPNT_vMUL(e,a,v)  { e.x=a.x*v; e.y=a.y*v; e.z=a.z*v; }
#define _GPNT_vDIV(e,a,v)  { float invf = 1.f/v; e.x=a.x*invf; e.y=a.y*invf; e.z=a.z*invf; }

#define _GCOL_vADD(e,c1,c2)  { e.r=c1.r+c2.r; e.g=c1.g+c2.g; e.b=c1.b+c2.b; e.a=c1.a+c2.a; }
#define _GCOL_vSUB(e,c1,c2)  { e.r=c1.r-c2.r; e.g=c1.g-c2.g; e.b=c1.b-c2.b; e.a=c1.a-c2.a; }
#define _GCOL_vMULF(e,c1,v)  { e.r=c1.r*v; e.g=c1.g*v; e.b=c1.b*v; }
#define _GCOL_vMULC(e,c1,c2) { e.r=c1.r*c2.r; e.g=c1.g*c2.g; e.b=c1.b*c2.b; e.a=c1.a*c2.a; }
