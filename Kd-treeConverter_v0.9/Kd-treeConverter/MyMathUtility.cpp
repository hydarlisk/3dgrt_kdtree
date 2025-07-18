/**************************************************************
  File name: MyMathUtility.cpp
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#include <stdio.h>
#include <math.h>
#include <float.h>

#include "MyMathUtility.h"

void dMyVecCrossProduct(double *v1, double *v2, double *v) {
	v[0] =   ( (v1[1] * v2[2]) - (v1[2] * v2[1]) );
	v[1] = - ( (v1[0] * v2[2]) - (v1[2] * v2[0]) );
	v[2] =   ( (v1[0] * v2[1]) - (v1[1] * v2[0]) );
}

void fMyVecCrossProduct(float *v1, float *v2, float *v) {
	v[0] =   ( (v1[1] * v2[2]) - (v1[2] * v2[1]) );
	v[1] = - ( (v1[0] * v2[2]) - (v1[2] * v2[0]) );
	v[2] =   ( (v1[0] * v2[1]) - (v1[1] * v2[0]) );
}

void dMyVecNormalize(double *v) {
	double length;

	length = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
	if (length < 0.0000001) {
		printf("dMyVecNormalize: almost null vector with length %15.7e.\n", length); // Need to modify
//		printf("dMyVecNormalize: (%10.7e, %10.7e, %10.7e)\n", v[0], v[1], v[2]);
	}
    for (int i = 0; i < 3; i++)
		v[i] /= length;
}

void fMyVecNormalize(float *v) {
	double length;

	length = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
	if (length < 0.00000001) {
		//printf("fMyVecNormalize: almost null vector with length %15.7e.\n", length); // Need to modify
//		printf("fMyVecNormalize: (%10.7e, %10.7e, %10.7e)\n", v[0], v[1], v[2]);
		v[0] = v[1] = v[2] = 0.0f;
		return;
	}

    for (int i = 0; i < 3; i++)
		v[i] = (float) (((double) v[i])/length);
}

float fMyVecLength(float *v) {
	return( (float) sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) );
}

double dMyVecLength(double *v) {
	return( sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) );
}

double dMyVecDotProduct(double *v1, double *v2) {
	return( v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] ); 
}

float fMyVecDotProduct(float *v1, float *v2) {
	return( v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] );
} 

void fMyVecElaborateFaceNormal(float *v0, float *v1, float *v2, float *face_norm) {
	int i;
	float min_dotprod;
	double v01[3], v02[3], facenorm[3], tmp;

		min_dotprod = FLT_MAX;

		for (i = 0; i < 3; i++) {
			v01[i] = (double) *(v1 + i) - *(v0 + i);
			v02[i] = (double) *(v2 + i) - *(v0 + i);
		}
		dMyVecNormalize(v01); 
		dMyVecNormalize(v02); 

		tmp = dMyVecDotProduct(v01, v02);
//		a0 = acos(tmp)*TO_DEGREE;

		if (fabs(tmp) < min_dotprod) {
			min_dotprod = fabs(tmp);
			dMyVecCrossProduct(v01, v02, facenorm);
			dMyVecNormalize(facenorm);
		}

		for (i = 0; i < 3; i++) {
			v01[i] = (double) *(v2 + i) - *(v1 + i);
			v02[i] = (double) *(v0 + i) - *(v1 + i);
		}
		dMyVecNormalize(v01); 
		dMyVecNormalize(v02); 

		tmp = dMyVecDotProduct(v01, v02);
//		a1 = acos(tmp)*TO_DEGREE;

		if (fabs(tmp) < min_dotprod) {
			min_dotprod = fabs(tmp);
			dMyVecCrossProduct(v01, v02, facenorm);
			dMyVecNormalize(facenorm);
		}

		for (i = 0; i < 3; i++) {
			v01[i] = (double) *(v0 + i) - *(v2 + i);
			v02[i] = (double) *(v1 + i) - *(v2 + i);
		}
		dMyVecNormalize(v01); 
		dMyVecNormalize(v02); 

		tmp = dMyVecDotProduct(v01, v02);
//		a2 = acos(tmp)*TO_DEGREE;

		if (fabs(tmp) < min_dotprod) {
			min_dotprod = fabs(tmp);
			dMyVecCrossProduct(v01, v02, facenorm);
			dMyVecNormalize(facenorm);
		}

		*(face_norm + 0) = (float) facenorm[0];
		*(face_norm + 1) = (float) facenorm[1];
		*(face_norm + 2) = (float) facenorm[2];
}