/**************************************************************
  File name: MyMathUtility.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#define TO_DEGREE 57.29577951308232
#define TO_RADIAN  0.01745329251994330

#define MyABS(x)     (((x) < 0)? -(x) : (x))
#define MyMAX(a,b)   (((a) > (b)) ? (a) : (b))
#define MyMIN(a,b)   (((a) < (b)) ? (a) : (b))

void dMyVecCrossProduct(double *, double *, double *);
void dMyVecNormalize(double *); 
double dMyVecLength(double *v);
double dMyVecDotProduct(double *, double *); 

void fMyVecCrossProduct(float *, float *, float *);
void fMyVecNormalize(float *); 
float fMyVecLength(float *);
float fMyVecDotProduct(float *, float *);  
void fMyVecElaborateFaceNormal(float *, float *, float *, float *);

