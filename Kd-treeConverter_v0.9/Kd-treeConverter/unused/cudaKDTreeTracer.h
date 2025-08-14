#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cmath>
#include "sgrt_interface.h"
#include "RayTraversal.h"
#include "cuCommonDefs.cuh"
#include "SGRTx2Lib/cuda_math.h"

int g_render_width = 800;
int g_render_height = 600;

// CUDA 에러 체크 함수
void checkCudaErrors(cudaError err);
void copyCompositeObjectToCUDA(const CompositeObject* hostObject, CUDACompositeObject*& cudaObject);
void initCudaRendering(CompositeObject& compositeObject, float* frameBuffer, bool* renderFlag);