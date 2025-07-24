#pragma once
#include "cuCommonDefs.cuh"

void launchCudaRender(const cuCamera& cam, const cuKdTree& kd, uchar4* resultImage);
