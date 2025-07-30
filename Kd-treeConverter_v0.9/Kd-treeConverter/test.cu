#include <stdio.h>
#include <cuda_runtime.h>

__global__ void testK() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	printf("%d %d // ", x, y);
}

void kernelTestFuncion(){
	dim3 blockDim(16, 16);
	dim3 gridDim(4,4);
	testK <<< gridDim, blockDim >>> ();
}