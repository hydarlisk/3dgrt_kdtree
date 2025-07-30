#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void testK() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    printf("%d %d // ", x, y);
}

void kernelTestFuncion() {
    dim3 blockDim(16, 16);
    dim3 gridDim(4, 4);
    testK << < gridDim, blockDim >> > ();
}

void cudaCopyTest() {
    cudaArray* d_kdtree_nodes;
    int tree_node_count = 1292643; // 문제가 되는 값 사용

    cudaChannelFormatDesc node_desc = cudaCreateChannelDesc<uint2>();
    CUDA_CHECK(cudaMallocArray(&d_kdtree_nodes, &node_desc, tree_node_count, 0));

    std::cout << "cudaMallocArray test succeeded!" << std::endl;

    // 할당된 배열 해제
    CUDA_CHECK(cudaFreeArray(d_kdtree_nodes));
}