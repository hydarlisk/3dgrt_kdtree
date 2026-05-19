#pragma once
#include <vector>
#include "Kd-treeConstructor.h"

// CUDA 작업의 최종 결과를 담을 구조체
struct CudaClipResult {
    double total_contrib_L;
    double total_contrib_R;
};

// CPU(Kd-treeConstructor.cpp)에서 호출할 메인 인터페이스 함수
CudaClipResult calculate_clipped_contributions_cuda(
    const std::vector<const PrimList*>& active_triangles,
    const BoundingBox& left_bbox,
    const BoundingBox& right_bbox
);