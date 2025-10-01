#pragma once

#include <vector>
#include "../Kd-treeConverter.h" // ExtendedVertex, MyMIN, MyMAX 등을 위함
#include "../Kd-treeConstructor.h" // BoundingBox를 위함

namespace KdTreeClipper {
    // 다각형의 면적을 계산하는 함수
    double calculate_polygon_area(const std::vector<ExtendedVertex>& polygon);

    // BoundingBox를 AABB_Clipping 라이브러리 형식으로 변환
    BoundingBox convert_to_internal_aabb(const BoundingBox& bbox);

    // 삼각형을 AABB에 맞춰 클리핑하는 메인 함수
    void clip_triangle_against_AABB(
        const ExtendedVertex triangle[3],
        const BoundingBox& bbox,
        std::vector<ExtendedVertex>& out_polygon
    );
}