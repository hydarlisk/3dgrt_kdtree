#pragma once

#include <vector>
#include "../Kd-treeConverter.h" // ExtendedVertex, MyMIN, MyMAX 등을 위함
#include "../Kd-treeConstructor.h" // BoundingBox를 위함

namespace KdTreeClipper {
    // 다각형의 면적을 계산하는 함수
    double calculate_polygon_area(const std::vector<ExtendedVertex>& polygon);

    // 삼각형을 AABB에 맞춰 클리핑하는 메인 함수
    void clip_triangle_against_AABB(
        const ExtendedVertex triangle[3],
        const BoundingBox& bbox,
        std::vector<ExtendedVertex>& out_polygon
    );
}