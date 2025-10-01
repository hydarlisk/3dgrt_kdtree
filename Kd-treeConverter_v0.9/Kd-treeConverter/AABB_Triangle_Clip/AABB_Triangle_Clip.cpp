#include "../Kd-treeConverter.h"
#include "AABB_Triangle_Clip.h"
#include "../MyMathUtility.h" // dMyVecCrossProduct, dMyVecLength 등을 위함
#include <cmath>

namespace KdTreeClipper {

    enum Axis { _X = 0, _Y, _Z };
    enum Side { _MIN = 0, _MAX };

    // 점이 경계 내부에 있는지 확인하는 내부 함수
    inline bool is_inside(float coord, float boundary, Side side) {
        if (side == _MIN) return coord >= boundary;
        return coord <= boundary;
    }

    // 폴리곤을 하나의 평면으로 클리핑하는 내부 함수
    void clip_polygon_against_plane(
        const std::vector<ExtendedVertex>& in_polygon,
        std::vector<ExtendedVertex>& out_polygon,
        float boundary, Axis axis, Side side)
    {
        out_polygon.clear();
        if (in_polygon.empty()) return;

        for (size_t i = 0; i < in_polygon.size(); ++i) {
            const ExtendedVertex& p1 = in_polygon[i];
            const ExtendedVertex& p2 = in_polygon[(i + 1) % in_polygon.size()];

            bool p1_inside = is_inside(p1.vertex[axis], boundary, side);
            bool p2_inside = is_inside(p2.vertex[axis], boundary, side);

            if (p1_inside) {
                out_polygon.push_back(p1);
            }

            if (p1_inside != p2_inside) {
                float t = (boundary - p1.vertex[axis]) / (p2.vertex[axis] - p1.vertex[axis]);
                ExtendedVertex intersection;
                for (int j = 0; j < 3; ++j) {
                    intersection.vertex[j] = p1.vertex[j] + t * (p2.vertex[j] - p1.vertex[j]);
                }
                intersection.material_ID = p1.material_ID; // material ID 상속
                out_polygon.push_back(intersection);
            }
        }
    }

    // 메인 클리핑 함수 구현
    void clip_triangle_against_AABB(
        const ExtendedVertex triangle[3],
        const BoundingBox& bbox,
        std::vector<ExtendedVertex>& out_polygon)
    {
        std::vector<ExtendedVertex> pg[2];
        int current = 0;

        pg[current].assign(triangle, triangle + 3);

        // X, Y, Z 축 순서로 6개의 평면에 대해 클리핑 수행
        clip_polygon_against_plane(pg[current], pg[1 - current], bbox.min[0], _X, _MIN); current = 1 - current;
        clip_polygon_against_plane(pg[current], pg[1 - current], bbox.max[0], _X, _MAX); current = 1 - current;
        clip_polygon_against_plane(pg[current], pg[1 - current], bbox.min[1], _Y, _MIN); current = 1 - current;
        clip_polygon_against_plane(pg[current], pg[1 - current], bbox.max[1], _Y, _MAX); current = 1 - current;
        clip_polygon_against_plane(pg[current], pg[1 - current], bbox.min[2], _Z, _MIN); current = 1 - current;
        clip_polygon_against_plane(pg[current], pg[1 - current], bbox.max[2], _Z, _MAX); current = 1 - current;

        out_polygon = pg[current];
    }

    // 클리핑 후 생성된 3D 폴리곤의 면적 계산
    double calculate_polygon_area(const std::vector<ExtendedVertex>& polygon) {
        if (polygon.size() < 3) {
            return 0.0;
        }

        double total_area = 0.0;
        const ExtendedVertex& v0 = polygon[0];

        for (size_t i = 1; i < polygon.size() - 1; ++i) {
            const ExtendedVertex& v1 = polygon[i];
            const ExtendedVertex& v2 = polygon[i + 1];

            double edge1[3] = { (double)v1.vertex[0] - v0.vertex[0], (double)v1.vertex[1] - v0.vertex[1], (double)v1.vertex[2] - v0.vertex[2] };
            double edge2[3] = { (double)v2.vertex[0] - v0.vertex[0], (double)v2.vertex[1] - v0.vertex[1], (double)v2.vertex[2] - v0.vertex[2] };

            double cross_product[3];
            dMyVecCrossProduct(edge1, edge2, cross_product);
            total_area += dMyVecLength(cross_product);
        }

        return total_area / 2.0;
    }
}