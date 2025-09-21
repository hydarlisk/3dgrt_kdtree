#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "ParallelKdTreeBuilder.h"
#include "Kd-treeConstructor.h"

// Helper structure to bundle triangle data with its tag for sorting
struct TaggedTriangle {
    TriangleList triangle;
    unsigned int tag;
    unsigned int originalIndex; // To build the final offset list
};

// --- Helper functions from the paper with O(1) complexity ---

// Calculate level (depth) from node index
inline int level(unsigned int i) {
    if (i == (unsigned int)-1) return -1;
    if (i == 0) return 0;
    // Using __lzcnt intrinsic for performance if available (MSVC, GCC, Clang)
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, i + 1);
    return index;
#else
    return 31 - __builtin_clz(i + 1);
#endif
}

// Calculate the total number of nodes in subtree s for a tree with N total nodes (ss(s))
inline unsigned int ss(unsigned int s, unsigned int N) {
    if (s >= N) return 0;
    int L = level(N - 1) + 1;
    int l = level(s);

    if (l >= L - 1) return 1;

    unsigned int fllc_s = (s + 1) << (L - 1 - l);
    fllc_s -= 1;

    unsigned int num_inner_nodes = (1 << (L - 1 - l)) - 1;

    unsigned int num_nodes_on_lowest_level = 0;
    if (fllc_s < N) {
        unsigned int max_nodes_in_subtree_on_lowest = 1 << (L - 1 - l);
        num_nodes_on_lowest_level = std::min((unsigned int)N - fllc_s, max_nodes_in_subtree_on_lowest);
    }

    return num_inner_nodes + num_nodes_on_lowest_level;
}

// Calculate the starting index in the sorted array for elements belonging to subtree s at level l (sb(s,l))
inline unsigned int sb(unsigned int s, int l, unsigned int N) {
    if (l <= 0) return 0;
    int L = level(N - 1) + 1;

    unsigned int first_node_on_level_l = (1 << l) - 1;
    if (s < first_node_on_level_l) return s;

    unsigned int sb_s = first_node_on_level_l;
    unsigned int nls_s = s - first_node_on_level_l; // number of left siblings

    if (L - 1 - l > 0) {
        sb_s += nls_s * ((1 << (L - 1 - l)) - 1); // inner nodes of left siblings
        unsigned int first_node_on_last_level = (1 << (L - 1)) - 1;
        if (N > first_node_on_last_level) {
            unsigned int nodes_on_last = N - first_node_on_last_level;
            unsigned int nodes_per_sibling_on_last = (1u << (L - 1 - l));
            sb_s += std::min(nls_s * nodes_per_sibling_on_last, nodes_on_last);
        }
    }

    return sb_s;
}

bool build_kdtree_parallel(CompositeObject* c_object) {
    unsigned int num_triangles = c_object->n_triangles;
    if (num_triangles == 0) return false;

    std::vector<TaggedTriangle> taggedTriangles(num_triangles);
    for (unsigned int i = 0; i < num_triangles; ++i) {
        taggedTriangles[i] = { g_pTriangleInfos[i], 0, i };
    }

    int num_levels = level(num_triangles - 1) + 1;

    for (int l = 0; l < num_levels; ++l) {
        int split_axis = l % 3;
        std::sort(taggedTriangles.begin(), taggedTriangles.end(),
            [split_axis](const TaggedTriangle& a, const TaggedTriangle& b) {
                if (a.tag != b.tag) {
                    return a.tag < b.tag;
                }
                float center_a = a.triangle.AABB.min[split_axis] + a.triangle.AABB.max[split_axis];
                float center_b = b.triangle.AABB.min[split_axis] + b.triangle.AABB.max[split_axis];
                return center_a < center_b;
            });

        if (l == num_levels - 1) break;

        std::vector<unsigned int> next_tags(num_triangles);
        unsigned int first_node_on_level = (1 << l) - 1;

#pragma omp parallel for
        for (int i = 0; i < (int)num_triangles; ++i) {
            unsigned int currentTag = taggedTriangles[i].tag;
            if (currentTag < first_node_on_level) {
                next_tags[i] = currentTag;
                continue;
            }

            unsigned int left_child = 2 * currentTag + 1;
            unsigned int right_child = 2 * currentTag + 2;

            unsigned int segment_begin = sb(currentTag, l, num_triangles);
            unsigned int num_nodes_in_left_subtree = ss(left_child, num_triangles);
            unsigned int pivot_pos = segment_begin + num_nodes_in_left_subtree;

            // [수정된 부분] 논문의 알고리즘을 정확히 구현한 3-way 분기
            if (i < (int)pivot_pos) {
                // 피벗보다 왼쪽에 있으면 왼쪽 자식으로
                next_tags[i] = left_child;
            }
            else if (i > (int)pivot_pos) {
                // 피벗보다 오른쪽에 있으면 오른쪽 자식으로
                next_tags[i] = right_child;
            }
            else {
                // 정확히 피벗 위치에 있으면 태그를 변경하지 않고 현재 노드에 고정
                next_tags[i] = currentTag;
            }
        }

        for (size_t i = 0; i < num_triangles; ++i) {
            taggedTriangles[i].tag = next_tags[i];
        }
    }

    // 최종 정렬: 태그를 인덱스로 사용하여 최종 위치로 재배치합니다.
    std::vector<TaggedTriangle> final_triangles(num_triangles);
    for (const auto& tt : taggedTriangles) {
        if (tt.tag < num_triangles) { // 안전장치
            final_triangles[tt.tag] = tt;
        }
    }

    g_iKdTree_Node_Count = num_triangles;
    g_pKdTree_Node_Array = new KdTreeNode[g_iKdTree_Node_Count];
    g_iKdTree_TriOffset_Count = num_triangles;
    g_pKdTree_TriOffset_Array = new unsigned int[g_iKdTree_TriOffset_Count];

    // 이제 final_triangles 배열은 최종 노드 순서(0, 1, 2, ...)대로 정렬되어 있습니다.
    for (unsigned int i = 0; i < num_triangles; ++i) {
        unsigned int left_child_idx = 2 * i + 1;

        if (left_child_idx < num_triangles) {
            // Internal node
            int l = level(i);
            int split_axis = l % 3;
            // 분할 위치는 현재 노드에 해당하는 삼각형의 중심점으로 설정합니다.
            float split_pos = (final_triangles[i].triangle.AABB.min[split_axis] + final_triangles[i].triangle.AABB.max[split_axis]) * 0.5f;
            setInnerNode(&g_pKdTree_Node_Array[i], split_axis, left_child_idx, split_pos);
        }
        else {
            // Leaf node
            setLeafNode(&g_pKdTree_Node_Array[i], 1, i);
        }
        // Offset list에는 최종 위치에 맞는 원래 삼각형의 인덱스를 저장합니다.
        g_pKdTree_TriOffset_Array[i] = final_triangles[i].originalIndex;
    }

    g_iKdTree_Level = num_levels - 1;
    g_iKdTree_LeafNode_Count = (num_triangles + 1) / 2;
    g_iKdTree_EmptyNode_Count = 0;
    g_iKdTree_MaxTriInLeafNode_Count = 1;

    return true;
}

