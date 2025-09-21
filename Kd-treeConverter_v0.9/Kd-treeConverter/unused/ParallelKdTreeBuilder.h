#pragma once

#include "Kd-treeConverter.h"

/**
 * @brief "GPU-friendly, Parallel..." 논문에 기반하여
 * 완전한 좌편향(left-balanced) k-d 트리를 생성합니다.
 * 이 함수는 재귀 호출 없이 반복적인 정렬을 통해 트리를 구성합니다.
 * @param c_object CompositeObject에 대한 포인터. 이 객체의 삼각형 데이터로
 * k-d 트리를 빌드하고, 결과를 c_object->kd_tree에 저장합니다.
 * @return 성공 시 true, 실패 시 false를 반환합니다.
 */
bool build_kdtree_parallel(CompositeObject* c_object);
