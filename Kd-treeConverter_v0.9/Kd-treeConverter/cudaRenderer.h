#pragma once

// C++ 코드에서 참조할 구조체들
#include "Kd-treeConverter.h"
#include "OpenGLStuffs.h"

/**
 * @brief CompositeObject를 사용하여 CUDA 렌더링을 시작하는 메인 함수.
 * @param object 렌더링할 CompositeObject (Kd-tree 포함).
 * @param camera 현재 카메라 정보.
 * @param width 렌더링할 프레임버퍼의 너비.
 * @param height 렌더링할 프레임버퍼의 높이.
 * @param out_framebuffer [out] 렌더링 결과가 저장될 Host 메모리의 프레임버퍼 포인터.
 * @param is_done [out] 렌더링 완료 여부를 나타내는 플래그.
 */
void launchCudaRender(
    const CompositeObject& object,
    const Camera& camera,
    int width,
    int height,
    float*& out_framebuffer,
    bool& is_done
);