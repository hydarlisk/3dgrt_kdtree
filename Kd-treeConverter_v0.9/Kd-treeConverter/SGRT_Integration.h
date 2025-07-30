#pragma once

#include "Kd-treeConverter.h"
#include "OpenGLStuffs.h"

// 전방 선언으로 불필요한 헤더 포함 최소화
//struct _CompositeObject;
//struct Camera;

// 렌더링을 실행하고 결과를 반환하는 유일한 Public 함수
void renderWithSGRT(
    const CompositeObject& object,
    const Camera& camera,
    int width,
    int height,
    float*& out_framebuffer,
    bool& is_done
);