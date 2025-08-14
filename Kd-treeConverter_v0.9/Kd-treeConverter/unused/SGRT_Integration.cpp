#include "SGRTx2Lib/GScene.h"
#include "SGRTx2Lib/GPolygonObject.h"
#include "SGRTx2Lib/GMaterial.h"
#include "SGRTx2Lib/GKDTreeStructure.h"
#include "SGRTx2Lib/GGPUExperimentalRayTracer.h" // << 핵심 클래스
#include "SGRTx2Lib/GKDTreeOption.h"
#include "SGRTx2Lib/GTriangleWrapper.h"
#include "SGRTx2Lib/GTriangleWrapperList.h"
#include "SGRT_Integration.h"
#include <vector>
#include <iostream>

class GCompositeObjectAdapter : public GPolygonObject {
private:
    const CompositeObject* m_compositeObject;

public:
    // 생성자: 원본 CompositeObject 데이터를 포인터로 받음
    GCompositeObjectAdapter(const CompositeObject* obj) : m_compositeObject(obj) {
        // GObject의 멤버 변수인 m_BoundingBox를 CompositeObject의 것으로 설정
        m_BoundingBox.setMin(GVector(obj->AABB[XMIN], obj->AABB[YMIN], obj->AABB[ZMIN]));
        m_BoundingBox.setMax(GVector(obj->AABB[XMAX], obj->AABB[YMAX], obj->AABB[ZMAX]));
    }

    // GPolygonObject의 순수 가상 함수들을 오버라이드

    // GObject에서 상속된 순수 가상 함수
    virtual GError validObject() override { return errorNo; }
    virtual GError convertToWorldObject() override { return errorNo; }

    // GPolygonObject에서 상속된 가상 함수
    virtual int getTriangleCount() const override {
        return m_compositeObject ? m_compositeObject->n_triangles : 0;
    }

    // SGRT의 Kd-tree 빌더가 이 함수를 호출하여 삼각형 리스트를 채워달라고 요청함
    virtual void getTriangleList(GTriangleWrapperList* pList, int objectIndex, int& triangleIndexOffset) const override {
        if (!m_compositeObject || !pList) return;

        // CompositeObject의 모든 삼각형을 순회
        for (int i = 0; i < m_compositeObject->n_triangles; ++i) {
            const ExtendedVertex& v0 = m_compositeObject->extended_vertices[i * 3 + 0];
            const ExtendedVertex& v1 = m_compositeObject->extended_vertices[i * 3 + 1];
            const ExtendedVertex& v2 = m_compositeObject->extended_vertices[i * 3 + 2];

            // GTriangleWrapper 객체를 동적으로 생성하여 리스트에 추가
            // (GTriangleWrapperList 소멸자가 내부 포인터를 해제하므로 new 사용이 올바름)
            GTriangleWrapper* wrapper = new GTriangleWrapper();
            wrapper->p0 = v0.vertex;
            wrapper->p1 = v1.vertex;
            wrapper->p2 = v2.vertex;
            wrapper->n0 = v0.normal;
            wrapper->n1 = v1.normal;
            wrapper->n2 = v2.normal;
            wrapper->uv0 = nullptr; // UV 좌표는 없음
            wrapper->uv1 = nullptr;
            wrapper->uv2 = nullptr;
            wrapper->m_pObject = const_cast<GCompositeObjectAdapter*>(this);
            wrapper->index = triangleIndexOffset++; // Scene 전체에서의 고유 삼각형 인덱스
            wrapper->objectIndexInScene = objectIndex;
            wrapper->bSelected = true; // 기본적으로 선택된 것으로 처리

            // 삼각형 BBox 계산
            GVector bmin, bmax;
            bmin.x = std::min({ v0.vertex[0], v1.vertex[0], v2.vertex[0] });
            bmin.y = std::min({ v0.vertex[1], v1.vertex[1], v2.vertex[1] });
            bmin.z = std::min({ v0.vertex[2], v1.vertex[2], v2.vertex[2] });
            bmax.x = std::max({ v0.vertex[0], v1.vertex[0], v2.vertex[0] });
            bmax.y = std::max({ v0.vertex[1], v1.vertex[1], v2.vertex[1] });
            bmax.z = std::max({ v0.vertex[2], v1.vertex[2], v2.vertex[2] });
            wrapper->m_BBox.setMin(bmin);
            wrapper->m_BBox.setMax(bmax);

            // GTriangleWrapperList.h에 정의된 addTriangleWrapper 함수 사용
            pList->addTriangleWrapper(wrapper);
        }
    }
};

// =================================================================================
// 어댑터 클래스를 사용하여 CompositeObject를 GScene으로 변환하는 함수
// =================================================================================
GScene* convertToGScene(const CompositeObject& object, const Camera& camera, int width, int height) {
    GScene* scene = new GScene();

    // Scene 설정
    scene->setResolution(width, height);
    scene->setSuperSampling(1, 1);
    scene->setGPUBlockSize(16, 16);
    scene->setMaxReflectionDepth(3);
    scene->setEnableShadow(true);

    // GCompositeObjectAdapter 객체를 생성하여 Scene에 추가
    if (object.n_triangles > 0) {
        GCompositeObjectAdapter* adapterObject = new GCompositeObjectAdapter(&object);

        GMaterial* mat = new GMaterial();
        mat->m_Diffuse = GColor(0.8f, 0.7f, 0.6f);
        mat->m_Specular = GColor(0.2f, 0.2f, 0.2f);
        mat->m_fReflection = 0.05f;
        adapterObject->getMaterial()->operator=(*mat); // GObject::getMaterial()은 포인터를 반환
        delete mat;

        scene->addObject(adapterObject);
    }

    // GCamera.h에 정의된 함수들을 사용하여 카메라 정보 설정
    GCamera* renderCamera = scene->getRenderCamera();
    renderCamera->setCameraPos(
        GVector(camera.pos[0], camera.pos[1], camera.pos[2]), // eye
        GVector(0, 0, 0), // view (임시로 원점)
        GVector(camera.vaxis[0], camera.vaxis[1], camera.vaxis[2])  // up
    );
    renderCamera->setPerspective(camera.fovy, camera.aspect, camera.near_c, camera.far_c);

    return scene;
}

// =================================================================================
// 최종 렌더링 함수
// =================================================================================
void renderWithSGRT(const CompositeObject& object, const Camera& camera, int width, int height, float*& out_framebuffer, bool& is_done) {
    is_done = false;
    std::cout << "--- SGRT Integration: Starting Render ---" << std::endl;

    // CompositeObject -> GScene 변환
    GScene* scene = convertToGScene(object, camera, width, height);
    if (!scene || scene->getObjectCount() == 0) {
        std::cerr << "[SGRT Error] Failed to convert to GScene or scene is empty." << std::endl;
        delete scene;
        return;
    }
    std::cout << "[SGRT] Step 1: Converted to GScene." << std::endl;

    // GScene::convertRenderScene() 호출하여 Kd-tree 빌드
    scene->setUseSpatialStructure(USE_KDTREE);
    if (scene->convertRenderScene() != errorNo) {
        std::cerr << "[SGRT Error] GScene::convertRenderScene() failed." << std::endl;
        delete scene;
        return;
    }
    std::cout << "[SGRT] Step 2: Kd-tree built via convertRenderScene()." << std::endl;

    // GGPUExperimentalRayTracer로 렌더링 실행
    GGPUExperimentalRayTracer* tracer = new GGPUExperimentalRayTracer();
    std::cout << "[SGRT] Step 3: Calling tracer->rendering()..." << std::endl;
    if (tracer->rendering(scene, false) != errorNo) {
        std::cerr << "[SGRT Error] Tracer->rendering() failed." << std::endl;
        delete tracer;
        delete scene;
        return;
    }
    std::cout << "[SGRT] Step 4: Rendering finished." << std::endl;

    // GImageBuffer.h에 정의된 getBuffer()로 결과 프레임버퍼 가져오기
    GImageBuffer* imageBuffer = scene->getImageBuffer();
    if (imageBuffer && imageBuffer->getBuffer()) {
        if (out_framebuffer) delete[] out_framebuffer;
        size_t bufferSize = static_cast<size_t>(width) * height * 3 * sizeof(float);
        out_framebuffer = new float[width * height * 3];
        memcpy(out_framebuffer, imageBuffer->getBuffer(), bufferSize);
        is_done = true;
        std::cout << "[SGRT] Step 5: Framebuffer copied." << std::endl;
    }
    else {
        std::cerr << "[SGRT Error] Failed to get framebuffer from scene." << std::endl;
    }

    // 메모리 해제
    delete tracer;
    delete scene;
    std::cout << "--- SGRT Integration: Finished ---" << std::endl;
}