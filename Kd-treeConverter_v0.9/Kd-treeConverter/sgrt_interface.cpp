// Kd-treeConverter 프로젝트에서 SGRTx2Lib CUDA 렌더러를 사용해 CompositeObject를 렌더링

#include "sgrt_interface.h"

//#include "cudaRenderPipeline.h"            // SGRTx2Lib CUDA 렌더러 초기화 및 실행
//#include "cudaRenderCommon.cuh"            // KdTreeNode, TriangleAccel, ExtendedVertex 등

//#include "GGPUExperimentalRayTracer.h"       // SGRT 렌더러
//#include "GSurfaceMesh.h"
//#include "GGPUKdTreeAccel.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cmath>
//#include "SGRTx2Lib/cudaRenderCommon.cuh"
//#include "SGRTx2Lib/cudaRenderPipelineCommonKernel.cu"
//#include "SGRTx2Lib/GScene.h"
//#include "SGRTx2Lib/GKDTreeStructure.h"
//#include "SGRTx2Lib/GGPUExperimentalRayTracer.h"

//using namespace KDTConverter;

//void SGRT_RenderFromCompositeObject(const CompositeObject* obj) {
//    const int width = 1280, height = 720;
//    printf("[SGRT] Initializing CUDA renderer...\n");
//    initCudaRenderPipeline(width, height);  // 화면 크기 등 초기화
//
//    printf("[SGRT] Uploading geometry...\n");
//
//    // GPU 메모리 할당 및 복사
//    int nVerts = obj->n_triangles * 3;
//    ExtendedVertex* d_vertices;
//    cudaMalloc(&d_vertices, sizeof(ExtendedVertex) * nVerts);
//    cudaMemcpy(d_vertices, obj->extended_vertices, sizeof(ExtendedVertex) * nVerts, cudaMemcpyHostToDevice);
//
//    KdTreeNode* d_nodes;
//    cudaMalloc(&d_nodes, sizeof(KdTreeNode) * obj->kd_tree->tree_node_count);
//    cudaMemcpy(d_nodes, obj->kd_tree->tree, sizeof(KdTreeNode) * obj->kd_tree->tree_node_count, cudaMemcpyHostToDevice);
//
//    TriAccel* d_tri_accel;
//    cudaMalloc(&d_tri_accel, sizeof(TriAccel) * obj->kd_tree->tri_offset_count);
//    cudaMemcpy(d_tri_accel, obj->kd_tree->tri_accel_list, sizeof(TriAccel) * obj->kd_tree->tri_offset_count, cudaMemcpyHostToDevice);
//
//    unsigned int* d_indices;
//    cudaMalloc(&d_indices, sizeof(unsigned int) * obj->kd_tree->tri_offset_count);
//    cudaMemcpy(d_indices, obj->kd_tree->tri_offset_list, sizeof(unsigned int) * obj->kd_tree->tri_offset_count, cudaMemcpyHostToDevice);
//
//    // SGRT 내부 전역 포인터에 연결 (cudaRenderPipeline.cu 내부에서 선언된 전역 포인터라고 가정)
//    setSGRTScenePointers(d_vertices, d_nodes, d_tri_accel, d_indices,
//        obj->n_triangles,
//        obj->kd_tree->tree_node_count,
//        obj->kd_tree->tri_offset_count,
//        obj->kd_tree->tri_offset_count);
//
//    printf("[SGRT] Launching CUDA render pipeline...\n");
//    launchCudaRenderPipeline();
//
//    printf("[SGRT] Saving output image...\n");
//    saveFramebufferToPPM("output.ppm");
//    printf("[SGRT] Done.\n");
//}
//
//void loadCompositeObjectToSGRT(const CompositeObject* obj, GGPUExperimentalRayTracer& tracer) {
//    printf("[SGRT] Converting CompositeObject to SGRT scene...\n");
//
//    // 1. SurfaceMesh 생성
//    GSurfaceMesh* surface = new GSurfaceMesh();
//    surface->create(obj->n_triangles * 3);
//
//    for (int i = 0; i < obj->n_triangles * 3; ++i) {
//        for (int j = 0; j < 3; ++j)
//            surface->vertices[i].position[j] = obj->extended_vertices[i].vertex[j];
//        for (int j = 0; j < 3; ++j)
//            surface->vertices[i].normal[j] = obj->extended_vertices[i].normal[j];
//        surface->vertices[i].materialID = obj->extended_vertices[i].material_ID;
//    }
//
//    // 2. GGPUKdTreeAccel 생성
//    GGPUKdTreeAccel* accel = new GGPUKdTreeAccel();
//
//    accel->create(obj->kd_tree->tree_node_count,
//        obj->kd_tree->tri_offset_count,
//        obj->kd_tree->tri_offset_count);
//
//    cudaMemcpy(accel->d_nodes, obj->kd_tree->tree,
//        sizeof(KdTreeNode) * obj->kd_tree->tree_node_count, cudaMemcpyHostToDevice);
//
//    cudaMemcpy(accel->d_triangleIndices, obj->kd_tree->tri_offset_list,
//        sizeof(unsigned int) * obj->kd_tree->tri_offset_count, cudaMemcpyHostToDevice);
//
//    cudaMemcpy(accel->d_triangles, obj->kd_tree->tri_accel_list,
//        sizeof(TriAccel) * obj->kd_tree->tri_offset_count, cudaMemcpyHostToDevice);
//
//    // 3. 트레이서 설정
//    tracer.setAccel(accel);
//    tracer.setSurface(surface);
//    tracer.setCameraLookAt({ 0, 0, -2 }, { 0, 0, 0 }, { 0, 1, 0 });
//    tracer.setResolution(1280, 720);
//
//    printf("[SGRT] Scene ready.\n");
//}

void convertCompositeObjectToGSceneAndKdTree(const CompositeObject& compObj, GScene& outScene) {
    // 폴리곤 오브젝트 생성
    GPolygonObject* obj = new GPolygonObject();

    // TriangleWrapperList 생성
    GTriangleWrapperList* wrapperList = new GTriangleWrapperList();

    const ExtendedVertex* ev = compObj.extended_vertices;

    for (int i = 0; i < compObj.n_triangles; ++i) {
        GTriangleWrapper* tri;
        GPoint v0(ev[3 * i + 0].vertex[0], ev[3 * i + 0].vertex[1], ev[3 * i + 0].vertex[2]);
        GPoint v1(ev[3 * i + 1].vertex[0], ev[3 * i + 1].vertex[1], ev[3 * i + 1].vertex[2]);
        GPoint v2(ev[3 * i + 2].vertex[0], ev[3 * i + 2].vertex[1], ev[3 * i + 2].vertex[2]);

        tri->setPoint(v0, v1, v2);
        wrapperList->addTriangleWrapper(tri);
    }

    obj->getTriangleList(wrapperList, 0, 0);
    outScene.addObject(obj);
    //outScene.setKdTreeLoadFilePath;

    printf("Convert Composite Object to GScene Done.\n");
}

void UploadCompositeObjectToDevice(const CompositeObject& compObj) {
    // 1. AABB
    cuBoundingBox aabb;
    aabb.min_max[0] = make_float4(compObj.AABB[XMIN], compObj.AABB[YMIN], compObj.AABB[ZMIN], 0.0f);
    aabb.min_max[1] = make_float4(compObj.AABB[XMAX], compObj.AABB[YMAX], compObj.AABB[ZMAX], 0.0f);
    cudaMemcpyToSymbol(g_SceneBBox, aabb, sizeof(cuBoundingBox));  // AABB로 직접 변환 필요

    ExtendedVertex* d_vertices;
    cudaMalloc(&d_vertices, sizeof(compObj.extended_vertices) * sizeof(ExtendedVertex));
    cudaMemcpy(d_vertices, compObj.vertices, compObj.numVertices * sizeof(ExtendedVertex), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(g_Vertices, &d_vertices, sizeof(ExtendedVertex*));

    // 2. kd-tree 노드 -> inKdTreeNodeTex
    cudaBindTexture(NULL, inKdTreeNodeTex, compObj.kdTreeNodes, sizeof(kdtreeNode) * compObj.kdTreeNodeCount);

    // 3. Object offset list -> inObjectOffsetListTex
    cudaBindTexture(NULL, inObjectOffsetListTex, compObj.objIndexList, sizeof(uint32_t) * compObj.objListSize);

    // 4. Triangle 데이터 / Material 데이터 등
    // getObjectMaterial(), singlePassIntersectRoutine() 에서 접근 가능한 글로벌 배열에 업로드
    cudaMemcpyToSymbol(devTriangleData, compObj.triangleList, sizeof(Triangle) * compObj.triangleCount);
    cudaMemcpyToSymbol(devMaterialList, compObj.materialList, sizeof(Material) * compObj.triangleCount);
}

// 임시: CompositeObject를 SGRTx2Lib의 GScene으로 변환
// 필요한 데이터 타입: ExtendedVertex, GTriangle 등은 SGRTx2Lib 내부 구조 기반
GScene* convertCompositeObjectToScene(CompositeObject* obj)
{
    if (!obj || obj->n_triangles <= 0) return nullptr;

    // 1) GScene 생성
    GScene* scene = new GScene();

    // 2) GTriangleWrapperList 생성
    GTriangleWrapperList* triList = new GTriangleWrapperList();

    // 3. 삼각형을 GTriangleWrapper로 변환
    for (int i = 0; i < obj->n_triangles; ++i) {
        const TriAccel& tri = obj->kd_tree->tri_accel_list[i];

        int idx0 = tri.indexInObject + 0;
        int idx1 = tri.indexInObject + 1;
        int idx2 = tri.indexInObject + 2;

        // 동적 wrapper 생성
        GTriangleWrapper* wrapper = new GTriangleWrapper();

        // vertex 복사
        memcpy(wrapper->p0, obj->extended_vertices[idx0].vertex, sizeof(float) * 3);
        memcpy(wrapper->p1, obj->extended_vertices[idx1].vertex, sizeof(float) * 3);
        memcpy(wrapper->p2, obj->extended_vertices[idx2].vertex, sizeof(float) * 3);

        wrapper->indexInObject = i;
        wrapper->m_mailBoxId = -1;

        triList->addTriangleWrapper(wrapper);
    }

    // 4) GKDTreeStructure 생성 및 설정
    GKDTreeStructure* kd = new GKDTreeStructure(scene);

    kd->setKdTreeNodeCount(obj->kd_tree->tree_node_count);
    kd->setKdTreeNode(new kdtreeNode[kd->getKdTreeNodeCount()]);
    memcpy(kd->getKdTreeNode(),
        obj->kd_tree->tree,
        sizeof(kdtreeNode)* kd->getKdTreeNodeCount());

    kd->setTriangleOffset(obj->kd_tree->tri_offset_count);
    kd->setTriangleOffsetList(new unsigned int[kd->getTriangleOffset()]);
    memcpy(kd->getTriangleOffsetList(),
        obj->kd_tree->tri_offset_list,
        sizeof(unsigned int)* kd->getTriangleOffset());

    kd->setBBoxMin(GPoint(obj->AABB[0], obj->AABB[1], obj->AABB[2]));
    kd->setBBoxMax(GPoint(obj->AABB[3], obj->AABB[4], obj->AABB[5]));

    kd->setSceneTriangleCount(obj->n_triangles);
    kd->setSceneTriangleList(triList);

    // 5) Scene에 연결
    scene->setSceneKDTree(kd);

    return scene;
}
//GScene* convertCompositeObjectToScene(CompositeObject* obj)
//{
//    if (!obj || obj->n_triangles <= 0) return nullptr;
//
//    // 1) GScene 생성
//    GScene* scene = new GScene();
//
//    // 2) GTriangleWrapperList 생성
//    //    - GTriangleWrapperList는 SGRTx2Lib에 존재
//    GTriangleWrapperList* triList = new GTriangleWrapperList();
//
//    // 3. 각 삼각형을 GTriangleWrapper로 변환
//    for (int i = 0; i < obj->n_triangles; ++i) {
//        const TriAccel& tri = obj->kd_tree->tri_accel_list[i];
//
//        int idx0 = tri.indexInObject + 0;
//        int idx1 = tri.indexInObject + 1;
//        int idx2 = tri.indexInObject + 2;
//
//        float* v0 = new float[3] {
//            obj->extended_vertices[idx0].vertex[0],
//                obj->extended_vertices[idx0].vertex[1],
//                obj->extended_vertices[idx0].vertex[2]
//            };
//        float* v1 = new float[3] {
//            obj->extended_vertices[idx1].vertex[0],
//                obj->extended_vertices[idx1].vertex[1],
//                obj->extended_vertices[idx1].vertex[2]
//            };
//        float* v2 = new float[3] {
//            obj->extended_vertices[idx2].vertex[0],
//                obj->extended_vertices[idx2].vertex[1],
//                obj->extended_vertices[idx2].vertex[2]
//            };
//
//        GTriangleWrapper wrapper;
//        wrapper.p0 = v0;
//        wrapper.p1 = v1;
//        wrapper.p2 = v2;
//        wrapper.indexInObject = i;
//        wrapper.m_mailBoxId = -1;
//
//        triList->addTriangleWrapper(&wrapper);
//    }
//
//    // 4) GKDTreeStructure 생성 및 데이터 복사
//    GKDTreeStructure* kd = new GKDTreeStructure(scene);
//
//    // 4-1) KdTree 노드 복사
//    kd->setKdTreeNodeCount(obj->kd_tree->tree_node_count);
//    kd->setKdTreeNode(new kdtreeNode[kd->getKdTreeNodeCount()]);
//    memcpy(kd->getKdTreeNode(),
//        obj->kd_tree->tree,
//        sizeof(kdtreeNode) * kd->getKdTreeNodeCount());
//
//    // 4-2) Triangle offset 리스트 복사
//    kd->setTriangleOffset(obj->kd_tree->tri_offset_count);
//    kd->setTriangleOffsetList(new unsigned int[kd->getTriangleOffset()]);
//    memcpy(kd->getTriangleOffsetList(),
//        obj->kd_tree->tri_offset_list,
//        sizeof(unsigned int) * kd->getTriangleOffset());
//
//
//    // 4-3) 씬 전체 AABB 설정
//    kd->setBBoxMin(
//        GPoint(obj->AABB[0], obj->AABB[1], obj->AABB[2]));
//    kd->setBBoxMax(
//        GPoint(obj->AABB[3], obj->AABB[4], obj->AABB[5]));
//
//    // 4-4) 래퍼 리스트 연결
//    kd->setSceneTriangleCount(obj->n_triangles);
//    kd->setSceneTriangleList(triList);
//
//    // 5) Scene에 Kd-tree 연결
//    scene->setSceneKDTree(kd);
//
//    return scene;
//}

//void upload_composite_object_to_cuda(CompositeObject* h_obj, CompositeObject* d_obj_out) {
//    // ExtendedVertex
//    int n_vtx = h_obj->n_triangles * 3;
//    cudaMalloc(&d_obj_out->extended_vertices, sizeof(ExtendedVertex) * n_vtx);
//    cudaMemcpy(d_obj_out->extended_vertices, h_obj->extended_vertices,
//        sizeof(ExtendedVertex) * n_vtx, cudaMemcpyHostToDevice);
//
//    // TriAccel
//    int n_tris = h_obj->kd_tree->tri_offset_count;
//    TriAccel* d_triaccel;
//    cudaMalloc(&d_triaccel, sizeof(TriAccel) * n_tris);
//    cudaMemcpy(d_triaccel, h_obj->kd_tree->tri_accel_list,
//        sizeof(TriAccel) * n_tris, cudaMemcpyHostToDevice);
//
//    // KDTreeNode
//    KdTreeNode* d_nodes;
//    int n_nodes = h_obj->kd_tree->tree_node_count;
//    cudaMalloc(&d_nodes, sizeof(KdTreeNode) * n_nodes);
//    cudaMemcpy(d_nodes, h_obj->kd_tree->tree,
//        sizeof(KdTreeNode) * n_nodes, cudaMemcpyHostToDevice);
//
//    // Offset List
//    unsigned int* d_offset;
//    cudaMalloc(&d_offset, sizeof(unsigned int) * h_obj->kd_tree->tri_offset_count);
//    cudaMemcpy(d_offset, h_obj->kd_tree->tri_offset_list,
//        sizeof(unsigned int) * h_obj->kd_tree->tri_offset_count, cudaMemcpyHostToDevice);
//
//    // KDTree 복사
//    KdTree* d_kdtree;
//    cudaMalloc(&d_kdtree, sizeof(KdTree));
//    KdTree temp = *h_obj->kd_tree;
//    temp.tri_accel_list = d_triaccel;
//    temp.tree = d_nodes;
//    temp.tri_offset_list = d_offset;
//    cudaMemcpy(d_kdtree, &temp, sizeof(KdTree), cudaMemcpyHostToDevice);
//
//    // CompositeObject 구성
//    CompositeObject d_obj = *h_obj;
//    d_obj.kd_tree = d_kdtree;
//    cudaMemcpy(d_obj_out, &d_obj, sizeof(CompositeObject), cudaMemcpyHostToDevice);
//}
//
//void deep_copy_composite_object_to_cuda(const CompositeObject* h_obj, CompositeObject** d_obj_out) {
//    // [1] 디바이스 측 CompositeObject 구조체 자체 할당
//    CompositeObject h_copy = *h_obj;
//    CompositeObject* d_obj;
//    cudaMalloc((void**)&d_obj, sizeof(CompositeObject));
//
//    // [2] ExtendedVertex 복사
//    size_t vertex_count = h_obj->n_triangles * 3; // 삼각형당 3개 정점
//    ExtendedVertex* d_vertices;
//    cudaMalloc(&d_vertices, sizeof(ExtendedVertex) * vertex_count);
//    cudaMemcpy(d_vertices, h_obj->extended_vertices, sizeof(ExtendedVertex) * vertex_count, cudaMemcpyHostToDevice);
//    h_copy.extended_vertices = d_vertices;
//
//    // [3] KDTree 복사
//    KdTree* d_kdtree;
//    cudaMalloc(&d_kdtree, sizeof(KdTree));
//
//    // [3-1] KDTree 구조체 복사 (host -> 임시 host copy)
//    KdTree h_tree_copy = *h_obj->kd_tree;
//
//    // [3-2] KDTree 내부 배열 복사
//    KdTreeNode* d_tree_nodes;
//    cudaMalloc(&d_tree_nodes, sizeof(KdTreeNode) * h_tree_copy.tree_node_count);
//    cudaMemcpy(d_tree_nodes, h_tree_copy.tree, sizeof(KdTreeNode) * h_tree_copy.tree_node_count, cudaMemcpyHostToDevice);
//    h_tree_copy.tree = d_tree_nodes;
//
//    unsigned int* d_tri_offsets;
//    cudaMalloc(&d_tri_offsets, sizeof(unsigned int) * h_tree_copy.tri_offset_count);
//    cudaMemcpy(d_tri_offsets, h_tree_copy.tri_offset_list, sizeof(unsigned int) * h_tree_copy.tri_offset_count, cudaMemcpyHostToDevice);
//    h_tree_copy.tri_offset_list = d_tri_offsets;
//
//    TriAccel* d_tri_accels;
//    cudaMalloc(&d_tri_accels, sizeof(TriAccel) * h_tree_copy.tri_offset_count);
//    cudaMemcpy(d_tri_accels, h_tree_copy.tri_accel_list, sizeof(TriAccel) * h_tree_copy.tri_offset_count, cudaMemcpyHostToDevice);
//    h_tree_copy.tri_accel_list = d_tri_accels;
//
//    // [3-3] AABB 복사
//    // AABB는 float[6] 이므로 memcpy로 가능 (자동 포함)
//
//    // [3-4] KDTree를 디바이스에 복사
//    cudaMemcpy(d_kdtree, &h_tree_copy, sizeof(KdTree), cudaMemcpyHostToDevice);
//    h_copy.kd_tree = d_kdtree;
//
//    // [4] 최종적으로 CompositeObject 전체 복사
//    cudaMemcpy(d_obj, &h_copy, sizeof(CompositeObject), cudaMemcpyHostToDevice);
//
//    // [5] 결과 포인터 반환
//    *d_obj_out = d_obj;
//}
//
//void save_as_ppm(const float* framebuffer, int width, int height, const char* filename) {
//    FILE* fp = fopen(filename, "wb");
//    if (!fp) {
//        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
//        return;
//    }
//
//    // Write PPM header
//    fprintf(fp, "P6\n%d %d\n255\n", width, height);
//
//    // Convert float RGB to 8-bit and write
//    for (int y = 0; y < height; ++y) {
//        for (int x = 0; x < width; ++x) {
//            int idx = 3 * (y * width + x);
//            unsigned char r = static_cast<unsigned char>(fminf(fmaxf(framebuffer[idx + 0], 0.0f), 1.0f) * 255.0f);
//            unsigned char g = static_cast<unsigned char>(fminf(fmaxf(framebuffer[idx + 1], 0.0f), 1.0f) * 255.0f);
//            unsigned char b = static_cast<unsigned char>(fminf(fmaxf(framebuffer[idx + 2], 0.0f), 1.0f) * 255.0f);
//
//            fwrite(&r, 1, 1, fp);
//            fwrite(&g, 1, 1, fp);
//            fwrite(&b, 1, 1, fp);
//        }
//    }
//
//    fclose(fp);
//    printf("Saved framebuffer to: %s\n", filename);
//}
