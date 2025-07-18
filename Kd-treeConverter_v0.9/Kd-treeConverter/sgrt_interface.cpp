// Kd-treeConverter 프로젝트에서 SGRTx2Lib CUDA 렌더러를 사용해 CompositeObject를 렌더링

#include "sgrt_interface.h"

//#include "cudaRenderPipeline.h"            // SGRTx2Lib CUDA 렌더러 초기화 및 실행
//#include "cudaRenderCommon.cuh"            // KdTreeNode, TriangleAccel, ExtendedVertex 등

//#include "GGPUExperimentalRayTracer.h"       // SGRT 렌더러
//#include "GSurfaceMesh.h"
//#include "GGPUKdTreeAccel.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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

void upload_composite_object_to_cuda(CompositeObject* h_obj, CompositeObject* d_obj_out) {
    // ExtendedVertex
    int n_vtx = h_obj->n_triangles * 3;
    cudaMalloc(&d_obj_out->extended_vertices, sizeof(ExtendedVertex) * n_vtx);
    cudaMemcpy(d_obj_out->extended_vertices, h_obj->extended_vertices,
        sizeof(ExtendedVertex) * n_vtx, cudaMemcpyHostToDevice);

    // TriAccel
    int n_tris = h_obj->kd_tree->tri_offset_count;
    TriAccel* d_triaccel;
    cudaMalloc(&d_triaccel, sizeof(TriAccel) * n_tris);
    cudaMemcpy(d_triaccel, h_obj->kd_tree->tri_accel_list,
        sizeof(TriAccel) * n_tris, cudaMemcpyHostToDevice);

    // KDTreeNode
    KdTreeNode* d_nodes;
    int n_nodes = h_obj->kd_tree->tree_node_count;
    cudaMalloc(&d_nodes, sizeof(KdTreeNode) * n_nodes);
    cudaMemcpy(d_nodes, h_obj->kd_tree->tree,
        sizeof(KdTreeNode) * n_nodes, cudaMemcpyHostToDevice);

    // Offset List
    unsigned int* d_offset;
    cudaMalloc(&d_offset, sizeof(unsigned int) * h_obj->kd_tree->tri_offset_count);
    cudaMemcpy(d_offset, h_obj->kd_tree->tri_offset_list,
        sizeof(unsigned int) * h_obj->kd_tree->tri_offset_count, cudaMemcpyHostToDevice);

    // KDTree 복사
    KdTree* d_kdtree;
    cudaMalloc(&d_kdtree, sizeof(KdTree));
    KdTree temp = *h_obj->kd_tree;
    temp.tri_accel_list = d_triaccel;
    temp.tree = d_nodes;
    temp.tri_offset_list = d_offset;
    cudaMemcpy(d_kdtree, &temp, sizeof(KdTree), cudaMemcpyHostToDevice);

    // CompositeObject 구성
    CompositeObject d_obj = *h_obj;
    d_obj.kd_tree = d_kdtree;
    cudaMemcpy(d_obj_out, &d_obj, sizeof(CompositeObject), cudaMemcpyHostToDevice);
}

void deep_copy_composite_object_to_cuda(const CompositeObject* h_obj, CompositeObject** d_obj_out) {
    // [1] 디바이스 측 CompositeObject 구조체 자체 할당
    CompositeObject h_copy = *h_obj;
    CompositeObject* d_obj;
    cudaMalloc((void**)&d_obj, sizeof(CompositeObject));

    // [2] ExtendedVertex 복사
    size_t vertex_count = h_obj->n_triangles * 3; // 삼각형당 3개 정점
    ExtendedVertex* d_vertices;
    cudaMalloc(&d_vertices, sizeof(ExtendedVertex) * vertex_count);
    cudaMemcpy(d_vertices, h_obj->extended_vertices, sizeof(ExtendedVertex) * vertex_count, cudaMemcpyHostToDevice);
    h_copy.extended_vertices = d_vertices;

    // [3] KDTree 복사
    KdTree* d_kdtree;
    cudaMalloc(&d_kdtree, sizeof(KdTree));

    // [3-1] KDTree 구조체 복사 (host -> 임시 host copy)
    KdTree h_tree_copy = *h_obj->kd_tree;

    // [3-2] KDTree 내부 배열 복사
    KdTreeNode* d_tree_nodes;
    cudaMalloc(&d_tree_nodes, sizeof(KdTreeNode) * h_tree_copy.tree_node_count);
    cudaMemcpy(d_tree_nodes, h_tree_copy.tree, sizeof(KdTreeNode) * h_tree_copy.tree_node_count, cudaMemcpyHostToDevice);
    h_tree_copy.tree = d_tree_nodes;

    unsigned int* d_tri_offsets;
    cudaMalloc(&d_tri_offsets, sizeof(unsigned int) * h_tree_copy.tri_offset_count);
    cudaMemcpy(d_tri_offsets, h_tree_copy.tri_offset_list, sizeof(unsigned int) * h_tree_copy.tri_offset_count, cudaMemcpyHostToDevice);
    h_tree_copy.tri_offset_list = d_tri_offsets;

    TriAccel* d_tri_accels;
    cudaMalloc(&d_tri_accels, sizeof(TriAccel) * h_tree_copy.tri_offset_count);
    cudaMemcpy(d_tri_accels, h_tree_copy.tri_accel_list, sizeof(TriAccel) * h_tree_copy.tri_offset_count, cudaMemcpyHostToDevice);
    h_tree_copy.tri_accel_list = d_tri_accels;

    // [3-3] AABB 복사
    // AABB는 float[6] 이므로 memcpy로 가능 (자동 포함)

    // [3-4] KDTree를 디바이스에 복사
    cudaMemcpy(d_kdtree, &h_tree_copy, sizeof(KdTree), cudaMemcpyHostToDevice);
    h_copy.kd_tree = d_kdtree;

    // [4] 최종적으로 CompositeObject 전체 복사
    cudaMemcpy(d_obj, &h_copy, sizeof(CompositeObject), cudaMemcpyHostToDevice);

    // [5] 결과 포인터 반환
    *d_obj_out = d_obj;
}

void save_as_ppm(const float* framebuffer, int width, int height, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // Convert float RGB to 8-bit and write
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = 3 * (y * width + x);
            unsigned char r = static_cast<unsigned char>(fminf(fmaxf(framebuffer[idx + 0], 0.0f), 1.0f) * 255.0f);
            unsigned char g = static_cast<unsigned char>(fminf(fmaxf(framebuffer[idx + 1], 0.0f), 1.0f) * 255.0f);
            unsigned char b = static_cast<unsigned char>(fminf(fmaxf(framebuffer[idx + 2], 0.0f), 1.0f) * 255.0f);

            fwrite(&r, 1, 1, fp);
            fwrite(&g, 1, 1, fp);
            fwrite(&b, 1, 1, fp);
        }
    }

    fclose(fp);
    printf("Saved framebuffer to: %s\n", filename);
}
