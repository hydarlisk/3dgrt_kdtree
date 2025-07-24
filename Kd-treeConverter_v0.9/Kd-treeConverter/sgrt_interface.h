#pragma once

#include "Kd-treeConverter.h"
//#include "Kd-treeConverterMain.h"
//#include "SGRTx2Lib/GScene.h"
//using namespace KDTConverter;

// CUDA에서 사용할 TriAccel 구조체
typedef struct {
    float n_u;
    float n_v;
    float n_d;
    unsigned int k : 2;
    unsigned int isTransparent : 1;
    unsigned int mbox : 29;
    float b_nu;
    float b_nv;
    float b_d;
    int indexInObject;
    float c_nu;
    float c_nv;
    float c_d;
    int material_ID;
    float N[3];
    int pad;
} CUDATriAccel;

// CUDA에서 사용할 KdTreeNode 구조체
typedef struct {
    unsigned int x;
    unsigned int y;
} CUDAKdTreeNode;

// CUDA에서 사용할 KdTree 구조체
typedef struct {
    CUDAKdTreeNode* tree;
    int tree_node_count;
    unsigned int* tri_offset_list;
    int tri_offset_count;
    CUDATriAccel* tri_accel_list;
    float AABB[6];
} CUDAKdTree;

// CUDA에서 사용할 ExtendedVertex 구조체
typedef struct {
    float vertex[3];
    float normal[3];
    int material_ID;
    char pad[4];
} CUDAExtendedVertex;

// CUDA에서 사용할 CompositeObject 구조체
typedef struct {
    int n_triangles;
    float AABB[6];
    CUDAExtendedVertex* extended_vertices;
    CUDAKdTree* kd_tree;
} CUDACompositeObject;

//void convertCompositeObjectToGSceneAndKdTree(const CompositeObject& compObj, GScene& outScene);
//void UploadCompositeObjectToDevice(const CompositeObject& compObj);
//void setSGRTScenePointers(ExtendedVertex*, KdTreeNode*, TriAccel*, unsigned int*, int n_triangles, int tree_node_count, int tri_accel_count, int tri_offset_count);
//void SGRT_RenderFromCompositeObject(const CompositeObject* obj);
//GScene* convertCompositeObjectToScene(CompositeObject* obj);
//GScene* convertCompositeObjectToGScene(const CompositeObject* compObj);
//void upload_composite_object_to_cuda(CompositeObject* h_obj, CompositeObject* d_obj_out);
//void deep_copy_composite_object_to_cuda(const CompositeObject* h_obj, CompositeObject** d_obj_out);
//void save_as_ppm(const float* framebuffer, int width, int height, const char* filename);
