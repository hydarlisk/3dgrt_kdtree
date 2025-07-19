#pragma once

#include "Kd-treeConverter.h"
#include "Kd-treeConverterMain.h"
#include "SGRTx2Lib/GScene.h"

#ifdef __cplusplus
extern "C" {
#endif
    struct SceneInfo {
        int iResolutionX;
        int iResolutionY;
        int iSuperSamplingX;
        int iSuperSamplingY;
        int iBlockSizeX;
        int iBlockSizeY;
        bool bUseShadow;
        bool bUseReflection;
        bool bUseRefraction;
        int maxDepth;
    };

	//void setSGRTScenePointers(ExtendedVertex*, KdTreeNode*, TriAccel*, unsigned int*, int n_triangles, int tree_node_count, int tri_accel_count, int tri_offset_count);
	//void SGRT_RenderFromCompositeObject(const CompositeObject* obj);
    GScene* convertCompositeObjectToScene(CompositeObject* compObj)
    //GScene* convertCompositeObjectToGScene(const CompositeObject* compObj);
    void upload_composite_object_to_cuda(CompositeObject* h_obj, CompositeObject* d_obj_out);
    void deep_copy_composite_object_to_cuda(const CompositeObject* h_obj, CompositeObject** d_obj_out);
    void save_as_ppm(const float* framebuffer, int width, int height, const char* filename);
#ifdef __cplusplus
}
#endif