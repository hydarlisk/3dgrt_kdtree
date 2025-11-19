/**************************************************************
  File name: Kd-treeConverterMain.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/
#pragma once

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#define SMALL_OFFSET 1.0
#define ZOOM_SENSITIVITY 0.25
#define GLOBALROTATION_SENSITIVITY 0.25
#define MIN_FOVY 0.5
#define MAX_FOVY 90.0
#define EPSILON 0.00001

typedef enum { FLAT_SHADING, SMOOTH_SHADING } OpenGLShadingMode;
typedef enum { LINE, FILL } OpenGLPolygonMode;
 
typedef struct _UIParameters {

	int bounding_box_display_mode;
	OpenGLShadingMode OpenGL_shading_mode;
	OpenGLPolygonMode OpenGL_polygon_mode;
	int camera_zoom_mode, camera_global_rotation_mode;
	int left_button_pressed, right_button_pressed;
	int prevx, prevy; //prevy for a later use
	int main_window_ID;
	int main_menu_ID;

	char *SL_KDT_CONFIG_filename;

	char mesh_geom_files_dir[256];
	char kd_tree_dump_dir[256];
	char kd_tree_filename[256];
	int	kd_tree_dump_format;
	char i_geometry_filename[256];
	char **mesh_geom_filenames;
	int *mesh_geom_mat_IDs;
	int n_mesh_geoms;


	CompositeObject poly_model;

	int composite_object_read;
} UIParameters;

void append_mesh_geom_to_composite_object(CompositeObject* c_object, MeshGeom* mesh_geom, int mat_type, float* AABB);
void set_kd_tree_leaf_node();


// --- Gaussian Scale에 따른 구조 정의. ---
// C++17의 std::array를 사용합니다. (float[3] 대신)
using Vertex = std::array<float, 3>;
// .obj는 1-based, C++은 0-based이므로 int를 사용.
using Face = std::array<int, 3>;

//#define MIN_OPACITY_FOR_20GON 0.7f
//const float MIN_OPACITY_FOR_20GON = 0.7f;
#if USE_KERNEL_SCALE
const float T_8 = 0.20f;  // ~ Bin 0 (0.18): 대부분의 가우시안
const float T_20 = 0.40f;  // ~ Bin 2 (0.37): 약간 큰 것들
const float T_80 = 0.75f;  // ~ Bin 3 (0.74)
const float T_162 = 1.15f;  // ~ Bin 5 (1.12)
const float T_264 = 1.50f;  // ~ Bin 7 (1.49)
const float T_320 = 2.00f;  // ~ Bin 9 (1.86)
const float T_420 = 3.00f;  // ~ Bin 15
const float T_544 = 4.00f;  // ~ Bin 21
const float T_684 = 5.00f;  // ~ Bin 26
const float T_760 = 7.00f;  // ~ Bin 36
const float T_840 = 9.00f;  // ~ Bin 48
const float T_924 = 11.00f; // ~ Bin 58
const float T_1012 = 13.00f; // ~ Bin 69
const float T_1104 = 15.00f; // ~ Bin 80
// 15.0 이상은 1280면체 (Bin 81 ~ 99)
#else
const float T_8 = 0.12f;  // (Bin 0) 8면체
const float T_20 = 0.24f;  // (Bin 1) 20면체
const float T_80 = 0.35f;  // (Bin 2) 80면체
const float T_162 = 0.47f;  // (Bin 3) 162면체
const float T_264 = 0.59f;  // (Bin 4) 264면체
const float T_320 = 0.70f;  // (Bin 5) 320면체
const float T_420 = 0.82f;  // (Bin 6) 420면체
const float T_544 = 0.94f;  // (Bin 7) 544면체
const float T_684 = 1.05f;  // (Bin 8) 684면체
const float T_760 = 1.29f;  // (Bin 9-10) 760면체
const float T_840 = 1.52f;  // (Bin 11-12) 840면체
const float T_924 = 1.87f;  // (Bin 13-15) 924면체
const float T_1012 = 2.34f;  // (Bin 16-19) 1012면체
const float T_1104 = 3.05f;  // (Bin 20-25) 1104면체
// 3.05f 이상은 1280면체
#endif

// --- (추가) 8면체 (Octahedron) ---
#define OCTA_NUM_VRT 6
#define OCTA_NUM_TRI 8
const float octaHedraDiag = 1.7320508075688774; // s / sqrt(2)

std::vector<Vertex> g_OctaVertices = {
	{0.0f, 0.0f, -octaHedraDiag}, {0.0f, octaHedraDiag, 0.0f}, {-octaHedraDiag, 0.0f, 0.0f},
	{0.0f, -octaHedraDiag, 0.0f}, {octaHedraDiag, 0.0f, 0.0f}, {0.0f, 0.0f, octaHedraDiag}
};
std::vector<Face> g_OctaFaces = {
	{2, 1, 0}, {1, 4, 0}, {4, 3, 0}, {3, 2, 0},
	{4, 1, 5}, {3, 4, 5}, {2, 3, 5}, {1, 2, 5}
};


// --- 20면체 (OctaIcosahedronhedron) ---
#define icosaHedronNumVrt 12
#define icosaHedronNumTri 20

const float goldenRatio = 1.618033988749895f;
const float unitspherefactor = 0.5257311121191335703f;

const float icosaEdge = 1.323169076499215f;

std::vector<Vertex> g_IcoVertices = {
	{-1, goldenRatio, 0}, {1, goldenRatio, 0}, {0, 1, -goldenRatio},
	{-goldenRatio, 0, -1}, {-goldenRatio, 0, 1}, {0, 1, goldenRatio},
	{goldenRatio, 0, 1}, {0, -1, goldenRatio}, {-1, -goldenRatio, 0},
	{0, -1, -goldenRatio}, {goldenRatio, 0, -1}, {1, -goldenRatio, 0} };
std::vector<Face> g_IcoFaces = {
	{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}, {0, 5, 1},
	{6, 1, 5}, {6, 5, 7}, {6, 7, 11}, {6, 11, 10}, {6, 10, 1},
	{8, 4, 3}, {8, 3, 9}, {8, 9, 11}, {8, 11, 7}, {8, 7, 4},
	{9, 3, 2}, {9, 2, 10}, {9, 10, 11},
	{5, 4, 7}, {1, 10, 2} };

// 80면체 이상
std::vector<Vertex> g_LOD_80_Vertices; // 80.obj
std::vector<Face>   g_LOD_80_Faces;
std::vector<Vertex> g_LOD_162_Vertices; // 162.obj
std::vector<Face>   g_LOD_162_Faces;
std::vector<Vertex> g_LOD_264_Vertices; // 264.obj
std::vector<Face>   g_LOD_264_Faces;
std::vector<Vertex> g_LOD_320_Vertices; // 320.obj
std::vector<Face>   g_LOD_320_Faces;
std::vector<Vertex> g_LOD_420_Vertices; // 420.obj
std::vector<Face>   g_LOD_420_Faces;
std::vector<Vertex> g_LOD_544_Vertices; // 544.obj
std::vector<Face>   g_LOD_544_Faces;
std::vector<Vertex> g_LOD_684_Vertices; // 684.obj
std::vector<Face>   g_LOD_684_Faces;
std::vector<Vertex> g_LOD_760_Vertices; // 760.obj
std::vector<Face>   g_LOD_760_Faces;
std::vector<Vertex> g_LOD_840_Vertices; // 840.obj
std::vector<Face>   g_LOD_840_Faces;
std::vector<Vertex> g_LOD_924_Vertices; // 924.obj
std::vector<Face>   g_LOD_924_Faces;
std::vector<Vertex> g_LOD_1012_Vertices; // 1012.obj
std::vector<Face>   g_LOD_1012_Faces;
std::vector<Vertex> g_LOD_1104_Vertices; // 1104.obj
std::vector<Face>   g_LOD_1104_Faces;
std::vector<Vertex> g_LOD_1280_Vertices; // 1280.obj
std::vector<Face>   g_LOD_1280_Faces;