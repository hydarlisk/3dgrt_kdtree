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
#define MAX_FOVY 100.0
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
float T_8 = 0.20f;  // ~ Bin 0 (0.18): 대부분의 가우시안
float T_20 = 0.40f;  // ~ Bin 2 (0.37): 약간 큰 것들
float T_80 = 0.75f;  // ~ Bin 3 (0.74)
float T_162 = 1.15f;  // ~ Bin 5 (1.12)
float T_264 = 1.50f;  // ~ Bin 7 (1.49)
float T_320 = 2.00f;  // ~ Bin 9 (1.86)
float T_420 = 3.00f;  // ~ Bin 15

#elif LESS_TRI
const float T_LOW = 2.00f;
const float T_MID = 10.00f;
const float T_HIGH = 15.00f;
const float T_8 = 1.5f;  // (Bin 0) 8면체
const float T_20 = 1.00f;  // (Bin 1) 20면체
const float T_80 = 2.00f;  // (Bin 2) 80면체
#else
//const float T_8 = 0.12f;  // (Bin 0) 8면체
//const float T_20 = 0.24f;  // (Bin 1) 20면체
//const float T_80 = 0.35f;  // (Bin 2) 80면체
//const float T_162 = 0.47f;  // (Bin 3) 162면체
//const float T_264 = 0.59f;  // (Bin 4) 264면체
//const float T_320 = 0.70f;  // (Bin 5) 320면체
//const float T_420 = 0.82f;  // (Bin 6) 420면체

float T_8 = 0.22f;  // (Scale < 0.22) -> 8면체 (약 1,350,000개)
// [구간 2: 중간 크기 방어] - 히스토그램 Bin 1~2 (~0.56) 커버
float T_20 = 0.45f;  // (Scale < 0.45) -> 20면체 (약 140,000개)
// [구간 3: 큰 입자 적응형 분할] - 히스토그램 롱테일(Long-tail) 구간
// Scale이 커질수록 더 많은 면을 사용하여 삼각형 크기(0.3)를 유지합니다.
float T_80 = 0.85f;  // (Scale < 0.85) -> 80면체
float T_162 = 1.20f;  // (Scale < 1.20) -> 162면체
float T_264 = 1.60f;  // (Scale < 1.60) -> 264면체
float T_320 = 2.00f;  // (Scale < 2.00) -> 320면체
float T_420 = 2.60f;  // (Scale < 2.60) -> 420면체
#endif

// --- (추가) 8면체 (Octahedron) ---
#define OCTA_NUM_VRT 6
#define OCTA_NUM_TRI 8
//#if ADAPTIVE_MESH
const float octaHedraDiag = 1.0f;
//#else
//const float octaHedraDiag = 1.7320508075688774; // s / sqrt(2)
//#endif

//#if USE_KERNEL_SCALE
//const float octaHedraDiag = 1.5115226281523f; //1.0f / (0.5f * icosaEdge);
//#else
//const float octaHedraDiag = 1.9021130325903f;
//#endif

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

const float icosaEdge = 1.323169076499215f;
const float unitspherefactor = 0.5257311121191335703f;

//const float ICO_X = 0.525731112119133606f;
//const float ICO_Z = 0.850650808352039932f;
//std::vector<Vertex> g_IcoVertices = {
//	{-ICO_X, ICO_Z, 0}, {ICO_X, ICO_Z, 0}, {0, ICO_X, -ICO_Z},
//	{-ICO_Z, 0, -ICO_X}, {-ICO_Z, 0, ICO_X}, {0, ICO_X, ICO_Z},
//	{ICO_Z, 0, ICO_X}, {0, -ICO_X, ICO_Z}, {-ICO_X, -ICO_Z, 0},
//	{0, -ICO_X, -ICO_Z}, {ICO_Z, 0, -ICO_X}, {ICO_X, -ICO_Z, 0}
//};
const float goldenRatio = 1.618033988749895f;
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
//std::vector<Vertex> g_LOD_544_Vertices; // 544.obj
//std::vector<Face>   g_LOD_544_Faces;
//std::vector<Vertex> g_LOD_684_Vertices; // 684.obj
//std::vector<Face>   g_LOD_684_Faces;
//std::vector<Vertex> g_LOD_760_Vertices; // 760.obj
//std::vector<Face>   g_LOD_760_Faces;
//std::vector<Vertex> g_LOD_840_Vertices; // 840.obj
//std::vector<Face>   g_LOD_840_Faces;
//std::vector<Vertex> g_LOD_924_Vertices; // 924.obj
//std::vector<Face>   g_LOD_924_Faces;
//std::vector<Vertex> g_LOD_1012_Vertices; // 1012.obj
//std::vector<Face>   g_LOD_1012_Faces;
//std::vector<Vertex> g_LOD_1104_Vertices; // 1104.obj
//std::vector<Face>   g_LOD_1104_Faces;
//std::vector<Vertex> g_LOD_1280_Vertices; // 1280.obj
//std::vector<Face>   g_LOD_1280_Faces;