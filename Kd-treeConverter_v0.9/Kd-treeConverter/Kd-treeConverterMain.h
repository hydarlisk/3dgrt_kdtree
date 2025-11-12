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
// .obj는 1-based, C++은 0-based이므로 int를 사용합니다.
using Face = std::array<int, 3>;

const float MESH_THRESHOLD_8 = 0.12f;  // 8면체 (Octahedron)
const float MESH_THRESHOLD_20 = 0.35f; // 20면체 (Icosahedron)
const float MESH_THRESHOLD_80 = 1.0f;  // 80면체 (L1 Subdivision)
const float MESH_THRESHOLD_320 = 3.0f; // 320면체 (L2 Subdivision)
// 3.0 이상은 1280면체 (L3 Subdivision)

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

// 80면체 이상 (Icosahedron Level 1 Subdivision)
std::vector<Vertex> g_L1_Vertices; // 80-gon (2.obj)
std::vector<Face>   g_L1_Faces;
std::vector<Vertex> g_L2_Vertices; // 320-gon (3.obj)
std::vector<Face>   g_L2_Faces;
std::vector<Vertex> g_L3_Vertices; // 1280-gon (4.obj)
std::vector<Face>   g_L3_Faces;
