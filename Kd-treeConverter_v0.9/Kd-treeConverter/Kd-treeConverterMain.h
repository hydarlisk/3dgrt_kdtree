/**************************************************************
  File name: Kd-treeConverterMain.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#define MAIN_WINDOW_WIDTH 900
#define MAIN_WINDOW_HEIGHT 600

#define SMALL_OFFSET 1.0
#define ZOOM_SENSITIVITY 0.25
#define GLOBALROTATION_SENSITIVITY 0.25
#define MIN_FOVY 0.5
#define MAX_FOVY 90.0
#define EPSILON 0.00001
//shyun
#define SCENE_NUM 0

#if SCENE_NUM == 0
	#define MODEL_PATH "../../Data/Obj/hotdog_3dgrt.obj"
	#define KDTREE_PATH "../../Data/Obj/hotdog_tree.kdt"
	#define IGEOM_PATH "../../Data/Obj/hotdog_igeom.bin"
#elif SCENE_NUM == 1
	#define MODEL_PATH "../../Data/ply/hotdog/hotdog_3dgrt.ply"
	#define KDTREE_PATH "../../Data/ply/hotdog/hotdog_tree.kdt"
	#define IGEOM_PATH "../../Data/ply/hotdog/hotdog_igeom.bin"
#elif SCENE_NUM == 2
	#define MODEL_PATH "../../Data/ply/lego/lego_3dgrt.ply"
	#define KDTREE_PATH "../../Data/ply/lego/lego_tree.kdt"
	#define IGEOM_PATH "../../Data/ply/lego/lego_igeom.bin"
#elif SCENE_NUM == 3
	#define MODEL_PATH "../../Data/ply/bonsai/bonsai_3dgrt.ply"
	#define KDTREE_PATH "../../Data/ply/bonsai/bonsai_tree.kdt"
	#define IGEOM_PATH "../../Data/ply/bonsai/bonsai_igeom.bin"
#elif SCENE_NUM == 4
	#define MODEL_PATH "../../Data/ply/chair/chair_3dgrt.ply"
	#define KDTREE_PATH "../../Data/ply/chair/chair_tree.kdt"
	#define IGEOM_PATH "../../Data/ply/chair/chair_igeom.bin"
#elif SCENE_NUM == 5
	#define MODEL_PATH "../../Data/ply/flowers/flowers_3dgrt.ply"
	#define KDTREE_PATH "../../Data/ply/flowers/flowers_tree.kdt"
	#define IGEOM_PATH "../../Data/ply/flowers/flowers_igeom.bin"
#endif
//shyun end
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