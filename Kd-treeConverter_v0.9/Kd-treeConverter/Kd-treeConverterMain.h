/**************************************************************
  File name: Kd-treeConverterMain.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

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