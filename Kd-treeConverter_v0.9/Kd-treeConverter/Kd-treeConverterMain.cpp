/**************************************************************
  File name: Kd-treeConverterMain.cpp
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include <fstream>
#include <sstream>
#include <vector>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Kd-treeConstructor.h"
#include "Kd-treeConverter.h"
#include "Kd-treeConverterMain.h"
#include "SLMeshDataIO.h"
#include "OpenGLStuffs.h"
#include "MyMathUtility.h"

//shyun
#include <cuda_gl_interop.h>
//#include "sgrt_interface.h"
//#include "SGRT_Integration.h"
#include "test.h"
#include "cudaRenderer.h"
//#include "cudaKDTreeTracer.h"
//#include "cudaRayTracingKernel.cu"
//#include "SGRTx2Lib/GKDTreeStructure.h"
//#include "SGRTx2Lib/GGPURayTracer.h"
//#include "SGRTx2Lib/GGPUExperimentalRayTracer.h"
//using namespace KDTConverter;
//using namespace KDTConstructor;
bool render_gaussian = false;
float* g_render_framebuffer = nullptr;
int g_render_width = MAIN_WINDOW_WIDTH;
int g_render_height = MAIN_WINDOW_HEIGHT;
bool g_cuda_rendering_done = false;
bool g_cuda_interactive_mode = false; // CUDA 인터랙티브 모드 활성화 플래그
bool g_camera_dirty = true;           // 카메라가 변경되었는지 확인하는 플래그
//shyun end
UIParameters uip;
Camera camera;
KdTree kd_tree;

GLuint buf_obj;

void load_poly_model_into_OpenGL(void) {
	/* suffering memory problem for the entire model
	glGenBuffers(1, &buf_obj);
	glBindBuffer(GL_ARRAY_BUFFER, buf_obj);
	glBufferData(GL_ARRAY_BUFFER, uip.poly_model.n_triangles*3*sizeof(ExtendedVertex), 
					uip.poly_model.exteded_vertices, GL_STATIC_DRAW);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
 
	glVertexPointer(3, GL_FLOAT, sizeof(ExtendedVertex), BUFFER_OFFSET(0));
	glNormalPointer(GL_FLOAT, sizeof(ExtendedVertex), BUFFER_OFFSET(3));
	*/
	//camera.pos[0] = uip.poly_model.AABB[XMAX];
	//camera.pos[1] = uip.poly_model.AABB[YMAX];
	//camera.pos[2] = uip.poly_model.AABB[ZMAX];
}
 
void display(void) {
	// CUDA 렌더링이 완료되었으면 프레임버퍼를 화면에 그립니다.
	if ((g_cuda_interactive_mode || g_cuda_rendering_done) && g_render_framebuffer != nullptr) {
		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);

		glMatrixMode(GL_PROJECTION); // 2D 렌더링을 위해 Projection 행렬을 초기화
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// glDrawPixels는 좌하단이 기준이므로 y좌표를 뒤집을 필요가 없음
		glRasterPos2f(-1.0f, -1.0f);
		glDrawPixels(g_render_width, g_render_height, GL_RGB, GL_FLOAT, g_render_framebuffer);

		glEnable(GL_DEPTH_TEST);
		glutSwapBuffers();
		return; // CUDA 결과를 그렸으므로 나머지 OpenGL 렌더링은 건너뜁니다.
	}

	int i;
	ExtendedVertex *ptr_ev;
 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 	glEnable(GL_DEPTH_TEST);
 
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	//glTranslatef(-(uip.poly_model.AABB[XMIN]+uip.poly_model.AABB[XMAX])/2.0,
	//				-(uip.poly_model.AABB[YMIN]+uip.poly_model.AABB[YMAX])/2.0, 
	//				-(uip.poly_model.AABB[ZMIN]+uip.poly_model.AABB[ZMAX])/2.0);

	draw_axes(100.0);

	if (uip.composite_object_read == 1) {
		
		if (uip.bounding_box_display_mode)
			draw_AABB(uip.poly_model.AABB);

		/* suffering memory problem for the entire model
		glBindBuffer(GL_ARRAY_BUFFER, buf_obj);
		glColor3f(1.0, 0.7, 0.1);
 		glDrawArrays(GL_TRIANGLES, 0, 3*uip.poly_model.n_triangles);
		*/
		 	
		// use an old way of drawing
		glColor3f(1.0, 0.7, 0.1);

		ptr_ev = uip.poly_model.extended_vertices;
		glBegin(GL_TRIANGLES);
	 	for (i = 0; i < uip.poly_model.n_triangles; i++) {
	 		//glNormal3fv(ptr_ev->normal);  
			glVertex3fv(ptr_ev->vertex); 
			ptr_ev++;

	 		//glNormal3fv(ptr_ev->normal);  
			glVertex3fv(ptr_ev->vertex); 
			ptr_ev++;

	 		//glNormal3fv(ptr_ev->normal);  
			glVertex3fv(ptr_ev->vertex); 
			ptr_ev++;
		}
		glEnd();
		// use an old way of drawing
	}

	glPopMatrix();
	glutSwapBuffers(); 
}

void keyboard(unsigned char key, int x, int y) {
	static int bf_culling = 0;
	switch (key) {
		case 'b':
			uip.bounding_box_display_mode = 1 - uip.bounding_box_display_mode;
			glutPostRedisplay();
			break;
		case 'c':
			if (bf_culling = 1 - bf_culling)  glEnable(GL_CULL_FACE);
			else glDisable(GL_CULL_FACE);
			glutPostRedisplay();
			break;
		case 'p':
			if (uip.OpenGL_polygon_mode == FILL) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				uip.OpenGL_polygon_mode = LINE;
				turn_off_parallel_OpenGL_head_light();
				glDisable(GL_LIGHTING);
			}
			else {
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				uip.OpenGL_polygon_mode = FILL;
				turn_on_parallel_OpenGL_head_light();
				glEnable(GL_LIGHTING); 
			}
			glutPostRedisplay();
			break;
		case 's':
			if (uip.OpenGL_shading_mode == FLAT_SHADING) {
				glShadeModel(GL_SMOOTH);
				uip.OpenGL_shading_mode = SMOOTH_SHADING;
			}
			else {
				glShadeModel(GL_FLAT);
				uip.OpenGL_shading_mode = FLAT_SHADING;			
			}
			glutPostRedisplay();
			break;
		case 'q':
			exit(0);
			break;
	}
}

void reshape(int width, int height) {
	g_render_width = width;   // 전역 변수 업데이트
	g_render_height = height; // 전역 변수 업데이트

	glViewport(0, 0, width, height);

	camera.aspect = (double) width/ height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(camera.fovy, camera.aspect, camera.near_c, camera.far_c);
}


void mousepress(int button, int state, int x, int y) {
	if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
		if (glutGetModifiers() == GLUT_ACTIVE_SHIFT) {
			uip.camera_zoom_mode = 1;
		}
		else if  (glutGetModifiers() == GLUT_ACTIVE_CTRL) {
			uip.camera_global_rotation_mode = 1;
		}			
		uip.left_button_pressed  = 1;
		uip.prevx = x, uip.prevy = y;
	}
	else if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_UP)) {
		uip.camera_zoom_mode = uip.camera_global_rotation_mode = 0;
		uip.left_button_pressed = 0;
	}
}

void mousemove(int x, int y) {
	int delx, dely;
	double length, u_length, v_length;
	float d[3], dir[3];
	float R[16], tmpx, tmpy, tmpz;
 
	if (uip.left_button_pressed) {
		g_camera_dirty = true; //shyun
		delx = x - uip.prevx, dely = uip.prevy - y;
		uip.prevx = x, uip.prevy = y;

		if (uip.camera_zoom_mode) {
			camera.fovy += delx*ZOOM_SENSITIVITY;
			if (camera.fovy < MIN_FOVY)
				camera.fovy = MIN_FOVY;
			if (camera.fovy > MAX_FOVY) 
				camera.fovy = MAX_FOVY;
			
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			gluPerspective(camera.fovy, camera.aspect, camera.near_c, camera.far_c);
			glutPostRedisplay();
		}
		else if (uip.camera_global_rotation_mode) {
			length = sqrt( (double) (delx*delx + dely*dely) );
			if (length < EPSILON) return;
			u_length = (double) delx/length, v_length = (double) dely/length;

			for (int i = 0; i < 3; i++) 
				d[i] = u_length*camera.uaxis[i] + v_length*camera.vaxis[i];
			fMyVecCrossProduct(&(camera.naxis[0]), d, dir);

			// Let's borrow the OpenGL's function to setup the rotation matrix R
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
			glRotatef(GLOBALROTATION_SENSITIVITY*length, dir[0], dir[1], dir[2]);
			glGetFloatv(GL_MODELVIEW_MATRIX, R);
			glPopMatrix();

			camera.uaxis[0] = R[0]*(tmpx=camera.uaxis[0]) + R[4]*(tmpy=camera.uaxis[1])
				+ R[8]*(tmpz=camera.uaxis[2]);
			camera.uaxis[1] = R[1]*tmpx + R[5]*tmpy + R[9]*tmpz;
			camera.uaxis[2] = R[2]*tmpx + R[6]*tmpy + R[10]*tmpz;
			fMyVecNormalize(camera.uaxis);

			camera.vaxis[0] = R[0]*(tmpx=camera.vaxis[0]) + R[4]*(tmpy=camera.vaxis[1])
				+ R[8]*(tmpz=camera.vaxis[2]);
			camera.vaxis[1] = R[1]*tmpx + R[5]*tmpy + R[9]*tmpz;
			camera.vaxis[2] = R[2]*tmpx + R[6]*tmpy + R[10]*tmpz;
			fMyVecNormalize(camera.vaxis);

			fMyVecCrossProduct(camera.uaxis, camera.vaxis, camera.naxis);

			set_rotate_mat(&camera);

			camera.pos[0] = R[0]*(tmpx=camera.pos[0]) + R[4]*(tmpy=camera.pos[1])
				+ R[8]*(tmpz=camera.pos[2]);
			camera.pos[1] = R[1]*tmpx + R[5]*tmpy + R[9]*tmpz;
			camera.pos[2] = R[2]*tmpx + R[6]*tmpy + R[10]*tmpz;
			
			// Now, update the Viewing Transformation
			glMatrixMode(GL_MODELVIEW);  
			glLoadIdentity();
 			glMultMatrixf(camera.mat);
 			glTranslatef(-camera.pos[X], -camera.pos[Y], -camera.pos[Z]);
			fprintf(stdout, "camera.pos: %f %f %f\n", camera.pos[0], camera.pos[1], camera.pos[2]);

			glutPostRedisplay();
		}
	}
}

void init_OpenGL_RC(void) {
	// glewInit();

	initialize_camera(&camera);
	glClearColor(0.2, 0.2, 0.2, 1.0);

	// Front face: Gold
	set_OpenGL_material(GL_FRONT, 0.24725, 0.1995, 0.0745,  0.75164, 0.60648, 0.22648, 
		0.628281, 0.555802, 0.366065,  0.4); 
	// Back face: Siver 
	set_OpenGL_material(GL_BACK, 0.19225, 0.19225, 0.19225,  0.50754, 0.50754, 0.50754, 
		0.508273, 0.508273, 0.508273,  0.4);

	set_parallel_OpenGL_head_light_color();

	set_OpenGL_light_model();

	if (uip.OpenGL_polygon_mode == LINE) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDisable(GL_LIGHTING);
	}
	else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_LIGHTING); 
		turn_on_parallel_OpenGL_head_light();
		glEnable(GL_LIGHTING);
	}

	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
//	glEnable(GL_CULL_FACE);

	if (uip.OpenGL_shading_mode == FLAT_SHADING) {
		glShadeModel(GL_FLAT);
	}
	else {
		glShadeModel(GL_SMOOTH);	
	}

	glMatrixMode(GL_MODELVIEW); 
	glLoadIdentity();
	set_parallel_OpenGL_head_light_position(); 

 	glMultMatrixf(camera.mat);
 	glTranslatef(-camera.pos[X], -camera.pos[Y], -camera.pos[Z]);
}


void clean_up_system(void) {
	// free memory and etc
	glutDestroyWindow(uip.main_window_ID); 
}


#define N_SL_KDT_CONFIG_COMMANDS 12
const char SL_KDT_CONFIG_indicator[] = "!SL_KDT_CONFIG_1.0";
const char SL_KDT_CONFIG_commands[N_SL_KDT_CONFIG_COMMANDS][256] = { "$INPUT_DIRECTORY", "$OUTPUT_DIRECTORY",
		"$KD_TREE_FILENAME", "$KD_TREE_DUMP_FORMAT", "$I_GEOMETRY_FILENAME", "$MESH_FILE_LIST", "$END",
		"$KD_TREE_TRAVL_COST", "$KD_TREE_ISECT_COST", "$KD_TREE_MAX_LEVEL", "$KD_TREE_MIN_TRIANGLE", 
		"$KD_TREE_EMTPY_BONUS" };
typedef enum _SL_KDT_CONFIG_command_ID {
	CMD_INPUT_DIRECTORY = 0, CMD_OUTPUT_DIRECTORY, CMD_KD_TREE_FILENAME, CMD_KD_TREE_DUMP_FORMAT,
	CMD_I_GEOMETRY_FILENAME, CMD_MESH_FILE_LIST, CMD_END, CMD_KD_TREE_TRAVL_COST, CMD_KD_TREE_ISECT_COST, 
	CMD_KD_TREE_MAX_LEVEL, CMD_KD_TREE_MIN_TRIANGLE, CMD_KD_TREE_EMTPY_BONUS, CMD_COMMENT, CMD_NULL
} SL_KDT_CONFIG_command_ID;

bool read_OBJ_geom_file(const char* filename, MeshGeom* mesh_geom) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		fprintf(stderr, "[OBJ] Cannot open file: %s\n", filename);
		return false;
	}

	std::vector<float> vertices;
	std::vector<unsigned int> indices;

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);

		if (line.substr(0, 2) == "v ") {
			char v;
			float x, y, z;
			iss >> v >> x >> y >> z;

			// 초기화할 때 normal 값도 기본값 0.0f로 넣음
			vertices.push_back(x);
			vertices.push_back(y);
			vertices.push_back(z);
			vertices.push_back(0.0f); // normal.x
			vertices.push_back(0.0f); // normal.y
			vertices.push_back(0.0f); // normal.z
		}
		else if (line.substr(0, 2) == "f ") {
			char f;
			int i1, i2, i3;
			iss >> f >> i1 >> i2 >> i3;

			// obj는 1-based index
			indices.push_back(i1 - 1);
			indices.push_back(i2 - 1);
			indices.push_back(i3 - 1);
		}
	}

	file.close();

	if (vertices.empty() || indices.empty()) {
		fprintf(stderr, "[OBJ] Invalid obj file or no geometry found.\n");
		return false;
	}

	// MeshGeom 구조체에 복사
	mesh_geom->nvertices = (int)(vertices.size() / 6);
	mesh_geom->nfaces = (int)(indices.size() / 3);

	mesh_geom->vertices = (float*)malloc(vertices.size() * sizeof(float));
	memcpy(mesh_geom->vertices, vertices.data(), vertices.size() * sizeof(float));

	mesh_geom->faces = (unsigned int*)malloc(indices.size() * sizeof(unsigned int));
	memcpy(mesh_geom->faces, indices.data(), indices.size() * sizeof(unsigned int));

	// AABB 계산
	float xmin = FLT_MAX, xmax = -FLT_MAX;
	float ymin = FLT_MAX, ymax = -FLT_MAX;
	float zmin = FLT_MAX, zmax = -FLT_MAX;

	for (int i = 0; i < mesh_geom->nvertices; ++i) {
		float* v = &mesh_geom->vertices[i * 6];
		if (v[0] < xmin) xmin = v[0];
		if (v[0] > xmax) xmax = v[0];
		if (v[1] < ymin) ymin = v[1];
		if (v[1] > ymax) ymax = v[1];
		if (v[2] < zmin) zmin = v[2];
		if (v[2] > zmax) zmax = v[2];
	}

	mesh_geom->AABB[XMIN] = xmin;
	mesh_geom->AABB[XMAX] = xmax;
	mesh_geom->AABB[YMIN] = ymin;
	mesh_geom->AABB[YMAX] = ymax;
	mesh_geom->AABB[ZMIN] = zmin;
	mesh_geom->AABB[ZMAX] = zmax;

	return true;
}

int read_OBJ_and_build_kdtree(const char* obj_filename)
{
	printf("> Reading OBJ File and building KD-tree: %s\n\n", obj_filename);

	MeshGeom mesh_geom;
	if (!read_OBJ_geom_file(obj_filename, &mesh_geom)) {
		fprintf(stderr, "Failed to read OBJ file: %s\n", obj_filename);
		return 0;
	}

	// Initialize uip.poly_model
	uip.poly_model.n_triangles = 0;
	for (int i = 0; i < 6; ++i) {
		uip.poly_model.AABB[i] = (i % 2 == 0) ? FLT_MAX : -FLT_MAX;
	}

	// Allocate space for ExtendedVertex
	if ((uip.poly_model.extended_vertices = (ExtendedVertex*)malloc(3 * mesh_geom.nfaces * sizeof(ExtendedVertex))) == NULL) {
		fprintf(stderr, "Memory allocation failed for extended_vertices\n");
		return 0;
	}

	append_mesh_geom_to_composite_object(&uip.poly_model, &mesh_geom, 0, uip.poly_model.AABB);
	printf("[DEBUG] n_triangles: %d\n", uip.poly_model.n_triangles);
	for (int i = 0; i < 3; ++i) {
		printf("Vertex[%d]: %f %f %f\n", i,
			uip.poly_model.extended_vertices[i].vertex[0],
			uip.poly_model.extended_vertices[i].vertex[1],
			uip.poly_model.extended_vertices[i].vertex[2]);
	}
	// KD-tree 생성
	if (uip.poly_model.kd_tree != NULL)
		delete uip.poly_model.kd_tree;

	uip.poly_model.kd_tree = new KdTree();

	//build_kd_tree_for_composite_object(&uip.poly_model);
	//printf("> KD-tree built successfully!\n");

	free(mesh_geom.vertices);
	free(mesh_geom.faces);
	printf("AABB: X [%f, %f] Y [%f, %f] Z [%f, %f]\n",
		uip.poly_model.AABB[XMIN], uip.poly_model.AABB[XMAX],
		uip.poly_model.AABB[YMIN], uip.poly_model.AABB[YMAX],
		uip.poly_model.AABB[ZMIN], uip.poly_model.AABB[ZMAX]);

	return 1;
}

bool load_obj_to_composite_object(const char* filename, CompositeObject* c_object) {
	std::ifstream infile(filename);
	if (!infile.is_open()) {
		fprintf(stderr, "Failed to open OBJ file: %s\n", filename);
		return false;
	}

	std::vector<float> vertices;
	std::vector<unsigned int> indices;
	std::string line;
	while (std::getline(infile, line)) {
		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;
		if (prefix == "v") {
			float x, y, z;
			iss >> x >> y >> z;
			vertices.push_back(x);
			vertices.push_back(y);
			vertices.push_back(z);
		}
		else if (prefix == "f") {
			unsigned int i1, i2, i3;
			iss >> i1 >> i2 >> i3;
			// .obj는 1-based index이므로 -1
			indices.push_back(i1 - 1);
			indices.push_back(i2 - 1);
			indices.push_back(i3 - 1);
		}
	}
	infile.close();

	size_t n_triangles = indices.size() / 3;
	c_object->n_triangles = static_cast<int>(n_triangles);
	c_object->extended_vertices = new ExtendedVertex[3 * n_triangles];

	// AABB 초기화
	for (int i = 0; i < 3; ++i) {
		c_object->AABB[2 * i + 0] = FLT_MAX;
		c_object->AABB[2 * i + 1] = -FLT_MAX;
	}

	for (size_t t = 0; t < n_triangles; ++t) {
		for (int k = 0; k < 3; ++k) {
			unsigned int vidx = indices[3 * t + k];
			float* pos = &vertices[3 * vidx];

			for (int i = 0; i < 3; ++i) {
				c_object->AABB[2 * i + 0] = fmin(c_object->AABB[2 * i + 0], pos[i]);
				c_object->AABB[2 * i + 1] = fmax(c_object->AABB[2 * i + 1], pos[i]);
			}

			ExtendedVertex& ev = c_object->extended_vertices[3 * t + k];
			ev.vertex[0] = pos[0];
			ev.vertex[1] = pos[1];
			ev.vertex[2] = pos[2];
			ev.normal[0] = 0.0f;  // 필요시 노멀 계산 가능, gaussian index 넣어야할듯
			ev.normal[1] = 0.0f;
			ev.normal[2] = 0.0f;
			ev.material_ID = 0;
		}
	}
	printf("AABB: X [%f, %f] Y [%f, %f] Z [%f, %f]\n",
		c_object->AABB[XMIN], c_object->AABB[XMAX],
		c_object->AABB[YMIN], c_object->AABB[YMAX],
		c_object->AABB[ZMIN], c_object->AABB[ZMAX]);
	return true;
}

SL_KDT_CONFIG_command_ID query_SL_KDT_CONFIG_command_ID(const char *command) {
	int i;

	if (command[0] == '#') return CMD_COMMENT;

	for (i = 0; i < N_SL_KDT_CONFIG_COMMANDS; i++)
	if (strstr(command, SL_KDT_CONFIG_commands[i]))
		return (SL_KDT_CONFIG_command_ID)i;

	return CMD_NULL;
}
typedef struct _S_Element {
	char string[256];
	int id;
	struct _S_Element *next;
} S_Element;
S_Element *filelist;

void append_mesh_geom_to_composite_object(CompositeObject *c_object, 
											MeshGeom *mesh_geom, int mat_type, float *AABB) {
	int i, face;
	float *vertex;
	unsigned int *ptr_index;
	ExtendedVertex *ptr_next;
	static int iii = 0;

	if (mesh_geom->AABB[XMIN] < AABB[XMIN]) AABB[XMIN] = mesh_geom->AABB[XMIN];
	if (mesh_geom->AABB[XMAX] > AABB[XMAX]) AABB[XMAX] = mesh_geom->AABB[XMAX];
	if (mesh_geom->AABB[YMIN] < AABB[YMIN]) AABB[YMIN] = mesh_geom->AABB[YMIN];
	if (mesh_geom->AABB[YMAX] > AABB[YMAX]) AABB[YMAX] = mesh_geom->AABB[YMAX];
	if (mesh_geom->AABB[ZMIN] < AABB[ZMIN]) AABB[ZMIN] = mesh_geom->AABB[ZMIN];
	if (mesh_geom->AABB[ZMAX] > AABB[ZMAX]) AABB[ZMAX] = mesh_geom->AABB[ZMAX];

	ptr_next = c_object->extended_vertices + 3*c_object->n_triangles;
	for (face = 0; face < mesh_geom->nfaces; face++) {
		ptr_index = mesh_geom->faces + 3*face;
		for (i = 0; i < 3; i++) {
			vertex = mesh_geom->vertices + *ptr_index*6;

	 		memcpy(ptr_next->vertex, vertex, 3*sizeof(float));
	 		memcpy(ptr_next->normal, vertex+3, 3*sizeof(float));
			ptr_next->material_ID = mat_type;

			ptr_index++;
			ptr_next++;
		}
		c_object->n_triangles++;
	}
}


int read_SL_KDT_CONFIG_file(void) {

	int i, n_total_faces, error_mode = 1;
	FILE *fp;
	size_t length;
	char command_buf[256], *ptr_c, tmp_buf[256], mat_ID_s[32];
	SL_KDT_CONFIG_command_ID current_command;
	char meshfilename[512];
	S_Element *ptr_se;
	MeshGeom cur_mesh_geom;

	fprintf(stdout, "> Reading SL_KDT_Config File and prepair geometry: %s\n\n",
		uip.SL_KDT_CONFIG_filename);

	filelist = (S_Element *)NULL;
	uip.poly_model.n_triangles = 0;
	uip.poly_model.AABB[XMIN] = uip.poly_model.AABB[YMIN] = uip.poly_model.AABB[ZMIN] = FLT_MAX;
	uip.poly_model.AABB[XMAX] = uip.poly_model.AABB[YMAX] = uip.poly_model.AABB[ZMAX] = -FLT_MAX;

	n_total_faces = 0;

	// First round
	if ((fp = fopen(uip.SL_KDT_CONFIG_filename, "r")) == NULL) {
		fprintf(stderr, "r_SL_KDT_CONFIG_f: (Error) cannot read the SL config file %s. Quitting...\n",
			uip.SL_KDT_CONFIG_filename);
		return 0;
	}

	fgets(command_buf, 511, fp);
	command_buf[18] = '\0';

	if (strcmp(SL_KDT_CONFIG_indicator, command_buf) != 0) {
		fprintf(stderr, "r_SL_KDT_CONFIG_f: (Error) the file type is not %s. Quitting...\n",
			SL_KDT_CONFIG_indicator);
		fclose(fp);
		return 0;
	}

	while (fgets(command_buf, 511, fp) != NULL) {
		current_command = query_SL_KDT_CONFIG_command_ID(command_buf);
		if ((current_command == CMD_COMMENT) || (current_command == CMD_NULL)) continue;

		switch (current_command) {
		case CMD_INPUT_DIRECTORY:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_INPUT_DIRECTORY]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_INPUT_DIRECTORY]), "%s", tmp_buf);
			strcpy(uip.mesh_geom_files_dir, tmp_buf);
			fprintf(stdout, "  * Mesh Geom Files Dir = %s\n", uip.mesh_geom_files_dir);
			break;
		case CMD_OUTPUT_DIRECTORY:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_OUTPUT_DIRECTORY]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_OUTPUT_DIRECTORY]), "%s", tmp_buf);
			strcpy(uip.kd_tree_dump_dir, tmp_buf);
			fprintf(stdout, "  * Kd-tree Dump Dir = %s\n", uip.kd_tree_dump_dir);
			break;
		case CMD_KD_TREE_FILENAME:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_KD_TREE_FILENAME]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_KD_TREE_FILENAME]), "%s", tmp_buf);
			strcpy(uip.kd_tree_filename, tmp_buf);
			fprintf(stdout, "  * Kd-tree Filename = %s\n", uip.kd_tree_filename);
			break;
		case CMD_KD_TREE_DUMP_FORMAT:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_KD_TREE_DUMP_FORMAT]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_KD_TREE_DUMP_FORMAT]), "%s", tmp_buf);
			if (strcmp("BINARY", tmp_buf) == 0) {
				uip.kd_tree_dump_format = KD_TREE_DUMP_IN_BINARY;
				fprintf(stdout, "  * Kd-tree dump format = BINARY\n");
			}
			else { // must check syntax error
				uip.kd_tree_dump_format = KD_TREE_DUMP_IN_ASCII;
				fprintf(stdout, "  * Kd-tree dump format = ASCII\n");
			}
			break;
		case CMD_I_GEOMETRY_FILENAME:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_I_GEOMETRY_FILENAME]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_I_GEOMETRY_FILENAME]), "%s", tmp_buf);
			strcpy(uip.i_geometry_filename, tmp_buf);
			fprintf(stdout, "  * I-geometry Filename = %s\n", uip.i_geometry_filename);
			break;

		case CMD_KD_TREE_TRAVL_COST:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_KD_TREE_TRAVL_COST]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_KD_TREE_TRAVL_COST]), "%f", &v_KD_TREE_TRAVL_COST);
			fprintf(stdout, "  * Kd-tree travel cost = %f\n", v_KD_TREE_TRAVL_COST);
			break;
		case CMD_KD_TREE_ISECT_COST:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_KD_TREE_ISECT_COST]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_KD_TREE_ISECT_COST]), "%f", &v_KD_TREE_ISECT_COST);
			fprintf(stdout, "  * Kd-tree intersection cost = %f\n", v_KD_TREE_ISECT_COST);
			break;
		case CMD_KD_TREE_MAX_LEVEL:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_KD_TREE_MAX_LEVEL]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_KD_TREE_MAX_LEVEL]), "%u", &v_KD_TREE_MAX_LEVEL);
			fprintf(stdout, "  * Kd-tree max level = %u\n", v_KD_TREE_MAX_LEVEL);
			break;
		case CMD_KD_TREE_MIN_TRIANGLE:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_KD_TREE_MIN_TRIANGLE]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_KD_TREE_MIN_TRIANGLE]), "%u", &v_KD_TREE_MIN_TRIANGLE);
			fprintf(stdout, "  * Kd-tree min # of triangles per leaf = %u\n", v_KD_TREE_MIN_TRIANGLE);
			break;
		case CMD_KD_TREE_EMTPY_BONUS:
			ptr_c = strstr(command_buf, SL_KDT_CONFIG_commands[CMD_KD_TREE_EMTPY_BONUS]);
			sscanf(ptr_c + strlen(SL_KDT_CONFIG_commands[CMD_KD_TREE_EMTPY_BONUS]), "%f", &v_KD_TREE_EMTPY_BONUS);
			fprintf(stdout, "  * Kd-tree empty bonus = %f\n", v_KD_TREE_EMTPY_BONUS);
			break;

		case CMD_MESH_FILE_LIST:
			uip.n_mesh_geoms = 0;
			while (fgets(command_buf, 511, fp) != NULL) {
				current_command = query_SL_KDT_CONFIG_command_ID(command_buf);
				if (current_command == CMD_COMMENT) continue;
				if (current_command == CMD_END) {
					error_mode = 0;
					break;
				}
				if ((ptr_c = strstr(command_buf, ".mesh")) == NULL) {
					fprintf(stderr, "r_SL_KDT_CONFIG_f: (Error) the filename %s is wrong. Skipping this file...\n",
						command_buf);
				}
				else {
					sscanf(command_buf, "%s %s", tmp_buf, mat_ID_s);

					ptr_se = (S_Element *) malloc(sizeof(S_Element));
					strcpy(ptr_se->string, tmp_buf);
					ptr_se->id = atoi(mat_ID_s);

					ptr_se->next = filelist;
					filelist = ptr_se;
					uip.n_mesh_geoms += 1;

					strcpy(meshfilename, uip.mesh_geom_files_dir);
					strcat(meshfilename, "/");
					strcat(meshfilename, filelist->string);

					n_total_faces += fetch_face_numbers_SL_mesh_geom_file(meshfilename);
				}
			} // while (fgets(command_buf, 511, fp) != NULL) {
			if ((current_command != CMD_END) || (uip.n_mesh_geoms == 0)) {
				fprintf(stdout, "r_SL_FNC_CONFIG_f: (Error) found a syntax error. Quitting...\n\n");
				fclose(fp);
				return 0;
			}
			uip.mesh_geom_filenames = (char **) malloc(uip.n_mesh_geoms*sizeof(char *));
			uip.mesh_geom_mat_IDs = (int *) malloc(uip.n_mesh_geoms*sizeof(int));

			ptr_se = filelist;
			for (i = uip.n_mesh_geoms - 1; i >= 0; i--) {
				uip.mesh_geom_filenames[i] = ptr_se->string;
				uip.mesh_geom_mat_IDs[i] = ptr_se->id;
				ptr_se = ptr_se->next;
			}
			fprintf(stdout, "\n");
			break;
		}
	}
	fclose(fp);
	if (error_mode == 1) {
		fprintf(stdout, "r_SL_KDT_CONFIG_f: (Error) found a syntax error. Quitting...\n\n");
		fclose(fp);
		return 0;
	}
	if ((uip.poly_model.extended_vertices
		= (ExtendedVertex *)malloc(3 * n_total_faces*sizeof(ExtendedVertex))) == NULL) {
		fprintf(stderr, "r_SL_KDT_CONFIG_f: (Error) memory allocation error c101\n");
		exit(-1);
	}

	for (i = 0; i < uip.n_mesh_geoms; i++) {
		strcpy(meshfilename, uip.mesh_geom_files_dir);
		strcat(meshfilename, "/");
		strcat(meshfilename, uip.mesh_geom_filenames[i]);

		if (!read_SL_mesh_geom_file_kd_tree(meshfilename, &cur_mesh_geom))
			continue;
		append_mesh_geom_to_composite_object(&uip.poly_model, &cur_mesh_geom, uip.mesh_geom_mat_IDs[i], uip.poly_model.AABB);
		free(cur_mesh_geom.vertices);
		free(cur_mesh_geom.faces);
	}

	fprintf(stdout, "> Done!\n\n");
	return 1;
}


void main_menu_action(int selection) {
	char full_kd_tree_file_name[512];
	char full_i_geometry_file_name[512];

	switch (selection) {
	case 0:
		g_cuda_rendering_done = !g_cuda_rendering_done;
		printf(g_cuda_rendering_done ? "->CUDA Rendering\n":"I-Geom Rendering\n");
		glutPostRedisplay();
		break;
	case 100:
		render_gaussian = false;
		//		uip.SL_KDT_CONFIG_filename = "../../Data/Configurations/YP_ALL.SL_KDT_config";
		//		uip.SL_KDT_CONFIG_filename = "../../Data/Configurations/AN_ALL.SL_KDT_config";
		//		uip.SL_KDT_CONFIG_filename = "../../Data/Configurations/YP_ALL_LED.SL_KDT_config";
		//  	uip.SL_KDT_CONFIG_filename = "../../Data/Configurations/TL/TL_ALL.SL_KDT_config";
		uip.SL_KDT_CONFIG_filename = "../../Data/Configurations/TL/TL_SOME_LED.SL_KDT_config";
		//		.SL_KDT_CONFIG_filename = "../../Data/Configurations/TL/TL_ALL_LED.SL_KDT_config";
		//      uip.SL_KDT_CONFIG_filename = "../../Data/Configurations/AN_ALL_LED.SL_KDT_config";
		read_SL_KDT_CONFIG_file();
		uip.composite_object_read = 1;

		if (0) {
			fprintf(stdout, "m_m_a: kd_tree_dump_dir = %s.\n", uip.kd_tree_dump_dir);
			fprintf(stdout, "m_m_a: kd_tree_file_name = %s.\n", uip.kd_tree_filename);
			if (uip.kd_tree_dump_format == KD_TREE_DUMP_IN_BINARY)
				fprintf(stdout, "m_m_a: kd_tree_dump_format = BINARY\n");
			else
				fprintf(stdout, "m_m_a: kd_tree_dump_format = ASCII\n");
		}
		load_poly_model_into_OpenGL();
		g_cuda_rendering_done = false;
		glutPostRedisplay();
		break;
	case 200:
		build_kd_tree_for_composite_object(&uip.poly_model);
		if (uip.poly_model.kd_tree->tri_accel_list == NULL) printf("tri_accel_list NULL\n");
		else {
			printf("triangle num: %d\n", uip.poly_model.n_triangles);
			printf("tri_accel_list size: %d", sizeof(uip.poly_model.kd_tree->tri_accel_list) / sizeof(*(uip.poly_model.kd_tree->tri_accel_list)));
		}
		break;
	case 300:
		strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
		strcat(full_kd_tree_file_name, "/");
		strcat(full_kd_tree_file_name, uip.kd_tree_filename);

		strcpy(full_i_geometry_file_name, uip.kd_tree_dump_dir);
		strcat(full_i_geometry_file_name, "/");
		strcat(full_i_geometry_file_name, uip.i_geometry_filename);

		if (render_gaussian) {
			dump_kd_tree_for_composite_object(
				&uip.poly_model,
				"../../Data/Obj/hotdog_tree.kdt",         // 저장할 kd-tree
				KD_TREE_DUMP_IN_BINARY,    // 저장 포맷
				"../../Data/Obj/hotdog_igeom.bin"         // 저장할 geometry
			);
		}
		else {
			dump_kd_tree_for_composite_object(&uip.poly_model, full_kd_tree_file_name,
				uip.kd_tree_dump_format, full_i_geometry_file_name);
		}
		break;
		case 400:
			strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
			printf("uip.kd_tree_dump_dir:%s\n", uip.kd_tree_dump_dir);
			strcat(full_kd_tree_file_name, "/");
			strcat(full_kd_tree_file_name, uip.kd_tree_filename);
			printf("uip.kd_tree_filename:%s\n", uip.kd_tree_filename);
			if (render_gaussian) {
				strcpy(full_kd_tree_file_name, "../../Data/Obj/hotdog_tree.kdt");
				uip.kd_tree_dump_format = KD_TREE_DUMP_IN_BINARY;
			}
			printf("full_kd_tree_file_name:%s\n", full_kd_tree_file_name);
			read_kd_tree_from_file(&uip.poly_model, full_kd_tree_file_name, uip.kd_tree_dump_format);
			glutPostRedisplay();
			break;
		case 500: {
			render_gaussian = true;
			const char* obj_path = "../../Data/Obj/hotdog_3dgrt.obj";  // obj 경로

			if (!read_OBJ_and_build_kdtree(obj_path)) {
				fprintf(stderr, "Failed to load obj and build Kd-tree\n");
				return;
			}

			//CompositeObject obj_model;
			//if (!load_obj_to_composite_object(obj_path, &obj_model)) {
			//	fprintf(stderr, "Failed to load .obj file.\n");
			//	break;
			//}
			//uip.poly_model = obj_model;

			uip.composite_object_read = 1;

			//fprintf(stdout, "Successfully loaded .obj model. Building Kd-tree...\n");
			//build_kd_tree_for_composite_object(&obj_model);

			//dump_kd_tree_for_composite_object(
			//	&obj_model,
			//	"../../Data/Obj/hotdog_tree.kdt",         // 저장할 kd-tree
			//	KD_TREE_DUMP_IN_BINARY,    // 저장 포맷
			//	"../../Data/Obj/hotdog_igeom.bin"         // 저장할 geometry
			//);


			//printf("uip, AABB: X [%f, %f] Y [%f, %f] Z [%f, %f]\n",
			//	uip.poly_model.AABB[XMIN], uip.poly_model.AABB[XMAX],
			//	uip.poly_model.AABB[YMIN], uip.poly_model.AABB[YMAX],
			//	uip.poly_model.AABB[ZMIN], uip.poly_model.AABB[ZMAX]);

			load_poly_model_into_OpenGL();
			g_cuda_rendering_done = false;
			glutPostRedisplay();
			printf("draw DONE\n");
			break;
		}
		case 600: {
			//CUDA rendering
			fprintf(stdout, "CUDA ray tracing Render using SGRT with kd-tree\n");
			if (!uip.composite_object_read) {
				fprintf(stderr, "CompositeObject not loaded.\n");
				break;
			}
			if (uip.poly_model.n_triangles == 0) {
				fprintf(stdout, "No triangles in CompositeObject\n");
				break;
			}
			if (uip.poly_model.kd_tree == NULL) {
				fprintf(stdout, "No kd-tree in CompositeObject\n");
				break;
			}
			//TODO: CUDA rendering*****************************************
			
			/*/test.h
			kernelTestFuncion();
			cudaCopyTest();
			printf("test done\n");*/

			/*//SGRT
			renderWithSGRT(
				uip.poly_model,
				camera,
				g_render_width,
				g_render_height,
				g_render_framebuffer,
				g_cuda_rendering_done
			);

			if (g_cuda_rendering_done) {
				printf("SGRT rendering complete. Refreshing display...\n");
				glutPostRedisplay();
			}
			else {
				fprintf(stderr, "SGRT rendering failed.\n");
			}*/

			//cudaRenderer.h
			//renderWithCuda(const CompositeObject & object, const Camera & camera, int width, int height, float*& out_framebuffer, bool& is_done)
			renderWithCuda(
				uip.poly_model,
				camera,
				g_render_width,
				g_render_height,
				g_render_framebuffer,
				g_cuda_rendering_done
			);
			if (g_cuda_rendering_done) {
				printf("CUDA rendering complete. Refreshing display...\n");
				glutPostRedisplay();
			}
			else {
				fprintf(stderr, "CUDA rendering failed.\n");
			}
			
			/*/cudaKDTreeTracer
			initCudaRendering(uip.poly_model, g_render_framebuffer, &g_cuda_rendering_done);*/

			/*GScene* scene = new GScene();
			scene->setKdTreeLoadFilePath("../../Data/Obj/hotdog_tree.kdt");
			scene->convertRenderScene();
			scene->buildObjectKdTree();
			GKDTreeStructure* kdTree = new GKDTreeStructure(scene);
			kdTree->initialize();
			scene->setSceneKDTree(kdTree);

			GGPUExperimentalRayTracer* rayTracer = new GGPUExperimentalRayTracer();

			rayTracer->rendering(scene, false);*/
			

			/*CompositeObject& obj = uip.poly_model;
			GScene scn;

			convertCompositeObjectToGSceneAndKdTree(obj, scn);
			GKDTreeStructure* kdTree = new GKDTreeStructure(&scn);
			kdTree->initialize();
			scn.setSceneKDTree(kdTree);

			GGPUExperimentalRayTracer* rayTracer = new GGPUExperimentalRayTracer();

			rayTracer->rendering(&scn, false);*/

			/*UploadCompositeObjectToDevice(uip.poly_model);
			LaunchRenderKernel(g_render_framebuffer, g_render_width, g_render_height);*/

			/*CompositeObject* compObj = &uip.poly_model;
			GScene* scene = convertCompositeObjectToScene(compObj);
			if (!scene) {
				printf("[SGRT] Failed to convert CompositeObject to GScene.\n");
				return;
			}

			scene->convertRenderScene();

			GGPUExperimentalRayTracer* tracer;
			//tracer.setScene(scene);//TODO
			GError err = tracer->rendering(scene, false);

			if (err != errorNo)
				printf("[SGRT] Rendering failed: %d\n", err);
			else
				printf("[SGRT] Rendering succeeded.\n");

			delete scene;*/

			/*GScene* scene = convertCompositeObjectToGScene(&uip.poly_model);
			GGPUExperimentalRayTracer raytracer;
			GError err = raytracer.rendering(scene, false);

			if (err != errorNo) {
				printf("CUDA rendering failed with error %d\n", err);
				break;
			}

			// 결과 프레임버퍼 가져오기
			g_render_framebuffer = raytracer.getFrameBufferPointer();
			g_render_width = raytracer.getFrameBufferWidth();
			g_render_height = raytracer.getFrameBufferHeight();
			g_cuda_rendering_done = 1;

			printf("Rendering done. Displaying on screen...\n");
			glutPostRedisplay();
			break;*/

			/*GKdTreeAccel* kdAccel = new GKdTreeAccel();
			kdAccel->setFromCompositeObject(&obj_model);
			
			GScene* scene = new GScene();
			scene->initalize(); // scene 내부 변수 초기화

			scene->setResolution(800, 600);
			scene->setSuperSampling(1, 1);
			scene->setGPUBlockSize(8, 8);
			scene->setAccelStructure(kdAccel);
			scene->setMaxReflectionDepth(1);
			scene->setEnableShadow(false);
			scene->setEnableLocalShading(false);
			scene->setUseTexture(false);

			GGPUExperimentalRayTracer* renderer = new GGPUExperimentalRayTracer();
			GError err = renderer->rendering(scene, false);
			if (err != errorNo) {
				printf("CUDA rendering failed: %s\n", GErrorManager::getGErrorString(err));
			}
			break;

			// [1] 카메라 설정
			cuCamera camera;
			camera.eye = make_float3(0, 0, -5);
			camera.u = make_float3(1, 0, 0);
			camera.v = make_float3(0, 1, 0);
			camera.n = make_float3(0, 0, 1);
			camera.fnear = 1.0f;
			camera.startPoint = make_float3(-1, 1, 0);
			camera.stepX = 2.0f / 800;
			camera.stepY = 2.0f / 600;

			// [2] 장면 정보
			SceneInfo info = {};
			info.iResolutionX = 800;
			info.iResolutionY = 600;
			info.iBlockSizeX = 8;
			info.iBlockSizeY = 8;
			info.iSuperSamplingX = 1;
			info.iSuperSamplingY = 1;

			//// [3] CompositeObject → CUDA에 업로드
			//CompositeObject* d_obj;
			//cudaMalloc(&d_obj, sizeof(CompositeObject));
			//upload_composite_object_to_cuda(&uip.poly_model, d_obj);
			// [3] CompositeObject → CUDA에 깊은 복사로 업로드
			CompositeObject* d_obj = nullptr;
			deep_copy_composite_object_to_cuda(&uip.poly_model, &d_obj);  // 새로 구현한 함수 사용

			// [4] 프레임버퍼 준비
			float* d_framebuffer;
			cudaMalloc(&d_framebuffer, sizeof(float) * 800 * 600 * 3);
			cudaMemset(d_framebuffer, 0, sizeof(float) * 800 * 600 * 3);

			// [5] CUDA 커널 호출
			dim3 block(8, 8);
			dim3 grid((800 + 7) / 8, (600 + 7) / 8);
			singlePassRayTracingKernel <<< grid, block >>> (
				d_framebuffer, 1, 0, 0, false,
				camera, info, d_obj
				);
			cudaDeviceSynchronize();

			// [6] 결과 복사
			float* h_framebuffer = new float[800 * 600 * 3];
			cudaMemcpy(h_framebuffer, d_framebuffer, sizeof(float) * 800 * 600 * 3, cudaMemcpyDeviceToHost);
			//save_as_ppm(h_framebuffer, 800, 600, "output_kdtree.ppm");*/
			//*************************************************************
			break;
		}
		case 700:
			g_cuda_interactive_mode = !g_cuda_interactive_mode; // 인터랙티브 모드 토글
			if (g_cuda_interactive_mode) {
				g_camera_dirty = true; // 모드를 켜는 즉시 한 번 렌더링하도록 설정
				printf("CUDA Interactive Mode: ON\n");
			}
			else {
				printf("CUDA Interactive Mode: OFF\n");
				// 인터랙티브 모드를 끄면 다시 OpenGL 뷰로 돌아가도록 화면 갱신
				glutPostRedisplay();
			}
			break;
		case 999:
			exit(0);
			clean_up_system();
			break;
	}
}

void register_callbacks_and_create_menu(void) {
	glutDisplayFunc(display); 
	glutKeyboardFunc(keyboard); 
	glutReshapeFunc(reshape);
	glutMouseFunc(mousepress); 
	glutMotionFunc(mousemove);
   
	uip.main_menu_ID = glutCreateMenu(main_menu_action);
	glutAddMenuEntry("ChangeMode", 0);
	glutAddMenuEntry("1. Read SL_KDT_Config File and Prepair I-Geometry", 100); 
	glutAddMenuEntry("2. Construct Kd-tree from I-Geometry", 200);  
	glutAddMenuEntry("3. Dump Kd-tree and I-Geometry to Files", 300);
	glutAddMenuEntry("4. Read Kd-tree from File", 400);
	glutAddMenuEntry("5. Read .obj File and Prepair I-Geometry", 500);
	glutAddMenuEntry("6. CUDA Rendering (One-shot)", 600);
	glutAddMenuEntry("7. CUDA Rendering (Interactive Toggle)", 700); // 메뉴 추가
	glutAddMenuEntry("Exit", 999); 

	glutAttachMenu(GLUT_RIGHT_BUTTON); 
}


void print_OpenGL_GLSL_GLEW_version(void) {
// Need to include proper header files
#define GL_SHADING_LANGUAGE_VERSION		0x8B8C
#define GL_MAJOR_VERSION                0x821B
#define GL_MINOR_VERSION                0x821C

	const GLubyte *renderer = glGetString(GL_RENDERER);
	const GLubyte *vendor = glGetString(GL_VENDOR);
	const GLubyte *version = glGetString(GL_VERSION);
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

 	GLint major, minor;
 	glGetIntegerv(GL_MAJOR_VERSION, &major);
 	glGetIntegerv(GL_MINOR_VERSION, &minor);

	printf("===========================================================\n");
	printf("GL Vendor      : %s\n", vendor);
	printf("GL Renderer    : %s\n", renderer);
	printf("GL Version  (string)   : %s\n", version);
 	printf("GL Version  (integer)  : %d.%d\n", major, minor);
	printf("GLSL Version  : %s\n", glslVersion);
	printf("===========================================================\n\n");

}

void init_KDT_system(void) {
	uip.camera_zoom_mode = 0;
	uip.camera_global_rotation_mode = 0;
	uip.bounding_box_display_mode = 1;
	uip.OpenGL_shading_mode = SMOOTH_SHADING;
	uip.OpenGL_polygon_mode = FILL;
	uip.left_button_pressed = 0;
	uip.right_button_pressed = 0;
	uip.composite_object_read = 0;

	v_KD_TREE_TRAVL_COST = 1.0;
	v_KD_TREE_ISECT_COST = 1.5;
	v_KD_TREE_MAX_LEVEL = 100;
	v_KD_TREE_MIN_TRIANGLE = 4;
	v_KD_TREE_EMTPY_BONUS = 0.9;
}

void initialize_glew(void) {
	GLenum error;

	glewExperimental = GL_TRUE;

	error = glewInit();
	if (error != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(error));
		exit(-1);
	}
	fprintf(stdout, "*********************************************************\n");
	fprintf(stdout, " - GLEW version supported: %s\n", glewGetString(GLEW_VERSION));
	fprintf(stdout, " - OpenGL renderer: %s\n", glGetString(GL_RENDERER));
	fprintf(stdout, " - OpenGL version supported: %s\n", glGetString(GL_VERSION));
	fprintf(stdout, "*********************************************************\n\n");
}


void show_greetings(void) {
	fprintf(stdout, "/***********************************************************/\n");
	fprintf(stdout, "/*                                                         */\n");
	fprintf(stdout, "/*    Mesh-to-Kd-Tree Converter SW                         */\n");
	fprintf(stdout, "/*    -----------------------------                        */\n");
	fprintf(stdout, "/*     - Date: December 1, 2014                            */\n");
	fprintf(stdout, "/*     - Version: 1.0.0_glut                               */\n");
	fprintf(stdout, "/*                                                         */\n");
	fprintf(stdout, "/*                                                         */\n");
	fprintf(stdout, "/***********************************************************/\n\n");
}

void idle() {
	// 인터랙티브 모드가 켜져 있고, 카메라가 변경되었을 때만 다시 렌더링
	if (g_cuda_interactive_mode && g_camera_dirty) {
		g_camera_dirty = false; // 플래그 리셋

		// CUDA 렌더링 실행 (기존 렌더링 함수 재사용)
		renderWithCuda(uip.poly_model, camera, g_render_width, g_render_height, g_render_framebuffer, g_cuda_rendering_done);

		glutPostRedisplay(); // 화면 갱신 요청
	}
}

void main(int argc, char **argv) {
	cudaGLSetGLDevice(0);
	init_KDT_system();
	glutInit (&argc, argv); 
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);   
	glutInitWindowSize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT);
	glutInitContextVersion(4, 0);
	glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
	uip.main_window_ID = glutCreateWindow("SL Mesh-to-Kd-Tree Converter SW: Verion 1.0_glut");

	if (!initCuda()) {
		fprintf(stderr, "Failed to initialize CUDA. Exiting.\n");
		system("pause");
		exit(1);
	}

	initialize_glew(); 
	register_callbacks_and_create_menu();
	glutIdleFunc(idle);

	init_OpenGL_RC(); 
	print_OpenGL_GLSL_GLEW_version();
	show_greetings();

	glutMainLoop ();
}


