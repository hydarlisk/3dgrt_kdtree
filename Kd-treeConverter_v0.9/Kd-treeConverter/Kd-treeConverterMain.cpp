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
#include <vector>
//#include "sgrt_interface.h"
//#include "SGRT_Integration.h"
#include "test.h"
#include "cudaRenderer.h"
//#include "SGRTx2Lib/cuda_math.h"
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
std::vector<Gaussian> g_gaussians; // 전역 변수로 가우시안 데이터를 저장할 벡터 선언
#include <sys/stat.h>

void check_ply_file_size(const char* filename) {
	struct stat st;
	if (stat(filename, &st) != 0) {
		printf("Failed to stat file\n");
		return;
	}
	printf("Actual file size: %lld bytes\n", (long long)st.st_size);
}

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
	glTranslatef(-(uip.poly_model.AABB[XMIN]+uip.poly_model.AABB[XMAX])/2.0,
					-(uip.poly_model.AABB[YMIN]+uip.poly_model.AABB[YMAX])/2.0, 
					-(uip.poly_model.AABB[ZMIN]+uip.poly_model.AABB[ZMAX])/2.0);

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

//shyun
bool loadGaussiansFromPly(const char* filename, std::vector<Gaussian>& gaussians) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "Error: Cannot open .ply file: %s\n", filename);
		return false;
	}

	// --- 1. 헤더 파싱 ---
	std::string line;
	int vertex_count = 0;
	while (std::getline(file, line)) {
		if (line == "end_header") {
			break;
		}
		// "element vertex" 라인을 찾아 가우시안의 총 개수를 파악
		if (line.rfind("element vertex", 0) == 0) {
			sscanf(line.c_str(), "element vertex %d", &vertex_count);
		}
	}

	if (vertex_count == 0) {
		fprintf(stderr, "Error: No vertex elements found in .ply header.\n");
		return false;
	}

	// --- 2. 바이너리 데이터 읽기 ---
	// .ply 파일에 저장된 순서와 타입을 그대로 반영한 임시 구조체
	struct PlyGaussian {
		float pos[3];
		float normal[3]; // 사용하지 않지만 파일에 포함되어 있음
		float f_dc[3];
		float opacity;
		float scale[3];
		float rot[4];
	};

	gaussians.resize(vertex_count);
	for (int i = 0; i < vertex_count; ++i) {
		PlyGaussian ply_gauss;
		file.read(reinterpret_cast<char*>(&ply_gauss), sizeof(PlyGaussian));

		if (file.eof()) break; // 파일 끝에 도달하면 중단

		// 임시 구조체에서 최종 Gaussian 구조체로 데이터 복사 및 변환
		gaussians[i].pos[0] = ply_gauss.pos[0];
		gaussians[i].pos[1] = ply_gauss.pos[1];
		gaussians[i].pos[2] = ply_gauss.pos[2];

		// 중요: opacity는 sigmoid 함수를 적용해야 0~1 사이 값으로 변환됨
		gaussians[i].opacity = 1.0f / (1.0f + expf(-ply_gauss.opacity));

		// 중요: scale은 exp 함수를 적용해야 실제 크기 값이 됨
		gaussians[i].scale[0] = expf(ply_gauss.scale[0]);
		gaussians[i].scale[1] = expf(ply_gauss.scale[1]);
		gaussians[i].scale[2] = expf(ply_gauss.scale[2]);

		// 쿼터니언은 정규화(normalize) 필요
		float norm = sqrt(ply_gauss.rot[0] * ply_gauss.rot[0] + ply_gauss.rot[1] * ply_gauss.rot[1] + ply_gauss.rot[2] * ply_gauss.rot[2] + ply_gauss.rot[3] * ply_gauss.rot[3]);
		gaussians[i].rot[0] = ply_gauss.rot[0] / norm; // w
		gaussians[i].rot[1] = ply_gauss.rot[1] / norm; // x
		gaussians[i].rot[2] = ply_gauss.rot[2] / norm; // y
		gaussians[i].rot[3] = ply_gauss.rot[3] / norm; // z

		// DC 색상값은 그대로 복사 (SH의 추가적인 요소는 일단 무시)
		gaussians[i].f_dc[0] = ply_gauss.f_dc[0];
		gaussians[i].f_dc[1] = ply_gauss.f_dc[1];
		gaussians[i].f_dc[2] = ply_gauss.f_dc[2];
	}

	file.close();

	// 로드 성공 후, 카메라 위치 등을 초기화하기 위해 AABB 계산 (선택적)
	if (!gaussians.empty()) {
		uip.poly_model.AABB[XMIN] = uip.poly_model.AABB[YMIN] = uip.poly_model.AABB[ZMIN] = FLT_MAX;
		uip.poly_model.AABB[XMAX] = uip.poly_model.AABB[YMAX] = uip.poly_model.AABB[ZMAX] = -FLT_MAX;
		for (const auto& g : gaussians) {
			uip.poly_model.AABB[XMIN] = fminf(uip.poly_model.AABB[XMIN], g.pos[0]);
			uip.poly_model.AABB[XMAX] = fmaxf(uip.poly_model.AABB[XMAX], g.pos[0]);
			uip.poly_model.AABB[YMIN] = fminf(uip.poly_model.AABB[YMIN], g.pos[1]);
			uip.poly_model.AABB[YMAX] = fmaxf(uip.poly_model.AABB[YMAX], g.pos[1]);
			uip.poly_model.AABB[ZMIN] = fminf(uip.poly_model.AABB[ZMIN], g.pos[2]);
			uip.poly_model.AABB[ZMAX] = fmaxf(uip.poly_model.AABB[ZMAX], g.pos[2]);
		}
	}

	fprintf(stdout, "Successfully loaded %zu gaussians.\n", gaussians.size());
	return true;
}

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

int read_OBJ_file(const char* obj_filename)
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

/*bool load_obj_to_composite_object(const char* filename, CompositeObject* c_object) {
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
}*/
//shyun end

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

	case 200: {
		render_gaussian = true;
		const char* file_path = MODEL_PATH;  // obj 경로
#if SCENE_NUM < 1
		if (!read_OBJ_file(file_path)) {
			fprintf(stderr, "Failed to load obj file\n");
			return;
		}
#else
		g_gaussians.clear();
		// 3DGS 학습 결과물을 로드
		if (!loadGaussiansFromPly(file_path, g_gaussians)) {
			fprintf(stderr, "Failed to load ply file\n");
			return;
		}
#endif

		uip.composite_object_read = 1;

		load_poly_model_into_OpenGL();
		g_cuda_rendering_done = false;
		glutPostRedisplay();
		printf("draw DONE\n");
		break;
	}
	case 300:
#if SCENE_NUM < 1
		build_kd_tree_for_composite_object(&uip.poly_model);
		if (uip.poly_model.kd_tree->tri_accel_list == NULL) printf("tri_accel_list NULL\n");
		else {
			printf("triangle num: %d\n", uip.poly_model.n_triangles);
			printf("tri_accel_list size: %d", sizeof(uip.poly_model.kd_tree->tri_accel_list) / sizeof(*(uip.poly_model.kd_tree->tri_accel_list)));
		}
#else
		if (g_gaussians.empty()) {
			printf("Error: No Gaussians loaded. Please load a .ply file first (Menu 800).\n");
			break;
		}
		printf("\n> Constructing Kd-tree from %zu Gaussian particles...\n", g_gaussians.size());

		// 1. 새로 만든 가우시안용 초기화 함수 호출
		if (!initialize_kdtree_for_gaussians(g_gaussians)) {
			fprintf(stderr, "Error: Failed to initialize k-d tree for Gaussians.\n");
			break;
		}

		// 2. 기존의 재귀 빌드 함수 호출 (g_pTriangleInfos에는 이제 가우시안 정보가 들어있음)
		build_kd_tree_recursive(g_bEdge, g_pTriangleInfos, g_iTriangleSize, g_root_AABB, 0, &(g_pKdTree_Node_Array[0]));

		// 3. 결과 출력
		fprintf(stdout, "  - Kd-tree for Gaussians constructed successfully!\n");
		fprintf(stdout, "   * Tree Level: %d\n", g_iKdTree_Level);
		fprintf(stdout, "   * Node Count (All,Leaf,Empty) : %5d, %5d, %5d(%.1f%%)\n",
			g_iKdTree_Node_Count, g_iKdTree_LeafNode_Count, g_iKdTree_EmptyNode_Count,
			100.0f * g_iKdTree_EmptyNode_Count / g_iKdTree_Node_Count);
		fprintf(stdout, "   * Maximum Gaussians in LeafNode: %d\n\n", g_iKdTree_MaxTriInLeafNode_Count);
		fprintf(stdout, "\n> Done!\n\n");

#endif
		break;
	case 400:
		strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
		strcat(full_kd_tree_file_name, "/");
		strcat(full_kd_tree_file_name, uip.kd_tree_filename);

		strcpy(full_i_geometry_file_name, uip.kd_tree_dump_dir);
		strcat(full_i_geometry_file_name, "/");
		strcat(full_i_geometry_file_name, uip.i_geometry_filename);

		if (render_gaussian) {
			dump_kd_tree_for_composite_object(
				&uip.poly_model,
				KDTREE_PATH,         // 저장할 kd-tree
				KD_TREE_DUMP_IN_BINARY,    // 저장 포맷
				IGEOM_PATH         // 저장할 geometry
			);
		}
		else {
			dump_kd_tree_for_composite_object(&uip.poly_model, full_kd_tree_file_name,
				uip.kd_tree_dump_format, full_i_geometry_file_name);
		}
		break;
	case 500:
		strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
		printf("uip.kd_tree_dump_dir:%s\n", uip.kd_tree_dump_dir);
		strcat(full_kd_tree_file_name, "/");
		strcat(full_kd_tree_file_name, uip.kd_tree_filename);
		printf("uip.kd_tree_filename:%s\n", uip.kd_tree_filename);
		if (render_gaussian) {
			strcpy(full_kd_tree_file_name, KDTREE_PATH);
			uip.kd_tree_dump_format = KD_TREE_DUMP_IN_BINARY;
		}
		printf("full_kd_tree_file_name:%s\n", full_kd_tree_file_name);
		read_kd_tree_from_file(&uip.poly_model, full_kd_tree_file_name, uip.kd_tree_dump_format);
		glutPostRedisplay();
		break;
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
		//cudaRenderer.h
		//renderWithCuda(const CompositeObject & object, const Camera & camera, int width, int height, float*& out_framebuffer, bool& is_done)
#if SCENE_NUM < 1
		renderObjWithCuda(
			uip.poly_model,
			camera,
			g_render_width,
			g_render_height,
			g_render_framebuffer,
			g_cuda_rendering_done
		);
#else
		renderGaussiansWithCuda(
			uip.poly_model,
			camera,
			g_render_width,
			g_render_height,
			g_render_framebuffer,
			g_cuda_rendering_done
		);
#endif
		if (g_cuda_rendering_done) {
			printf("CUDA rendering complete. Refreshing display...\n");
			glutPostRedisplay();
		}
		else {
			fprintf(stderr, "CUDA rendering failed.\n");
		}
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
#if SCENE_NUM < 1
	glutAddMenuEntry("5. Read .obj File and Prepair I-Geometry", 200);
#else
	glutAddMenuEntry("5. Read .ply File and Prepair I-Geometry", 200);
#endif
	glutAddMenuEntry("2. Construct Kd-tree from I-Geometry", 300);  
	glutAddMenuEntry("3. Dump Kd-tree and I-Geometry to Files", 400);
	glutAddMenuEntry("4. Read Kd-tree from File", 500);
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
#if SCENE_NUM < 1
		renderObjWithCuda(uip.poly_model, camera, g_render_width, g_render_height, g_render_framebuffer, g_cuda_rendering_done);
#else
		renderGaussiansWithCuda(uip.poly_model, camera, g_render_width, g_render_height, g_render_framebuffer, g_cuda_rendering_done);
#endif

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


