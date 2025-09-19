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
#include <map>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cuda_gl_interop.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "test.h"
#include "cudaRenderer.h"
//#include "SGRTx2Lib/cuda_math.h"
char* ply_file_path;
char* ply_kdtree_path;
char* ply_igeom_path;
char* ply_to_obj;

//cudaEvent_t start_ev, stop_ev;
char* kdtree_build_path;
int submenu[5] = { 101,1012,102,104,105 };

bool render_gaussian = false;
int g_render_width = RENDERING_WIDTH;
int g_render_height = RENDERING_HEIGHT;
bool g_cuda_rendering_done = false;
bool g_cuda_interactive_mode = false; // CUDA 인터랙티브 모드 활성화 플래그
bool g_camera_dirty = true;           // 카메라가 변경되었는지 확인하는 플래그
std::vector<Gaussian> g_gaussians;	  // 전역 변수로 가우시안 데이터를 저장할 벡터

float g_fps = 0.0f; // FPS를 저장할 전역 변수
float total_fps = 0.0f;
int frame_count = 0;
bool measure_fps_interval = false;

GLuint pbo;
struct cudaGraphicsResource* pbo_cuda_resource;
GLuint result_texture_id; // 렌더링 결과를 담을 텍스춰 ID
GLuint quad_vao;          // 화면 전체 사각형 VAO
//GLuint quad_vbo, quad_ebo;

#if USE_STACK > SHORT_STACK
cu_traceState* g_d_global_stack = nullptr;
#endif

int visualize_kdtree_mode = 6;
float3* h_debug_buffer1_main = nullptr;
float3* h_debug_buffer2_main = nullptr;
int max_debug_values[6] = { 0, };
#if LEAF_NODE_DEBUG
ExtendedVertex* original_vertices = nullptr;
int num_original_vertices = 0;
std::vector<LeafNodeInfo> leaf_nodes; // 모든 리프 노드 정보
int selected_leaf_index = -1;       // 현재 선택된 리프 노드 인덱스 (-1은 전체 보기)
ExtendedVertex* leaf_display_vertices = nullptr; // 리프 시각화용 임시 정점 버퍼
BoundingBox original_model_AABB;
int largest_leaf_index = -1; // 가장 큰 리프 노드의 인덱스를 저장
#endif
void setup_interop_resources() {
	// PBO 생성 (기존 코드와 유사)
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, g_render_width * g_render_height * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&pbo_cuda_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	// 렌더링 결과를 담을 텍스춰 생성
	glGenTextures(1, &result_texture_id);
	glBindTexture(GL_TEXTURE_2D, result_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, g_render_width, g_render_height, 0, GL_RGB, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	// 화면 전체를 덮는 사각형 VAO/VBO 생성 (간단한 버전)
	glGenVertexArrays(1, &quad_vao);
}

// FPS를 화면 좌측 상단에 그리는 함수
void draw_fps() {
	glDisable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	// 텍스트 색상 설정
	glColor3f(1.0f, 1.0f, 0.0f); // 노란색

	// 텍스트 위치 설정 (좌측 상단)
	glRasterPos2f(-0.98f, 0.95f);

	char fps_string[32];
	sprintf(fps_string, "FPS: %.2f", g_fps);

	for (char* c = fps_string; *c != '\0'; c++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
	}

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_LIGHTING);
}

void print_current_time(const char* com) {
	auto now = std::chrono::system_clock::now();

	auto in_time_t = std::chrono::system_clock::to_time_t(now);

	std::tm buf;
#ifdef _MSC_VER
	localtime_s(&buf, &in_time_t);
#else
	buf = *std::localtime(&in_time_t);
#endif
	std::cout << com << " time: " << std::put_time(&buf, "%Y-%m-%d %H:%M:%S") << std::endl;
}

bool timerRunning = false;
void timer_callback(int value) {
	if (timerRunning) {
		g_camera_dirty = true;

		glutPostRedisplay();
	}

	glutTimerFunc(1000 / 60, timer_callback, 0);
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
	// kd-tree debug 인자에 따른 hitmap
	if (visualize_kdtree_mode < 6 && h_debug_buffer1_main != nullptr && h_debug_buffer2_main != nullptr) {
		//printf("[DEBUG] Visualizing heatmap. Max node visits value: %d\n", max_debug_values[0]);

		float* color_buffer = new float[g_render_width * g_render_height * 3];
		for (int i = 0; i < g_render_width * g_render_height; i++) {
			float normalized_value = 0.0f;
			if (max_debug_values[visualize_kdtree_mode] > 0) {
				switch (visualize_kdtree_mode) {
					case 0:
						normalized_value = (float)((int)h_debug_buffer1_main[i].x) / max_debug_values[0];
						break;
					case 1:
						normalized_value = (float)((int)h_debug_buffer1_main[i].y) / max_debug_values[1];
						break;
					case 2:
						normalized_value = (float)((int)h_debug_buffer1_main[i].z) / max_debug_values[2];
						break;
					case 3:
						normalized_value = (float)((int)h_debug_buffer2_main[i].x) / max_debug_values[3];
						break;
					case 4:
						normalized_value = (float)((int)h_debug_buffer2_main[i].y) / max_debug_values[4];
						break;
					case 5:
						normalized_value = (float)((int)h_debug_buffer2_main[i].z) / max_debug_values[5];
						break;
				}
				
			}

			// Grayscale: 값이 클수록 밝아짐 (흰색)
			int pixel_idx = (g_render_height - 1 - (i / g_render_width)) * g_render_width + (i % g_render_width); // y좌표 뒤집기
			color_buffer[pixel_idx * 3 + 0] = normalized_value; // R
			color_buffer[pixel_idx * 3 + 1] = normalized_value; // G
			color_buffer[pixel_idx * 3 + 2] = normalized_value; // B
		}

		glClear(GL_COLOR_BUFFER_BIT);
		//glRasterPos2f(-1.0f, -1.0f);
		//glDrawPixels(g_render_width, g_render_height, GL_RGB, GL_FLOAT, color_buffer);
		// 
		// 2D 렌더링을 위해 행렬 상태를 초기화
		glMatrixMode(GL_PROJECTION);
		glPushMatrix(); // 현재 3D Projection 행렬을 스택에 저장
		glLoadIdentity();   // Projection 행렬을 단위 행렬로 초기화

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix(); // 현재 3D ModelView 행렬을 스택에 저장
		glLoadIdentity();   // ModelView 행렬을 단위 행렬로 초기화
		// ----------------------------------------------------------------

		glRasterPos2f(-1.0f, -1.0f);
		glDrawPixels(g_render_width, g_render_height, GL_RGB, GL_FLOAT, color_buffer);

		// 그리기 끝난 후 원래 행렬 상태로 복원
		glMatrixMode(GL_PROJECTION);
		glPopMatrix(); // 저장했던 Projection 행렬 복원

		glMatrixMode(GL_MODELVIEW);
		glPopMatrix(); // 저장했던 ModelView 행렬 복원
		// ----------------------------------------------------

		delete[] color_buffer; // 메모리 해제
		draw_fps(); // FPS
	}
	// CUDA 렌더링 완료 => 프레임버퍼를 화면에 그림
	else if (g_cuda_interactive_mode || g_cuda_rendering_done) {
		//glClear(GL_COLOR_BUFFER_BIT);
		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);

		glMatrixMode(GL_PROJECTION); // 2D 렌더링을 위해 Projection 행렬을 초기화
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

		/*/ glDrawPixels는 좌하단이 기준이므로 y좌표를 뒤집을 필요가 없음
		glRasterPos2f(-1.0f, -1.0f);
		//glDrawPixels(g_render_width, g_render_height, GL_RGB, GL_FLOAT, g_render_framebuffer);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glDrawPixels(g_render_width, g_render_height, GL_RGB, GL_FLOAT, (GLvoid*)0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glEnable(GL_DEPTH_TEST);*/
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, result_texture_id);

		// 화면 전체에 사각형 그리기
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
		glEnd();

		glDisable(GL_TEXTURE_2D);

		draw_fps(); // FPS
	}
	else {

		int i;
		ExtendedVertex* ptr_ev;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(-(uip.poly_model.AABB[XMIN] + uip.poly_model.AABB[XMAX]) / 2.0,
			-(uip.poly_model.AABB[YMIN] + uip.poly_model.AABB[YMAX]) / 2.0,
			-(uip.poly_model.AABB[ZMIN] + uip.poly_model.AABB[ZMAX]) / 2.0);

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
	}

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
		case 't':
			timerRunning = !timerRunning;
			break;
		case 'f':
			measure_fps_interval = !measure_fps_interval;
			timerRunning = true;
			total_fps = 0.0f;
			frame_count = 0;
			break;
#if LEAF_NODE_DEBUG
		case 'm':
			if (largest_leaf_index == -1) {
				printf("Largest leaf index not found. Please extract leaf data first.\n");
				break;
			}
			selected_leaf_index = largest_leaf_index;
			set_kd_tree_leaf_node();
			g_camera_dirty = true;
			glutPostRedisplay();
			break;
		case'i':
			if (leaf_nodes.empty()) {
				printf("Leaf node data is not extracted yet.\n");
				break;
			}
			printf("input node index: ");
			fscanf(stdin, "%d", &selected_leaf_index);
			set_kd_tree_leaf_node();
			g_camera_dirty = true; // 뷰가 변경되었으므로 다시 그리도록 플래그 설정
			glutPostRedisplay();
			break;
		case 'j':
			if (leaf_nodes.empty()) {
				printf("Leaf node data is not extracted yet.\n");
				break;
			}
			selected_leaf_index--;
			set_kd_tree_leaf_node();
			g_camera_dirty = true; // 뷰가 변경되었으므로 다시 그리도록 플래그 설정
			glutPostRedisplay();
			break;
		case 'k':
			if (leaf_nodes.empty()) {
				printf("Leaf node data is not extracted yet.\n");
				break;
			}
			selected_leaf_index = -1;
			set_kd_tree_leaf_node();
			g_camera_dirty = true; // 뷰가 변경되었으므로 다시 그리도록 플래그 설정
			glutPostRedisplay();
			break;
		case 'l':
			if (leaf_nodes.empty()) {
				printf("Leaf node data is not extracted yet.\n");
				break;
			}
			selected_leaf_index++;
			set_kd_tree_leaf_node();
			g_camera_dirty = true; // 뷰가 변경되었으므로 다시 그리도록 플래그 설정
			glutPostRedisplay();
			break;
#endif
#if HIT_AND_NODE_COUNT_DEBUG
		case 'v':
			visualize_kdtree_mode = (visualize_kdtree_mode + 1) % 7;
			g_camera_dirty = true;
			printf("Kd-tree visualization mode: ");
			switch (visualize_kdtree_mode) {
				case 0:
					printf("max_node_visits\n");
					break;
				case 1:
					printf("max_leaf_visits\n");
					break;
				case 2:
					printf("max_intersection_tests\n");
					break;
				case 3:
					printf("max_hits_found\n");
					break;
				case 4:
					printf("max_blend_ops\n");
					break;
				case 5:
					printf("max_max_sort_size\n");
					break;
				case 6:
					printf("OFF\n");
					break;
			}
			glutPostRedisplay();
			break;
#endif
		case 'q':
			exit(0);
			break;
	}
}

void reshape(int width, int height) {
	//g_render_width = width;   // 전역 변수 업데이트
	//g_render_height = height; // 전역 변수 업데이트

	glViewport(0, 0, width, height);

	camera.aspect = (double) width/ height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(camera.fovy, camera.aspect, camera.near_c, camera.far_c);

	if (g_cuda_interactive_mode) {
		g_camera_dirty = true;
	}
}


void mousepress(int button, int state, int x, int y) {
	if ((button == GLUT_LEFT_BUTTON) && (state == GLUT_DOWN)) {
		if (glutGetModifiers() == GLUT_ACTIVE_SHIFT) {
			uip.camera_zoom_mode = 1;
		}
		else/* if  (glutGetModifiers() == GLUT_ACTIVE_CTRL)*/ {
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
			//fprintf(stdout, "camera.pos: %f %f %f\n", camera.pos[0], camera.pos[1], camera.pos[2]);

			glutPostRedisplay();
		}
	}
}

void init_OpenGL_RC(void) {
	// glewInit();

	initialize_camera(&camera);
	//glClearColor(0.2, 0.2, 0.2, 1.0);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0);

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
	cleanupCudaResources();
	glutDestroyWindow(uip.main_window_ID);
#if HIT_AND_NODE_COUNT_DEBUG
	if (h_debug_buffer1_main != nullptr) {
		delete[] h_debug_buffer1_main;
		h_debug_buffer1_main = nullptr;
	}
	if (h_debug_buffer2_main != nullptr) {
		delete[] h_debug_buffer2_main;
		h_debug_buffer2_main = nullptr;
	}
#endif
#if LEAF_NODE_DEBUG
	if (original_vertices != nullptr) {
		delete[] original_vertices;
		original_vertices = nullptr;
	}
	if (leaf_display_vertices != nullptr) {
		delete[] leaf_display_vertices;
		leaf_display_vertices = nullptr;
	}
#endif
#if USE_STACK > SHORT_STACK
	if (g_d_global_stack) cudaFree(g_d_global_stack);
	g_d_global_stack = nullptr;
#endif
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
#if LEAF_NODE_DEBUG
void set_kd_tree_leaf_node() {
	if (selected_leaf_index >= (int)leaf_nodes.size()) {
		selected_leaf_index = -1; // -1은 전체 보기로 돌아감을 의미
	}
	if (selected_leaf_index < -1) {
		selected_leaf_index = (int)leaf_nodes.size() - 1;
	}

	//  이전에 사용한 임시 버퍼가 있다면 메모리를 해제
	if (leaf_display_vertices != nullptr) {
		delete[] leaf_display_vertices;
		leaf_display_vertices = nullptr;
	}

	//  '전체 보기' 모드 처리
	if (selected_leaf_index == -1) {
		printf("Displaying all %d triangles.\n", num_original_vertices / 3);
		uip.poly_model.extended_vertices = original_vertices;
		uip.poly_model.n_triangles = num_original_vertices / 3;
		uip.poly_model.AABB[XMIN] = original_model_AABB.min[0];
		uip.poly_model.AABB[YMIN] = original_model_AABB.min[1];
		uip.poly_model.AABB[ZMIN] = original_model_AABB.min[2];
		uip.poly_model.AABB[XMAX] = original_model_AABB.max[0];
		uip.poly_model.AABB[YMAX] = original_model_AABB.max[1];
		uip.poly_model.AABB[ZMAX] = original_model_AABB.max[2];
	}
	//  '단일 리프 노드 보기' 모드 처리
	else {
		const auto& selected_leaf = leaf_nodes[selected_leaf_index];
		const auto& indices = selected_leaf.triangle_indices;

		printf("Displaying leaf %d / %zu (%zu triangles)\n",
			selected_leaf_index, leaf_nodes.size(), indices.size());

		uip.poly_model.AABB[XMIN] = selected_leaf.aabb.min[0];
		uip.poly_model.AABB[YMIN] = selected_leaf.aabb.min[1];
		uip.poly_model.AABB[ZMIN] = selected_leaf.aabb.min[2];
		uip.poly_model.AABB[XMAX] = selected_leaf.aabb.max[0];
		uip.poly_model.AABB[YMAX] = selected_leaf.aabb.max[1];
		uip.poly_model.AABB[ZMAX] = selected_leaf.aabb.max[2];

		if (!indices.empty()) {
			// 선택된 리프의 삼각형들을 담을 임시 버퍼를 새로 할당
			int num_leaf_vertices = indices.size() * 3;
			leaf_display_vertices = new ExtendedVertex[num_leaf_vertices];

			// 원본 정점 데이터에서 해당 삼각형들만 임시 버퍼로 복사
			for (size_t i = 0; i < indices.size(); ++i) {
				unsigned int tri_idx = indices[i];
				// tri_idx번째 삼각형(정점 3개)을 통째로 복사
				memcpy(&leaf_display_vertices[i * 3], &original_vertices[tri_idx * 3], sizeof(ExtendedVertex) * 3);
			}

			// display() 함수가 임시 버퍼를 그리도록 포인터를 교체
			uip.poly_model.extended_vertices = leaf_display_vertices;
			uip.poly_model.n_triangles = indices.size();
		}
		else {
			// 빈 리프 노드일 경우, 그릴 삼각형이 없음을 명시
			uip.poly_model.n_triangles = 0;
		}
	}

	printf("AABB: X [%f, %f] Y [%f, %f] Z [%f, %f]\n",
		uip.poly_model.AABB[XMIN], uip.poly_model.AABB[XMAX],
		uip.poly_model.AABB[YMIN], uip.poly_model.AABB[YMAX],
		uip.poly_model.AABB[ZMIN], uip.poly_model.AABB[ZMAX]);
}
#endif

void printKdTreeLeafNodeInfo() {
	if (!uip.poly_model.kd_tree) {
		printf("Kd-tree is not available.\n");
		return;
	}

	printf("\n--- Analyzing triangles per leaf node ---\n");

	// 모든 리프 노드의 삼각형 개수 수집
	std::vector<unsigned int> triangle_counts;
	collectTriangleCounts_recursive(uip.poly_model.kd_tree, 0, triangle_counts);

	if (triangle_counts.empty()) {
		printf(" -> No leaf nodes found in the tree.\n");
		printf("-------------------------------------------\n\n");
		return;
	}

	// 기본 통계 계산
	const unsigned int leaf_count = triangle_counts.size();
	const unsigned long long total_triangles = std::accumulate(triangle_counts.begin(), triangle_counts.end(), 0ULL);
	const auto minmax = std::minmax_element(triangle_counts.begin(), triangle_counts.end());
	const unsigned int min_val = *minmax.first;
	const unsigned int max_val = *minmax.second;
	const float avg_triangles = (float)total_triangles / leaf_count;

	printf(" -> Total Leaf Nodes Found: %u\n", leaf_count);
	printf(" -> Max triangles in a leaf: %u\n", max_val);
	printf(" -> Min triangles in a leaf: %u\n", min_val);
	printf(" -> Avg triangles per leaf: %.2f\n", avg_triangles);
	printf("\n--- Histogram of Triangles per Leaf ---\n");

	// 히스토그램 생성
	const int num_bins = 20; // 히스토그램 막대 개수 (조정 가능)
	std::vector<unsigned int> bins(num_bins, 0);

	// 최소값과 최대값이 같을 경우 bin_size가 0이 되는 것을 방지
	const float range = static_cast<float>(max_val - min_val);
	const float bin_size = (range > 0) ? (range / num_bins) : 1.0f;

	for (unsigned int count : triangle_counts) {
		int bin_index = (range > 0) ? static_cast<int>((count - min_val) / bin_size) : 0;
		// 마지막 bin에 최대값을 포함시키기 위한 처리
		if (bin_index >= num_bins) bin_index = num_bins - 1;
		bins[bin_index]++;
	}

	// 히스토그램 출력
	const unsigned int max_bin_count = *std::max_element(bins.begin(), bins.end());
	const int max_bar_width = 50; // 히스토그램 막대의 최대 너비 (조정 가능)

	for (int i = 0; i < num_bins; ++i) {
		// 각 bin의 값 범위를 계산
		unsigned int bin_start = min_val + static_cast<unsigned int>(i * bin_size);
		unsigned int bin_end = min_val + static_cast<unsigned int>((i + 1) * bin_size) - 1;
		if (i == num_bins - 1) bin_end = max_val;


		printf(" [%5u - %5u] | %-7u | ", bin_start, bin_end, bins[i]);

		int bar_width = 0;
		if (max_bin_count > 0) {
			bar_width = static_cast<int>(((float)bins[i] / max_bin_count) * max_bar_width);
		}

		for (int j = 0; j < bar_width; ++j) {
			printf("*");
		}
		printf("\n");
	}
	printf("-------------------------------------------\n\n");
}

inline void fMyVecNormalize4D(float v[4]) {
	float len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
	if (len_sq > 0.00001f) { // 0으로 나누는 것을 방지
		float len_inv = 1.0f / sqrtf(len_sq);
		v[0] *= len_inv;
		v[1] *= len_inv;
		v[2] *= len_inv;
		v[3] *= len_inv;
	}
}

void transform_vector_by_matrix_transpose(const float v[3], const float3x3& rot, float result[3]) {
	result[0] = v[0] * rot.m[0][0] + v[1] * rot.m[1][0] + v[2] * rot.m[2][0];
	result[1] = v[0] * rot.m[0][1] + v[1] * rot.m[1][1] + v[2] * rot.m[2][1];
	result[2] = v[0] * rot.m[0][2] + v[1] * rot.m[1][2] + v[2] * rot.m[2][2];
}

// 쿼터니언(w,x,y,z)을 전치된 회전 행렬로 변환
void quaternionToMatrixTranspose(const float q[4], float3x3& rot) {
	float w = q[0], x = q[1], y = q[2], z = q[3];
	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, xz = x * z, yz = y * z;
	float wx = w * x, wy = w * y, wz = w * z;

	rot.m[0][0] = 1.0f - 2.0f * (yy + zz); rot.m[1][0] = 2.0f * (xy - wz);		  rot.m[2][0] = 2.0f * (xz + wy);
	rot.m[0][1] = 2.0f * (xy + wz);		   rot.m[1][1] = 1.0f - 2.0f * (xx + zz); rot.m[2][1] = 2.0f * (yz - wx);
	rot.m[0][2] = 2.0f * (xz - wy);		   rot.m[1][2] = 2.0f * (yz + wx);		  rot.m[2][2] = 1.0f - 2.0f * (xx + yy);
}

void rotate_vector_by_quaternion(float v_out[3], float v[3], const float q[4]) {
	// v_out = v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v)
	// t = 2 * cross(q.xyz, v); v' = v + q.w * t + cross(q.xyz, t);

	float qv[3], qqv[3];
	float q_vec[3] = { q[1], q[2], q[3] };

	// u = 2.0f * (q_vec X v)
	fMyVecCrossProduct(q_vec, v, qv);
	for (int i = 0; i < 3; ++i) qv[i] *= 2.0f;

	// v_out = v + q[0] * u + (q_vec X u)
	fMyVecCrossProduct(q_vec, qv, qqv);
	for (int i = 0; i < 3; ++i) {
		v_out[i] = v[i] + q[0] * qv[i] + qqv[i];
	}

	//float q_vec[3] = { q[1], q[2], q[3] };
	//float VcR[3];
	//fMyVecCrossProduct(q_vec, v, VcR);

	//for (int i = 0; i < 3; ++i) {
	//	v_out[i] = (q[0] * q[0] - fMyVecDotProduct(q_vec, q_vec)) * v[i]
	//		+ 2.0f * q_vec[i] * fMyVecDotProduct(q_vec, v)
	//		+ 2.0f * q[0] * VcR[i];
	//}
}


bool loadGaussiansFromPly(const char* filename, std::vector<Gaussian>& gaussians) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "Error: Cannot open PLY file %s\n", filename);
		return false;
	}

	// --- 헤더 파싱 ---
	std::string line;
	long num_vertices = 0;

	// 프로퍼티의 순서와 오프셋을 저장할 맵
	std::map<std::string, int> property_offsets;
	int current_offset = 0;

	std::vector<std::string> properties_order;

	while (std::getline(file, line) && line != "end_header") {
		std::stringstream ss(line);
		std::string token;
		ss >> token;
		if (token == "element" && (ss >> token, token == "vertex")) {
			ss >> num_vertices;
		}
		else if (token == "property") {
			std::string type, name;
			ss >> type >> name;
			properties_order.push_back(name); // 실제 파일에 기록된 순서 저장

			int type_size = 0;
			if (type == "float") type_size = 4;
			else if (type == "double") type_size = 8;
			else if (type == "uchar") type_size = 1;
			// ... 다른 타입 추가 가능 ...

			property_offsets[name] = current_offset;
			current_offset += type_size;
		}
	}

	if (num_vertices == 0) return false;
	const int vertex_byte_size = current_offset; // 한 정점 데이터의 총 크기

	gaussians.clear();
	gaussians.reserve(num_vertices);

	// ---  헤더 정보에 기반한 바이너리 데이터 읽기 ---
	std::vector<char> buffer(vertex_byte_size); // 한 정점 크기의 버퍼 생성

	for (long i = 0; i < num_vertices; ++i) {
		file.read(buffer.data(), vertex_byte_size);
		if (!file) {
			fprintf(stderr, "Error reading vertex data at index %ld\n", i);
			return false;
		}

		Gaussian g;

		// 오프셋 맵을 사용하여 버퍼에서 직접 데이터 추출
		// 헤더에 명시된 순서와 관계없이 이름으로 정확한 위치를 찾아감
		memcpy(g.pos, &buffer[property_offsets["x"]], sizeof(float) * 3);
		memcpy(g.f_dc, &buffer[property_offsets["f_dc_0"]], sizeof(float) * 3);

		// f_rest는 f_rest_0부터 f_rest_44까지 순차적으로 복사
		if (property_offsets.count("f_rest_0")) {
			memcpy(g.f_rest, &buffer[property_offsets["f_rest_0"]], sizeof(float) * 45);
		}

		memcpy(&g.opacity, &buffer[property_offsets["opacity"]], sizeof(float));
		g.opacity = 1.0f / (1.0f + expf(-g.opacity));

		memcpy(g.scale, &buffer[property_offsets["scale_0"]], sizeof(float) * 3);
		g.scale[0] = expf(g.scale[0]);
		g.scale[1] = expf(g.scale[1]);
		g.scale[2] = expf(g.scale[2]);

#if QUATERNION
		memcpy(g.rot, &buffer[property_offsets["rot_0"]], sizeof(float) * 4);
		fMyVecNormalize4D(g.rot);
#else
		float quat[4];
		memcpy(quat, &buffer[property_offsets["rot_0"]], sizeof(float) * 4);
		fMyVecNormalize4D(quat);
		quaternionToMatrixTranspose(quat, g.rot_matrix);
#endif

		gaussians.push_back(g);
	}

	fprintf(stderr, "Robustly loaded %zu gaussians based on PLY header.\n", gaussians.size());
	return true;
}

float kernelScale_final(float density, float minResponse, float kernel_degree) {
	const float responseModulation = (1 & (1 << 0)) != 0 ? density : 1.0f;
	const float min_response = fminf(minResponse / responseModulation, 0.97f);

	// kernelDegree < 0 (Bump Kernel)
	if (kernel_degree < 0) {
		const float k = std::fabs(kernel_degree);
		const float s = 1.0 / std::pow(3.0, k);
		const float ks = std::pow((1.f / (std::log(min_response) - 1.f) + 1.f) / s, 1.f / k);
		return ks;
	}

	// kernelDegree == 0 (Linear Kernel)
	if (kernel_degree == 0) {
		return ((1.0f - min_response) / 3.0f) / -0.329630334487f;
	}

	// kernelDegree > 0 (Generalized Gaussian Kernel)
	const float b = kernel_degree;
	const float a = -4.5f / std::pow(3.0f, b);

	return std::pow(std::log(min_response) / a, 1.0f / b);
}

void create_composite_object_from_gaussians(
	const std::vector<Gaussian>& gaussians,
	float alpha_min = ALPHA_MIN,
	float kernel_degree = KERNEL_DEGREE
) {
	if (gaussians.empty()) {
		printf("Gaussian list is empty. Nothing to create.\n");
		return;
	}

	if (uip.poly_model.extended_vertices != nullptr) {
		free(uip.poly_model.extended_vertices);
		uip.poly_model.extended_vertices = nullptr;
	}

#if DEBUG_SIGMA_HISTOGRAM
	int sigma_histogram[DEBUG_SIGMA_HISTOGRAM] = { 0 };
#endif

	// 메모리 할당
	long num_gaussians = gaussians.size();
	long num_total_triangles = num_gaussians * icosaHedronNumTri;
	long num_total_vertices = num_total_triangles * 3;
	num_total_triangles = 0;

	uip.poly_model.n_triangles = 0; // 시작은 0
	uip.poly_model.extended_vertices = (ExtendedVertex*)malloc(num_total_vertices * sizeof(ExtendedVertex));
	if (uip.poly_model.extended_vertices == NULL) {
		fprintf(stderr, "Fatal Error: Memory allocation failed for %ld vertices!\n", num_total_vertices);
		exit(1);
	}
	ExtendedVertex* current_vertex_ptr = uip.poly_model.extended_vertices;

	// AABB 초기화
	uip.poly_model.AABB[XMIN] = uip.poly_model.AABB[YMIN] = uip.poly_model.AABB[ZMIN] = FLT_MAX;
	uip.poly_model.AABB[XMAX] = uip.poly_model.AABB[YMAX] = uip.poly_model.AABB[ZMAX] = -FLT_MAX;

	//const float ICOSA_VRT_SCALE = 0.5f * icosaEdge;
	int cnt_sigma = 0;
	int cnt_scale = 0;
	float k_iso_max = 0;

	// 모든 가우시안에 대해 20면체 생성
	for (long i = 0; i < num_gaussians; ++i) {
		const Gaussian& g = gaussians[i];

		//sigma(density) 계산
		const float sigma = g.opacity;
		//const float sigma = 1.0f / (1.0f + expf(-g.opacity));

#if DEBUG_SIGMA_HISTOGRAM
		int bin_index = static_cast<int>(sigma * DEBUG_SIGMA_HISTOGRAM);
		if (bin_index >= DEBUG_SIGMA_HISTOGRAM) { // sigma가 1.0일 경우를 대비한 안전장치
			bin_index = DEBUG_SIGMA_HISTOGRAM - 1;
		}
		sigma_histogram[bin_index]++;
#endif
		if (sigma < SIGMA_THRESHOLD) { cnt_sigma++; continue; }
		float k_iso = 0.0f;

#if USE_KERNEL_SCALE
		//if (sigma / alpha_min > 1.0f)
		// kernelScale_final 함수를 호출하여 k_iso 계산
		k_iso = kernelScale_final(sigma, alpha_min, kernel_degree);
		//printf("%d, k_iso: %f\n", i, k_iso);
		//k_iso = fminf(k_iso, 3.0f);

		float final_scale[3] = {
			g.scale[0] * k_iso * 0.5f * icosaEdge,
			g.scale[1] * k_iso * 0.5f * icosaEdge,
			g.scale[2] * k_iso * 0.5f * icosaEdge
			//g.scale[0] * k_iso * unitspherefactor,
			//g.scale[1] * k_iso * unitspherefactor,
			//g.scale[2] * k_iso * unitspherefactor
		};
#else
		if (sigma / alpha_min > 1.0f) {
			k_iso = sqrtf(2.0f * logf(sigma / alpha_min));
		}
		float final_scale[3] = {
			g.scale[0] * k_iso * unitspherefactor,
			g.scale[1] * k_iso * unitspherefactor,
			g.scale[2] * k_iso * unitspherefactor
		};
#endif
		k_iso_max = fmaxf(k_iso_max, k_iso);

		//if (final_scale[0] < 1e-6f && final_scale[1] < 1e-6f && final_scale[2] < 1e-6f) { cnt_scale++; continue; }

		num_total_triangles += icosaHedronNumTri;
		// 아이코사헤드론의 20개 면(삼각형)을 생성
		for (int j = 0; j < icosaHedronNumTri; ++j) {
			const int* face_indices = ICO_FACES[j];

			// 3개의 정점을 변환하여 저장
			for (int l = 0; l < 3; ++l) {
				// 원본 단위 아이코사헤드론 정점
				const float* v_cano = ICO_VERTICES[face_indices[l]];

				// 스케일, 회전, 이동 변환 적용
				float v_scaled[3], v_rotated[3], v_final[3];

				// 스케일 적용 (비등방성 S * 등방성 k)
				v_scaled[0] = v_cano[0] * final_scale[0];
				v_scaled[1] = v_cano[1] * final_scale[1];
				v_scaled[2] = v_cano[2] * final_scale[2];
				//printf("%f, %f, %f\n", expf(g.scale[0]), expf(g.scale[1]), expf(g.scale[2]));

				// 회전 적용
#if QUATERNION
				rotate_vector_by_quaternion(v_rotated, v_scaled, g.rot);
#else
				transform_vector_by_matrix_transpose(v_scaled, g.rot_matrix, v_rotated);
#endif

				// 위치(Translate) 적용
				v_final[0] = v_rotated[0] + g.pos[0];
				v_final[1] = v_rotated[1] + g.pos[1];
				v_final[2] = v_rotated[2] + g.pos[2];
				//printf("%f, %f, %f\n", g.pos[0], g.pos[1], g.pos[2]);

				// ExtendedVertex 데이터 채우기
				memcpy(current_vertex_ptr->vertex, v_final, sizeof(float) * 3);
				current_vertex_ptr->material_ID = i; // 가우시안 인덱스를 저장
				// 노멀은 일단 0으로 초기화 (필요 시 계산 가능)
				//memset(current_vertex_ptr->normal, 0, sizeof(float) * 3);

				// AABB 업데이트
				uip.poly_model.AABB[XMIN] = fminf(uip.poly_model.AABB[XMIN], v_final[0]);
				uip.poly_model.AABB[XMAX] = fmaxf(uip.poly_model.AABB[XMAX], v_final[0]);
				uip.poly_model.AABB[YMIN] = fminf(uip.poly_model.AABB[YMIN], v_final[1]);
				uip.poly_model.AABB[YMAX] = fmaxf(uip.poly_model.AABB[YMAX], v_final[1]);
				uip.poly_model.AABB[ZMIN] = fminf(uip.poly_model.AABB[ZMIN], v_final[2]);
				uip.poly_model.AABB[ZMAX] = fmaxf(uip.poly_model.AABB[ZMAX], v_final[2]);

				current_vertex_ptr++;
			}
		}
	}
	long num_final_vertices = num_total_triangles * 3;

#if DEBUG_SIGMA_HISTOGRAM
	printf("\n--- Sigma (Opacity) Histogram ---\n");
	int max_count = 0;
	for (int i = 0; i < DEBUG_SIGMA_HISTOGRAM; ++i) {
		if (sigma_histogram[i] > max_count) {
			max_count = sigma_histogram[i];
		}
	}

	const int MAX_BAR_WIDTH = 50; // 막대그래프의 최대 너비
	for (int i = 0; i < DEBUG_SIGMA_HISTOGRAM; ++i) {
		float min_range = (float)i / DEBUG_SIGMA_HISTOGRAM;
		float max_range = (float)(i + 1) / DEBUG_SIGMA_HISTOGRAM;
		int bar_width = 0;
		if (max_count > 0) {
			bar_width = static_cast<int>((float)sigma_histogram[i] / max_count * MAX_BAR_WIDTH);
		}
		printf("Bin %3d [%.2f-%.2f): %-7d |", i, min_range, max_range, sigma_histogram[i]);
		for (int j = 0; j < bar_width; ++j) {
			printf("#");
		}
		printf("\n");
	}
	printf("---------------------------------\n\n");
#endif

	printf("k_iso_max: %f\n", k_iso_max);
	printf("delete by SIGMA cnt: %d\n", cnt_sigma);
	printf("delete by SCALE cnt: %d\n", cnt_scale);
	printf("\n");
	printf("vtx_cnt_temp: %d\n", num_total_vertices);
	printf("vtx_cnt_real: %d\n", num_final_vertices);
	uip.poly_model.extended_vertices = (ExtendedVertex*)realloc(uip.poly_model.extended_vertices, num_final_vertices * sizeof(ExtendedVertex));

	uip.poly_model.n_triangles = num_total_triangles;
	uip.composite_object_read = 1;
	//printf("\nSuccessfully created CompositeObject with %d triangles from %ld Gaussians.\n\n", uip.poly_model.n_triangles, num_gaussians);
	printf("\nSuccessfully created CompositeObject with %d triangles from %ld Gaussians.\n\n", uip.poly_model.n_triangles, num_gaussians - (cnt_sigma + cnt_scale));
}

// 축-각도 표현을 쿼터니언으로 변환
void fMyQuatFromAngleAxis(float q[4], float angle_rad, const float axis[3]) {
	float s = sinf(angle_rad / 2.0f);
	q[0] = cosf(angle_rad / 2.0f); // w
	q[1] = axis[0] * s;             // x
	q[2] = axis[1] * s;             // y
	q[3] = axis[2] * s;             // z
}

// 두 쿼터니언을 곱 (result = q1 * q2)
void fMyQuatMul(float result[4], const float q1[4], const float q2[4]) {
	result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]; // w
	result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]; // x
	result[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]; // y
	result[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]; // z
}

void matrix_multiply(float3x3& C, const float3x3& A, const float3x3& B) {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			C.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j];
		}
	}
}

void fMyQuatInv(float* q) {
	q[1] *= -1;
	q[2] *= -1;
	q[3] *= -1;
}

// CompositeObject의 모든 정점을 축 기준으로 회전시키는 함수
void rotate_composite_object(std::vector<Gaussian>& gaussians, float angle_degrees, float axis_x, float axis_y, float axis_z) {
	// 회전축 벡터 정규화
	float axis_vec[3] = { axis_x, axis_y, axis_z };
	fMyVecNormalize(axis_vec);
	float ux = axis_vec[0];
	float uy = axis_vec[1];
	float uz = axis_vec[2];

	// 회전 행렬 계산을 위한 값들 준비
	float angle_rad = angle_degrees * M_PI / 180.0f;
	float cos_theta = cosf(angle_rad);
	float sin_theta = sinf(angle_rad);
	float one_minus_cos = 1.0f - cos_theta;

#if QUATERNION
	// 가우시안 방향(rot) 회전을 위한 쿼터니언 생성
	float rot_quat[4];
	fMyQuatFromAngleAxis(rot_quat, angle_rad, axis_vec);
#else
	// 임의 축 회전 행렬 (Row-major)
	float R[3][3];
	R[0][0] = cos_theta + ux * ux * one_minus_cos;
	R[0][1] = ux * uy * one_minus_cos - uz * sin_theta;
	R[0][2] = ux * uz * one_minus_cos + uy * sin_theta;

	R[1][0] = uy * ux * one_minus_cos + uz * sin_theta;
	R[1][1] = cos_theta + uy * uy * one_minus_cos;
	R[1][2] = uy * uz * one_minus_cos - ux * sin_theta;

	R[2][0] = uz * ux * one_minus_cos - uy * sin_theta;
	R[2][1] = uz * uy * one_minus_cos + ux * sin_theta;
	R[2][2] = cos_theta + uz * uz * one_minus_cos;
	float3x3 R_T;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			R_T.m[i][j] = R[j][i];
		}
	}
#endif
	// --- 원본 가우시안 데이터 회전 ---
	for (size_t i = 0; i < gaussians.size(); ++i) {
#if QUATERNION
		// 가우시안 위치 회전
		rotate_vector_by_quaternion(gaussians[i].pos, gaussians[i].pos, rot_quat);

		// 가우시안 방향(쿼터니언) 회전
		float current_rot[4];
		memcpy(current_rot, gaussians[i].rot, sizeof(float) * 4);

		float new_rot[4];
		fMyQuatMul(new_rot, rot_quat, current_rot);
		memcpy(gaussians[i].rot, new_rot, sizeof(float) * 4);
		fMyVecNormalize4D(gaussians[i].rot);
#else
		// 가우시안 위치 회전
		float* pos = gaussians[i].pos;
		float ox = pos[0], oy = pos[1], oz = pos[2];
		pos[0] = ox * R[0][0] + oy * R[0][1] + oz * R[0][2];
		pos[1] = ox * R[1][0] + oy * R[1][1] + oz * R[1][2];
		pos[2] = ox * R[2][0] + oy * R[2][1] + oz * R[2][2];

		float3x3 old_matrix = gaussians[i].rot_matrix;
		matrix_multiply(gaussians[i].rot_matrix, old_matrix, R_T);
#endif
	}

	// 모든 정점을 순회하며 회전 변환 적용
	int total_vertices = uip.poly_model.n_triangles * 3;
	for (int i = 0; i < total_vertices; ++i) {
		float* v = uip.poly_model.extended_vertices[i].vertex;
#if QUATERNION
		rotate_vector_by_quaternion(v, v, rot_quat);
#else
		float ox = v[0], oy = v[1], oz = v[2]; // 원본 좌표

		v[0] = ox * R[0][0] + oy * R[0][1] + oz * R[0][2];
		v[1] = ox * R[1][0] + oy * R[1][1] + oz * R[1][2];
		v[2] = ox * R[2][0] + oy * R[2][1] + oz * R[2][2];
#endif
	}

	// AABB 다시 계산
	uip.poly_model.AABB[XMIN] = uip.poly_model.AABB[YMIN] = uip.poly_model.AABB[ZMIN] = FLT_MAX;
	uip.poly_model.AABB[XMAX] = uip.poly_model.AABB[YMAX] = uip.poly_model.AABB[ZMAX] = -FLT_MAX;
	for (int i = 0; i < total_vertices; ++i) {
		float* v = uip.poly_model.extended_vertices[i].vertex;
		uip.poly_model.AABB[XMIN] = fminf(uip.poly_model.AABB[XMIN], v[0]);
		uip.poly_model.AABB[XMAX] = fmaxf(uip.poly_model.AABB[XMAX], v[0]);
		uip.poly_model.AABB[YMIN] = fminf(uip.poly_model.AABB[YMIN], v[1]);
		uip.poly_model.AABB[YMAX] = fmaxf(uip.poly_model.AABB[YMAX], v[1]);
		uip.poly_model.AABB[ZMIN] = fminf(uip.poly_model.AABB[ZMIN], v[2]);
		uip.poly_model.AABB[ZMAX] = fmaxf(uip.poly_model.AABB[ZMAX], v[2]);
	}

	printf("CompositeObject rotated by %.1f degrees around axis (%.2f, %.2f, %.2f).\n",
		angle_degrees, ux, uy, uz);
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

	free(mesh_geom.vertices);
	free(mesh_geom.faces);
	printf("AABB: X [%f, %f] Y [%f, %f] Z [%f, %f]\n",
		uip.poly_model.AABB[XMIN], uip.poly_model.AABB[XMAX],
		uip.poly_model.AABB[YMIN], uip.poly_model.AABB[YMAX],
		uip.poly_model.AABB[ZMIN], uip.poly_model.AABB[ZMAX]);

	return 1;
}

bool save_composite_object_to_obj(const CompositeObject& object, const char* filename) {
	// 파일 스트림 열기
	std::ofstream outFile(filename);
	if (!outFile.is_open()) {
		fprintf(stderr, "Error: Cannot open file for writing: %s\n", filename);
		return false;
	}

	// 파일 헤더 주석 작성
	outFile << "# OBJ file generated from a CompositeObject structure\n";
	outFile << "# Total Triangles: " << object.n_triangles << "\n";
	const int total_vertices = object.n_triangles * 3;
	outFile << "# Total Vertices in Array: " << total_vertices << "\n\n";

	// 정점(vertex) 및 법선(vertex normal) 데이터 작성
	for (int i = 0; i < total_vertices; ++i) {
		const ExtendedVertex& v = object.extended_vertices[i];

		// 정점 좌표 (v x y z)
		outFile << "v " << v.vertex[0] << " " << v.vertex[1] << " " << v.vertex[2] << "\n";

		// 정점 법선 (vn x y z)
		//outFile << "vn " << v.normal[0] << " " << v.normal[1] << " " << v.normal[2] << "\n";
	}

	outFile << "\n"; // 데이터 섹션 구분을 위한 공백 라인

	// 면(face) 데이터 작성
	// OBJ 파일의 인덱스는 1부터 시작하므로, C++ 배열 인덱스에 1을 더해줘야 
	for (int i = 0; i < object.n_triangles; ++i) {
		// 현재 삼각형을 구성하는 세 정점의 시작 인덱스
		const int v1_idx = 3 * i + 1;
		const int v2_idx = 3 * i + 2;
		const int v3_idx = 3 * i + 3;

		// 면 정보 (f v1//vn1 v2//vn2 v3//vn3)
		// 각 정점과 법선이 1:1로 매칭되므로, 정점 인덱스와 법선 인덱스는 동일
		//outFile << "f " << v1_idx << "//" << v1_idx << " "
		//	<< v2_idx << "//" << v2_idx << " "
		//	<< v3_idx << "//" << v3_idx << "\n";
		outFile << "f " << v1_idx << " " << v2_idx << " " << v3_idx << "\n";
	}

	// 파일 닫기 및 완료 메시지
	outFile.close();
	printf("Successfully saved CompositeObject to %s\n", filename);

	return true;
}
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
	 		//memcpy(ptr_next->normal, vertex+3, 3*sizeof(float));
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

void subMenuHandler(int value) {
	render_gaussian = true;
	g_gaussians.clear();
	switch (value) {
	case 101: printf("Hotdog selected\n");
		ply_file_path = "../../Data/ply/hotdog/hotdog_3dgrt.ply";
#if !ROTATION
		ply_kdtree_path = "../../Data/ply/hotdog/hotdog_tree.kdt";
		ply_igeom_path = "../../Data/ply/hotdog/hotdog_igeom.bin";
		ply_to_obj = "../../Data/ply/hotdog/hotdog_3dgrt_new.obj";
		kdtree_build_path = "../../Data/ply/hotdog/hotdog_3dgrt_kdt.txt";
#else
		ply_kdtree_path = "../../Data/ply/hotdog/hotdog_tree_rot.kdt";
		ply_igeom_path = "../../Data/ply/hotdog/hotdog_igeom_rot.bin";
		ply_to_obj = "../../Data/ply/hotdog/hotdog_3dgrt_new_rot.obj";
		kdtree_build_path = "../../Data/ply/hotdog/hotdog_3dgrt_kdt_rot.txt";
#endif
		break;
	case 1012: printf("Hotdog2 selected\n");
		ply_file_path = "../../Data/ply/hotdog2/hotdog_3dgrt2.ply";
#if !ROTATION
		ply_kdtree_path = "../../Data/ply/hotdog2/hotdog2_tree.kdt";
		ply_igeom_path = "../../Data/ply/hotdog2/hotdog2_igeom.bin";
		ply_to_obj = "../../Data/ply/hotdog2/hotdog_3dgrt2_new.obj";
		kdtree_build_path = "../../Data/ply/hotdog2/hotdog2_3dgrt_kdt.txt";
#else
		ply_kdtree_path = "../../Data/ply/hotdog2/hotdog2_tree_rot.kdt";
		ply_igeom_path = "../../Data/ply/hotdog2/hotdog2_igeom_rot.bin";
		ply_to_obj = "../../Data/ply/hotdog2/hotdog_3dgrt2_new_rot.obj";
		kdtree_build_path = "../../Data/ply/hotdog2/hotdog2_3dgrt_kdt_rot.txt";
#endif
		break;
	case 102: printf("Lego selected\n");
		ply_file_path = "../../Data/ply/lego/lego_3dgrt.ply";
#if !ROTATION
		ply_kdtree_path = "../../Data/ply/lego/lego_tree.kdt";
		ply_igeom_path = "../../Data/ply/lego/lego_igeom.bin";
		ply_to_obj = "../../Data/ply/lego/lego_3dgrt_new.obj";
		kdtree_build_path = "../../Data/ply/lego/lego_3dgrt_kdt.txt";
#else
		ply_kdtree_path = "../../Data/ply/lego/lego_tree_rot.kdt";
		ply_igeom_path = "../../Data/ply/lego/lego_igeom_rot.bin";
		ply_to_obj = "../../Data/ply/lego/lego_3dgrt_new_rot.obj";
		kdtree_build_path = "../../Data/ply/lego/lego_3dgrt_kdt_rot.txt";
#endif
		break;
	case 103: printf("Bonsai selected\n");
		ply_file_path = "../../Data/ply/bonsai/bonsai.ply";
#if !ROTATION
		ply_kdtree_path = "../../Data/ply/bonsai/bonsai_tree.kdt";
		ply_igeom_path = "../../Data/ply/bonsai/bonsai_igeom.bin";
		ply_to_obj = "../../Data/ply/bonsai/bonsai_3dgrt_new.obj";
		kdtree_build_path = "../../Data/ply/bonsai/bonsai_3dgrt_kdt.txt";
#else
		ply_kdtree_path = "../../Data/ply/bonsai/bonsai_tree_rot.kdt";
		ply_igeom_path = "../../Data/ply/bonsai/bonsai_igeom_rot.bin";
		ply_to_obj = "../../Data/ply/bonsai/bonsai_3dgrt_new_rot.obj";
		kdtree_build_path = "../../Data/ply/bonsai/bonsai_3dgrt_kdt_rot.txt";
#endif
		break;
	case 104: printf("Chair selected\n");
		ply_file_path = "../../Data/ply/chair/chair_3dgrt.ply";
#if !ROTATION
		ply_kdtree_path = "../../Data/ply/chair/chair_tree.kdt";
		ply_igeom_path = "../../Data/ply/chair/chair_igeom.bin";
		ply_to_obj = "../../Data/ply/chair/chair_3dgrt_new.obj";
		kdtree_build_path = "../../Data/ply/chair/chair_3dgrt_kdt.txt";
#else
		ply_kdtree_path = "../../Data/ply/chair/chair_tree_rot.kdt";
		ply_igeom_path = "../../Data/ply/chair/chair_igeom_rot.bin";
		ply_to_obj = "../../Data/ply/chair/chair_3dgrt_new_rot.obj";
		kdtree_build_path = "../../Data/ply/chair/chair_3dgrt_kdt_rot.txt";
#endif
		break;
	case 105: printf("Flowers selected\n");
		ply_file_path = "../../Data/ply/flowers/flowers.ply";
#if !ROTATION
		ply_kdtree_path = "../../Data/ply/flowers/flowers_tree.kdt";
		ply_igeom_path = "../../Data/ply/flowers/flowers_igeom.bin";
		ply_to_obj = "../../Data/ply/flowers/flowers_3dgrt_new.obj";
		kdtree_build_path = "../../Data/ply/flowers/flowers_3dgrt_kdt.txt";
#else
		ply_kdtree_path = "../../Data/ply/flowers/flowers_tree_rot.kdt";
		ply_igeom_path = "../../Data/ply/flowers/flowers_igeom_rot.bin";
		ply_to_obj = "../../Data/ply/flowers/flowers_3dgrt_new_rot.obj";
		kdtree_build_path = "../../Data/ply/flowers/flowers_3dgrt_kdt_rot.txt";
#endif
		break;
	}
	//printf("%s\n", ply_file_path);
	// 3DGS 학습 결과물을 로드
	if (!loadGaussiansFromPly(ply_file_path, g_gaussians)) {
		fprintf(stderr, "Failed to load ply file\n");
		return;
	}
	create_composite_object_from_gaussians(g_gaussians);
#if ROTATION
	//printf("%s\n%s\n%s\n", ply_kdtree_path, ply_igeom_path, ply_to_obj);
	rotate_composite_object(g_gaussians, 45.0f, 1.0f, 1.0f, 1.0f);
#endif
	uip.composite_object_read = 1;
#if LEAF_NODE_DEBUG
	if (original_vertices != nullptr) {
		delete[] original_vertices;
		original_vertices = nullptr;
	}
	num_original_vertices = uip.poly_model.n_triangles * 3;
	if (num_original_vertices > 0) {
		original_vertices = new ExtendedVertex[num_original_vertices];

		size_t total_bytes = num_original_vertices * sizeof(ExtendedVertex);
		memcpy(original_vertices, uip.poly_model.extended_vertices, total_bytes);

		original_model_AABB.min[0] = uip.poly_model.AABB[XMIN];
		original_model_AABB.min[1] = uip.poly_model.AABB[YMIN];
		original_model_AABB.min[2] = uip.poly_model.AABB[ZMIN];
		original_model_AABB.max[0] = uip.poly_model.AABB[XMAX];
		original_model_AABB.max[1] = uip.poly_model.AABB[YMAX];
		original_model_AABB.max[2] = uip.poly_model.AABB[ZMAX];
	}
#endif
	load_poly_model_into_OpenGL();
	g_cuda_rendering_done = false;
	glutPostRedisplay();
}

void main_menu_action(int selection) {
	char full_kd_tree_file_name[512];
	char full_i_geometry_file_name[512];
	fprintf(stdout, "\n");
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

	case 200: { // obj loader
		render_gaussian = true;
		const char* file_path = MODEL_PATH;  // obj 경로
		if (!read_OBJ_file(file_path)) {
			fprintf(stderr, "Failed to load obj file\n");
			return;
		}
		ply_file_path = MODEL_PATH;
		ply_kdtree_path = KDTREE_PATH;
		ply_igeom_path = IGEOM_PATH;
		ply_to_obj = "../../Data/obj/hotdog_3dgrt_new.obj";

		uip.composite_object_read = 1;

		load_poly_model_into_OpenGL();
		g_cuda_rendering_done = false;
		glutPostRedisplay();
		printf("draw DONE\n");
		break;
	}
	case 300: // construct kd-tree
		print_current_time("kdtree build start");

		build_kd_tree_for_composite_object(&uip.poly_model);

		if (uip.poly_model.kd_tree->tri_accel_list == NULL) printf("tri_accel_list NULL\n");
		else {
			printf("triangle num: %d\n", uip.poly_model.n_triangles);
			//printf("tri_accel_list size: %d\n", sizeof(uip.poly_model.kd_tree->tri_accel_list) / sizeof(*(uip.poly_model.kd_tree->tri_accel_list)));
		}
		print_current_time("kdtree build end");
		printKdTreeLeafNodeInfo();
#if LEAF_NODE_DEBUG
		leaf_nodes = extract_all_leaf_data(&uip.poly_model, largest_leaf_index);
#endif
		break;
	case 400: // dump kd-tree
		strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
		strcat(full_kd_tree_file_name, "/");
		strcat(full_kd_tree_file_name, uip.kd_tree_filename);

		strcpy(full_i_geometry_file_name, uip.kd_tree_dump_dir);
		strcat(full_i_geometry_file_name, "/");
		strcat(full_i_geometry_file_name, uip.i_geometry_filename);

		if (render_gaussian) {
			dump_kd_tree_for_composite_object(
				&uip.poly_model,
				ply_kdtree_path,         // 저장할 kd-tree
				KD_TREE_DUMP_IN_BINARY,    // 저장 포맷
				ply_igeom_path         // 저장할 geometry
			);
		}
		else {
			dump_kd_tree_for_composite_object(&uip.poly_model, full_kd_tree_file_name,
				uip.kd_tree_dump_format, full_i_geometry_file_name);
		}
		break;
	case 500: // load kd-tree
		strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
		printf("uip.kd_tree_dump_dir:%s\n", uip.kd_tree_dump_dir);
		strcat(full_kd_tree_file_name, "/");
		strcat(full_kd_tree_file_name, uip.kd_tree_filename);
		printf("uip.kd_tree_filename:%s\n", uip.kd_tree_filename);
		if (render_gaussian) {
			strcpy(full_kd_tree_file_name, ply_kdtree_path);
			uip.kd_tree_dump_format = KD_TREE_DUMP_IN_BINARY;
		}
		printf("full_kd_tree_file_name:%s\n", full_kd_tree_file_name);
		read_kd_tree_from_file(&uip.poly_model, full_kd_tree_file_name, uip.kd_tree_dump_format);

		printKdTreeLeafNodeInfo();
#if LEAF_NODE_DEBUG
		leaf_nodes = extract_all_leaf_data(&uip.poly_model, largest_leaf_index);
#endif
		glutPostRedisplay();
		break;
	case 600:
		g_cuda_interactive_mode = !g_cuda_interactive_mode; // 인터랙티브 모드 토글
		if (g_cuda_interactive_mode) {
			//if (g_d_render_framebuffer) {
			//	cudaFree(g_d_render_framebuffer);
			//}
			//cudaMalloc((void**)&g_d_render_framebuffer, (size_t)g_render_width * g_render_height * 3 * sizeof(float));

			renderGaussianWithCudaSetup(uip.poly_model, g_gaussians);
#if USE_STACK > SHORT_STACK
			if (g_d_global_stack) cudaFree(g_d_global_stack);
			cudaMalloc((void**)&g_d_global_stack, (size_t)g_render_width * g_render_height * MAX_GLOBAL_STACK_DEPTH * sizeof(cu_traceState));
#endif
			g_camera_dirty = true; // 모드를 켜는 즉시 한 번 렌더링하도록 설정
			printf("CUDA Interactive Mode: ON\n");
		}
		else {
#if USE_STACK > SHORT_STACK
			if (g_d_global_stack) cudaFree(g_d_global_stack);
			g_d_global_stack = nullptr;
#endif

			printf("CUDA Interactive Mode: OFF\n");
			// 인터랙티브 모드를 끄면 다시 OpenGL 뷰로 돌아가도록 화면 갱신
			glutPostRedisplay();
		}
		break;
	case 700:
		fprintf(stdout, "dump .obj file\n");
		if (uip.composite_object_read) {
			save_composite_object_to_obj(uip.poly_model, ply_to_obj);
			fprintf(stdout, "Done!\n");
		}
		else {
			fprintf(stderr, "Error: No composite object loaded to save.\n");
		}
		break;
	case 800:
		print_current_time("all_build_start\n");
		for (int i = 0; i < 5; i++) {
			subMenuHandler(submenu[i]);

			build_kd_tree_for_composite_object2(&uip.poly_model, kdtree_build_path);

			dump_kd_tree_for_composite_object(
				&uip.poly_model,
				ply_kdtree_path,         // 저장할 kd-tree
				KD_TREE_DUMP_IN_BINARY,    // 저장 포맷
				ply_igeom_path         // 저장할 geometry
			);
		}
		print_current_time("all_build_end\n");
		break;
	case 999:
		clean_up_system();
		exit(0);
		break;
	}
}

void register_callbacks_and_create_menu(void) {
	glutDisplayFunc(display); 
	glutKeyboardFunc(keyboard); 
	glutReshapeFunc(reshape);
	glutMouseFunc(mousepress); 
	glutMotionFunc(mousemove);

	int submenu = glutCreateMenu(subMenuHandler);
	glutAddMenuEntry("hotdog", 101);
	glutAddMenuEntry("hotdog2", 1012);
	glutAddMenuEntry("lego", 102);
	glutAddMenuEntry("bonsai", 103);
	glutAddMenuEntry("chair", 104);
	glutAddMenuEntry("flowers", 105);

	uip.main_menu_ID = glutCreateMenu(main_menu_action);
	glutAddMenuEntry("ChangeMode", 0);
	glutAddMenuEntry("1. Read SL_KDT_Config File and Prepair I-Geometry", 100);
	glutAddMenuEntry("2. Read .obj File and Prepair I-Geometry", 200);
	//glutAddMenuEntry("2. Read .ply File and Prepair I-Geometry", 200);
	glutAddSubMenu("2. Read .ply File and Prepair I-Geometry", submenu);
	glutAddMenuEntry("3. Construct Kd-tree from I-Geometry", 300);  
	glutAddMenuEntry("4. Dump Kd-tree and I-Geometry to Files", 400);
	glutAddMenuEntry("4-1. Dump I-Geometry to obj File", 700);
	glutAddMenuEntry("5. Read Kd-tree from File", 500);
	glutAddMenuEntry("7. CUDA Rendering (Interactive Toggle)", 600);
	glutAddMenuEntry("8. ply all build", 800);
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

	//v_KD_TREE_TRAVL_COST = 1.0;
	v_KD_TREE_TRAVL_COST = TRAVL_COST;
	//v_KD_TREE_ISECT_COST = 1.5;
	v_KD_TREE_ISECT_COST = ISCET_COST;
	//v_KD_TREE_MAX_LEVEL = 100;
	v_KD_TREE_MAX_LEVEL = MAX_LEVEL;
	//v_KD_TREE_MIN_TRIANGLE = 4;
	v_KD_TREE_MIN_TRIANGLE = MIN_TRI;
	//v_KD_TREE_EMTPY_BONUS = 0.9;
	v_KD_TREE_EMTPY_BONUS = EMTPY_BONUS;
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

		// PBO를 CUDA에서 사용할 수 있도록 매핑
		float* d_pbo_ptr;
		cudaGraphicsMapResources(1, &pbo_cuda_resource, 0);
		size_t num_bytes;// = (size_t)g_render_width * g_render_height * 3 * sizeof(float);
		cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, pbo_cuda_resource);

		// CUDA 렌더링 실행 (기존 렌더링 함수 재사용)
#if SCENE_NUM < 1
		renderObjWithCuda(uip.poly_model, camera, g_render_width, g_render_height, d_pbo_ptr, g_cuda_rendering_done);
#else
		g_fps = renderGaussianWithCudaFrame(camera, g_render_width, g_render_height, d_pbo_ptr
#if HIT_AND_NODE_COUNT_DEBUG
			, h_debug_buffer1_main, h_debug_buffer2_main
#endif
#if USE_STACK > SHORT_STACK
			, g_d_global_stack
#endif
#if LEAF_NODE_DEBUG
			, uip.poly_model
#endif
		);

		if (measure_fps_interval) {
			if(++frame_count > MEASURE_START_FRAME) total_fps += g_fps;
			if (frame_count >= MEASURE_END_FRAME) {
				printf("avg FPS for %d-%d frame : %f\n", MEASURE_START_FRAME, MEASURE_END_FRAME, (float)(total_fps / (MEASURE_END_FRAME - MEASURE_START_FRAME)));
				frame_count = 0;
				total_fps = 0.0f;
				measure_fps_interval = false;
				timerRunning = false;
			}
		}

		// CUDA → OpenGL 동기화 해제
		cudaGraphicsUnmapResources(1, &pbo_cuda_resource, 0);

		g_cuda_rendering_done = true;

#if HIT_AND_NODE_COUNT_DEBUG
		// 최댓값을 계산하여 전역 변수에 저장
		// TODO: MyMAX로 변환
		memset(max_debug_values, 0, sizeof(max_debug_values));
		if (h_debug_buffer1_main != nullptr) {
			for (int i = 0; i < g_render_width * g_render_height; i++) {
				max_debug_values[0] = MyMAX(max_debug_values[0], (int)h_debug_buffer1_main[i].x);
				max_debug_values[1] = MyMAX(max_debug_values[1], (int)h_debug_buffer1_main[i].y);
				max_debug_values[2] = MyMAX(max_debug_values[2], (int)h_debug_buffer1_main[i].z);
			}
			for (int i = 0; i < g_render_width * g_render_height; i++) {
				max_debug_values[3] = MyMAX(max_debug_values[3], (int)h_debug_buffer2_main[i].x);
				max_debug_values[4] = MyMAX(max_debug_values[4], (int)h_debug_buffer2_main[i].y);
				max_debug_values[5] = MyMAX(max_debug_values[5], (int)h_debug_buffer2_main[i].z);
			}
		}
#endif
#endif

		// PBO의 내용을 텍스춰로 복사
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, result_texture_id);
		// PBO 버퍼의 데이터를 현재 바인딩된 2D 텍스춰로 전송
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_render_width, g_render_height, GL_RGB, GL_FLOAT, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glutPostRedisplay(); // 화면 갱신 요청
	}
}

void main(int argc, char **argv) {
	init_KDT_system();
	glutInit (&argc, argv); 
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);   
	glutInitWindowSize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT);
	glutInitContextVersion(4, 0);
	glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
	uip.main_window_ID = glutCreateWindow("Ply-to-Kd-Tree Converter-Tracer SW: Verion 1.0_glut");
	initialize_glew();
//shyun
	cudaGLSetGLDevice(0);
	if (!initCuda()) {
		fprintf(stderr, "Failed to initialize CUDA. Exiting.\n");
		system("pause");
		exit(1);
	}

	setup_interop_resources();
//shyun end

	register_callbacks_and_create_menu();
	glutIdleFunc(idle);

	init_OpenGL_RC(); 
	print_OpenGL_GLSL_GLEW_version();
	show_greetings();

	glutTimerFunc(16, timer_callback, 0);

	glutMainLoop ();
}


