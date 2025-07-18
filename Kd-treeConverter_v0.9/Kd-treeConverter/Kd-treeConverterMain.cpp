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

#include "sgrt_interface.h"
#include "cudaRayTracingKernel.cu"
#include "SGRTx2Lib/GGPURayTracer.h"

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
			printf("camera.pos: %f %f %f\n", camera.pos[0], camera.pos[1], camera.pos[2]);

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

	switch(selection) {
		case 100:
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

			glutPostRedisplay();
			break;
		case 200:
			build_kd_tree_for_composite_object(&uip.poly_model);
			break;
		case 300:
			strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
			strcat(full_kd_tree_file_name, "/");
			strcat(full_kd_tree_file_name, uip.kd_tree_filename);

			strcpy(full_i_geometry_file_name, uip.kd_tree_dump_dir);
			strcat(full_i_geometry_file_name, "/");
			strcat(full_i_geometry_file_name, uip.i_geometry_filename);

		
			dump_kd_tree_for_composite_object(&uip.poly_model, full_kd_tree_file_name,
				uip.kd_tree_dump_format, full_i_geometry_file_name);
			break;
		case 400:
			strcpy(full_kd_tree_file_name, uip.kd_tree_dump_dir);
			printf("uip.kd_tree_dump_dir:%s\n", uip.kd_tree_dump_dir);
			strcat(full_kd_tree_file_name, "/");
			strcat(full_kd_tree_file_name, uip.kd_tree_filename);
			printf("uip.kd_tree_filename:%s\n", uip.kd_tree_filename);
			printf("full_kd_tree_file_name:%s\n", full_kd_tree_file_name);
			strcpy(full_kd_tree_file_name, "../../Data/Obj/hotdog_tree.kdt");
			printf("full_kd_tree_file_name:%s\n", full_kd_tree_file_name);
			read_kd_tree_from_file(&uip.poly_model, full_kd_tree_file_name, uip.kd_tree_dump_format);
			glutPostRedisplay();
			break;
		case 500: {
			const char* obj_path = "../../Data/Obj/hotdog_3dgrt.obj";  // obj 경로
			CompositeObject obj_model;

			if (!load_obj_to_composite_object(obj_path, &obj_model)) {
				fprintf(stderr, "Failed to load .obj file.\n");
				break;
			}

			fprintf(stdout, "Successfully loaded .obj model. Building Kd-tree...\n");
			build_kd_tree_for_composite_object(&obj_model);

			dump_kd_tree_for_composite_object(
				&obj_model,
				"../../Data/Obj/hotdog_tree.kdt",         // 저장할 kd-tree
				KD_TREE_DUMP_IN_BINARY,    // 저장 포맷
				"../../Data/Obj/hotdog_igeom.bin"         // 저장할 geometry
			);

			uip.poly_model = obj_model;
			uip.composite_object_read = 1;

			printf("uip, AABB: X [%f, %f] Y [%f, %f] Z [%f, %f]\n",
				uip.poly_model.AABB[XMIN], uip.poly_model.AABB[XMAX],
				uip.poly_model.AABB[YMIN], uip.poly_model.AABB[YMAX],
				uip.poly_model.AABB[ZMIN], uip.poly_model.AABB[ZMAX]);

			glutPostRedisplay();
			break;
		}
		case 600: {
			//CUDA rendering
			fprintf(stdout, "CUDA ray tracing Render with kd-tree\n");
			if (&uip.poly_model == NULL) {
				fprintf(stdout, "Dosen't exist kd-tree\n");
				break;
			}

			//TODO: CUDA rendering*****************************************
			GScene* scene = convertCompositeObjectToGScene(&uip.poly_model);

			// [2] GGPUExperimentalRayTracer 초기화 및 렌더링
			GGPUExperimentalRayTracer raytracer;
			GError err = raytracer.rendering(scene, false);

			if (err != errorNo) {
				printf("Rendering failed with error %d\n", err);
			}
			else {
				printf("Rendering done. Check framebuffer or saved file.\n");
			}

			delete scene;  // 적절한 해제 필요
			break;

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
		}
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
	glutAddMenuEntry("1. Read SL_KDT_Config File and Prepair I-Geometry", 100); 
	glutAddMenuEntry("2. Construct Kd-tree from I-Geometry", 200);  
	glutAddMenuEntry("3. Dump Kd-tree and I-Geometry to Files", 300);
	glutAddMenuEntry("4. Read Kd-tree from File", 400);
	glutAddMenuEntry("5. Read & Construct & Dump & Rendering .obj File and Prepair I-Geometry", 500);
	//glutAddMenuEntry("6. CUDA Rendering", 600);
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

void main(int argc, char **argv) {

	init_KDT_system();
	glutInit (&argc, argv); 
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);   
	glutInitWindowSize(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT);
	glutInitContextVersion(4, 0);
	glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
	uip.main_window_ID = glutCreateWindow("SL Mesh-to-Kd-Tree Converter SW: Verion 1.0_glut");
	initialize_glew(); 
	register_callbacks_and_create_menu();

	init_OpenGL_RC(); 
	print_OpenGL_GLSL_GLEW_version();
	show_greetings();

	glutMainLoop ();
}


