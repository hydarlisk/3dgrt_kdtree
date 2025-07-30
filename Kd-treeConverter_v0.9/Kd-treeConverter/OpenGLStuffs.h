#pragma once
/**************************************************************
  File name: OpenGLStuffs.h
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#define X 0
#define Y 1
#define Z 2

#define XMIN 0
#define XMAX 1
#define YMIN 2
#define YMAX 3
#define ZMIN 4
#define ZMAX 5

#define BUFFER_OFFSET(bytes) ((GLuint *) NULL + (bytes))

typedef struct _cam {
	float pos[3];
	float uaxis[3], vaxis[3], naxis[3];
	GLfloat mat[16];
	GLfloat mat_inv[16];
	int move, upanddown;
	GLdouble fovy, aspect, near_c, far_c;
} Camera;

void set_rotate_mat(Camera *); 
void initialize_camera(Camera *);
void draw_axes(GLfloat);
void draw_AABB(GLfloat *);
void set_OpenGL_material(GLenum, GLfloat, GLfloat, GLfloat, 
						 GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat, GLfloat);
void set_parallel_OpenGL_head_light_color(void);
void set_parallel_OpenGL_head_light_position(void);
void turn_on_parallel_OpenGL_head_light(void);
void turn_off_parallel_OpenGL_head_light(void);

void set_OpenGL_light_model(void);