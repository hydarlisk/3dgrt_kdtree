/**************************************************************
  File name: OpenGLStuffs.cpp
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#include <stdio.h>
#include <math.h>
#include <GL/glut.h>

#include "OpenGLStuffs.h"
#include "Kd-treeConverter.h"

extern int g_render_width;
extern int g_render_height;

void set_rotate_mat(Camera *cam) {
	#define M(row,col)  cam->mat[col*4+row] // C/C++ way
    M(0,0)=cam->uaxis[X]; M(0,1)=cam->uaxis[Y]; M(0,2)=cam->uaxis[Z]; M(0,3)=0.0;
    M(1,0)=cam->vaxis[X]; M(1,1)=cam->vaxis[Y]; M(1,2)=cam->vaxis[Z]; M(1,3)=0.0;
    M(2,0)=cam->naxis[X]; M(2,1)=cam->naxis[Y]; M(2,2)=cam->naxis[Z]; M(2,3)=0.0;
    M(3,0)=0.0;  M(3,1)=0.0;  M(3,2)=0.0;  M(3,3)=1.0;
}

void initialize_camera(Camera *cam) {
#define i_c_POS_X 5.0
#define i_c_POS_Y 5.0
#define i_c_POS_Z 5.0

	GLfloat matrix[16];

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
    gluLookAt(i_c_POS_X, i_c_POS_Y, i_c_POS_Z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	glPopMatrix();

	cam->pos[X] = i_c_POS_X; cam->pos[Y]= i_c_POS_Y;  cam->pos[Z] = i_c_POS_Z;

	cam->uaxis[X] = matrix[0]; cam->uaxis[Y] = matrix[4]; cam->uaxis[Z] = matrix[8];
	cam->vaxis[X] = matrix[1]; cam->vaxis[Y] = matrix[5]; cam->vaxis[Z] = matrix[9];
	cam->naxis[X] = matrix[2]; cam->naxis[Y] = matrix[6]; cam->naxis[Z] = matrix[10];
 
	set_rotate_mat(cam);

	cam->move = 0;
	cam->fovy = 20.0, cam->aspect = 1.0; cam->near_c = 1.0; cam->far_c = 10000.0;
	//cam->fovy = FOV_Y, cam->aspect = g_render_width / g_render_height; cam->near_c = NEAR_PLANE; cam->far_c = FAR_PLANE;
} 

 void draw_axes(GLfloat scale) {
	 GLboolean lighting_enabled;

	 lighting_enabled = glIsEnabled(GL_LIGHTING);
	 if (lighting_enabled)  
		 glDisable(GL_LIGHTING);

	 glLineWidth(3.0);
	 glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(scale, 0.0, 0.0);

		glColor3f(0.0, 1.0, 0.0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, scale, 0.0);

		glColor3f(0.0, 0.0, 1.0);
		glVertex3f(0.0, 0.0, 0.0);
		glVertex3f(0.0, 0.0, scale);
	glEnd();
	glLineWidth(1.0);

	 if (lighting_enabled)  
		 glEnable(GL_LIGHTING);
}

void draw_AABB(GLfloat *AABB) {	 
	
	GLboolean lighting_enabled;

	lighting_enabled = glIsEnabled(GL_LIGHTING);

	if (lighting_enabled)  
		 glDisable(GL_LIGHTING);

	glLineWidth(2.0);

	glColor3f(0.0, 1.0, 0.0);
	glBegin(GL_LINE_LOOP);
	glVertex3f(AABB[XMIN], AABB[YMIN], AABB[ZMIN]);
	glVertex3f(AABB[XMAX], AABB[YMIN], AABB[ZMIN]);
	glVertex3f(AABB[XMAX], AABB[YMIN], AABB[ZMAX]);
	glVertex3f(AABB[XMIN], AABB[YMIN], AABB[ZMAX]);
	glEnd();
 
	glBegin(GL_LINE_LOOP);
	glVertex3f(AABB[XMIN], AABB[YMAX], AABB[ZMIN]);
	glVertex3f(AABB[XMAX], AABB[YMAX], AABB[ZMIN]);
	glVertex3f(AABB[XMAX], AABB[YMAX], AABB[ZMAX]);
	glVertex3f(AABB[XMIN], AABB[YMAX], AABB[ZMAX]);
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(AABB[XMIN], AABB[YMIN], AABB[ZMIN]);
	glVertex3f(AABB[XMIN], AABB[YMAX], AABB[ZMIN]);

	glVertex3f(AABB[XMAX], AABB[YMIN], AABB[ZMIN]);
	glVertex3f(AABB[XMAX], AABB[YMAX], AABB[ZMIN]);

	glVertex3f(AABB[XMAX], AABB[YMIN], AABB[ZMAX]);
	glVertex3f(AABB[XMAX], AABB[YMAX], AABB[ZMAX]);

	glVertex3f(AABB[XMIN], AABB[YMIN], AABB[ZMAX]);
	glVertex3f(AABB[XMIN], AABB[YMAX], AABB[ZMAX]);

	glEnd();
	glLineWidth(1.0);

	if (lighting_enabled)  
		 glEnable(GL_LIGHTING);
}

void set_OpenGL_material(GLenum which_face, GLfloat ambr, GLfloat ambg, GLfloat ambb,
					 GLfloat difr, GLfloat difg, GLfloat difb,
					 GLfloat specr, GLfloat specg, GLfloat specb, GLfloat shine) {
   GLfloat color[4];

   color[0] = ambr; color[1] = ambg; color[2] = ambb; color[3] = 1.0;
   glMaterialfv(which_face, GL_AMBIENT, color);

   color[0] = difr; color[1] = difg; color[2] = difb;
   glMaterialfv(which_face, GL_DIFFUSE, color);

   color[0] = specr; color[1] = specg; color[2] = specb;
   glMaterialfv(which_face, GL_SPECULAR, color);
   glMaterialf(which_face, GL_SHININESS, shine * 128.0);
}

void set_parallel_OpenGL_head_light_color(void) {
	GLfloat color_amb[4] = { 0.23, 0.23, 0.23, 1.0 };
 	GLfloat color_diff_spec[4] = { 0.95, 0.95, 0.95, 1.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, color_amb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, color_diff_spec);
	glLightfv(GL_LIGHT0, GL_SPECULAR, color_diff_spec);
}

void set_parallel_OpenGL_head_light_position(void) {
	GLfloat position[4];

	position[0] = 0.0; position[1] = 0.0; position[2] = 0.0; position[3] = 1.0;
	glLightfv(GL_LIGHT0, GL_POSITION, position);
}

void turn_on_parallel_OpenGL_head_light(void) {
 	glEnable(GL_LIGHT0); 
}

void turn_off_parallel_OpenGL_head_light(void) {
	glDisable(GL_LIGHT0); 
}

void set_OpenGL_light_model(void) {
	GLfloat global_ambient_color[4] = {0.23, 0.23, 0.23, 1.0 };

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient_color);
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0);
}