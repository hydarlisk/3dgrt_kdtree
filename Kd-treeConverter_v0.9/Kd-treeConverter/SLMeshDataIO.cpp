/**************************************************************
  File name: SLMeshDataIO.cpp
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "Kd-treeConverter.h"
#include "MyMathUtility.h"
//using namespace KDTConverter;

int fetch_face_numbers_SL_mesh_geom_file(const char *filename) {
	int tmp;
	FILE *fp;

//	fprintf(stdout, "f_f_n_SL_m_g_f: fetching the triangle number from  the SL mesh geometry file = %s\n", filename);
	if ((fp = fopen(filename, "r")) == NULL) {
		fprintf(stderr, "f_f_n_SL_m_g_f: cannot open the file %s. Just skipping...\n", filename);
		return 0;
	}
	fscanf(fp, "Total Number of Node = %d\n", &tmp);
	fscanf(fp, "Total Number of Element = %d\n", &tmp);
	fclose(fp);
	fprintf(stdout, "  * # of triangles in %s = %d\n", filename, tmp);
	return tmp;
}

int read_SL_mesh_geom_file_kd_tree(const char *filename, MeshGeom *meshgeom) {
	int i;
	char buf0[32], buf1[32];
	unsigned int *ptr_ui;
	float *ptr_f;
	FILE *fp;

//	printf("r_SL_m_c_f_k_t: reading the SL mesh geometry file = %s\n", filename);
	printf("  - Reading %s\n", filename);
	if ((fp = fopen(filename, "r")) == NULL) {
		return(0);
	}
	
	fscanf(fp, "Total Number of Node = %d\n", &(meshgeom->nvertices));
	printf("    * # of vertices = %d\n", meshgeom->nvertices);
	fscanf(fp, "Total Number of Element = %d\n", &(meshgeom->nfaces));
	printf("    * # of (triangular) faces = %d\n", meshgeom->nfaces);

	if ((meshgeom->vertices = (float *) malloc(meshgeom->nvertices*6*sizeof(float))) == NULL) {
		fprintf(stderr, "r_SL_m_c_f_k_t: (Error) memory allocation error (c100)\n");
		exit(-1);
	}
	
	ptr_f = meshgeom->AABB;
	ptr_f[XMIN] = ptr_f[YMIN] = ptr_f[ZMIN] = FLT_MAX;
	ptr_f[XMAX] = ptr_f[YMAX] = ptr_f[ZMAX] = -FLT_MAX;

	ptr_f = meshgeom->vertices;
	for (i = 1; i <= meshgeom->nvertices; i++) {
		fscanf(fp, "%s %s %f %f %f %f %f %f\n", buf0, buf1, ptr_f, ptr_f+1, ptr_f+2, 
			ptr_f+3, ptr_f+4, ptr_f+5);

		if (ptr_f[X] < meshgeom->AABB[XMIN]) meshgeom->AABB[XMIN] = ptr_f[X];
		if (ptr_f[X] > meshgeom->AABB[XMAX]) meshgeom->AABB[XMAX] = ptr_f[X];
		if (ptr_f[Y] < meshgeom->AABB[YMIN]) meshgeom->AABB[YMIN] = ptr_f[Y];
		if (ptr_f[Y] > meshgeom->AABB[YMAX]) meshgeom->AABB[YMAX] = ptr_f[Y];
		if (ptr_f[Z] < meshgeom->AABB[ZMIN]) meshgeom->AABB[ZMIN] = ptr_f[Z];
		if (ptr_f[Z] > meshgeom->AABB[ZMAX]) meshgeom->AABB[ZMAX] = ptr_f[Z];

		ptr_f += 6;
	}
	
	if ((meshgeom->faces = (unsigned int *) malloc(meshgeom->nfaces*3*sizeof(unsigned int))) == NULL) {
		fprintf(stderr, "r_SL_m_c_f_k_t: (Error) memory allocation error (c101)\n");
		exit(-1);
	}

    ptr_ui = meshgeom->faces;
	for (i = 1; i <= meshgeom->nfaces; i++) {
		fscanf(fp, "%s %s %u %u %u\n", buf0, buf1, ptr_ui, ptr_ui+1, ptr_ui+2);
		*ptr_ui -= 1; *(ptr_ui+1) -= 1; *(ptr_ui+2) -= 1;
 	 	
		ptr_ui += 3;
	}
	fclose(fp);

	ptr_f = meshgeom->AABB;
	printf("  - Done!\n\n");

	return(1);
}

