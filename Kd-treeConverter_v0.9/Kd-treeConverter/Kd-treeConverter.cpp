/**************************************************************
  File name: Kd-treeConverter.cpp
  Version: 1.0
  Date: November 1, 2014
 **************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "Kd-treeConverter.h"
//#include "Kd-treeConverterMain.h"
#include "Kd-treeConstructor.h"
#include "RayTraversal.h"
#include "MyMathUtility.h"
//using namespace KDTConstructor;
//using namespace KDTConverter;
int build_kd_tree_for_composite_object(CompositeObject *c_object) {
	// Returns 1 if a kd-tree was constructed successfully, or 0 otherwise.
	// Input: "c_object->n_triangles" & "c_object->extended_vertices"
	// Output: "c_object->kd_tree"

	fprintf(stdout, "\n> Constructing Kd-tree from I-geometry\n");
	fprintf(stdout, "\n  * # of triangles: %d\n", c_object->n_triangles);
	fprintf(stdout, "\n  * AABB: [%7.2f, %7.2f] x [%7.2f, %7.2f] x [%7.2f, %7.2f]\n", c_object->AABB[0], c_object->AABB[1],
		c_object->AABB[2], c_object->AABB[3], c_object->AABB[4], c_object->AABB[5]); 
	fprintf(stdout, "\n  * Empty Bonus: %7.3f\n  * Travel Cost: %7.3f\n  * Intersection Cost: %7.3f\n  * Max Tree Level: %d\n  * Min # of Triangles per Leaf: %d\n",
			v_KD_TREE_EMTPY_BONUS, v_KD_TREE_TRAVL_COST, v_KD_TREE_ISECT_COST, v_KD_TREE_MAX_LEVEL, v_KD_TREE_MIN_TRIANGLE );

	// allocate memory and initialize data
	if (initialize_kd_tree(c_object) == 0) {
		fprintf(stderr, "Kd-tree construction error.\n");
		return 0;
	}

	// build kd-tree
	fprintf(stdout, "\n  - Building a kd-tree\n");
	build_kd_tree_recursive(g_bEdge, g_pTriangleInfos, g_iTriangleSize, g_root_AABB, 0,  &(g_pKdTree_Node_Array[0]));
	fprintf(stdout, "  - Done!\n\n");
	fprintf(stdout, "   * Tree Level: %d\n", g_iKdTree_Level);
	fprintf(stdout, "   * Node Count (All,Leaf,Empty) : %5d, %5d, %5d(%.1f%%)\n",
									g_iKdTree_Node_Count, g_iKdTree_LeafNode_Count, g_iKdTree_EmptyNode_Count,
									100.0f * g_iKdTree_EmptyNode_Count / g_iKdTree_Node_Count);
	fprintf(stdout, "   * Maximum Tri# in LeafNode: %d\n", g_iKdTree_MaxTriInLeafNode_Count);

	fprintf(stdout, "\n  - Building a kd-tree triangle accerlaration list\n");
	// build triangle acceleration
	TriAccel *pTriAcc = NULL;
	pTriAcc = (TriAccel*)_aligned_malloc(c_object->n_triangles * sizeof(TriAccel), 16);
	build_TriAccList(c_object, pTriAcc);
	fprintf(stdout, "  - Done!\n");

	c_object->kd_tree = new KdTree;
	c_object->kd_tree->tree = g_pKdTree_Node_Array;
	c_object->kd_tree->tree_node_count = g_iKdTree_Node_Count;
	c_object->kd_tree->tri_offset_list = g_pKdTree_TriOffset_Array;
	c_object->kd_tree->tri_offset_count = g_iKdTree_TriOffset_Count;
	c_object->kd_tree->tri_accel_list = pTriAcc;
	fprintf(stdout, "\n> Done!\n\n");
	return 1;
}

void dump_kd_tree_for_composite_object(CompositeObject *c_object, const char *filename,
									   int dump_format, const char *filename_igeom) {
	// Dump the kd-tree "c_object->kd_tree" into the file "filename" in "dump_format" type:
	//    dump_format == KD_TREE_DUMP_IN_ASCII  --> ASCII format
	//    dump_format == KD_TREE_DUMP_IN_BINARY --> in Binary format	
	// Dump the indexed geometry "c_object->extended_vertices" into the file "filename_igeom" in binary format

	FILE* fp;
	int i;
	fprintf(stdout, "> Dumping kd-tree and i-geometry to file: \n          kd-tree =%s\n          i-geometry = %s\n\n", filename, filename_igeom);
	if (dump_format == KD_TREE_DUMP_IN_BINARY) {
		fprintf(stdout, "   * Kd-tree format: BINARY\n");

		if ((fp = fopen(filename, "wb")) == NULL) {
			fprintf(stderr, "d_k_t_f_c_o: (Error) cannot open the file %s...\n", filename);
			exit(-1);
		}
		
		// Dump kd-tree node info
		int nTreeNodeCount = c_object->kd_tree->tree_node_count;
		fwrite(&nTreeNodeCount, 4, 1, fp);
		fwrite(c_object->kd_tree->tree, 8, nTreeNodeCount, fp);

		// Dump kd-tree triangle offset in leafnode
		int nTriOffCount = c_object->kd_tree->tri_offset_count;
		fwrite(&nTriOffCount, 4, 1, fp);
	 	fwrite(c_object->kd_tree->tri_offset_list, 4, nTriOffCount, fp);

	} 
	else {
		fprintf(stdout, "   * Kd-tree format: ASCII\n");

		if ((fp = fopen(filename, "w" )) == NULL) {
			fprintf(stderr, "d_k_t_f_c_o: (Error) cannot open the file %s...\n", filename);
			exit(-1);
		}
		const char axis_label[3][2] = { "X", "Y", "Z" };

		// Dump kd-tree node info
		int nTreeNodeCount = c_object->kd_tree->tree_node_count;
		fprintf(fp, "n%d\n", nTreeNodeCount);

		KdTreeNode* node = c_object->kd_tree->tree;
		for (i = 0; i < nTreeNodeCount; i++, node++) {
			if (IS_LEAF(*node) == 0) {
				int   left_child = FIRST_CHILD_OFFSET(*node);
				int   split_axis = SPLIT_AXIS(*node);
				float split_pos  = SPLIT_POS(*node);
				fprintf(fp, "%s|%f %d<>%d\n", axis_label[split_axis], split_pos, left_child, left_child+1);
			} else {
				int offset   = OBJECTLIST_OFFSET(*node);
				int leaftris = OBJECT_SIZE(*node);
				fprintf(fp, "L cnt:%d off:%d\n", leaftris, offset);
			}
		}

		// Dump kd-tree triangle offset in leafnode
		int nTriOffCount = c_object->kd_tree->tri_offset_count;
		fprintf(fp, "o%d\n", nTriOffCount);

		for (i = 0; i < nTriOffCount; i++) {
			int triID = c_object->kd_tree->tri_offset_list[i];
			fprintf(fp, "%d\n", triID);
		}

	}
	fclose(fp);

	if ((fp = fopen(filename_igeom, "wb")) == NULL) {
		fprintf(stderr, "d_k_t_f_c_o: (Error) cannot open the file %s...\n", filename_igeom);
		exit(-1);
	}
	fprintf(stdout, "   * I-geometry format: BINARY\n");

	fwrite(&(c_object->n_triangles), sizeof(int), 1, fp);
	fwrite(c_object->AABB, sizeof(float), 6, fp);
	fwrite(c_object->extended_vertices, sizeof(ExtendedVertex), 3 * c_object->n_triangles, fp);
	fclose(fp);

	fprintf(stdout, "\n> Done!\n\n");
}

int read_kd_tree_from_file(CompositeObject *c_object, const char *filename, int dump_format) {
	// Return 1 if a kd-tree was read successfully, or 0 otherwise.
	// Input: "filename" 
	// Output: "kd_tree"

	FILE *fp;
	int i;

	if (dump_format == KD_TREE_DUMP_IN_BINARY) {
		fp = fopen( filename, "rb" );
		if (fp == NULL) {
			fprintf(stderr, "%s kd-tree File Load Error!\n", filename );
			return 0;
		}

		// Load kd-tree node info
		int nTreeNodeCount = 0;
		fread( &nTreeNodeCount, 4, 1, fp );
		g_iKdTree_Node_CountAlloc = g_iKdTree_Node_Count = nTreeNodeCount;
		g_pKdTree_Node_Array = new KdTreeNode[ g_iKdTree_Node_CountAlloc ];
		KdTreeNode* node = &g_pKdTree_Node_Array[0];
		fread( g_pKdTree_Node_Array, 8, nTreeNodeCount, fp );

		// Load kd-tree node info
		int nTriOffCount = 0;
		fread( &nTriOffCount, 4, 1, fp );
		g_iKdTree_TriOffset_CountAlloc = g_iKdTree_TriOffset_Count = nTriOffCount;
		g_pKdTree_TriOffset_Array = new unsigned int [ g_iKdTree_TriOffset_CountAlloc ];
		fread( g_pKdTree_TriOffset_Array, 4, nTriOffCount, fp );

	} else {

		fp = fopen( filename, "r" );
		if (fp == NULL) {
			fprintf(stderr, "%s kd-tree File Load Error!\n", filename );
			return 0;
		}

		char data[1024] = { 0x00, };
		char data_a[64], data_b[64];

		// Load kd-tree node info
		int nTreeNodeCount = 0;
		fgets( data, 1024, fp );
		sscanf(data+1,"%d", &nTreeNodeCount);
		g_iKdTree_Node_CountAlloc = g_iKdTree_Node_Count = nTreeNodeCount;
		g_pKdTree_Node_Array = new KdTreeNode[ g_iKdTree_Node_CountAlloc ];

		KdTreeNode* n = (KdTreeNode*)g_pKdTree_Node_Array;
		for ( i = 0; i < nTreeNodeCount; i++, n++ )	{
			fgets( data, 1024, fp );
			if (data[0] == 'L') {
				int offset;
				int leaftris;
				sscanf(data+2, "%s %s", data_a, data_b);
				sscanf(data_a+4, "%d", &leaftris);
				sscanf(data_b+4, "%d", &offset);
				setLeafNode( n, leaftris, offset );
			} else {
				int   left_child;
				int   split_axis;
				float split_pos;
				split_axis = (data[0] - 'X');
				sscanf(data+2, "%f %d", &split_pos, &left_child);
				setInnerNode( n, split_axis, left_child, split_pos );
			}
		}

		// Load kd-tree triangle offset in leafnode
		int nTriOffCount;
		fgets( data, 1024, fp );
		sscanf(data+1, "%d", &nTriOffCount);
	
		g_iKdTree_TriOffset_CountAlloc = g_iKdTree_TriOffset_Count = nTriOffCount;
		g_pKdTree_TriOffset_Array = new unsigned int [ g_iKdTree_TriOffset_CountAlloc ];

		for ( i = 0; i < nTriOffCount; i++ ) {
			fgets( data, 1024, fp );
			int triID;
			sscanf(data, "%d", &triID);
			g_pKdTree_TriOffset_Array[i] = triID;
		}

	}

	// build triangle acceleration
	TriAccel *pTriAcc = NULL;
	pTriAcc = (TriAccel*)_aligned_malloc(c_object->n_triangles * sizeof(TriAccel), 16);
	build_TriAccList(c_object, pTriAcc);

	c_object->kd_tree = new KdTree;
	c_object->kd_tree->tree = g_pKdTree_Node_Array;
	c_object->kd_tree->tree_node_count = g_iKdTree_Node_Count;
	c_object->kd_tree->tri_offset_list = g_pKdTree_TriOffset_Array;
	c_object->kd_tree->tri_offset_count = g_iKdTree_TriOffset_Count;
	c_object->kd_tree->tri_accel_list = pTriAcc;
	printf("->n_triangles: %d\n", c_object->n_triangles);
	if (pTriAcc == NULL) printf("triaccNULL\n");

	fprintf(stdout, "Reading Kd-tree is completed.\n");

	return 1;
}

int find_ray_object_intersection(KdTree *kd_tree, Ray *ray, float *point, float *normal) {
	// Find the first hit between "kd-tree" and "ray"
	// Input: "kd_tree" & "ray"
	// Output: "point" & "normal" & 
	//          returns "material_ID" value of Extended Vertex if an intersection was found
	//		              -1                                   if no intersection was found

	Hit is;

	ray->Depth = 0;
	InitRay(0, ray);
	TraceRay(0, kd_tree, &is);

	if (is.tacc == 0) {
		return -1;
	} else {
		point[0] = is.pf[0];
		point[1] = is.pf[1];
		point[2] = is.pf[2];
		normal[0] = is.nf[0];
		normal[1] = is.nf[1];
		normal[2] = is.nf[2];
		return is.material_ID;
	}
}