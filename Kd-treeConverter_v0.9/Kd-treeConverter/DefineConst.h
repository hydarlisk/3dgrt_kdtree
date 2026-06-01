#pragma once

/* Gaussian Configs */
#define KERNEL_MIN_RESPONSE 0.0113f
#define KERNEL_DEGREE 4.0f
#define SPH_EVAL_DEGREE 3
#define GAUSSIAN_DEGREE 4
#define OPACITY_THRESHOLD 0.95f					// 충족할때까지 kd-tree 탐색
#define MAX_HITS 256								// leaf node에서 blending을 위한 최대 sort 크기

#define ADAPTIVE_KERNEL_CLAMPING 1
#define SIGMA_THRESHOLD_MODE 1
#define ADAPTIVE_MESH false

/* camera */
#define CAM_MOVE_SPEED 0.5
#define CAM_ROT_SPEED 0.1
#define CAM_MOVE_SHIFT 0.01


#define NEAR_PLANE 0.005f
#define FAR_PLANE 20.00f

#define EPSILON 0.00001f


/************* JS add ************/
/* data format */
#define JS_BIN true
#define KDT_VERSION 1

#define OCCLUDE_MIN_OPACITY 1
#define OCCLUDE_MIN_OPACITY_TRI 1

/* rendering */
#define UPLOAD_INV_SCALE 1
#define UPLOAD_INV_KSCALE 1
#define VOLUME_ISECT 1
////////////////////////////////////


/* stack type */
#define SHORT_STACK 0
#define HYBRID_STACK 1
#define GLOBAL_STACK 2

#define MAX_GLOBAL_STACK_DEPTH 64

/* primitive type */
#define TRI 0
#define ELLIPSOID 1
#define ELLIPSOID_BY_TRI 2

/* assets */
#define HOTDOG 0
#define BICYCLE 1
#define ROOM 2
#define LEGO 3