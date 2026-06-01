#pragma once


//#define SOFT_SPLIT_THRESHOLD 128				// kd-tree 강제분할(완화)
#define SOFT_SPLIT_THRESHOLD2 256				// kd-tree 강제분할(완화)
//#define LESS_TRI false

#define BLEND_SELECT false

#define SAH_OPACITY 0
#define CLIP_AREA false							// 부모 노드의 AABB로 삼각형 면적 clip
#define SAH_MAXIMIZE false
//0 - P_s * N_s																									//235
//1 - P_s * SUM(sigma)																							//227
#define TRANSPARENCY false
//2 - P_s * SUM(sigma(i) * area(i))																				//187
//2 - P_s * SUM(sigma(i) * area_clip_parent(i))																	//154
//21 - P_s * SUM(sigma(i) * area(i) * OPACITY_PENALTY)															//182
//21 - P_s * SUM(sigma(i) * area_clip_parent(i) * OPACITY_PENALTY)												//71
#define OPACITY_PENALTY 10.0f					//for SAH 21
//22 - P_s * ( SUM(sigma(i) * area(i)) + HYBRID_BETA * N_s)														//196
//22 - P_s * ( SUM(sigma(i) * area_clip_parent(i)) + HYBRID_BETA * N_s)											//
#define HYBRID_BETA 0.3f						//for SAH 22, 23
//23 - P_s * ( (1-HYBRID_BETA) * SUM(sigma(i) * area(i))_normalize + HYBRID_BETA * N_s_normalize)				//194
//23 - P_s * ( (1-HYBRID_BETA) * SUM(sigma(i) * area_clip_parent(i))_normalize + HYBRID_BETA * N_s_normalize)	//
//3 - P_s * SUM(sigma(i) * area(i) / MAX(area(V_s))																//4
//3 - P_s * SUM(sigma(i) * area_clip_parent(i) / MAX(area(V_s))													//
//4 - P_s * SUM(sigma(i) * area(i) / MAX(area(V))																//208
//4 - P_s * SUM(sigma(i) * area_clip_parent(i) / MAX(area(V))													//
//5 - SUM(sigma(i) * area(i) / MAX(area(V))																		//
//5 - SUM(sigma(i) * area_clip_parent(i) / MAX(area_clip_parent_in_V))											//
//6 - P_s * SUM(sigma(i) * area(i_real)): (실패)
//7 - P_s * SUM(sigma(i) * (A_tri_leaf_AABB/A_leaf_AABB)): (실패)
//8 - P_s * SUM(sigma(i) * (A_tri_clip_parent_AABB/A_parent_AABB))												//
//81 - P_s * SUM(sigma(i) * area(i) * (A_tri_clip_parent_AABB/A_parent_AABB)): TODO								//
//81 - P_s * SUM(sigma(i) * area_clip_parent(i) * (A_tri_clip_parent_AABB/A_parent_AABB)): TODO					//
//9 - P_s * SUM(sigma(i) * (A_tri_clip_parent_AABB/A_tri_origin_AABB)): TODO
//91 - P_s * SUM(sigma(i) * area(i) * (A_tri_clip_parent_AABB/A_tri_origin_AABB)): TODO
//91 - P_s * SUM(sigma(i) * area_clip_parent(i) * (A_tri_clip_parent_AABB/A_tri_origin_AABB)): TODO
//10 - P_s * ( SUM(sigma(i)) / Volume(leaf_AABB) ): (실패)
//101 - ( SUM(sigma(i)) / Volume(leaf_AABB) ): (실패)
//1000 - 일정 레벨까지 1, 이후 0
#define HYBRID_SAH_DEPTH_THRESHOLD 5			//for SAH 1000, 2000
//1001 - 일정 갯수까지 1, 이후 0
#define HYBRID_SAH_TRIANGLE_THRESHOLD 10000		//for SAH 1001, 2001
//20 - 4V_s/S_s * N_s
//201 - 4V_s/S_s * ( SUM(sigma(i)) )
//in 20, 201 P_s * ()
#define MUL_PROP false
//2000 - 일정 레벨까지 20, 이후 0
//2001 - 일정 갯수까지 20, 이후 0
//2010 - 일정 레벨까지 201, 이후 0
//2011 - 일정 갯수까지 201, 이후 0