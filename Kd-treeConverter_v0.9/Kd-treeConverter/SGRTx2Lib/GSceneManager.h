#pragma once

#include "GBase.h"
#include "GScene.h"
#include "GPolygonObject.h"
#include "GRaySetLight.h"

/**
 *	파일 관리자.
 *	by graphicsian
 */
typedef struct _vertex_info {
	float vertex[3];
	float normal[3];
	float texcoord[3];
} GVertexInfo;

/**
 *	OBJ Data Format 에서 NORMAL 을 합해서
 *	새로운 NORMAL 정보를 만들때 사용할 임시 구조체.
 */
typedef struct _obj_normal_weight_ {
	GVector normal;
	int index;
} GOBJNormalWeight;

class  GSceneManager
{
private:
	char m_SceneBasePath[ 1024 ];

public:
	GSceneManager(void);
	~GSceneManager(void);

public:
	/**
	 *	Data 파일을 로드해서 Scene 을 만들어서 리턴한다.
	 */
	GError loadScene( const char* szFileName, GScene** ppScene );

	/**
	 *	현재 Scene 을 file 로 저장한다.
	 */
	GError saveScene( const char* szFileName, GScene* pScene );

private:
	GError load914File( const char* szFileName, GScene* ppScene );
	GError loadOBJFile( const char* szFileName, GScene* ppScene );
	GError loadBinaryFile( const char* szFileName, GScene* ppScene );

	bool readLine( char *buffer, int size, FILE *fp );

	bool readVersion( char *buffer, int size, FILE *fp, GScene* pScene );
	bool readSceneInfo( char *buffer, int size, FILE *fp, GScene* pScene );
	bool readCamera( char *buffer, int size, FILE *fp, GScene* pScene );
	bool readGlobal( char *buffer, int size, FILE *fp, GScene* pScene );
	bool readLight( char *buffer, int size, FILE *fp, GScene* pScene );

	bool readTriangleObject( char *buffer, int size, FILE *fp, GScene* pScene );

	bool readVirtualLight( char *data, int size, FILE *fp, GScene* pScene );
	bool readPointLight( char *data, int size, FILE *fp, GScene* pScene );
	bool readRaySetLight( char *data, int size, FILE *fp, GScene* pScene );
	bool loadRaySetDataFile( GRaySetLight *pRaySetLight, const char *datafile );

	bool readObjectVertex( char *data, int size, FILE *fp, GPolygonObject* pObject, int count, bool bUv );
	bool readObjectIndex( char *data, int size, FILE *fp, GPolygonObject* pObject, int count, int baseIndex );

	bool loadMeshData( GPolygonObject* pObject, const char* filename );
	bool readPhotonMappingOption( char *data, int size, FILE *fp, GScene* pScene );

	bool loadDatFile( const char*szFileName, GPolygonObject *pObject );

	bool readImportObject( char *data, int size, FILE *fp, GScene* pScene );
};




