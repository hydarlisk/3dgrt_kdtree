#pragma once

#include <vector>

using namespace std;

/**
 *	OBJECT File Format 을 Load 하는 클래스.
 *	삼각형으로만 구성된 face 를 로드할수 있다.
 *
 *	by graphicsian.
 */

/**
 *	object 파일의 material 정보를 표현.
 */
class GOBJMaterial {

public:
	char *pName;				//	material name

	int illum;
	float kr;					//	SGRTx2 에서 확장. raytracing 시 reflection 비율.
	float ka[3];
	float kd[3];
	float ks[3];
	float ke[3];
	float ns;
	float ni;
	float d;
	float tr;
	float tf[3];
	float sharpness;
	float density;

	char *pMapKaPath;
	char *pMapKdPath;
	char *pMapKsPath;
	char *pMapKePath;
	char *pMapNsPath;
	char *pMapDPath;
	char *pMapBumpPath;
	char *pBumpPath;

	GOBJMaterial() {

		kr = 0.0f;
		ka[0] = 0.0f; ka[1] = 0.0f; ka[2] = 0.0f;
		kd[0] = 0.0f; kd[1] = 0.0f; kd[2] = 0.0f;
		ks[0] = 0.0f; ks[1] = 0.0f; ks[2] = 0.0f;
		ke[0] = 0.0f; ke[1] = 0.0f; ke[2] = 0.0f;
		ns = 1.0f; ni = 1.0f; d = 1.0f; tr = 1.0f;
		tf[0] = 1.0f; tf[1] = 1.0f; tf[2] = 1.0f;
		sharpness = 1.0f;
		density = 0.0f;

		pName = NULL; 
		pMapKaPath = NULL; 
		pMapKdPath = NULL;
		pMapKsPath = NULL;
		pMapKePath = NULL;
		pMapNsPath = NULL;
		pMapDPath = NULL;
		pMapBumpPath = NULL;
		pBumpPath = NULL;
	}

	~GOBJMaterial() {
		if ( pName ) free( pName );
		if ( pMapKaPath ) free( pMapKaPath );
		if ( pMapKdPath ) free( pMapKdPath );
		if ( pMapKsPath ) free( pMapKsPath );
		if ( pMapKePath ) free( pMapKePath );
		if ( pMapNsPath ) free( pMapNsPath );
		if ( pMapDPath ) free( pMapDPath );
		if ( pMapBumpPath ) free( pMapBumpPath );
		if ( pBumpPath ) free( pBumpPath );
	}
};

class GOBJVector {
public:
	float x, y, z;
};

/**
 *	obj file 에 나타나는
 *	vertex, texture, normal 등 3개 data set 를 표현.
 */
class GOBJGeometryData {

public:
	vector<GOBJVector*> m_VertexList;
	vector<GOBJVector*> m_NormalList;
	vector<GOBJVector*> m_TextureCoordList;

	~GOBJGeometryData() {
		for ( int i = 0; i < (int)m_VertexList.size(); ++i ) {
			delete m_VertexList[ i ];
		}
		for ( int i = 0; i < (int)m_NormalList.size(); ++i ) {
			delete m_NormalList[ i ];
		}
		for ( int i = 0; i < (int)m_TextureCoordList.size(); ++i ) {
			delete m_TextureCoordList[ i ];
		}
	}

	void addVertex( float x, float y, float z ) {
		GOBJVector *data = new GOBJVector();
		data->x = x; data->y = y; data->z = z;
		m_VertexList.push_back( data );
	}

	void addNormal( float x, float y, float z ) {
		GOBJVector *data = new GOBJVector();
		data->x = x; data->y = y; data->z = z;
		m_NormalList.push_back( data );
	}

	void addTextureCoord( float x, float y, float z ) {
		GOBJVector *data = new GOBJVector();
		data->x = x; data->y = y; data->z = z;
		m_TextureCoordList.push_back( data );
	}
};

/**
 *	하나의 삼각형 face 를 위한 vertex, normal, texture 좌표데이터 index.
 *	실제데이터는 GOBJGeometryData 에 있음.
 */
class GOBJTriangleFace {
public:
	int v0, n0, t0;
	int v1, n1, t1;
	int v2, n2, t2;

	GOBJTriangleFace() {
		v0 = -1; n0 = -1; t0 = -1;
		v1 = -1; n1 = -1; t1 = -1;
		v2 = -1; n2 = -1; t2 = -1;
	}
};

/**
 *	Object 정보.
 */
class GOBJObject {
	
public:
	char *pName;
	GOBJMaterial *pMaterial;
	bool m_bNormal, m_bTextureCoord;

	vector<GOBJTriangleFace*> m_FaceList;

	GOBJObject() {
		pName = NULL;
		pMaterial = NULL;
		m_bNormal = false;
		m_bTextureCoord = false;
	}

	~GOBJObject() {
		if ( pName ) 
			free( pName );
		for ( int i = 0; i < (int) m_FaceList.size(); ++i ) {
			delete m_FaceList[i];
		}
		m_FaceList.clear();
	}

	void addFace( GOBJTriangleFace *face ) {
		m_FaceList.push_back( face );
	}

	bool isExistNormal() {
		return m_bNormal;
	}

	bool isExistTextureCoord() {
		return m_bTextureCoord;
	}

};

class GOBJFileLoader
{
private:
	vector<GOBJMaterial*> m_MaterialList;
	vector<GOBJObject*> m_ObjectList;
	GOBJGeometryData m_GeometryData;

public:
	GOBJFileLoader(void);
	~GOBJFileLoader(void);

	bool load( const char* filename );

	int getObjectCount();
	GOBJObject* getObject( int index );
	void printObjectInfo();
	GOBJGeometryData *getGeometryData();

private:
	bool loadMaterialFile( const char* filename );
	bool readLine( char *data, int size, FILE *fp );
	bool readMaterialInfo( const char* matname, FILE *fp );

	GOBJMaterial *getMaterial( const char* name );
	bool parsingFaceData( char* str, int *result );
	char *createPathFromParam( vector<char*> &valueList );
};
