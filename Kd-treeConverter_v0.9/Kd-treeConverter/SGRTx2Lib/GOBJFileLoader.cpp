#include "GOBJFileLoader.h"
#include "GUtil.h"

GOBJFileLoader::GOBJFileLoader(void)
{
}

GOBJFileLoader::~GOBJFileLoader(void)
{
	for ( int i = 0; i < (int) m_MaterialList.size(); ++i ) {
		delete m_MaterialList[ i ];
	}
	m_MaterialList.clear();

	for ( int i = 0; i < (int) m_ObjectList.size(); ++i ) {
		delete m_ObjectList[ i ];
	}
	m_ObjectList.clear();
}

bool GOBJFileLoader::load( const char *filename )
{
	char data[2048] = { 0x00, };
	vector<char*> valueList;
	FILE *fp = fopen( filename, "rt" );
	GOBJObject *pObject = NULL;
	bool bSuccess = true, bFirstFace = true;
	int nNoName_ObjIdx = 0;

	GLogManager::logging( LOG_INFO, " -> Loading Object File : %s", filename );

	if ( fp == NULL ) {
		GLogManager::logging( LOG_FATAL, " -> Object File Not Found : %s", filename );
		return false;
	}

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) {
			break;
		}

		if ( data[ 0 ] == '#' || data[ 0 ] == 0x00 ) continue;
		GUtil::getValueList( data, valueList );

		if ( _stricmp( valueList[0], "mtllib" ) == 0 ) {

			if ( (int) valueList.size() != 2 ) { 
				GLogManager::logging( LOG_ERROR, "load material file error" );
				bSuccess = false; break; 
			}

			char path[2048] = { 0x00, }, temp[2048] = { 0x00, };
			GUtil::getParentPath( filename, temp );
			sprintf( path, "%s%c%s", temp, FILE_SEPARATOR, valueList[1] );

			if ( !loadMaterialFile( path ) ) {
				GLogManager::logging( LOG_ERROR, "load material file error" );
				bSuccess = false; break;
			}

		} else if ( _stricmp( valueList[0], "g" ) == 0 ) {
			
			/** object 이름이 있을때만 parsing */
			if ( (int) valueList.size() == 2 ) {
				GLogManager::logging( LOG_DEBUG, " -> OBJ Load Object : %s", valueList[ 1 ] );
				pObject = new GOBJObject();
				pObject->pName = GUtil::copyString( valueList[ 1 ] );
				bFirstFace = true;
				m_ObjectList.push_back( pObject );
			}

		} else if ( _stricmp( valueList[0], "v" ) == 0 ) {
			
			if ( (int) valueList.size() != 4 ) { bSuccess = false; break; }
			m_GeometryData.addVertex( (float) atof( valueList[ 1 ] ),
									  (float) atof( valueList[ 2 ] ),
									  (float) atof( valueList[ 3 ] ) );

		} else if ( _stricmp( valueList[0], "vt" ) == 0 ) {

			if ( (int) valueList.size() != 3 && (int) valueList.size() != 4 ) { bSuccess = false; break; }
			if ( (int) valueList.size() == 3 ) {
				m_GeometryData.addTextureCoord( (float) atof( valueList[ 1 ] ),
										  (float) atof( valueList[ 2 ] ),
										  0.0f );
			}
			if ( (int) valueList.size() == 4 ) {
				m_GeometryData.addTextureCoord( (float) atof( valueList[ 1 ] ),
										  (float) atof( valueList[ 2 ] ),
										  (float) atof( valueList[ 3 ] ) );
			}

		} else if ( _stricmp( valueList[0], "vn" ) == 0 ) {

			if ( (int) valueList.size() != 4 ) { bSuccess = false; break; }
			m_GeometryData.addNormal( (float) atof( valueList[ 1 ] ),
									  (float) atof( valueList[ 2 ] ),
									  (float) atof( valueList[ 3 ] ) );

		} else if ( _stricmp( valueList[0], "usemtl" ) == 0 ) {

			if ( (int) valueList.size() != 2 ) { 
				GLogManager::logging( LOG_ERROR, "usemtl error." );
				bSuccess = false; break; 
			}
			if (pObject == NULL || pObject->pMaterial != NULL) {
				pObject = new GOBJObject();
				char s[32]; sprintf(s, "sgrtx2_%d", nNoName_ObjIdx++);
				pObject->pName = GUtil::copyString( s );
				bFirstFace = true;
				m_ObjectList.push_back( pObject );
			}

			pObject->pMaterial = getMaterial( valueList[1] );

		} else if( _stricmp( valueList[0], "f" ) == 0 ) {

			/** 
			 * 삼각형 or 사각형만 불러들인다.
			 */

			if ( (int) valueList.size() != 4 && (int) valueList.size() != 5 ) { 
				GLogManager::logging( LOG_ERROR, "Face must be triangle or quad. (count=%d)", valueList.size() );
				bSuccess = false; break; 
			}
			
			/**
			 *	삼각형일때.
			 */
			if ( (int) valueList.size() == 4 ) {
				/** face 데이터 parsing :  vertex/texture/normal */
				/** 
				 *	데이터상에서는 index 가 1 부터 시작하지만 
				 *	c 배열상의 index 는 0부터 시작하므로 -1 한다.
				 */
				GOBJTriangleFace *pFace = new GOBJTriangleFace();
				int indexResult[3];

				parsingFaceData( valueList[ 1 ], indexResult );
				pFace->v0 = indexResult[ 0 ] - 1;
				pFace->t0 = indexResult[ 1 ] - 1;
				pFace->n0 = indexResult[ 2 ] - 1;

				parsingFaceData( valueList[ 2 ], indexResult );
				pFace->v1 = indexResult[ 0 ] - 1;
				pFace->t1 = indexResult[ 1 ] - 1;
				pFace->n1 = indexResult[ 2 ] - 1;

				parsingFaceData( valueList[ 3 ], indexResult );
				pFace->v2 = indexResult[ 0 ] - 1;
				pFace->t2 = indexResult[ 1 ] - 1;
				pFace->n2 = indexResult[ 2 ] - 1;

				/** 첫 face 값을 체크해서 normal 과 texture coord 가 있는지를 설정 */
				if ( bFirstFace ) {
					bFirstFace = false;
					if ( pFace->n0 >= 0 && pFace->n1 >= 0 && pFace->n2 >= 0 )
						pObject->m_bNormal = true;
					if ( pFace->t0 >= 0 && pFace->t1 >= 0 && pFace->t2 >= 0 )
						pObject->m_bTextureCoord = true;
				}
				pObject->addFace( pFace );
			}

			/**
			 *	사각형일때. 삼각형 2개로 만든다.
			 */
			if ( (int) valueList.size() == 5 ) {

				/** face 데이터 parsing :  vertex/texture/normal */
				/** 
				 *	데이터상에서는 index 가 1 부터 시작하지만 
				 *	c 배열상의 index 는 0부터 시작하므로 -1 한다.
				 */
				GOBJTriangleFace *pFace = new GOBJTriangleFace();
				int indexResult[3];

				/** parsingFaceData 후에는 인자로 주어진 string 이 변하므로 백업 */
				char *p1 = GUtil::copyString( valueList[ 1 ] );
				char *p2 = GUtil::copyString( valueList[ 2 ] );
				char *p3 = GUtil::copyString( valueList[ 3 ] );
				char *p4 = GUtil::copyString( valueList[ 4 ] );

				parsingFaceData( valueList[ 1 ], indexResult );
				pFace->v0 = indexResult[ 0 ] - 1;
				pFace->t0 = indexResult[ 1 ] - 1;
				pFace->n0 = indexResult[ 2 ] - 1;

				parsingFaceData( valueList[ 2 ], indexResult );
				pFace->v1 = indexResult[ 0 ] - 1;
				pFace->t1 = indexResult[ 1 ] - 1;
				pFace->n1 = indexResult[ 2 ] - 1;

				parsingFaceData( valueList[ 3 ], indexResult );
				pFace->v2 = indexResult[ 0 ] - 1;
				pFace->t2 = indexResult[ 1 ] - 1;
				pFace->n2 = indexResult[ 2 ] - 1;

				/** 첫 face 값을 체크해서 normal 과 texture coord 가 있는지를 설정 */
				if ( bFirstFace ) {
					bFirstFace = false;
					if ( pFace->n0 >= 0 && pFace->n1 >= 0 && pFace->n2 >= 0 )
						pObject->m_bNormal = true;
					if ( pFace->t0 >= 0 && pFace->t1 >= 0 && pFace->t2 >= 0 )
						pObject->m_bTextureCoord = true;
				}
				pObject->addFace( pFace );

				pFace = new GOBJTriangleFace();

				parsingFaceData( p3, indexResult );
				pFace->v0 = indexResult[ 0 ] - 1;
				pFace->t0 = indexResult[ 1 ] - 1;
				pFace->n0 = indexResult[ 2 ] - 1;

				parsingFaceData( p4, indexResult );
				pFace->v1 = indexResult[ 0 ] - 1;
				pFace->t1 = indexResult[ 1 ] - 1;
				pFace->n1 = indexResult[ 2 ] - 1;

				parsingFaceData( p1, indexResult );
				pFace->v2 = indexResult[ 0 ] - 1;
				pFace->t2 = indexResult[ 1 ] - 1;
				pFace->n2 = indexResult[ 2 ] - 1;

				pObject->addFace( pFace );

				free( p1 ); free( p2 ); free( p3 ); free( p4 );
			}			

		}


	}

	GLogManager::logging( LOG_INFO, " -> Loaded Object File : %s", filename );

	fclose( fp );

	return bSuccess;
}

/**
 *	face 데이터를 parsing 해서, index 정보를 리턴한다.
 *	하나의 face 정보는 vertex/texture/normal index 로 구성되어 있다.
 *	해당 값이 없다면 -1 로 지정.
 */
bool GOBJFileLoader::parsingFaceData( char* str, int *result )
{
	int length = (int) strlen( str );
	int dataCount = 0;
	vector<char*> valueList;

	result[ 0 ] = 0; result[ 1 ] = 0; result[ 2 ] = 0;

	GUtil::getValueList( str, valueList, "/" );
	dataCount = (int) valueList.size();

	switch( dataCount ) {
		/** vertex only */
		case 1:
			result[ 0 ] = atoi( valueList[ 0 ] );
			result[ 1 ] = 0;
			result[ 2 ] = 0;
			return true;
		/** vertex//normal */
		case 2:
			result[ 0 ] = atoi( valueList[ 0 ] );
			result[ 1 ] = 0;
			result[ 2 ] = atoi( valueList[ 1 ] );
			return true;
		/** vertex/texture/normal */
		case 3:
			result[ 0 ] = atoi( valueList[ 0 ] );
			result[ 1 ] = atoi( valueList[ 1 ] );
			result[ 2 ] = atoi( valueList[ 2 ] );
			return true;
	}

	return false;
}

/**
 *	Load Material File.
 */
bool GOBJFileLoader::loadMaterialFile( const char *filename )
{
	char data[2048] = { 0x00, };
	vector<char*> valueList;
	bool bSuccess = true;

	GLogManager::logging( LOG_INFO, " -> Loading Material File : %s", filename );

	FILE *fp = fopen( filename, "rt" );

	if ( fp == NULL ) {
		GLogManager::logging( LOG_FATAL, " -> Material File Not Found: %s", filename );
		return false;
	}

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) {
			break;
		}

		/** 
		 *	주석이나 공백일때.
		 */
		if ( data[ 0 ] == '#' || data[ 0 ] == 0x00 ) continue;

		/** 새로운 material 정보 시작 */
		if ( _strnicmp( data, "newmtl", 6 ) == 0 ) {
			GUtil::getValueList( data, valueList );
			/** newmtl 뒤에 이름이 있어야 함. */
			if ( valueList.size() != 2 ) {
				bSuccess = false;
				break;
			}
			readMaterialInfo( valueList[1], fp );
		}
	}

	fclose( fp );

	GLogManager::logging( LOG_INFO, " -> Loaded Material File : %s", filename );

	return bSuccess;
}

bool GOBJFileLoader::readMaterialInfo( const char* matname, FILE *fp )
{
	char data[2048] = { 0x00, };
	vector<char*> valueList;
	bool bSuccess = true;

	GOBJMaterial *pMaterial = new GOBJMaterial();
	pMaterial->pName = GUtil::copyString( matname );

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) {
			break;
		}

		/** 
		 *	주석일때.
		 */
		if ( data[ 0 ] == '#' ) continue;

		/**
		 *	공백이 나타나면 material 정보 끝.
		 */
		if ( data[ 0 ] == 0x00 ) {
			break;
		}

		GUtil::getValueList( data, valueList );

		if ( _stricmp( valueList[0], "NS" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->ns = (float) atof( valueList[ 1 ] );
		} else if ( _stricmp( valueList[0], "NI" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->ni = (float) atof( valueList[ 1 ] );
		} else if ( _stricmp( valueList[0], "d" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->d = (float) atof( valueList[ 1 ] );
		} else if ( _stricmp( valueList[0], "Tr" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->tr = (float) atof( valueList[ 1 ] );
		} else if ( _stricmp( valueList[0], "Tf" ) == 0 ) {
			if ( valueList.size() != 4 ) { bSuccess = false; break;	}
			pMaterial->tf[0] = (float) atof( valueList[ 1 ] );
			pMaterial->tf[1] = (float) atof( valueList[ 2 ] );
			pMaterial->tf[2] = (float) atof( valueList[ 3 ] );
		} else if ( _stricmp( valueList[0], "illum" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->illum = atoi( valueList[ 1 ] );
		} else if ( _stricmp( valueList[0], "kr" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->kr = (float) atof( valueList[ 1 ] );
		} else if ( _stricmp( valueList[0], "Ka" ) == 0 ) {
			if ( valueList.size() != 4 ) { bSuccess = false; break;	}
			pMaterial->ka[0] = (float) atof( valueList[ 1 ] );
			pMaterial->ka[1] = (float) atof( valueList[ 2 ] );
			pMaterial->ka[2] = (float) atof( valueList[ 3 ] );
		} else if ( _stricmp( valueList[0], "Kd" ) == 0 ) {
			if ( valueList.size() != 4 ) { bSuccess = false; break;	}
			pMaterial->kd[0] = (float) atof( valueList[ 1 ] );
			pMaterial->kd[1] = (float) atof( valueList[ 2 ] );
			pMaterial->kd[2] = (float) atof( valueList[ 3 ] );
		} else if ( _stricmp( valueList[0], "Ks" ) == 0 ) {
			if ( valueList.size() != 4 ) { bSuccess = false; break;	}
			pMaterial->ks[0] = (float) atof( valueList[ 1 ] );
			pMaterial->ks[1] = (float) atof( valueList[ 2 ] );
			pMaterial->ks[2] = (float) atof( valueList[ 3 ] );
		} else if ( _stricmp( valueList[0], "Ke" ) == 0 ) {
			if ( valueList.size() != 4 ) { bSuccess = false; break;	}
			pMaterial->ke[0] = (float) atof( valueList[ 1 ] );
			pMaterial->ke[1] = (float) atof( valueList[ 2 ] );
			pMaterial->ke[2] = (float) atof( valueList[ 3 ] );
		} else if ( _stricmp( valueList[0], "sharpness" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->sharpness = (float) atof( valueList[ 1 ] );
		} else if ( _stricmp( valueList[0], "density" ) == 0 ) {
			if ( valueList.size() != 2 ) { bSuccess = false; break;	}
			pMaterial->density = (float) atof( valueList[ 1 ] );

		} else if ( _stricmp( valueList[0], "map_Ka" ) == 0 ) {
			/** 
			 *	경로명에 공백이 있을수도 있으니 인자가 1 이상있으면 
			 *	다 연결해서 하나의 문자열로 만든다. 
			 */
			pMaterial->pMapKaPath = createPathFromParam( valueList );

		} else if ( _stricmp( valueList[0], "map_Kd" ) == 0 ) {
			pMaterial->pMapKdPath = createPathFromParam( valueList );
		} else if ( _stricmp( valueList[0], "map_Ks" ) == 0 ) {
			pMaterial->pMapKsPath = createPathFromParam( valueList );
		} else if ( _stricmp( valueList[0], "map_Ke" ) == 0 ) {
			pMaterial->pMapKePath = createPathFromParam( valueList );
		} else if ( _stricmp( valueList[0], "map_Ns" ) == 0 ) {
			pMaterial->pMapNsPath = createPathFromParam( valueList );
		} else if ( _stricmp( valueList[0], "map_Bump" ) == 0 ) {
			pMaterial->pMapBumpPath = createPathFromParam( valueList );
		} else if ( _stricmp( valueList[0], "Bump" ) == 0 ) {
			pMaterial->pBumpPath = createPathFromParam( valueList );
		}
	}

	m_MaterialList.push_back( pMaterial );

	return bSuccess;

}

char *GOBJFileLoader::createPathFromParam( vector<char*> &valueList )
{
	unsigned int totalLength = 0;
	char *newLength = 0;

	for ( unsigned int i = 0; i < valueList.size(); ++i ) {
		totalLength += (unsigned int)strlen( valueList[i] ) + 1;
	}

	newLength = (char*) malloc( sizeof( char ) * totalLength );
	newLength[0] = 0x00;

	for ( unsigned int i = 1; i < valueList.size(); ++i ) {
		if ( i > 1 ) strcat( newLength, " " );
		strcat( newLength, valueList[ i ] );
	}

	return newLength;
}


/**
 *	주석, 공백등을 제거한 데이터를 만날때까지 loop 를 돌면서
 *	한줄을 읽어온다.
 */
bool GOBJFileLoader::readLine( char *data, int size, FILE *fp )
{
	while( !feof( fp ) ) {
		/**
		 *	데이터가 없다면 continue;
		 */
		if ( fgets( data, 2040, fp ) == NULL )
			continue;
			
		GUtil::removeCRLF( data );

		/** 
		 *	실제 데이터 일때만 리턴.
		 */	
		return true;
	}
	
	return false;
}

GOBJMaterial *GOBJFileLoader::getMaterial( const char* name )
{
	for ( int i = 0; i < (int) m_MaterialList.size(); ++i ) {
		if ( _stricmp( m_MaterialList[ i ]->pName, name ) == 0 )
			return m_MaterialList[ i ];
	}
	return NULL;
}

int GOBJFileLoader::getObjectCount()
{
	return (int) m_ObjectList.size();
}

GOBJObject* GOBJFileLoader::getObject( int index )
{
	return m_ObjectList[ index ];
}

void GOBJFileLoader::printObjectInfo()
{
	GLogManager::logging( LOG_INFO, "----------------------------------------------------------------------" );
	GLogManager::logging( LOG_INFO, " -> Object Count : %d", (int) m_ObjectList.size() );
	GLogManager::logging( LOG_INFO, "  -> Vertex Count = %d", (int) m_GeometryData.m_VertexList.size() );
	GLogManager::logging( LOG_INFO, "  -> Normal Count = %d", (int) m_GeometryData.m_NormalList.size() );
	GLogManager::logging( LOG_INFO, "  -> Texture Coord Count = %d", (int) m_GeometryData.m_TextureCoordList.size() );
	for ( int i = 0; i < (int) m_ObjectList.size(); ++i ) {
		GOBJObject *pObject = m_ObjectList[ i ];
		GLogManager::logging( LOG_INFO, "  -> Object = %s", pObject->pName );
		GLogManager::logging( LOG_INFO, "  -> Face Count = %d", (int) pObject->m_FaceList.size() );
	}
	GLogManager::logging( LOG_INFO, "----------------------------------------------------------------------" );

}

GOBJGeometryData* GOBJFileLoader::getGeometryData()
{
	return &m_GeometryData;
}