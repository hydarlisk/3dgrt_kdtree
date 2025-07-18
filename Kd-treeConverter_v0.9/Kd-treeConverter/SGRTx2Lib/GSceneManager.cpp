#include "GSceneManager.h"
#include "GUtil.h"
#include "GPointLight.h"
#include "GVirtualLight.h"
#include "GRaySetLight.h"
#include "GPolygonGeometry.h"
#include "GPhotonMappingOption.h"
#include "math.h"
#include "GOBJFileLoader.h"
#include "GTextureManager.h"

#define VERSION_TAG			"[VERSION]"
#define SCENEINFO_TAG		"[SCENEINFO]"
#define PHOTONMAPPING_TAG	"[PHOTONMAPPING]"
#define CAMERA_TAG			"[CAMERA]"
#define LIGHT_TAG			"[LIGHT]"
#define OBJECT_TAG			"[OBJECT]"
#define OBJECTFILE_TAG		"[OBJECTIMPORT]"
#define GLOBAL_TAG			"[GLOBAL]"
#define CLOSE_TAG			"[/]"

// ------------------------------------------------------------------------------------------------
// GSceneManager::GSceneManager
// ------------------------------------------------------------------------------------------------
GSceneManager::GSceneManager(void)
{
	m_SceneBasePath[ 0 ] = 0x00;
}

// ------------------------------------------------------------------------------------------------
// GSceneManager::~GSceneManager
// ------------------------------------------------------------------------------------------------
GSceneManager::~GSceneManager(void)
{
}


/**
 *	Data 파일을 로드해서 Scene 을 만들어서 리턴한다.
 */
GError GSceneManager::loadScene( const char* szFileName, GScene** ppScene )
{
	/** parent path 를 가져온다. */
	GUtil::getParentPath( szFileName, m_SceneBasePath );
	GLogManager::logging( LOG_INFO, " > Loading Scene File ( %s )", szFileName );

	/** 
	 *	TODO: 나중에 GTextureManager 를 Scene 안으로 통합해넣자 
	 *	Scene 이 바뀔때, Texture 를 초기화 해야 하므로.
	 */
	GTextureManager::getInstance()->clear();

	(*ppScene) = new GScene();
	(*ppScene)->setSceneBasePath( m_SceneBasePath );

	const char* ext = GUtil::getFileExt( szFileName );

	if ( ext == NULL || _stricmp( ext, "914" ) == 0 ) {
		return load914File( szFileName, (*ppScene) );
	} else if ( _stricmp( ext, "obj" ) == 0 ) {
		return loadOBJFile( szFileName, (*ppScene) );
	}

	delete (*ppScene);
	(*ppScene) = NULL;

	return errorFileDataError;
}

/**
 *	Text 형식의 914 File 을 로드한다.
 */
GError GSceneManager::load914File( const char* szFileName, GScene* pScene )
{
	char data[2048] = { 0x00, };
	GError error = errorFileDataError;
	FILE *fp = NULL;

	if ( ( fp = fopen( szFileName, "rt" ) ) == NULL ) {
		GLogManager::logging( LOG_DEBUG, "FileNotFound = %s", szFileName );
		return errorFileNotFound;
	}

	while( 1 ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;
	
		/** 마지막은 close_tag 로 끝나야 정상. */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			error = errorNo;
			break;
		}

		/** VERSION 정보 */
		if ( _stricmp( data, VERSION_TAG ) == 0 ) {
			if ( !readVersion( data, 2048, fp, pScene ) ) {
				error = errorFileInvalidVersion;
				break;
			}

		/** SCENE 정보	*/
		} else if ( _stricmp( data, SCENEINFO_TAG ) == 0 ) {
			if ( !readSceneInfo( data, 2048, fp, pScene ) ) {
				error = errorFileInvalidSceneInfo;
				break;
			}

		/** CAMERA 정보 */
		} else if ( _stricmp( data, CAMERA_TAG ) == 0 ) {
			if ( !readCamera( data, 2048, fp, pScene ) ) {
				error = errorFileInvalidCamera;
				break;
			}

		/** LIGHT 정보 */
		} else if ( _stricmp( data, LIGHT_TAG ) == 0 ) {
			if ( !readLight( data, 2048, fp, pScene ) ) {
				error = errorFileInvalidLight;
				break;
			}

		/** PHOTON MAPPING OPTION 정보 */
		} else if ( _stricmp( data, PHOTONMAPPING_TAG ) == 0 ) {
			if ( !readPhotonMappingOption( data, 2048, fp, pScene ) ) {
				error = errorFileDataError;
				break;
			}

		/** OBJECT 정보 */
		} else if ( _stricmp( data, OBJECT_TAG ) == 0 ) {
			if ( !readTriangleObject( data, 2048, fp, pScene ) ) {
				error = errorFileInvalidObject;
				break;
			}

		/** OBJECT FILE 데이터 */
		} else if ( _stricmp( data, OBJECTFILE_TAG ) == 0 ) {
			if ( !readImportObject( data, 2048, fp, pScene ) ) {
				error = errorFileInvalidObject;
				break;
			}

		/** GLOBAL 정보 */
		} else if ( _stricmp( data, GLOBAL_TAG ) == 0 ) {
			if ( !readGlobal( data, 2048, fp, pScene ) ) {
				error = errorFileInvalidGlobal;
				break;
			}

		} else {
			error = errorFileDataError;
			break;
		}

	}

	fclose( fp );

	pScene->initalize();

	return error;
}

/**
 *	VERSION_TAG Parsing.
 */
bool GSceneManager::readVersion( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 )
			return true;

		GUtil::getKeyValue( data, &key, &value );
		if ( _stricmp( key, "Version" ) == 0 ) {
			pScene->setVersion( value );
		}
	}

	return false;
}

/**
 *	SCENEINFO Parsing.
 */
bool GSceneManager::readSceneInfo( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;
	vector<char*> valueList;

	pScene->setOutputPath( m_SceneBasePath );

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 )
			return true;

		GUtil::getKeyValue( data, &key, &value );

		if ( _stricmp( key, "SceneName" ) == 0 ) {
			pScene->setSceneName( value );
		}

		if ( _stricmp( key, "OutputPath" ) == 0 ) {

			if ( strlen( value ) > 0 && value[0] == '.' ) {
				char temp[2048] = { 0x00, };
				sprintf( temp, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
				pScene->setOutputPath( temp );
			}

		}
		if ( _stricmp( key, "TexturePath" ) == 0 ) {
			pScene->setTexturePath( value );
		}
		if ( _stricmp( key, "Resolution" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 2 )
				break;
			pScene->setResolution( atoi( valueList[0] ), atoi( valueList[1] ) );
		}
		if ( _stricmp( key, "SuperSampling" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 2 )
				break;
			pScene->setSuperSampling( atoi( valueList[0] ), atoi( valueList[1] ) );
		}
		if ( _stricmp( key, "MaxReflectionDepth" ) == 0 ) {
			pScene->setMaxReflectionDepth( atoi( value ) );
		}
		if ( _stricmp( key, "FrontFace" ) == 0 ) {
			if ( _stricmp( value, "CCW" ) == 0 ) 
				pScene->setFrontFace( faceCCW );
			else 
				pScene->setFrontFace( faceCW );
		}
		if ( _stricmp( key, "BackFaceCulling" ) == 0 ) {
			if ( _stricmp( value, "Yes" ) == 0 ) 
				pScene->setBackFaceCulling( true );
			else 
				pScene->setBackFaceCulling( false );
		}
		if ( _stricmp( key, "Shadow" ) == 0 ) {
			if ( _stricmp( value, "Yes" ) == 0 ) 
				pScene->setEnableShadow( true );
			else 
				pScene->setEnableShadow( false );
		}
		if ( _stricmp( key, "UseTexture" ) == 0 ) {
			if ( _stricmp( value, "Yes" ) == 0 ) 
				pScene->setUseTexture( true );
			else 
				pScene->setUseTexture( false );
		}
		if ( _stricmp( key, "BloomingFilter" ) == 0 ) {
			if ( _stricmp( value, "Yes" ) == 0 ) 
				pScene->setBloomingFilter( true );
			else 
				pScene->setBloomingFilter( false );
		}
		if ( _stricmp( key, "BlurFilter" ) == 0 ) {
			if ( _stricmp( value, "Yes" ) == 0 ) 
				pScene->setUseBlurFilter( true );
			else 
				pScene->setUseBlurFilter( false );
		}
		if ( _stricmp( key, "BloomingRadius" ) == 0 ) {
			pScene->setBloomingRadius( (float) atof( value ) );
		}
		if ( _stricmp( key, "BloomingWeight" ) == 0 ) {
			pScene->setBloomingWeight( (float) atof( value ) );
		}
	}

	return false;
}

/**
 *	CAMERA Parsing.
 */
bool GSceneManager::readCamera( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;
	GVector eye, view, up;
	float fovy, ffar, fnear;
	vector<char*> valueList;
	GCamera *camera = new GCamera;

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			camera->setCameraPos( eye, view, up );
			camera->setPerspective( fovy, 
								   (float) pScene->getResolution().x / (float) pScene->getResolution().y, 
								   fnear, ffar );
			pScene->addCamera( camera );
			if( pScene->getCameraList()->size() == 1 )
				pScene->setRenderCamera( camera );
			pScene->setInitRenderCamera( camera );
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		if ( _stricmp( key, "position" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			eye.x = (float) atof( valueList[0] ); 
			eye.y = (float) atof( valueList[1] ); 
			eye.z = (float) atof( valueList[2] );
		}
		if ( _stricmp( key, "view" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			view.x = (float) atof( valueList[0] ); 
			view.y = (float) atof( valueList[1] ); 
			view.z = (float) atof( valueList[2] );
		}
		if ( _stricmp( key, "up" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			up.x = (float) atof( valueList[0] ); 
			up.y = (float) atof( valueList[1] ); 
			up.z = (float) atof( valueList[2] );
		}
		if ( _stricmp( key, "fovy" ) == 0 ) {
			fovy = (float) atof( value );
		}
		if ( _stricmp( key, "near" ) == 0 ) {
			fnear = (float) atof( value );
		}
		if ( _stricmp( key, "far" ) == 0 ) {
			ffar = (float) atof( value );
		}
	}

	delete camera;
	return false;
}

/**
 *	GLOBAL Parsing.
 */
bool GSceneManager::readGlobal( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;
	GVector eye, view, up;
	vector<char*> valueList;

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		if ( _stricmp( key, "ambient" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pScene->setGlobalAmbient( GColor( (float) atof( valueList[0] ),
									  (float) atof( valueList[1] ),
                                      (float) atof( valueList[2] ), 1.0 ) );
		}
	}

	return false;
}


/**
 *	Photon Mapping Option Parsing.
 */
bool GSceneManager::readPhotonMappingOption( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;
	vector<char*> valueList;
	GPhotonMappingOption photonMappingOption;

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			pScene->setPhotonMappingOption( photonMappingOption );
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		if ( _stricmp( key, "EmitPhotonPerIteration" ) == 0 ) {
			photonMappingOption.m_iEmitPhotonPerIteration = atoi( value );
		}
		if ( _stricmp( key, "MaxBoundPerIteration" ) == 0 ) {
			photonMappingOption.m_iMaxBound = atoi( value );
		}
		if ( _stricmp( key, "Iteration" ) == 0 ) {
			photonMappingOption.m_iIteration = atoi( value );
		}
		if ( _stricmp( key, "SceneLightPower" ) == 0 ) {
			photonMappingOption.m_fTotalSceneLightPower = (float) atof( value );
		}
		if ( _stricmp( key, "SaveDirectPhoton" ) == 0 ) {
			if ( _stricmp( value, "true" ) == 0 )
				photonMappingOption.m_bSaveDirectPhoton = true;
			else
				photonMappingOption.m_bSaveDirectPhoton = false;
		}
		if ( _stricmp( key, "DirectIllumination" ) == 0 ) {
			if ( _stricmp( value, "true" ) == 0 )
				photonMappingOption.m_bDirectIllumByPhotonMap = true;
			else
				photonMappingOption.m_bDirectIllumByPhotonMap = false;
			if ( photonMappingOption.m_bDirectIllumByPhotonMap ) {
				photonMappingOption.m_bSaveDirectPhoton = true;
			}
		}
		if ( _stricmp( key, "GlossyIllumination" ) == 0 ) {
			if ( _stricmp( value, "true" ) == 0 )
				photonMappingOption.m_bGlossyEffectPhotonMap = true;
			else
				photonMappingOption.m_bGlossyEffectPhotonMap = false;
			if ( photonMappingOption.m_bGlossyEffectPhotonMap ) {
				photonMappingOption.m_bSaveDirectPhoton = true;
			}
		}
		if ( _stricmp( key, "DensityEstimateMethod" ) == 0 ) {
			if ( _stricmp( value, "areaphoton" ) == 0 ) {
				photonMappingOption.m_eDensityMethod = densityAreaPhoton;
			} else {
				photonMappingOption.m_eDensityMethod = densityProjectedCircle;
			}
		}
		if ( _stricmp( key, "GridUnitLength" ) == 0 ) {
			photonMappingOption.m_fGridUnitLength = (float) atof( value );
		}
		if ( _stricmp( key, "SearchRadius" ) == 0 ) {
			photonMappingOption.m_fSearchRadius = (float) atof( value );
		}
	}

	return false;
}

/**
 *	Light Parsing
 */
bool GSceneManager::readLight( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;

	if( !readLine( data, 2048, fp ) ) 
		return false;

	/** 처음에 반드시 type 이 나와야 한다. */
	GUtil::getKeyValue( data, &key, &value );

	if ( _stricmp( key, "type" ) == 0 ) {

		if ( _stricmp( value, "point" ) == 0 ) {
			return readPointLight( data, size, fp, pScene );
		} else if ( _stricmp( value, "rayset" ) == 0 ) {
			return readRaySetLight( data, size, fp, pScene );
		} else if ( _stricmp( value, "virtual" ) == 0 ) {
			return readVirtualLight( data, size, fp, pScene );
		}

		return false;
	}

	return false;

}

/**
 *	LIGHT Parsing.
 */
bool GSceneManager::readPointLight( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;
	vector<char*> valueList;
	GPointLight *pointLight = new GPointLight();

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			pointLight->setPolygonType( typePolygonNone );
			pScene->addLight( pointLight );
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		if ( _stricmp( key, "position" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pointLight->setPosition( GPoint( (float) atof( valueList[0] ),
									  (float) atof( valueList[1] ),
                                      (float) atof( valueList[2] ) ) );
		}
		if ( _stricmp( key, "intensity" ) == 0 ) {
			pointLight->setIntensity( (float) atof( value ) );
		}
		if ( _stricmp( key, "disabled" ) == 0 ) {
			pointLight->setEnable( false );
		}
		if ( _stricmp( key, "emitphoton" ) == 0 ) {
			if ( _stricmp( value, "true" ) == 0 )
				pointLight->setUsePhoton( true );
			else
				pointLight->setUsePhoton( false );
		}
		if ( _stricmp( key, "color" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pointLight->setLightColor( 
				GColor( (float) atof( valueList[0] ), 
				        (float) atof( valueList[1] ), 
						(float) atof( valueList[2] ), 1.0f ) );
		}
	}

	delete pointLight;

	return false;
}

/**
 *	RaySet LIGHT Parsing.
 *	ray sample 데이터로 구성된 light.
 */
bool GSceneManager::readRaySetLight( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;
	vector<char*> valueList;
	GRaySetLight *raySetLight = new GRaySetLight();

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 *	만약 ray set data 가 없다면 에러.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			if ( raySetLight->getRaySetData() == NULL ) {
				GLogManager::logging( LOG_FATAL, "there is no ray set data." );
				return false;
			}
			raySetLight->convertToWorldObject();
			pScene->addLight( raySetLight );
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		/** rayset datafile 을 읽어들인다. */
		if ( _stricmp( key, "datafile" ) == 0 ) {
			char temppath[ 2048 ] = { 0x00, };
			sprintf( temppath, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
			if ( !loadRaySetDataFile( raySetLight, temppath ) ) {
				GLogManager::logging( LOG_FATAL, "fail to load rayset data file : %s", value );
				return false;
			}
		}

		/** rayset 의 hit 를 체크하기 위한 object data 있다면 읽어들인다. */
		if ( _stricmp( key, "meshfile" ) == 0 ) {
			char temppath[ 2048 ] = { 0x00, };
			sprintf( temppath, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
			if ( !loadMeshData( raySetLight, temppath ) ) {
				GLogManager::logging( LOG_FATAL, "fail to load polygon data file : %s", value );
				return false;
			}
			raySetLight->setPolygonType( typePolygonTriangle );
			raySetLight->setIntersection( true );
		}

		if ( _stricmp( key, "position" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->setPosition( GPoint( (float) atof( valueList[0] ),
									  (float) atof( valueList[1] ),
                                      (float) atof( valueList[2] ) ) );
		}

		if ( _stricmp( key, "intensity" ) == 0 ) {
			raySetLight->setIntensity( (float) atof( value ) );
		}
		if ( _stricmp( key, "random" ) == 0 ) {
			if ( _stricmp( value, "false" ) == 0 )
				raySetLight->setRandomMode( false );
			else
				raySetLight->setRandomMode( true );
		}

		if ( _stricmp( key, "emitphoton" ) == 0 ) {
			if ( _stricmp( value, "true" ) == 0 )
				raySetLight->setUsePhoton( true );
			else
				raySetLight->setUsePhoton( false );
		}

		if ( _stricmp( key, "color" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->setLightColor( 
				GColor( (float) atof( valueList[0] ), 
				        (float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}


		if ( _stricmp( key, "diffuse" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->getMaterial()->setDiffuse( 
				GColor( (float) atof( valueList[0] ), 
						(float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}
		if ( _stricmp( key, "specular" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->getMaterial()->setSpecular( 
				GColor( (float) atof( valueList[0] ), 
						(float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}
		if ( _stricmp( key, "ambient" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->getMaterial()->setAmbient( 
				GColor( (float) atof( valueList[0] ), 
						(float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}
		if ( _stricmp( key, "transparency" ) == 0 ) {
			raySetLight->getMaterial()->setTransparency( (float) atof( value ) );
		}
		if ( _stricmp( key, "reflection" ) == 0 ) {
			raySetLight->getMaterial()->setReflection( (float) atof( value ) );
		}
		if ( _stricmp( key, "roughness" ) == 0 ) {
			raySetLight->getMaterial()->setRoughness( (float) atof( value ) );
		}
		if ( _stricmp( key, "RefractionIndex" ) == 0 ) {
			raySetLight->getMaterial()->m_fRefractionIndex = (float) atof( value );
		}

		/**
		 *	Translate 일때.
		 */
		if ( _stricmp( key, "Translate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->setTranslate( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Rotate 일때.
		 */
		if ( _stricmp( key, "Rotate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->setRotate(
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Scale 일때.
		 */
		if ( _stricmp( key, "Scale" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			raySetLight->setScale( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}

	}

	delete raySetLight;

	return false;
}


/**
 *	RaySet LIGHT Parsing.
 *	ray sample 데이터로 구성된 light.
 */
bool GSceneManager::readVirtualLight( char *data, int size, FILE *fp, GScene* pScene )
{
	char *key, *value;
	vector<char*> valueList;
	GVirtualLight *virtualLight = new GVirtualLight();

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 *	만약 ray set data 가 없다면 에러.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			virtualLight->convertToWorldObject();
			pScene->addLight( virtualLight );
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		/** rayset 의 hit 를 체크하기 위한 object data 있다면 읽어들인다. */
		if ( _stricmp( key, "meshfile" ) == 0 ) {
			char temppath[ 2048 ] = { 0x00, };
			sprintf( temppath, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
			if ( !loadMeshData( virtualLight, temppath ) ) {
				GLogManager::logging( LOG_FATAL, "fail to load polygon data file : %s", value );
				return false;
			}
			virtualLight->setPolygonType( typePolygonTriangle );
			virtualLight->setIntersection( true );
		}

		if ( _stricmp( key, "position" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			virtualLight->setPosition( GPoint( (float) atof( valueList[0] ),
									  (float) atof( valueList[1] ),
                                      (float) atof( valueList[2] ) ) );
		}

		if ( _stricmp( key, "intensity" ) == 0 ) {
			virtualLight->setIntensity( (float) atof( value ) );
		}

		if ( _stricmp( key, "emitphoton" ) == 0 ) {
			if ( _stricmp( value, "true" ) == 0 )
				virtualLight->setUsePhoton( true );
			else
				virtualLight->setUsePhoton( false );
		}

		if ( _stricmp( key, "color" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			virtualLight->setLightColor( 
				GColor( (float) atof( valueList[0] ), 
				        (float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}

		/**
		 *	Translate 일때.
		 */
		if ( _stricmp( key, "Translate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			virtualLight->setTranslate( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Rotate 일때.
		 */
		if ( _stricmp( key, "Rotate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			virtualLight->setRotate(
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Scale 일때.
		 */
		if ( _stricmp( key, "Scale" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			virtualLight->setScale( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}

	}

	delete virtualLight;

	return false;
}

bool GSceneManager::loadRaySetDataFile( GRaySetLight *pRaySetLight, const char *datafile )
{
	FILE *fp = fopen( datafile, "rt" );
	float length = 0;
	int i = 0;
	if ( fp == NULL ) {
		return false;
	}

	int dataCount = 0;
	GRaySet *pRaySet = NULL;

	fscanf( fp, "%d\n", &dataCount );
	
	if ( dataCount > 0 ) {

		pRaySet = (GRaySet*) malloc( sizeof( GRaySet ) * dataCount );
		memset( pRaySet, 0x00, sizeof( GRaySet ) * dataCount );

		for ( i = 0; i < dataCount; ++i ) {
			fscanf( fp, "%f %f %f %f %f %f %f", pRaySet[ i ].pos, ( pRaySet[ i ].pos + 1 ), ( pRaySet[ i ].pos + 2 ),
												pRaySet[ i ].dir, ( pRaySet[ i ].dir + 1 ), ( pRaySet[ i ].dir + 2 ),
												&( pRaySet[ i ].power ) );

			/**
			 *	dir 은 normalize 되어야 하는데, 데이터가 혹시나
			 *	normalize 안되어 있을수 있으므로 여기서 normalize 한다.
			 */
			length = sqrtf( pRaySet[ i ].dir[ 0 ] * pRaySet[ i ].dir[ 0 ] +
						    pRaySet[ i ].dir[ 1 ] * pRaySet[ i ].dir[ 1 ] +
						    pRaySet[ i ].dir[ 2 ] * pRaySet[ i ].dir[ 2 ] );
			if ( length == 0 )
				break;

			pRaySet[ i ].dir[ 0 ] = pRaySet[ i ].dir[ 0 ] / length;
			pRaySet[ i ].dir[ 1 ] = pRaySet[ i ].dir[ 1 ] / length;
			pRaySet[ i ].dir[ 2 ] = pRaySet[ i ].dir[ 2 ] / length;

			if ( pRaySet[ i ].power <= 0.0f ) {
				break;
			}
		}

	}

	if ( i < dataCount ) {
		fclose( fp );
		return false;
	}

	pRaySetLight->setRaySetData( pRaySet, dataCount );

	fclose( fp );

	return true;
}


/**
 *	OBJECT Parsing.
 */
bool GSceneManager::readImportObject( char *data, int size, FILE *fp, GScene* pScene )
{
	char temp[1024] = { 0x00, };
	char *key, *value;
	vector<char*> valueList;
	char filetype[1024] = { 0x00, }, datafile[1024] = { 0x00, };
	char kdtree_load_file__SAH[1024] = { 0x00, };
	char kdtree_load_file_ESAH[1024] = { 0x00, };
	char kdtree_save_file__SAH[1024] = { 0x00, };

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 *	object 를 import 한다.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {

			if ( pScene->isKdTreeFileLoad() ) {
				int nType = pScene->getKdTreeFileType();

				if (nType == SAH) {
					pScene->setKdTreeLoadFilePath( kdtree_load_file__SAH );
					if (kdtree_load_file__SAH[0] == '\0') {
						pScene->setKdTreeFileLoad( false );
					} else {
						FILE *kdtfp = fopen( kdtree_load_file__SAH, "rt" );
						if (kdtfp == NULL) pScene->setKdTreeFileLoad( false );
						else fclose(kdtfp);
					}
				} else if (nType == EMPTY_SAH) {
					pScene->setKdTreeLoadFilePath( kdtree_load_file_ESAH );
					if (kdtree_load_file_ESAH[0] == '\0') {
						pScene->setKdTreeFileLoad( false );
					} else {
						FILE *kdtfp = fopen( kdtree_load_file_ESAH, "rt" );
						if (kdtfp == NULL) pScene->setKdTreeFileLoad( false );
						else fclose(kdtfp);
					}
				}
			}
			if ( pScene->isKdTreeFileSave() ) {
				if ( kdtree_save_file__SAH[0] == '\0' ) {
					pScene->setKdTreeFileSave( false );
				}
			}

			/** object file 일때 */
			if ( _stricmp( filetype, "object" ) == 0 ) {
				char objfile[2048] = { 0x00, };
				sprintf( objfile, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, datafile );
				loadOBJFile( objfile, pScene );
				return true;
			}

			GLogManager::logging( LOG_ERROR, "%s type is not supported", filetype );

			return false;
		}

		GUtil::getKeyValue( data, &key, &value );

		if ( _stricmp( key, "filetype" ) == 0 ) {
			strcpy( filetype, value );
		}
		if ( _stricmp( key, "datafile" ) == 0 ) {
			strcpy( datafile, value );
		}
		if ( _stricmp( key, "kdtree_load_type" ) == 0 ) {
			if ( _stricmp( value, "SAH" ) == 0)
				pScene->setKdTreeFileType( SAH );			// SAH
			else
				pScene->setKdTreeFileType( EMPTY_SAH );		// Empty SAH
		}
		if ( _stricmp( key, "kdtree_load" ) == 0 ) {
			if ( _stricmp( value, "No" ) == 0 ) {
				pScene->setKdTreeFileLoad( false );
			} else {
				pScene->setKdTreeFileLoad( true );
				if ( _stricmp( value, "SAH" ) == 0 )
					pScene->setKdTreeFileType( SAH );			// SAH
				else
					pScene->setKdTreeFileType( EMPTY_SAH );		// Empty SAH
			}
		}
		if ( _stricmp( key, "kdtree_load_file__SAH" ) == 0 ) {
			sprintf( kdtree_load_file__SAH, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
		}
		if ( _stricmp( key, "kdtree_load_file_ESAH" ) == 0 ) {
			sprintf( kdtree_load_file_ESAH, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
		}
		if ( _stricmp( key, "kdtree_save" ) == 0 ) {
			if ( _stricmp( value, "Yes" ) == 0 )
				pScene->setKdTreeFileSave( true );
			else
				pScene->setKdTreeFileSave( false );
		}
		if ( _stricmp( key, "kdtree_save_file__SAH" ) == 0 ) {
			sprintf( kdtree_save_file__SAH, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
			pScene->setKdTreeSaveFilePath( kdtree_save_file__SAH );
		}


	}

	return false;
}

/**
 *	OBJECT Parsing.
 */
bool GSceneManager::readTriangleObject( char *data, int size, FILE *fp, GScene* pScene )
{
	char temp[1024] = { 0x00, };
	char *key, *value;
	GPolygonGeometry *pTriangleGeometry = new GPolygonGeometry();
	vector<char*> valueList;

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			if ( pTriangleGeometry->getVertexArray() == NULL ) {
				GLogManager::logging( LOG_FATAL, "%s Object Data error!", pTriangleGeometry->getName() );
				break;
			}
			pTriangleGeometry->setPolygonType( typePolygonTriangle );
			pTriangleGeometry->convertToWorldObject();
			pScene->addObject( pTriangleGeometry );
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		if ( _stricmp( key, "name" ) == 0 ) {
			pTriangleGeometry->setName( value );
		}
		if ( _stricmp( key, "visible" ) == 0 ) {
			if ( _stricmp( value, "false" ) == 0 )
				pTriangleGeometry->setVisible( false );
		}
		/**
		 *	Scene Option 에 Texture 사용이 true 일때, Texture full path 를 저장한다.
		 */
		if ( _stricmp( key, "texture" ) == 0 ) {
			sprintf( temp, "%s%c%s%c%s", m_SceneBasePath,	FILE_SEPARATOR, 
								pScene->getTexturePath(), FILE_SEPARATOR, value );
			pTriangleGeometry->setTextureID( GTextureManager::getInstance()->addTexture( temp ) );
		}
		if ( _stricmp( key, "meshfile" ) == 0 ) {
			sprintf( temp, "%s%c%s", m_SceneBasePath, FILE_SEPARATOR, value );
			if ( !loadDatFile( temp, pTriangleGeometry ) ) {
				GLogManager::logging( LOG_FATAL, "%s Data File Error!", pTriangleGeometry->getName() );
				break;
			}
		}

		if ( _stricmp( key, "diffuse" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 ) {
				GLogManager::logging( LOG_FATAL, "object diffuse color error" );
				break;
			}
			pTriangleGeometry->getMaterial()->setDiffuse( 
				GColor( (float) atof( valueList[0] ), 
						(float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}
		if ( _stricmp( key, "specular" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 ) {
				GLogManager::logging( LOG_FATAL, "object diffuse color error" );
				break;
			}
			pTriangleGeometry->getMaterial()->setSpecular( 
				GColor( (float) atof( valueList[0] ), 
						(float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}
		if ( _stricmp( key, "ambient" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 ) {
				GLogManager::logging( LOG_FATAL, "object diffuse color error" );
				break;
			}
			pTriangleGeometry->getMaterial()->setAmbient( 
				GColor( (float) atof( valueList[0] ), 
						(float) atof( valueList[1] ), 
						(float) atof( valueList[2] ) ) );
		}
		if ( _stricmp( key, "transparency" ) == 0 ) {
			pTriangleGeometry->getMaterial()->setTransparency( (float) atof( value ) );
		}
		if ( _stricmp( key, "reflection" ) == 0 ) {
			pTriangleGeometry->getMaterial()->setReflection( (float) atof( value ) );
		}
		if ( _stricmp( key, "roughness" ) == 0 ) {
			pTriangleGeometry->getMaterial()->setRoughness( (float) atof( value ) );
		}
		if ( _stricmp( key, "RefractionIndex" ) == 0 ) {
			pTriangleGeometry->getMaterial()->m_fRefractionIndex = (float) atof( value );
		}
		
		/**
		 *	Translate 일때.
		 */
		if ( _stricmp( key, "Translate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pTriangleGeometry->setTranslate( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Rotate 일때.
		 */
		if ( _stricmp( key, "Rotate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pTriangleGeometry->setRotate(
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Scale 일때.
		 */
		if ( _stricmp( key, "Scale" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pTriangleGeometry->setScale( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}

		/** 
		 *	vertex data 를 읽는다. value 에는 vertex 가 몇개인지 정보가 들어있다.
		 */
		if ( _stricmp( key, "Vertex" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 2 )
				break;
			bool bUv = false;
			if ( _stricmp( valueList[1], "yes" ) == 0 || _stricmp( valueList[1], "1" ) == 0 )
				bUv = true;
			if ( !readObjectVertex( data, 2048, fp, pTriangleGeometry, atoi( valueList[0] ), bUv ) )
				break;
		}
		if ( _stricmp( key, "Index" ) == 0 ) {
			int baseIndex = 0;
			GUtil::getValueList( value, valueList );
			if ( valueList.size() == 2 )
				baseIndex = atoi( valueList[1] );

			if ( !readObjectIndex( data, 2048, fp, pTriangleGeometry, atoi( valueList[0] ), baseIndex ) )
				break;
		}
	}

	/** 에러가 있는것이므로 delete 한다. */
	GLogManager::logging( LOG_ERROR, "%s object is invalid", pTriangleGeometry->getName() );

	delete pTriangleGeometry;

	return false;
}


/**
 *	Vertex 와 index 정보만 있는 Polygon Data 를 읽어들여서 pObject 에 세팅.
 */
bool GSceneManager::loadMeshData( GPolygonObject* pObject, const char* filename )
{
	char data[2048] = { 0x00, };
	char *key, *value;
	vector<char*> valueList;

	FILE *fp = fopen( filename, "rt" );
	if ( fp == NULL )
		return false;

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/**
		 *	마지막 CLOSE_TAG 정보가 와야 정상적으로 끝나는것.
		 */
		if ( _stricmp( data, CLOSE_TAG ) == 0 ) {
			return true;
		}

		GUtil::getKeyValue( data, &key, &value );

		/**
		 *	Translate 일때.
		 */
		if ( _stricmp( key, "Translate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pObject->setTranslate( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Rotate 일때.
		 */
		if ( _stricmp( key, "Rotate" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pObject->setRotate(
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}
		/**
		 *	Scale 일때.
		 */
		if ( _stricmp( key, "Scale" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 3 )
				break;
			pObject->setScale( 
				GPoint( (float)atof( valueList[0] ), (float)atof( valueList[1] ), (float)atof( valueList[2] ) ) );
		}

		/** 
		 *	vertex data 를 읽는다. value 에는 vertex 가 몇개인지 정보가 들어있다.
		 */
		if ( _stricmp( key, "Vertex" ) == 0 ) {
			GUtil::getValueList( value, valueList );
			if ( valueList.size() != 2 )
				break;
			bool bUv = false;
			if ( _stricmp( valueList[1], "yes" ) == 0 || _stricmp( valueList[1], "1" ) == 0 )
				bUv = true;
			if ( !readObjectVertex( data, 2048, fp, pObject, atoi( valueList[0] ), bUv ) )
				break;
		}
		if ( _stricmp( key, "Index" ) == 0 ) {
			int baseIndex = 0;
			GUtil::getValueList( value, valueList );
			if ( valueList.size() == 2 )
				baseIndex = atoi( valueList[1] );

			if ( !readObjectIndex( data, 2048, fp, pObject, atoi( valueList[0] ), baseIndex ) )
				break;
		}
	}

	return false;
}

/** 
 *	%f %f %f 데이터로 이루어진 line 들을 vertex 개수만큼 읽어들인다.
 *	읽어들일때 BoudingBox 정보도 계산한다.
 */
bool GSceneManager::readObjectVertex( char *data, int size, FILE *fp, 
									 GPolygonObject* pObject, int count, bool bUV )
{
	GBoundingBox m_BBox;
	GVector bmin, bmax;
	float *vertexArray = NULL, *normalArray = NULL, *uvArray = NULL, *visibilityArray = NULL;

	vertexArray = (float*) malloc( sizeof( float ) * 3 * count );
	normalArray = (float*) malloc( sizeof( float ) * 3 * count );
	visibilityArray = (float*) malloc( sizeof( float ) * 3 * count );

	/**
	 *	데이터에 uv 좌표가 존재한다면.
	 */
	if ( bUV ) {
		uvArray = (float*) malloc( sizeof( float ) * 2 * count );
	}

	float v0, v1, v2, n0, n1, n2, u, v, length;
	int i = 0;

	for ( i = 0; i < count; ++i ) {
		visibilityArray[i*3+0] = 0.0f;
		visibilityArray[i*3+1] = 0.0f;
		visibilityArray[i*3+2] = 0.0f;

		if ( !readLine( data, 2048, fp ) ) 
			break;

		if ( bUV ) {
			sscanf( data, "%f %f %f %f %f %f %f %f", &v0, &v1, &v2, &n0, &n1, &n2, &u, &v );
			*( vertexArray + i * 3 + 0 ) = v0;
			*( vertexArray + i * 3 + 1 ) = v1;
			*( vertexArray + i * 3 + 2 ) = v2;

			/**
			 *	normal 은데이터가 혹시나
			 *	normalize 안되어 있을수 있으므로 여기서 normalize 한다.
			 */
			length = sqrtf( n0 * n0 + n1 * n1 + n2 * n2 );
			if ( length == 0 ) {
				GLogManager::logging( LOG_FATAL, 
									"%s object normal length is zero. (%f, %f, %f)", 
									pObject->getName(), n0, n1, n2 );
				n0 = 1.0f; n1 = 0.0f; n2 = 0.0f;
				length = 1.0f;
				//break;
			}

			*( normalArray + i * 3 + 0 ) = n0 / length;
			*( normalArray + i * 3 + 1 ) = n1 / length;
			*( normalArray + i * 3 + 2 ) = n2 / length;

			/** u, v 는 0.0~1.0 으로 normalize 시킨다. */
			u = u - (int)u;
			v = v - (int)v;

			if ( u < 0.0f ) u = 1.0f + u;
			if ( v < 0.0f ) v = 1.0f + v;

			*( uvArray + i * 2 + 0 ) = u;
			*( uvArray + i * 2 + 1 ) = v;

		} else {
			sscanf( data, "%f %f %f %f %f %f", &v0, &v1, &v2, &n0, &n1, &n2 );
			*( vertexArray + i * 3 + 0 ) = v0;
			*( vertexArray + i * 3 + 1 ) = v1;
			*( vertexArray + i * 3 + 2 ) = v2;
	
			/**
			 *	normal 은데이터가 혹시나
			 *	normalize 안되어 있을수 있으므로 여기서 normalize 한다.
			 */
			length = sqrtf( n0 * n0 + n1 * n1 + n2 * n2 );
			if ( length == 0 ) {
				GLogManager::logging( LOG_FATAL, 
									"%s object normal length is zero. (%f, %f, %f)", 
									pObject->getName(), n0, n1, n2 );
				n0 = 1.0f; n1 = 0.0f; n2 = 0.0f;
				length = 1.0f;
				//break;
			}
			
			*( normalArray + i * 3 + 0 ) = n0 / length;
			*( normalArray + i * 3 + 1 ) = n1 / length;
			*( normalArray + i * 3 + 2 ) = n2 / length;
		}
	
		if ( i == 0 ) {
			bmin.x = v0; bmin.y = v1; bmin.z = v2;
			bmax.x = v0; bmax.y = v1; bmax.z = v2;
		} else {
			bmin.x = min( bmin.x, v0 );
			bmin.y = min( bmin.y, v1 );
			bmin.z = min( bmin.z, v2 );
			bmax.x = max( bmax.x, v0 );
			bmax.y = max( bmax.y, v1 );
			bmax.z = max( bmax.z, v2 );
		}

	}

	if ( i < count ) {
		GLogManager::logging( LOG_FATAL, "object vertex read error" );
		delete vertexArray;
		delete normalArray;
		if ( bUV )
			delete uvArray;
		return false;
	}

	m_BBox.setMin( bmin );
	m_BBox.setMax( bmax );

	pObject->setVertexCount( count );
	pObject->setNormalArray( normalArray );
	pObject->setVertexArray( vertexArray );
	if ( bUV )
		pObject->setUVArray( uvArray );
	pObject->setVisibilityArray( visibilityArray );

	pObject->setBoundingBox( m_BBox );

	return true;
}

/** 
 *	%d 데이터로 이루어진 line 들을 index 개수만큼 읽어들인다.
 *	baseIndex 는 triangle 이 가리키는 vertex index 를 0부터 시작하는지 1부터
 *	시작하는지 여부. c언어 배열상으로는 무조건 0 부터 index 가 시작
 *	하게 해야 하므로, 내부에서 index 를 조정한다.
 */
bool GSceneManager::readObjectIndex( char *data, int size, FILE *fp, 
									 GPolygonObject* pObject, int count, int baseIndex )
{
	int *indexArray = (int*) malloc( sizeof( int ) * 3 * count );
	int v0, v1, v2, maxVertex = pObject->getVertexCount();
	int i;

	/** 
	 *	index 는 1부터 이지만, 배열상으로는 0 부터 vertex 가
	 *	들어가 있으므로 index 를 -1 씩 한다.
	 */
	for ( i = 0; i < count; ++i ) {
		if ( !readLine( data, 2048, fp ) ) 
			break;
		sscanf( data, "%d %d %d", &v0, &v1, &v2 );
		
		v0 = v0 - baseIndex; v1 = v1 - baseIndex; v2 = v2 - baseIndex;

		if ( v0 < 0 || v1 < 0 || v2 < 0 || v0 >= maxVertex || v1 >= maxVertex || v2 >= maxVertex ) {
			break;
		}

		*( indexArray + i * 3 + 0 ) = v0;
		*( indexArray + i * 3 + 1 ) = v1;
		*( indexArray + i * 3 + 2 ) = v2;
	}

	if ( i < count ) {
		GLogManager::logging( LOG_FATAL, "object index read error" );
		delete indexArray;
		return false;
	}

	pObject->setTriangleCount( count );
	pObject->setIndexArray( indexArray );

	return true;
}

/**
 *	OBJ File 를 로드해서 Scene 에 Object 를 추가한다.
 */
GError GSceneManager::loadOBJFile( const char* szFileName, GScene* pScene )
{
	char temp[2048] = { 0x00, };
	int objectCount = 0;

	GOBJObject					*pOBJObject;
	GOBJMaterial				*pOBJMaterial;
	GOBJGeometryData			*pOBJGeometryData;
	vector<GOBJNormalWeight>	*pOBJNormalWeightList;
	vector<GOBJNormalWeight>	*pDestOBJNormalWeightList;

	GOBJFileLoader objLoader;
	if ( !objLoader.load( szFileName ) ) {
		GLogManager::logging( LOG_FATAL, "Object File Load Error" );
		return errorFileDataError;
	}

	//objLoader.printObjectInfo();

	/** SGRTx2 Object 로 변환한다. */
	GLogManager::logging( LOG_INFO, " -> Convert to SGRTx2 Object" );
	
	pOBJGeometryData	= objLoader.getGeometryData();
	objectCount			= objLoader.getObjectCount();

	/** 
	 *	같은 vertex 를 공유하는 데이터의 normal 정보를 합치기 위해서 사용할 구조체 할당 
	 *	vertex 는 최대 pOBJGeometryData->m_VertexList.size() 개수만큼만 존재.
	 */
	pOBJNormalWeightList     = new vector<GOBJNormalWeight>[ pOBJGeometryData->m_VertexList.size() ];
	pDestOBJNormalWeightList = new vector<GOBJNormalWeight>[ pOBJGeometryData->m_VertexList.size() ];


	for ( int index = 0; index < objectCount; ++index ) {

		pOBJObject = objLoader.getObject( index );
		pOBJMaterial = pOBJObject->pMaterial;

		GPolygonGeometry *pObject = new GPolygonGeometry();

		pObject->setName( pOBJObject->pName );

		/** material convert */
		if ( pOBJMaterial != NULL ) {
			pObject->getMaterial()->setAmbient( GColor( pOBJMaterial->ka[0], pOBJMaterial->ka[1], pOBJMaterial->ka[2], 1.0 ) );
			pObject->getMaterial()->setDiffuse( GColor( pOBJMaterial->kd[0], pOBJMaterial->kd[1], pOBJMaterial->kd[2], 1.0 ) );
			pObject->getMaterial()->setSpecular( GColor( pOBJMaterial->ks[0], pOBJMaterial->ks[1], pOBJMaterial->ks[2], 1.0  ) );
			pObject->getMaterial()->setEmission( GColor( pOBJMaterial->ke[0], pOBJMaterial->ke[1], pOBJMaterial->ke[2], 1.0 ) );
			pObject->getMaterial()->m_fRoughness = 1.0f; //min( 1.0f, pOBJMaterial->ns / 250.0f );
			pObject->getMaterial()->m_fRefractionIndex = pOBJMaterial->ni;
			pObject->getMaterial()->setReflection( pOBJMaterial->kr );
			pObject->getMaterial()->setTransparency( 1.0f - pOBJMaterial->tr );
		}

		/** texture convert */
		if ( pOBJMaterial != NULL ) {
			if ( pOBJMaterial->pMapKdPath != NULL ) {
				sprintf( temp, "%s%c%s%c%s", m_SceneBasePath,	FILE_SEPARATOR, 
									pScene->getTexturePath(), FILE_SEPARATOR, pOBJMaterial->pMapKdPath );
				pObject->setTextureID( GTextureManager::getInstance()->addTexture( temp ) );
			}
		}

		/** bump texture 가 있을때. map_bump 항목만 사용 */
		if ( pOBJMaterial != NULL ) {
			if ( pOBJMaterial->pMapBumpPath != NULL ) {
				sprintf( temp, "%s%c%s%c%s", m_SceneBasePath, FILE_SEPARATOR, 
									pScene->getTexturePath(), FILE_SEPARATOR, pOBJMaterial->pMapBumpPath );
				pObject->setBumpTextureID( GTextureManager::getInstance()->addTexture( temp ) );
			}
		}


		/** geometry convert */
		/** 
		 *	object file 은 vertex 와 normal texture coord 를 각각 서로 따로 따로
		 *	구성해서 각각의 index 를 삼각형 face 가 지정하지만, SGRTx2 에서는
		 *	vertex, normal, texture coord 가 한 set 로 indexing 되므로
		 *	좀 복잡한 converting 을 수행해야 한다. 따라서, 중복되는 vertex 가 
		 *	있을수 있어도 face 의 데이터 갯수만큼 vertex, normal, texture 데이터를 그대로 생성하자.
		 */
		/** vertex data */
		int triangleCount = (int) pOBJObject->m_FaceList.size();
		int vertexCount = triangleCount * 3;
		float *vertexArray = (float*) malloc( sizeof( float ) * 3 * vertexCount );
		float *normalArray = NULL, *textureCoordArray = NULL, *visibilityArray = NULL, *colorArray = NULL;

		pObject->setVertexCount( vertexCount );
		pObject->setVertexArray( vertexArray );

		if ( pOBJObject->isExistNormal() ) {
			normalArray = (float*) malloc( sizeof( float ) * 3 * vertexCount );
			pObject->setNormalArray( normalArray );
		}
		if ( pOBJObject->isExistTextureCoord() ) {
			textureCoordArray = (float*) malloc( sizeof( float ) * 2 * vertexCount );
			pObject->setUVArray( textureCoordArray );
		}
		visibilityArray = (float*) malloc( sizeof( float ) * 3 * vertexCount );
		pObject->setVisibilityArray( visibilityArray );
		colorArray = (float*) malloc( sizeof( float ) * 3 * vertexCount );
		pObject->setColorArray( colorArray );
		
		for ( int i = 0; i < pOBJGeometryData->m_VertexList.size(); i++) {
			pOBJNormalWeightList[i].clear();
			pDestOBJNormalWeightList[i].clear();
		}

		/** index data */
		int *indexArray = (int*) malloc( sizeof( int ) * 3 * triangleCount );

		pObject->setTriangleCount( triangleCount );
		pObject->setIndexArray( indexArray );

		int index0, index1, index2;

		/** 삼각형 face 정보를 구성하면서 vertex, normal 과 texture coord 도 같이 구성 */
		for ( int i = 0; i < triangleCount; ++i ) {

			index0 = i * 3 + 0;
			index1 = i * 3 + 1;
			index2 = i * 3 + 2;

			indexArray[ index0 ] = index0;
			indexArray[ index1 ] = index1;
			indexArray[ index2 ] = index2;

			/** vertex, normal 과 texture 의 index 가 아니라 실제데이터를 넣는다. */
			vertexArray[ index0 * 3 + 0 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v0 ]->x;
			vertexArray[ index0 * 3 + 1 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v0 ]->y;
			vertexArray[ index0 * 3 + 2 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v0 ]->z;

			vertexArray[ index1 * 3 + 0 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v1 ]->x;
			vertexArray[ index1 * 3 + 1 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v1 ]->y;
			vertexArray[ index1 * 3 + 2 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v1 ]->z;

			vertexArray[ index2 * 3 + 0 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v2 ]->x;
			vertexArray[ index2 * 3 + 1 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v2 ]->y;
			vertexArray[ index2 * 3 + 2 ] = pOBJGeometryData->m_VertexList[ pOBJObject->m_FaceList[ i ]->v2 ]->z;

			if ( pOBJObject->isExistNormal() ) {

				normalArray[ index0 * 3 + 0 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n0 ]->x;
				normalArray[ index0 * 3 + 1 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n0 ]->y;
				normalArray[ index0 * 3 + 2 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n0 ]->z;

				normalArray[ index1 * 3 + 0 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n1 ]->x;
				normalArray[ index1 * 3 + 1 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n1 ]->y;
				normalArray[ index1 * 3 + 2 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n1 ]->z;

				normalArray[ index2 * 3 + 0 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n2 ]->x;
				normalArray[ index2 * 3 + 1 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n2 ]->y;
				normalArray[ index2 * 3 + 2 ] = pOBJGeometryData->m_NormalList[ pOBJObject->m_FaceList[ i ]->n2 ]->z;

				GOBJNormalWeight data;
				data.normal.x = normalArray[ index0 * 3 + 0 ];
				data.normal.y = normalArray[ index0 * 3 + 1 ];
				data.normal.z = normalArray[ index0 * 3 + 2 ];
				data.index = index0;
				pOBJNormalWeightList[ pOBJObject->m_FaceList[ i ]->v0 ].push_back( data );

				data.normal.x = normalArray[ index1 * 3 + 0 ];
				data.normal.y = normalArray[ index1 * 3 + 1 ];
				data.normal.z = normalArray[ index1 * 3 + 2 ];
				data.index = index1;
				pOBJNormalWeightList[ pOBJObject->m_FaceList[ i ]->v1 ].push_back( data );

				data.normal.x = normalArray[ index2 * 3 + 0 ];
				data.normal.y = normalArray[ index2 * 3 + 1 ];
				data.normal.z = normalArray[ index2 * 3 + 2 ];
				data.index = index2;
				pOBJNormalWeightList[ pOBJObject->m_FaceList[ i ]->v2 ].push_back( data );
			}

			if ( pOBJObject->isExistTextureCoord() ) {
				textureCoordArray[ index0 * 2 + 0 ] = pOBJGeometryData->m_TextureCoordList[ pOBJObject->m_FaceList[ i ]->t0 ]->x;
				textureCoordArray[ index0 * 2 + 1 ] = pOBJGeometryData->m_TextureCoordList[ pOBJObject->m_FaceList[ i ]->t0 ]->y;

				textureCoordArray[ index1 * 2 + 0 ] = pOBJGeometryData->m_TextureCoordList[ pOBJObject->m_FaceList[ i ]->t1 ]->x;
				textureCoordArray[ index1 * 2 + 1 ] = pOBJGeometryData->m_TextureCoordList[ pOBJObject->m_FaceList[ i ]->t1 ]->y;

				textureCoordArray[ index2 * 2 + 0 ] = pOBJGeometryData->m_TextureCoordList[ pOBJObject->m_FaceList[ i ]->t2 ]->x;
				textureCoordArray[ index2 * 2 + 1 ] = pOBJGeometryData->m_TextureCoordList[ pOBJObject->m_FaceList[ i ]->t2 ]->y;
			}

		}


		/**
		 *	normal 을 합한다. normal 을 합칠때, 현재 기준이 되는 normal 과 다른 normal 들과의
		 *	weight 를 계산해서 현재 normal 의 새로운 방향을 계산하는데 이용할지를 결정해서 적용한다.
		 */
		for ( int i = 0; i < (int) pOBJGeometryData->m_VertexList.size(); ++i ) {
			int normalCount = (int) pOBJNormalWeightList[ i ].size();
			for ( int j = 0; j < normalCount; ++j ) {
				GOBJNormalWeight data = pOBJNormalWeightList[ i ][ j ];
				for ( int k = 0; k < normalCount; ++k ) {
					if ( j == k ) continue;
					/** 
					 *	두개의 normal 의 cos 값이 0.5 보다 클때만 누적시킨다. 
					 *	누적된 결과가 다음 normal 들에게 영향을 주면 안되므로, 새로운 공간에 누적시켜야 한다.
					 */
					if ( pOBJNormalWeightList[ i ][ j ].normal.innerProduct( pOBJNormalWeightList[ i ][ k ].normal ) > 0.5 ) {
						data.normal += pOBJNormalWeightList[ i ][ k ].normal;
					}
				}
				data.normal = data.normal.normalize();
				pDestOBJNormalWeightList[ i ].push_back( data );
			}
		}

		/**
		 *	새로 계산된 object 데이터에 업데이트 시킨다.
		 */
		for ( int i = 0; i < (int) pOBJGeometryData->m_VertexList.size(); ++i ) {
			int normalCount = (int) pDestOBJNormalWeightList[ i ].size();
			for ( int j = 0; j < normalCount; ++j ) {
				int index = pDestOBJNormalWeightList[ i ][ j ].index;
				normalArray[ index * 3 + 0 ] = pDestOBJNormalWeightList[ i ][ j ].normal.x;
				normalArray[ index * 3 + 1 ] = pDestOBJNormalWeightList[ i ][ j ].normal.y;
				normalArray[ index * 3 + 2 ] = pDestOBJNormalWeightList[ i ][ j ].normal.z;
			}
		}

		//float *tempVisibilityArray = new float[pOBJGeometryData->m_VertexList.size()];

		//// Visibility 계산
		//for ( int i = 0; i < pObject->getVertexCount(); ++i )
		//	visibilityArray[i] = 0.0f;
		//for ( int i = 0; i < (int)pOBJGeometryData->m_VertexList.size(); ++i )
		//	tempVisibilityArray[i] = 0.0f;

		//for ( int i = 0; i < pObject->getTriangleCount(); ++i )
		//{
		//	int v0[3];
		//	v0[0] = pOBJObject->m_FaceList[i]->v0;
		//	v0[1] = pOBJObject->m_FaceList[i]->v1;
		//	v0[2] = pOBJObject->m_FaceList[i]->v2;

		//	for ( int j = 0; j < pObject->getTriangleCount(); ++j )
		//	{
		//		if( i == j )
		//			continue;
		//		int v1[3];
		//		v1[0] = pOBJObject->m_FaceList[i]->v0;
		//		v1[1] = pOBJObject->m_FaceList[i]->v1;
		//		v1[2] = pOBJObject->m_FaceList[i]->v2;

		//		bool share = false; //!< 공통된 vertex 를 가지고 있는지
		//		for( int i0 = 0; i0 < 3; i0++ )
		//		{
		//			for( int j0 = 0; j0 < 3; j0++ )
		//			{
		//				if( v0[i0] == v1[j0] )
		//					share = true;
		//			}
		//		}
		//		if( share )
		//		{
		//			// pObject 와 pOBJObject 의 triangle index 가 같다고 가정
		//			GVector center_i = pObject->getCenter( i );
		//			GVector center_j = pObject->getCenter( j );
		//			//pList->getTriangleWrapper( i )->visibility += max( getNormal( i ).innerProduct( (center_j - center_i).normalize() ), 0.0 );
		//			float visibility = max( pObject->getNormal( i ).innerProduct( (center_j - center_i).normalize() ), 0.0f );
		//			//faceVisibilityArray[i] = visibility;
		//			for( int i0 = 0; i0 < 3; i0++ )
		//				tempVisibilityArray[ v0[i0] ] += visibility;
		//		}
		//	}
		//}

		//for ( int i = 0; i < pObject->getTriangleCount(); ++i )
		//{
		//	int v0[3];
		//	v0[0] = pOBJObject->m_FaceList[i]->v0;
		//	v0[1] = pOBJObject->m_FaceList[i]->v1;
		//	v0[2] = pOBJObject->m_FaceList[i]->v2;

		//	for ( int j = 0; j < 3; ++j )
		//		visibilityArray[ i * 3 + j ] = tempVisibilityArray[ v0[j] ];
		//}

		//
		//for ( int i = 0; i < pObject->getTriangleCount(); ++i )
		//{
		//	const int *indices = pObject->getIndexArray() + i * 3 + 0;
		//	//m_pVisibilityArray[ indices[0] ] = pList->getTriangleWrapper( i )->visibility;
		//	//m_pVisibilityArray[ indices[1] ] = pList->getTriangleWrapper( i )->visibility;
		//	//m_pVisibilityArray[ indices[2] ] = pList->getTriangleWrapper( i )->visibility;

		//	for( int j = 0; j < 3; j++ )
		//	{
		//		colorArray[ indices[j] * 3 + 0 ] += visibilityArray[ indices[j] ];
		//		colorArray[ indices[j] * 3 + 1 ] += visibilityArray[ indices[j] ];
		//		colorArray[ indices[j] * 3 + 2 ] += visibilityArray[ indices[j] ];
		//		/*colorArray[ indices[j] * 3 + 0 ] = 1.0f;
		//		colorArray[ indices[j] * 3 + 1 ] = 1.0f;
		//		colorArray[ indices[j] * 3 + 2 ] = 1.0f;*/
		//	}
		//}

		//delete[] tempVisibilityArray;

		pObject->setPolygonType( typePolygonTriangle );
		pObject->convertToWorldObject();
		pScene->addObject( pObject );

	}

	delete[] pOBJNormalWeightList;
	delete[] pDestOBJNormalWeightList;

	GLogManager::logging( LOG_INFO, " -> Converted to SGRTx2 Object" );

	return errorNo;
}


/**
 *	DAT 파일 Format 을 읽어서 polygon 데이터를 구성한다.
 *	DAT 는 
 *
 *	하나의 face 를 구성하는 vertex 수와 데이터들로 구성되어 있다.
 *	face 는 삼각형 데이터라는 것을 가정.
 *
 *	vertex수
 *	x y z nx ny nz
 *  vertex수
 *	x y z nx ny nz
 *	....
 *
 *	급하게 만들어서 버그가 있을수 잇음.;; 시간날대 다시 확인
 *
 */
bool GSceneManager::loadDatFile( const char*szFileName, GPolygonObject *pObject )
{
	char data[2048] = { 0x00, };
	bool bSuccess = true;

	FILE *fp = fopen( szFileName, "rt" );
	if ( fp == NULL ) {
		GLogManager::logging( LOG_ERROR, "%s Data File Load Error!", szFileName );
		return false;
	}

	GLogManager::logging( LOG_INFO, "%s Data File Load Start.", szFileName );

	/** 
	 *	vertex 데이터가 몇개인지 모르므로 임시로 vertex list 를 
	 *	주욱 담아둘 vector
	 */
	vector<GVertexInfo> m_VertexList;
	m_VertexList.reserve( 300000 );

	GVertexInfo vertexInfo;
	int faceVertex;

	while( !feof( fp ) ) {

		if ( !readLine( data, 2048, fp ) ) 
			break;

		/** 
		 *	반복되는 데이터중 한 face 를 이루는 vertex 수 
		 *	무조건 3 이어야 한다.
		 */
		sscanf( data, "%d", &faceVertex );
		if ( faceVertex != 3 ) {
			bSuccess = false;
			GLogManager::logging( LOG_ERROR, "Face Vertex is not three!" );
			break;
		}
		for ( int i = 0; i < 3; ++i ) {
			if ( !readLine( data, 2048, fp ) ) {
				bSuccess = false;
				break;
			}
			sscanf( data, "%f %f %f %f %f %f", vertexInfo.vertex, 
									   vertexInfo.vertex + 1, 
									   vertexInfo.vertex + 2,
									   vertexInfo.normal,
									   vertexInfo.normal + 1,
									   vertexInfo.normal + 2 );
			m_VertexList.push_back( vertexInfo );
		}
	}

	GLogManager::logging( LOG_INFO, "%s Vertex Count = %d", szFileName, m_VertexList.size() );

	if ( bSuccess && m_VertexList.size() % 3 == 0 ) {

		int TriangleCount = (int) m_VertexList.size() / 3;

		/**
		 *	읽어들인 데이터로 삼각형 polygon object 를 구성한다.
		 */
		float* vertexArray = (float*) malloc( sizeof( float ) * m_VertexList.size() * 3 );
		float* normalArray = (float*) malloc( sizeof( float ) * m_VertexList.size() * 3 );
		int* indexArray = (int*) malloc( sizeof( int ) * TriangleCount * 3 );

		for ( int i = 0; i < (int) m_VertexList.size(); ++i ) {
			vertexArray[ i * 3 + 0 ] = m_VertexList[ i ].vertex[ 0 ];
			vertexArray[ i * 3 + 1 ] = m_VertexList[ i ].vertex[ 1 ];
			vertexArray[ i * 3 + 2 ] = m_VertexList[ i ].vertex[ 2 ];
			normalArray[ i * 3 + 0 ] = m_VertexList[ i ].normal[ 0 ];
			normalArray[ i * 3 + 1 ] = m_VertexList[ i ].normal[ 1 ];
			normalArray[ i * 3 + 2 ] = m_VertexList[ i ].normal[ 2 ];
		}

		/** vertexArray 순서대로 */
		for ( int i = 0; i < TriangleCount; ++i ) {
			indexArray[ i * 3 + 0 ] = i * 3 + 0;
			indexArray[ i * 3 + 1 ] = i * 3 + 1;
			indexArray[ i * 3 + 2 ] = i * 3 + 2;
		}

		pObject->setVertexCount( (int) m_VertexList.size() );
		pObject->setVertexArray( vertexArray );
		pObject->setNormalArray( normalArray );

		pObject->setTriangleCount( TriangleCount );
		pObject->setIndexArray( indexArray );

		bSuccess = true;

	} else {
		bSuccess = false;
		GLogManager::logging( LOG_ERROR, "vertex data count is not valid" );
	}

	m_VertexList.clear();

	fclose( fp );

	return bSuccess;
}

/**
 *	주석, 공백등을 제거한 데이터를 만날때까지 loop 를 돌면서
 *	한줄을 읽어온다.
 */
bool GSceneManager::readLine( char *data, int size, FILE *fp )
{
	while( !feof( fp ) ) {
		/**
		 *	데이터가 없다면 continue;
		 */
		if ( fgets( data, 2040, fp ) == NULL )
			continue;
			
		GUtil::removeCRLF( data );

		/** 
		 *	주석이나 공백일때.
		 */
		if ( data[ 0 ] == '#' || data[ 0 ] == 0x00 ) continue;
	
		/** 
		 *	실제 데이터 일때만 리턴.
		 */	
		return true;
	}
	
	return false;
}

/**
 *	Binary 형식의 File 을 로드한다.
 */
GError GSceneManager::loadBinaryFile( const char* szFileName, GScene* ppScene )
{
	return errorUnknown;
}

/**
 *	현재 Scene 을 file 로 저장한다.
 */
GError GSceneManager::saveScene( const char* szFileName, GScene* pScene )
{
	return errorUnknown;
}