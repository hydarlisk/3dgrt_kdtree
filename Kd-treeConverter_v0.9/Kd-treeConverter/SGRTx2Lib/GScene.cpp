#include "GScene.h"
#include "GPolygonObject.h"
#include "GObjectNumberGenerator.h"
#include "GTextureManager.h"
#include "GKDTreeStructure.h"
#include "GBVHStructure.h"
#include "GGridStructure.h"


GScene::GScene(void)
	: m_LoadedCameraCount(0)
{
	m_szSceneName[0] = 0x00;
	m_iSceneNumber = GObjectNumberGenerator::generateSceneNumber();

	m_szVersion[0] = 0x00;
	m_szOutputPath[0] = 0x00;
	m_szTexturePath[0] = 0x00;
	m_szSceneBasePath[0] = 0x00;

	m_Resolution.x = 800;
	m_Resolution.y = 600;
    m_SuperSampling.x = 1;
	m_SuperSampling.y = 1;
	m_RenderingBlock.x = 1;
	m_RenderingBlock.y = 1;

	m_FrontFace = faceCCW;

	m_iGeometryChangeTimestamp = 0;
	m_iLastConvertRenderScene = -1;

	m_LastUseSpatialStructure = USE_NONE;
	m_CurrUseSpatialStructure = USE_KDTREE;
	memset(m_iLastGeomTimestamp_For_SpatialStructure, 0x00, sizeof(m_iLastGeomTimestamp_For_SpatialStructure));

	m_bJittering		= true;
	m_bBackFaceCulling	= false;
	m_bUseTexture		= true;
	m_bLocalShading		= true;
	m_bShadow			= false;
	m_iMaxReflectionDepth = 3;

	m_pImageBuffer = NULL;
	m_pDirectIllumImageBuffer = NULL;
	m_pIndirectIllumImageBuffer = NULL;

	m_iCPUThreadCount = 4;
	m_GPUBlockSize.x = 4;
	m_GPUBlockSize.y = 64;
	m_CPUPacketSize.x = 1;
	m_CPUPacketSize.y = 1;

	m_fFPS = 0.0f;
	m_fFPS2 = 0.0f;

	m_bOpenGLLoadTexture = false;

	m_bKDTreeFileType = SAH;
	m_bKDTreeFileLoad = false;
	m_bKDTreeFileSave = false;
	m_szKDTreeLoadFilePath[0] = 0x00;
	m_szKDTreeSaveFilePath[0] = 0x00;
	m_pKDTree = NULL;
	m_pEmptyKDTree = NULL;
	m_pBVH = NULL;
	m_pGrid= NULL;

	m_bBloomingFilter = false;
	m_fBloomingRadius = 1.0f;
	m_fBloomingWeight = 0.03f;

	m_bAntialiasingFilter = false;
	m_bGrayScaleFilter = false;
	m_bEdgeDetectionFilter = false;
	m_bBlurFilter = false;

	m_AdaptiveSamplingType = adaptiveSubpixel;
	m_bSamplingDebugInfo = false;

	m_AdaptiveSamplingCompareType = COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL  | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
									COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
									COMPARE_ETC_REGION;

	setHighLevelAdaptiveSamplingThreshold();

	m_fOnlyColorThreshold = 0.1f;

	m_bRun10Times = false;
	strcpy(m_szProfileResultPath, "profile.out");
	m_bProfileFlag = false;
	m_bTestFlag = false;
}

GScene::~GScene(void)
{
	/** 
	 *	light 는 list 만 클리어 한다. 실제 light object 는
	 *	object list 에도 들어 있으므로, object 삭제시 삭제될 것이다.
	 */
	m_LightList.clear();

	//! delete camera
	//for ( int i = 0; i < (int)m_CameraList.size(); ++i )
	//	delete m_CameraList[i];
	//m_CameraList.clear();

	if ( m_pImageBuffer )
		delete m_pImageBuffer;
	if ( m_pDirectIllumImageBuffer )
		delete m_pDirectIllumImageBuffer;
	if ( m_pIndirectIllumImageBuffer )
		delete m_pIndirectIllumImageBuffer;
	if ( m_pKDTree )
		delete m_pKDTree;
	if ( m_pBVH )
		delete m_pBVH;
	if ( m_pGrid )
		delete m_pGrid;

	clearObject();
}

void GScene::setSceneName( const char* name )
{
	sprintf( m_szSceneName, "%s", name );
}

const char* GScene::getSceneName()
{
	return m_szSceneName;
}

int GScene::getSceneNumber()
{
	return m_iSceneNumber;
}

bool GScene::initalize()
{
	GTextureManager::getInstance()->loadAllTexture();

	return true;
}

void GScene::setKdTreeFileType( int type )
{
	m_bKDTreeFileType = type;
}

int GScene::getKdTreeFileType()
{
	return m_bKDTreeFileType;
}

void GScene::setKdTreeFileLoad( bool flag )
{
	m_bKDTreeFileLoad = flag;
}

bool GScene::isKdTreeFileLoad()
{
	return m_bKDTreeFileLoad;
}
	
void GScene::setKdTreeFileSave( bool flag )
{
	m_bKDTreeFileSave = flag;
}

bool GScene::isKdTreeFileSave()
{
	return m_bKDTreeFileSave;
}

void GScene::setKdTreeLoadFilePath( const char* path )
{
	strncpy( m_szKDTreeLoadFilePath, path, MAX_PATH_LENGTH );
	m_szKDTreeLoadFilePath[ MAX_PATH_LENGTH - 1 ] = 0x00;
}

const char* GScene::getKdTreeLoadFilePath()
{
	return m_szKDTreeLoadFilePath;
}

void GScene::setKdTreeSaveFilePath( const char* path )
{
	strncpy( m_szKDTreeSaveFilePath, path, MAX_PATH_LENGTH );
	m_szKDTreeSaveFilePath[ MAX_PATH_LENGTH - 1 ] = 0x00;
}

const char* GScene::getKdTreeSaveFilePath()
{
	return m_szKDTreeSaveFilePath;
}

void GScene::setEnableJittering( bool flag )
{
	m_bJittering = flag;
}

bool GScene::isEnableJittering()
{
	return m_bJittering;
}
	
void GScene::setEnableShadow( bool flag )
{
	m_bShadow = flag;
}

bool GScene::isEnableShadow()
{
	return m_bShadow;
}

void GScene::setCPUThreadCount( int count )
{
	m_iCPUThreadCount = count;
}

int GScene::getCPUThreadCount()
{
	return m_iCPUThreadCount;
}

void GScene::setCPUPacketSize( int x, int y )
{
	m_CPUPacketSize.x = x;
	m_CPUPacketSize.y = y;
}

void GScene::setPrimaryOIDRegionColorThreshold( float f )
{
	m_fPrimaryOIDRegionColorThreshold = f;
}

float GScene::getPrimaryOIDRegionColorThreshold()
{
	return m_fPrimaryOIDRegionColorThreshold;
}

void GScene::setPrimaryNormalRegionColorThreshold( float f )
{
	m_fPrimaryNormalRegionColorThreshold = f;
}

float GScene::getPrimaryNormalRegionColorThreshold()
{
	return m_fPrimaryNormalRegionColorThreshold;
}

void GScene::setPrimaryShadowRegionColorThreshold( float f )
{
	m_fPrimaryShadowRegionColorThreshold = f;
}

float GScene::getPrimaryShadowRegionColorThreshold()
{
	return m_fPrimaryShadowRegionColorThreshold;
}

void GScene::setPrimaryTextureRegionColorThreshold( float f )
{
	m_fPrimaryTextureRegionColorThreshold = f;
}

float GScene::getPrimaryTextureRegionColorThreshold()
{
	return m_fPrimaryTextureRegionColorThreshold;
}

void GScene::setSecondaryOIDRegionColorThreshold( float f )
{
	m_fSecondaryOIDRegionColorThreshold = f;
}

float GScene::getSecondaryOIDRegionColorThreshold()
{
	return m_fSecondaryOIDRegionColorThreshold;
}

void GScene::setSecondaryNormalRegionColorThreshold( float f )
{
	m_fSecondaryNormalRegionColorThreshold = f;
}

float GScene::getSecondaryNormalRegionColorThreshold()
{
	return m_fSecondaryNormalRegionColorThreshold;
}

void GScene::setSecondaryShadowRegionColorThreshold( float f )
{
	m_fSecondaryShadowRegionColorThreshold = f;
}

float GScene::getSecondaryShadowRegionColorThreshold()
{
	return m_fSecondaryShadowRegionColorThreshold;
}

void GScene::setSecondaryTextureRegionColorThreshold( float f )
{
	m_fSecondaryTextureRegionColorThreshold = f;
}

float GScene::getSecondaryTextureRegionColorThreshold()
{
	return m_fSecondaryTextureRegionColorThreshold;
}

void GScene::setEtcRegionColorThreshold( float f )
{
	m_fEtcRegionColorThreshold = f;
}

float GScene::getEtcRegionColorThreshold()
{
	return m_fEtcRegionColorThreshold;
}

void GScene::setOnlyColorThreshold( float f )
{
	m_fOnlyColorThreshold = f;
}

float GScene::getOnlyColorThreshold()
{
	return m_fOnlyColorThreshold;
}

void GScene::setAdaptiveDetectionStageTime( float t )
{
	m_fAdaptiveDetectionStageTime = t;
}

float GScene::getAdaptiveDetectionStageTime()
{
	return m_fAdaptiveDetectionStageTime;
}

void GScene::setAdaptiveRayRate( float r )
{
	m_fAdaptiveRayRate = r;
}

float GScene::getAdaptiveRayRate()
{
	return m_fAdaptiveRayRate;
}


void GScene::setLowLevelAdaptiveSamplingThreshold()
{
	setPrimaryOIDRegionColorThreshold( 0.05f );
	setPrimaryNormalRegionColorThreshold( 0.06f );
	setPrimaryShadowRegionColorThreshold( 1.0f );
	setPrimaryTextureRegionColorThreshold( 1.0f );

	setSecondaryOIDRegionColorThreshold( 0.05f );
	setSecondaryNormalRegionColorThreshold( 0.06f );
	setSecondaryShadowRegionColorThreshold( 1.0f );
	setSecondaryTextureRegionColorThreshold( 1.0f );

	setEtcRegionColorThreshold( 1.0f );

	setAdaptiveSamplingCompareType( 
		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
		COMPARE_ETC_REGION );
}

void GScene::setGoodLevelAdaptiveSamplingThreshold()
{
	setPrimaryOIDRegionColorThreshold( 0.05f );
	setPrimaryNormalRegionColorThreshold( 0.06f );
	setPrimaryShadowRegionColorThreshold( 0.06f );
	setPrimaryTextureRegionColorThreshold( 0.6f );

	setSecondaryOIDRegionColorThreshold( 0.05f );
	setSecondaryNormalRegionColorThreshold( 0.06f );
	setSecondaryShadowRegionColorThreshold( 0.06f );
	setSecondaryTextureRegionColorThreshold( 0.6f );

	setEtcRegionColorThreshold( 0.6f );

	setAdaptiveSamplingCompareType( 
		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
		COMPARE_ETC_REGION );
}

void GScene::setHighLevelAdaptiveSamplingThreshold()
{
	setPrimaryOIDRegionColorThreshold( 0.05f );
	setPrimaryNormalRegionColorThreshold( 0.06f );
	setPrimaryShadowRegionColorThreshold( 0.06f );
	setPrimaryTextureRegionColorThreshold( 0.6f );

	setSecondaryOIDRegionColorThreshold( 0.05f );
	setSecondaryNormalRegionColorThreshold( 0.06f );
	setSecondaryShadowRegionColorThreshold( 0.06f );
	setSecondaryTextureRegionColorThreshold( 0.6f );

	setEtcRegionColorThreshold( 0.6f );

	setAdaptiveSamplingCompareType( 
		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
		COMPARE_ETC_REGION );
}

void GScene::setVeryHighLevelAdaptiveSamplingThreshold()
{
	setPrimaryOIDRegionColorThreshold( 0.05f );
	setPrimaryNormalRegionColorThreshold( 0.06f );
	setPrimaryShadowRegionColorThreshold( 0.06f );
	setPrimaryTextureRegionColorThreshold( 0.07f );

	setSecondaryOIDRegionColorThreshold( 0.05f );
	setSecondaryNormalRegionColorThreshold( 0.06f );
	setSecondaryShadowRegionColorThreshold( 0.06f );
	setSecondaryTextureRegionColorThreshold( 0.07f );

	setEtcRegionColorThreshold( 0.07f );

	setAdaptiveSamplingCompareType( 
		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
		COMPARE_ETC_REGION );
}


void GScene::setVeryHighLevelExceptTextureAdaptiveSamplingThreshold( float oidColor, float tColor )
{
	setPrimaryOIDRegionColorThreshold( oidColor );
	setPrimaryNormalRegionColorThreshold( 0.06f );
	setPrimaryShadowRegionColorThreshold( 0.06f );
	setPrimaryTextureRegionColorThreshold( tColor );

	setSecondaryOIDRegionColorThreshold( oidColor );
	setSecondaryNormalRegionColorThreshold( 0.06f );
	setSecondaryShadowRegionColorThreshold( 0.06f );
	setSecondaryTextureRegionColorThreshold( tColor );

	setEtcRegionColorThreshold( tColor );

	setAdaptiveSamplingCompareType( 
		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
		COMPARE_ETC_REGION );
}

//
//void GScene::setLowLevelAdaptiveSamplingThreshold()
//{
//	setPrimaryOIDRegionColorThreshold( 0.0f );
//	setPrimaryNormalRegionColorThreshold( 0.3f );
//	setPrimaryShadowRegionColorThreshold( 0.3f );
//	setPrimaryTextureRegionColorThreshold( 1.0f );
//
//	setSecondaryOIDRegionColorThreshold( 0.0f );
//	setSecondaryNormalRegionColorThreshold( 0.3f );
//	setSecondaryShadowRegionColorThreshold( 0.3f );
//	setSecondaryTextureRegionColorThreshold( 1.0f );
//
//	setEtcRegionColorThreshold( 1.0f );
//
//	setAdaptiveSamplingCompareType( 
//		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
//		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE | COMPARE_ETC_REGION );
//}
//
//void GScene::setGoodLevelAdaptiveSamplingThreshold()
//{
//	setPrimaryOIDRegionColorThreshold( 0.1f );
//	setPrimaryNormalRegionColorThreshold( 0.1f );
//	setPrimaryShadowRegionColorThreshold( 0.1f );
//	setPrimaryTextureRegionColorThreshold( 0.5f );
//
//	setSecondaryOIDRegionColorThreshold( 0.1f );
//	setSecondaryNormalRegionColorThreshold( 0.1f );
//	setSecondaryShadowRegionColorThreshold( 0.1f );
//	setSecondaryTextureRegionColorThreshold( 0.5f );
//
//	setEtcRegionColorThreshold( 0.5f );
//
//	setAdaptiveSamplingCompareType( 
//		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
//		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
//		COMPARE_ETC_REGION );
//}
//
//void GScene::setHighLevelAdaptiveSamplingThreshold()
//{
//	setPrimaryOIDRegionColorThreshold( 0.05f );
//	setPrimaryNormalRegionColorThreshold( 0.1f );
//	setPrimaryShadowRegionColorThreshold( 0.1f );
//	setPrimaryTextureRegionColorThreshold( 0.3f );
//
//	setSecondaryOIDRegionColorThreshold( 0.05f );
//	setSecondaryNormalRegionColorThreshold( 0.1f );
//	setSecondaryShadowRegionColorThreshold( 0.1f );
//	setSecondaryTextureRegionColorThreshold( 0.3f );
//
//	setEtcRegionColorThreshold( 0.3f );
//
//	setAdaptiveSamplingCompareType( 
//		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
//		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
//		COMPARE_ETC_REGION );
//}
//
//void GScene::setVeryHighLevelAdaptiveSamplingThreshold()
//{
//	setPrimaryOIDRegionColorThreshold( 0.05f );
//	setPrimaryNormalRegionColorThreshold( 0.1f );
//	setPrimaryShadowRegionColorThreshold( 0.1f );
//	setPrimaryTextureRegionColorThreshold( 0.15f );
//
//	setSecondaryOIDRegionColorThreshold( 0.05f );
//	setSecondaryNormalRegionColorThreshold( 0.1f );
//	setSecondaryShadowRegionColorThreshold( 0.1f );
//	setSecondaryTextureRegionColorThreshold( 0.15f );
//
//	setEtcRegionColorThreshold( 0.15f );
//
//	setAdaptiveSamplingCompareType( 
//		COMPARE_PRIMARY_OID | COMPARE_PRIMARY_NORMAL | COMPARE_PRIMARY_SHADOW | COMPARE_PRIMARY_TEXTURE |
//		COMPARE_SECONDARY_OID | COMPARE_SECONDARY_NORMAL | COMPARE_SECONDARY_SHADOW | COMPARE_SECONDARY_TEXTURE |
//		COMPARE_ETC_REGION );
//}

void GScene::setAdaptiveSamplingType( enumAdaptiveSamplingType type )
{
	m_AdaptiveSamplingType = type;
}

enumAdaptiveSamplingType GScene::getAdaptiveSamplingType()
{
	return m_AdaptiveSamplingType;
}
	
void GScene::setAdaptiveSamplingCompareType( int compare )
{
	m_AdaptiveSamplingCompareType = compare;
}

int GScene::getAdaptiveSamplingCompareType()
{
	return m_AdaptiveSamplingCompareType;
}

void GScene::setEnableSamplingDebugInfo( bool flag )
{
	m_bSamplingDebugInfo = flag;
}

bool GScene::isEnableSamplingDebugInfo()
{
	return m_bSamplingDebugInfo;
}
	
void GScene::setEnableLocalShading( bool flag )
{
	m_bLocalShading = flag;
}

bool GScene::isEnableLocalShading()
{
	return m_bLocalShading;
}

GDimension GScene::getCPUPacketSize()
{
	return m_CPUPacketSize;
}

void GScene::setGPUBlockSize( int x, int y )
{
	m_GPUBlockSize.x = x;
	m_GPUBlockSize.y = y;
}

GDimension GScene::getGPUBlockSize()
{
	return m_GPUBlockSize;
}

void GScene::setUseAntialiasingFilter( bool flag )
{
	m_bAntialiasingFilter = flag;
}

bool GScene::isUseAntialiasingFilter()
{
	return m_bAntialiasingFilter;
}
	
void GScene::setUseBlurFilter( bool flag )
{
	m_bBlurFilter = flag;
}

bool GScene::isUseBlurFilter()
{
	return m_bBlurFilter;
}

void GScene::setUseEdgeDetectionFilter( bool flag )
{
	m_bEdgeDetectionFilter = flag;
}

bool GScene::isUseEdgeDetectionFilter()
{
	return m_bEdgeDetectionFilter;
}

void GScene::setUseGrayScaleFilter( bool flag )
{
	m_bGrayScaleFilter = flag;
}

bool GScene::isUseGrayScaleFilter()
{
	return m_bGrayScaleFilter;
}

void GScene::setOpenGLLoadTexture( bool flag )
{
	m_bOpenGLLoadTexture = flag;
}

bool GScene::isOpenGLLoadTexture()
{
	return m_bOpenGLLoadTexture;
}

float GScene::getFPS()
{
	return m_fFPS;
}

void GScene::setFPS( float fps )
{
	m_fFPS = fps;
}

float GScene::getFPS2()
{
	return m_fFPS2;
}

void GScene::setFPS2( float fps )
{
	m_fFPS2 = fps;
}

void GScene::setBloomingFilter( bool flag )
{
	m_bBloomingFilter = flag;
}

bool GScene::isBloomingFilter()
{
	return m_bBloomingFilter;
}

void GScene::setBloomingRadius( float radius )
{
	m_fBloomingRadius = radius;
}

float GScene::getBloomingRadius()
{
	return m_fBloomingRadius;
}

void GScene::setBloomingWeight( float w )
{
	m_fBloomingWeight = w;
}

float GScene::getBloomingWeight()
{
	return m_fBloomingWeight;
}

void GScene::setRenderingBlock( int x, int y )
{
	m_RenderingBlock.x = x;
	m_RenderingBlock.y = y;
}

GDimension GScene::getRenderingBlock()
{
	return m_RenderingBlock;
}

/**
 *	Scene 의 모든 Object 를 제거한다.
 */
void GScene::clearObject()
{
	vector<GObject*>::iterator iter;
	for ( iter = m_ObjectList.begin(); iter != m_ObjectList.end(); ++iter ) {
		delete (*iter);
	}
	m_ObjectList.clear();
	m_SelecteGObjectList.clear();
}

/** debugging 관련 object 만삭제 */
void GScene::clearDebugObject()
{
	vector<GObject*>::iterator iter;

	for ( iter = m_ObjectList.begin(); iter < m_ObjectList.end(); ) {
		if ( (*iter)->isDebugObject() ) {
			delete (*iter);
			iter = m_ObjectList.erase( iter );
		} else {
			++iter;
		}
	}

	m_SelecteGObjectList.clear();
}

/**
 *	Scene 에 Object 를 추가한다.
 *	해당 Object 의 필요정보를 세팅시키고, valid 한 object
 *	인지를 확인한다.
 */
GError GScene::addObject( GObject* pObject )
{
	GError error = pObject->validObject();

	if ( error != errorNo )
		return error;

	m_ObjectListCS.lock();
	m_ObjectList.push_back( pObject );
	m_ObjectListCS.unlock();

	if ( !pObject->isDebugObject() )
		m_iGeometryChangeTimestamp++;

	return errorNo;
}

/**
 *	Scene 에서 Object 를 제거한다.
 */
void GScene::removeObject( GObject* pObject )
{
	deselectAllObject();

	m_ObjectListCS.lock();
	vector<GObject*>::iterator iter;
	for ( iter = m_ObjectList.begin(); iter != m_ObjectList.end(); ++iter ) {
		if ( *iter == pObject ) {
			if ( !pObject->isDebugObject() )
				m_iGeometryChangeTimestamp++;
			delete pObject;
			m_ObjectList.erase( iter );
			break;
		}
	}
	m_ObjectListCS.unlock();
}

/**
 *	name 이름을 가진 object 를 삭제한다.
 */
void GScene::removeObject( const char* name )
{
	deselectAllObject();

	m_ObjectListCS.lock();
	vector<GObject*>::iterator iter;
	for ( iter = m_ObjectList.begin(); iter != m_ObjectList.end(); ++iter ) {
		if ( _stricmp( name, (*iter)->getName() ) == 0 ) {
			if ( !(*iter)->isDebugObject() )
				m_iGeometryChangeTimestamp++;
			delete (*iter);
			m_ObjectList.erase( iter );
			break;
		}
	}
	m_ObjectListCS.unlock();

}




GObject* GScene::getObject( int index )
{
	if ( index < 0 || index >= (int)m_ObjectList.size() )
		return NULL;
	return m_ObjectList[index];
}

int GScene::getObjectCount()
{
	return (int)m_ObjectList.size();
}

/**
 *	현재 Scene 의 모든 물체 목록을 리턴한다.
 */
const vector<GObject*>* GScene::getObjectList()
{
	return &m_ObjectList;
}
/**
 *	pObject 물체를 선택 목록에 추가한다.
 *	이미 있다면 추가하지 않는다.
 */
void GScene::selectObject( GObject *pObject )
{
	if ( pObject == NULL ) 
		return;

	for ( int i = 0; i < (int)m_SelecteGObjectList.size(); ++i ) {
		if ( m_SelecteGObjectList[i] == pObject )
			return;
	}
	
	pObject->setSelected( true );
	m_SelecteGObjectList.push_back( pObject );
}

/**
 *	현재 Scene 에서 선택된 물체의 갯수.
 */
int GScene::getSelectedGObjectCount()
{
	return (int)m_SelecteGObjectList.size();
}

/**
 *	현재 Scene 에서 선택된 물체를 리턴한다.
 *	여러개를 선택했다면 그중 첫번째 것만 리턴.
 *	모든 물체를 넘겨받으려면 getSelectedGObjectList() 함수를 사용.
 */
GObject* GScene::getSelectedGObject()
{
	if ( (int)m_SelecteGObjectList.size() > 0 ) {
		return m_SelecteGObjectList[0];
	}
	return NULL;
}

/**
 *	선택된 모든 물체를 deselect 시킨다.
 *	각 object 의 selected 는 false 로 해제시킨다.
 */
void GScene::deselectAllObject()
{
	for ( int i = 0; i < (int)m_SelecteGObjectList.size(); ++i ) {
		m_SelecteGObjectList[i]->setSelected( false );
	}
	m_SelecteGObjectList.clear();
}

/**
 *	물체를 deselect 시킨다.
 */
void GScene::deselectObject( GObject* pObject )
{
	if ( pObject == NULL )
		return;
	vector<GObject*>::iterator iter;
	for ( iter = m_SelecteGObjectList.begin(); iter != m_SelecteGObjectList.end(); ++iter ) {
		if ( (*iter) == pObject ) {
			pObject->setSelected( false );
			m_SelecteGObjectList.erase( iter );
			break;
		}
	}
}

/**
 *	현재 Scene 에서 선택된 물체 목록을 리턴한다.
 */
vector<GObject*>* const GScene::getSelectedGObjectList()
{
	return &m_SelecteGObjectList;
}

/**
 *	object Number 에 해당하는 Object 를 리턴한다.
 */
GObject* GScene::getObjectByNumber( int objectNumber )
{
	for ( int i = 0; i < (int)m_ObjectList.size(); ++i ) {
		if ( m_ObjectList[i]->getObjectNumber() == objectNumber )
			return m_ObjectList[i];
	}
	return NULL;
}

/**
*  center of selected objects
*/
GVector GScene::getCenterOfSelectedGObjects()
{
	GVector Center( 0.0f, 0.0f, 0.0f, 1.0f );

	vector<GObject*>::iterator iter;
	for( iter = m_SelecteGObjectList.begin(); iter != m_SelecteGObjectList.end(); ++iter )
	{
		GBoundingBox *BoundingBox = (*iter)->getBoundingBox();
		Center += *(*iter)->getMatrix() * (( BoundingBox->getMin() + BoundingBox->getMax() ) * 0.5f);
	}
	if( getSelectedGObjectCount() == 0 )
		return GVector( 0.0f, 0.0f, 0.0f, 1.0f );
	Center *= 1.0f/float( getSelectedGObjectCount() );
	return Center;
}

GBoundingBox GScene::getBoundingBoxOfSelectedGObjects()
{
	GVector Min( 1.0e+030f, 1.0e+030f, 1.0e+030f, 1.0f );
	GVector Max( -1.0e+030f, -1.0e+030f, -1.0e+030f, 1.0f );

	if( getSelectedGObjectCount() == 0 )
		return GBoundingBox( GVector(0.0f, 0.0f, 0.0f, 1.0f), GVector(0.0f, 0.0f, 0.0f, 1.0f) );

	vector<GObject*>::iterator iter;
	for( iter = m_SelecteGObjectList.begin(); iter != m_SelecteGObjectList.end(); ++iter )
	{
		GBoundingBox *BoundingBox = (*iter)->getBoundingBox();
		
		for( int i = 0; i < 3; i ++ )
		{
			if( BoundingBox->getMin().GetPointer()[i] < Min.GetPointer()[i] )
				Min.GetPointer()[i] = BoundingBox->getMin().GetPointer()[i];
			if( BoundingBox->getMax().GetPointer()[i] > Max.GetPointer()[i] )
				Max.GetPointer()[i] = BoundingBox->getMax().GetPointer()[i];
		}
	}
	return GBoundingBox( Min, Max );
}
	
/**
 *	Light 추가. Object 에도 추가한다.
 */
void GScene::addLight( GLight *pLight )
{
	m_ObjectListCS.lock();
	m_LightList.push_back( pLight );
	m_ObjectList.push_back( pLight );
	m_iGeometryChangeTimestamp++;
	m_ObjectListCS.unlock();
}

/**
 *	Light 목록 가져오기
 */
const vector<GLight*>* GScene::getLightList()
{
	return &m_LightList;
}

void GScene::addCamera( GCamera *pCamera )
{
	m_CameraList.push_back( pCamera );
}

const vector<GCamera*>* GScene::getCameraList()
{
	return &m_CameraList;
}

void GScene::setVersion( const char* version )
{
	strncpy( m_szVersion, version, 1000 );
	m_szVersion[ 1000 ] = 0x00;
}

const char* GScene::getVersion()
{
	return m_szVersion;
}
	
void GScene::setResolution( int width, int height )
{
	m_Resolution.x = width;
	m_Resolution.y = height;
}

GDimension GScene::getResolution()
{
	return m_Resolution;
}

void GScene::setSuperSampling( int x, int y )
{
	m_SuperSampling.x = x;
	m_SuperSampling.y = y;
}

void GScene::setMaxReflectionDepth( int depth )
{
	m_iMaxReflectionDepth = depth;
}

int GScene::getMaxReflectionDepth()
{
	return m_iMaxReflectionDepth;
}

GDimension GScene::getSuperSampling()
{
	return m_SuperSampling;
}

void GScene::setOutputPath( const char* path )
{
	strncpy( m_szOutputPath, path, MAX_PATH_LENGTH );
	m_szOutputPath[ MAX_PATH_LENGTH - 1 ] = 0x00;
}

void GScene::setTexturePath( const char* path )
{
	strncpy( m_szTexturePath, path, MAX_PATH_LENGTH );
	m_szTexturePath[ MAX_PATH_LENGTH - 1 ] = 0x00;
}

const char* GScene::getOutputPath()
{
	return m_szOutputPath;
}

const char* GScene::getTexturePath()
{
	return m_szTexturePath;
}

void GScene::setFrontFace( enumFrontFace face )
{
	m_FrontFace = face;
}

enumFrontFace GScene::getFrontFace()
{
	return m_FrontFace;
}

GCamera* GScene::getRenderCamera()
{
	return &m_RenderCamera;
}

GCamera* GScene::getInitRenderCamera()
{
	return &m_InitRenderCamera;
}

GTextureManager *GScene::getTextureManager()
{
	return GTextureManager::getInstance();
}

void GScene::setRenderCamera( const GCamera* pCamera )
{
	/** 세팅복사 */
	m_RenderCamera = (*pCamera);
}

void GScene::setInitRenderCamera( const GCamera* pCamera )
{
	/** 세팅복사 */
	m_InitRenderCamera = (*pCamera);
}

void GScene::setGlobalAmbient( GColor& color )
{
	m_globalAmbient = color;
}

GColor GScene::getGlobalAmbient()
{
	return m_globalAmbient;
}

GImageBuffer* GScene::getImageBuffer()
{
	return m_pImageBuffer;
}

GImageBuffer* GScene::getDirectIllumImageBuffer()
{
	return m_pDirectIllumImageBuffer;
}

GImageBuffer* GScene::getIndirectIllumImageBuffer()
{
	return m_pIndirectIllumImageBuffer;
}

GError GScene::convertRenderScene()
{
	GError error;

	/**
	 *	Rendering 결과를 저장할 Image Buffer 를 만든다.
	 *	이전 screen size 와 변했을때만 새로 생성.
	 */
	if ( m_pImageBuffer == NULL || m_pImageBuffer->getWidth() != m_Resolution.x ||
		 m_pImageBuffer->getWidth() != m_Resolution.y ) {
		
		if ( m_pImageBuffer != NULL )
			delete m_pImageBuffer;
		m_pImageBuffer = new GImageBuffer( m_Resolution.x, m_Resolution.y );

		if ( m_pDirectIllumImageBuffer != NULL )
			delete m_pDirectIllumImageBuffer;
		m_pDirectIllumImageBuffer = new GImageBuffer( m_Resolution.x, m_Resolution.y );

		if ( m_pIndirectIllumImageBuffer != NULL )
			delete m_pIndirectIllumImageBuffer;
		m_pIndirectIllumImageBuffer = new GImageBuffer( m_Resolution.x, m_Resolution.y );

	}

	/**
	 *	이전에 구성한 Spatial structure 의 변동사항이 없고,
	 *	이전에 변환한 이후에도 geometry 변동사항이 없는 경우 리턴
	 *	그 외 좌표계변환 및 Spatial structure 재구성
	 */
	if (m_iLastConvertRenderScene == m_iGeometryChangeTimestamp) {
		if (m_iLastGeomTimestamp_For_SpatialStructure[m_CurrUseSpatialStructure] == m_iGeometryChangeTimestamp)
			return errorNo;
	}

	/** 
	 *	object ( light 포함 ) 를 모두 world 좌표계로 보낸다. 
	 */
	for ( int i = 0; i < (int) m_ObjectList.size(); ++i ) {
		if ( ( error = m_ObjectList[ i ]->convertToWorldObject() ) != errorNo ) {
			return error;
		}
	}

	// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
	// USE_KDTREE
	// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
	if ( m_CurrUseSpatialStructure == USE_KDTREE ) {

		if ( m_pKDTree ) delete m_pKDTree;
		m_pKDTree = new GKDTreeStructure( this );

		GKDTreeOption *kdtree_option = new GKDTreeOption();	// 임시
		//kdtree_option->LoadFile()
		m_pKDTree->setKDTreeOption( kdtree_option );

		bool bKDTreeNewBuild = true;

		// 이미 구성된 KDTree 가 파일에 있다면 파일로부터 KDTree 를 읽는다
		if ( m_bKDTreeFileLoad ) {
			bKDTreeNewBuild = !(m_pKDTree->loadStructureFromFile( m_szKDTreeLoadFilePath ));
		}
		
		if ( bKDTreeNewBuild ) {
			// KDTree 를 구성한다.
			if ( m_pKDTree->initialize() != errorNo ) {
				return errorKDTree;
			}
		}

		// 구성된 KDTree 를 파일에 저장한다
		if ( m_bKDTreeFileSave ) {
			if ( m_bKDTreeFileType == SAH ) {
				// SAH 인 경우
				if ( bKDTreeNewBuild ) {
					// 새로 SAH 를 만들었거나
					m_pKDTree->saveStructureToFile( m_szKDTreeSaveFilePath );
				} else if ( strcmp(m_szKDTreeSaveFilePath, m_szKDTreeLoadFilePath) != 0 ) {
					// 혹은 SAH 를 로딩했지만, 저장하는 파일이름이 다른 경우
					m_pKDTree->saveStructureToFile( m_szKDTreeSaveFilePath );
				}
			} else if (m_bKDTreeFileType == EMPTY_SAH) {
				// EMPTY_SAH 인 경우
				m_pKDTree->saveStructureToFile( m_szKDTreeSaveFilePath );
			}
		}
	}
	// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
	// USE_BVH
	// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
	else if( m_CurrUseSpatialStructure == USE_BVH ) {

		if ( m_pBVH ) delete m_pBVH;
		m_pBVH = new GBVHStructure( this );

		if ( m_pBVH->initialize() != errorNo ) {
			return errorKDTree;
		}
	}
	// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
	// USE_GRID
	// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
	else if ( m_CurrUseSpatialStructure == USE_GRID ) {

		if(m_pGrid) delete m_pGrid;

		m_pGrid = new GGridStructure( this );
		if( m_pGrid->initialize() != errorNo ){
			return errorUnknown;//나중에 에러 추가 할 것.
		}
	}

	m_LastUseSpatialStructure = m_CurrUseSpatialStructure;
	m_iLastConvertRenderScene = m_iGeometryChangeTimestamp;
	m_iLastGeomTimestamp_For_SpatialStructure[m_CurrUseSpatialStructure] = m_iGeometryChangeTimestamp;

	return errorNo;
}

/**
 * Empty KD-Tree 를 만듦.
*/
GError GScene::buildEmptyKdTree()
{
	if( m_pKDTree != NULL )
		delete m_pKDTree;

	if( m_pEmptyKDTree != NULL )
		delete m_pEmptyKDTree;

	//m_pEmptyKDTree = new GKDTreeStructure();
	// 결과 : m_pEmptyKDTree

	return errorNo;
}

/**
 * Object 의 KD-Tree 를 만듦.
 * Empty KD-Tree 가 없을 경우 전체 scene 에 대해서 만든다.
 * Empty KD-Tree 를 삭제를 하면 안됨.
*/
GError GScene::buildObjectKdTree()
{
	if( m_pKDTree != NULL )
		delete m_pKDTree;

	if( m_pEmptyKDTree )
	{
		// Object 에 대해서 생성
	}
	else
	{
		// 전체 scene 에 대해서 생성

	}

	//m_pKDTree = new GKDTreeStructure();
	// 결과 : m_pKDTree

	return errorNo;
}

/**
 *	KDTree 를 리턴한다.
 */
GKDTreeStructure *GScene::getKDTreeStructure()
{
	return m_pKDTree;
}

/**
 *	BVH 를 리턴한다.
 */
GBVHStructure* GScene::getBVHStructure()
{
	return m_pBVH;
}

/**
 *	Grid 를 리턴한다.
 */
GGridStructure *GScene::getGridStructure(void)
{
	return m_pGrid;
}

void GScene::setUseSpatialStructure( enumSpatialStructureType type )
{
	m_CurrUseSpatialStructure = type;
}

/**
 *	Scene 안의 물체들중 intersection 을 지원해야 하고,
 *	trianglulation 을 지원하는 object 를 GTriangleWrapperList 로
 *	만들어 리턴한다.
 */
GTriangleWrapperList *GScene::createSceneTriangleList( GBoundingBox &bbox )
{
	int totalCount = 0;
	GTriangleWrapperList *pList = new GTriangleWrapperList();
	GPolygonObject* pTriangleObject = NULL;

	bbox.setMax( GVector( -1000000.0f, -1000000.0f, -1000000.0f ) );
	bbox.setMin( GVector( 1000000.0f, 1000000.0f, 1000000.0f ) );

	/**
	 *	Scene 안의 Object 들로부터 삼각형 정보를 얻어온다.
	 *	1. 먼저 총 삼각형의 개수를 구한다.
	 */
	for ( int i = 0; i < (int) m_ObjectList.size(); ++i ) {

		pTriangleObject = (GPolygonObject*) m_ObjectList[ i ];

		if ( !isSceneTriangleObject( pTriangleObject ) ) 
			continue;

		/**
		 *	Scene 전체의 삼각형들의 BoundingBox 를 구한다.
		 */
		bbox += (*pTriangleObject->getBoundingBox());
		totalCount += pTriangleObject->getTriangleCount();

	}

	/** 
	 *	2. scene 전체의 삼각형을 위한 공간을 할당하고, 구성한다. 
	 */
	pList->reserve( totalCount );
	totalCount = 0;

	for ( int i = 0; i < (int) m_ObjectList.size(); ++i ) {

		pTriangleObject = (GPolygonObject*) m_ObjectList[ i ];

		if ( !isSceneTriangleObject( pTriangleObject ) ) 
			continue;

		pTriangleObject->getTriangleList( pList, i, totalCount );

	}

	return pList;
}

bool GScene::isSceneTriangleObject( GPolygonObject *pObject )
{
	if ( !pObject->isVisible() || pObject->isDebugObject() || 
		 pObject->getPolygonType() != typePolygonTriangle ||
		 !pObject->isIntersection() ) return false;
	return true;
}

int GScene::getGeometryChangeTimestamp()
{
	return m_iGeometryChangeTimestamp;
}

void GScene::setBackFaceCulling( bool flag )
{
	m_bBackFaceCulling = flag;
}

bool GScene::isBackFaceCulling()
{
	return m_bBackFaceCulling;
}

void GScene::setUseTexture( bool flag )
{
	m_bUseTexture = flag;
}

bool GScene::isUseTexture()
{
	return m_bUseTexture;
}

void GScene::setPhotonMappingOption( GPhotonMappingOption option )
{
	m_PhotonMappingOption = option;
}

GPhotonMappingOption* GScene::getPhotonMappingOption()
{
	return &m_PhotonMappingOption;
}

GRayProfilerOption* GScene::getOpenGLOption()
{
	return &m_OpenGLOption;
}

void GScene::setSceneBasePath( const char* path )
{
	strncpy( m_szSceneBasePath, path, MAX_PATH_LENGTH - 1 );
	m_szSceneBasePath[ MAX_PATH_LENGTH - 1 ] = 0x00;
}

const char* GScene::getSceneBasePath()
{
	return m_szSceneBasePath;
}

bool GScene::saveImage( const char* postfix )
{
	char resultfile[ 2048 ] = { 0x00, };

	sprintf( resultfile, "%s%c%s%s.bmp", m_szOutputPath, FILE_SEPARATOR, "result", postfix );

	return m_pImageBuffer->saveImage( resultfile );
}

bool GScene::saveImageFullPath( const char* fullpath )
{
	return m_pImageBuffer->saveImage( fullpath );
}

void GScene::setRun10Times( bool flag )
{
	m_bRun10Times = flag;
}

bool GScene::IsRun10Times()
{
	return m_bRun10Times;
}

void GScene::setProfileResultPath( const char* path )
{
	strcpy( m_szProfileResultPath, path );
}

const char* GScene::getProfileResultPath()
{
	return m_szProfileResultPath;
}

void GScene::setProfileFlag( bool flag )
{
	m_bProfileFlag = flag;
}

bool GScene::IsProfileFlag()
{
	return m_bProfileFlag;
}

void GScene::setTestFlag( bool flag )
{
	m_bTestFlag = flag;
}

bool GScene::IsTestFlag()
{
	return m_bTestFlag;
}

/**
 *	Scene 안의 물체들중 intersection 을 지원해야 하고,
 *	trianglulation 을 지원하는 object 를 GTriangleWrapperList 로
 *	만들어 리턴한다.
 */
/*
GTriangleWrapperList *GScene::precalcBVHTriangleList( GBoundingBox &bbox )
{
	int totalCount = 0;
	int tri_index  = 0;
	GTriangleWrapperList *pList = new GTriangleWrapperList();
	GPolygonObject* pTriangleObject = NULL;

	bbox.setMax( GVector( -1000000.0f, -1000000.0f, -1000000.0f ) );
	bbox.setMin( GVector( 1000000.0f, 1000000.0f, 1000000.0f ) );

	///**
	// *	Scene 안의 Object 들로부터 삼각형 정보를 얻어온다.
	// *	1. 먼저 총 삼각형의 개수를 구한다.
	
	for ( int i = 0; i < (int) m_ObjectList.size(); ++i ) {

		pTriangleObject = (GPolygonObject*) m_ObjectList[ i ];

		if ( !isSceneTriangleObject( pTriangleObject ) ) 
			continue;

		// 각 삼각형들의 bounding box를 구한다.
		aryAABB[tri_index++] = (*pTriangleObject->getBoundingBox());

		///**
		// *	Scene 전체의 삼각형들의 BoundingBox 를 구한다.
	
		bbox += (*pTriangleObject->getBoundingBox());		
		totalCount += pTriangleObject->getTriangleCount();

	}

	//
	 //*	2. scene 전체의 삼각형을 위한 공간을 할당하고, 구성한다. 
	 
	pList->reserve( totalCount );
	totalCount = 0;

	for ( int i = 0; i < (int) m_ObjectList.size(); ++i ) {

		pTriangleObject = (GPolygonObject*) m_ObjectList[ i ];

		if ( !isSceneTriangleObject( pTriangleObject ) ) 
			continue;

		pTriangleObject->getTriangleList( pList, i, totalCount );

	}

	return pList;
}
*/
