#pragma once

//--------------------------------------------------------------------------//
//																			//
//	전체 Scene																//
//																			//
//--------------------------------------------------------------------------//
#include "GBase.h"
#include <vector>
#include "GObject.h"
#include "GLight.h"
#include "GCamera.h"
#include "GDimension.h"
#include "GTriangleWrapper.h"
#include "GSpatialStructure.h"
#include "GImageBuffer.h"
#include "GTriangleWrapperList.h"
#include "GPhotonMappingOption.h"
#include "GRayProfilerOption.h"
#include "GBoundingBox.h"
#include "GCriticalSection.h"

using namespace std;

#define MAX_PATH_LENGTH	2048

/**
 *	face 방향.
 */
typedef enum {
	faceCW = 0,			//	시계방향
	faceCCW = 1,		//	반시계방향
} enumFrontFace;

typedef enum {
	adaptiveFixed = 0,			//	추가적으로 super-sampling 해야할 영역에 무조건 4x4 를 쏘는 방법.
	adaptiveSubpixel = 1		//	추가적으로 super-sampling 해야할 영역의 sub 4 pixels 중 어떤걸 쏠지 결정하는 방법.
} enumAdaptiveSamplingType;

typedef enum {
	SAH  = 0,			//	SAH (기본값)
	EMPTY_SAH = 1,		//	EMPTY_SAH
} enumKdTreeType;

typedef enum {
	USE_NONE	  = 0,
	USE_KDTREE    = 1,		//	Kd-Tree (기본값)
	USE_BVH       = 2,		//	BVH
	USE_GRID      = 3,		//	Grid
} enumSpatialStructureType;

#define COMPARE_PRIMARY_OID					1
#define COMPARE_PRIMARY_NORMAL				2
#define COMPARE_PRIMARY_SHADOW				4
#define COMPARE_PRIMARY_TEXTURE				8

#define COMPARE_SECONDARY_OID				16
#define COMPARE_SECONDARY_NORMAL			32
#define COMPARE_SECONDARY_SHADOW			64
#define COMPARE_SECONDARY_TEXTURE			128

#define COMPARE_ETC_REGION					256

#define COMPARE_ONLY_PIXEL_COLOR			512

class GKDTreeStructure;						// Kd-Tree
class GBVHStructure;						// BVH
class GGridStructure;						// Grid

class GScene
{
private:
	int m_iSceneNumber;
	int m_LoadedCameraCount;

	char m_szSceneName[1024];
	/**
	 *	Scene Rendering Option
	 */
	char m_szVersion[MAX_PATH_LENGTH];
	
	char m_szSceneBasePath[MAX_PATH_LENGTH];
	char m_szOutputPath[MAX_PATH_LENGTH];
	char m_szTexturePath[MAX_PATH_LENGTH];

	int m_iGeometryChangeTimestamp;			//	Scene 의 Object, Light 정보가 변경될때마다+
	int m_iLastConvertRenderScene;

	enumSpatialStructureType m_LastUseSpatialStructure;
	enumSpatialStructureType m_CurrUseSpatialStructure;
	int m_iLastGeomTimestamp_For_SpatialStructure[16];		// enumSpatialStructureType 의 크기만큼 필요

	GDimension m_Resolution;
	GDimension m_SuperSampling;
	GDimension m_RenderingBlock;

	bool m_bJittering;
	bool m_bBackFaceCulling;
	bool m_bUseTexture;
	bool m_bLocalShading;
	bool m_bShadow;
	int m_iMaxReflectionDepth;
	enumFrontFace m_FrontFace;
	GColor m_globalAmbient;

	GImageBuffer *m_pImageBuffer;
	GImageBuffer *m_pDirectIllumImageBuffer;
	GImageBuffer *m_pIndirectIllumImageBuffer;

	int m_iCPUThreadCount;					// 쓰레드 갯수
	GDimension m_GPUBlockSize;				// GPU block  size
	GDimension m_CPUPacketSize;				// CPU packet size : 1x1, 2x2, 4x4

	float m_fFPS, m_fFPS2;
	bool m_bOpenGLLoadTexture;

	// ----------------------------------------------------------------------
	// Spatial structure
	// ----------------------------------------------------------------------
	int m_bKDTreeFileType;
	bool m_bKDTreeFileLoad;
	bool m_bKDTreeFileSave;
	char m_szKDTreeLoadFilePath[MAX_PATH_LENGTH];
	char m_szKDTreeSaveFilePath[MAX_PATH_LENGTH];
	GKDTreeStructure	*m_pKDTree;
	GKDTreeStructure	*m_pEmptyKDTree;
	GBVHStructure		*m_pBVH;
	GGridStructure		*m_pGrid;

	// ----------------------------------------------------------------------
	// Photon mapping
	// ----------------------------------------------------------------------
	GPhotonMappingOption m_PhotonMappingOption;
	GRayProfilerOption m_OpenGLOption;

	bool m_bAntialiasingFilter;
	bool m_bBloomingFilter;
	float m_fBloomingRadius;
	float m_fBloomingWeight;

	bool m_bGrayScaleFilter;
	bool m_bEdgeDetectionFilter;
	bool m_bBlurFilter;

	// ----------------------------------------------------------------------
	// Adaptive sampling
	// ----------------------------------------------------------------------
	enumAdaptiveSamplingType m_AdaptiveSamplingType;
	int m_AdaptiveSamplingCompareType;
	bool m_bSamplingDebugInfo;

	float m_fPrimaryOIDRegionColorThreshold;
	float m_fPrimaryNormalRegionColorThreshold;
	float m_fPrimaryShadowRegionColorThreshold;
	float m_fPrimaryTextureRegionColorThreshold;
	float m_fSecondaryOIDRegionColorThreshold;
	float m_fSecondaryNormalRegionColorThreshold;
	float m_fSecondaryShadowRegionColorThreshold;
	float m_fSecondaryTextureRegionColorThreshold;
	float m_fEtcRegionColorThreshold;
	float m_fOnlyColorThreshold;

	float m_fAdaptiveDetectionStageTime;
	float m_fAdaptiveRayRate;

protected:
	static int g_iObjectNumberGen;

	GCamera	m_RenderCamera;
	GCamera	m_InitRenderCamera;

	vector<GObject*> m_ObjectList;
	vector<GObject*> m_SelecteGObjectList;
	vector<GLight*>	 m_LightList;
	vector<GCamera*> m_CameraList;

	GCriticalSection	m_ObjectListCS;

public:
	GScene(void);
	virtual ~GScene(void);

	void setSceneName( const char* name );
	const char* getSceneName();

	int getSceneNumber();

	/**
	 *	Scene Rendering Option
	 */
	void setVersion( const char* version );
	const char* getVersion();
	void setSceneBasePath( const char* path );
	void setOutputPath( const char* path );
	void setTexturePath( const char* path );
	const char* getSceneBasePath();
	const char* getOutputPath();
	const char* getTexturePath();

	bool saveImage( const char* postfix );
	bool saveImageFullPath( const char* fullpath );

	int getGeometryChangeTimestamp();

	void setResolution( int width, int height );
	GDimension getResolution();
	void setSuperSampling( int x, int y );
	GDimension getSuperSampling();
	void setRenderingBlock( int x, int y );
	GDimension getRenderingBlock();

	void setEnableJittering( bool flag );
	bool isEnableJittering();
	void setBackFaceCulling( bool flag );
	bool isBackFaceCulling();
	void setUseTexture( bool flag );
	bool isUseTexture();
	void setEnableLocalShading( bool flag );
	bool isEnableLocalShading();
	void setEnableShadow( bool flag );
	bool isEnableShadow();
	void setMaxReflectionDepth( int depth );
	int getMaxReflectionDepth();
	void setFrontFace( enumFrontFace face );
	enumFrontFace getFrontFace();
	void setGlobalAmbient( GColor& color );
	GColor getGlobalAmbient();

	/**
	 *	Scene 을 렌더링한 결과를 가지고 있는 Image Buffer.
	 */
	GImageBuffer *getImageBuffer();
	GImageBuffer *getDirectIllumImageBuffer();
	GImageBuffer *getIndirectIllumImageBuffer();

	void setCPUThreadCount( int count );
	int getCPUThreadCount();
	void setGPUBlockSize( int x, int y );
	GDimension getGPUBlockSize();
	void setCPUPacketSize( int x, int y );
	GDimension getCPUPacketSize();

	void setOpenGLLoadTexture( bool flag );
	bool isOpenGLLoadTexture();

	float getFPS();
	void setFPS( float fps );
	float getFPS2();
	void setFPS2( float fps );

	// ----------------------------------------------------------------------
	// Spatial structure
	// ----------------------------------------------------------------------
	void setKdTreeFileType( int type );
	int getKdTreeFileType();
	void setKdTreeFileLoad( bool flag );
	bool isKdTreeFileLoad();
	void setKdTreeFileSave( bool flag );
	bool isKdTreeFileSave();
	void setKdTreeLoadFilePath( const char* path );
	const char* getKdTreeLoadFilePath();
	void setKdTreeSaveFilePath( const char* path );
	const char* getKdTreeSaveFilePath();

	GKDTreeStructure*	getKDTreeStructure(void);
	GBVHStructure*		getBVHStructure(void);
	GGridStructure*		getGridStructure(void);

	void setUseSpatialStructure( enumSpatialStructureType type );

	/**
	 * Empty KD-Tree 를 만듦.
	 * 만약 Object KD Tree 가 존재할 경우 Object KD Tree 를 삭제함.
	*/
	GError buildEmptyKdTree();

	/**.
	 * Object 의 KD-Tree 를 만듦
	 * Empty KD-Tree 가 없을 경우 전체 scene 에 대해서 만든다.
	 * Empty KD-Tree 를 삭제를 하면 안됨.
	*/
	GError buildObjectKdTree();

	// ----------------------------------------------------------------------
	// Adaptive sampling
	// ----------------------------------------------------------------------
	void setAdaptiveSamplingType( enumAdaptiveSamplingType type );
	enumAdaptiveSamplingType getAdaptiveSamplingType();

	void setPrimaryOIDRegionColorThreshold( float f );
	float getPrimaryOIDRegionColorThreshold();
	void setPrimaryNormalRegionColorThreshold( float f );
	float getPrimaryNormalRegionColorThreshold();
	void setPrimaryShadowRegionColorThreshold( float f );
	float getPrimaryShadowRegionColorThreshold();
	void setPrimaryTextureRegionColorThreshold( float f );
	float getPrimaryTextureRegionColorThreshold();

	void setSecondaryOIDRegionColorThreshold( float f );
	float getSecondaryOIDRegionColorThreshold();
	void setSecondaryNormalRegionColorThreshold( float f );
	float getSecondaryNormalRegionColorThreshold();
	void setSecondaryShadowRegionColorThreshold( float f );
	float getSecondaryShadowRegionColorThreshold();
	void setSecondaryTextureRegionColorThreshold( float f );
	float getSecondaryTextureRegionColorThreshold();

	void setEtcRegionColorThreshold( float f );
	float getEtcRegionColorThreshold();

	void setOnlyColorThreshold( float f );
	float getOnlyColorThreshold();

	void setAdaptiveDetectionStageTime( float t );
	float getAdaptiveDetectionStageTime();
	void setAdaptiveRayRate( float r );
	float getAdaptiveRayRate();

	void setLowLevelAdaptiveSamplingThreshold();
	void setGoodLevelAdaptiveSamplingThreshold();
	void setHighLevelAdaptiveSamplingThreshold();
	void setVeryHighLevelAdaptiveSamplingThreshold();
	void setVeryHighLevelExceptTextureAdaptiveSamplingThreshold( float primaryColor, float tColor );

	void setAdaptiveSamplingCompareType( int compare );
	int getAdaptiveSamplingCompareType();
	void setEnableSamplingDebugInfo( bool flag );
	bool isEnableSamplingDebugInfo();

	void setUseAntialiasingFilter( bool flag );
	bool isUseAntialiasingFilter();
	void setUseBlurFilter( bool flag );
	bool isUseBlurFilter();
	void setUseEdgeDetectionFilter( bool flag );
	bool isUseEdgeDetectionFilter();
	void setUseGrayScaleFilter( bool flag );
	bool isUseGrayScaleFilter();

	void setBloomingFilter( bool flag );
	bool isBloomingFilter();
	void setBloomingRadius( float radius );
	float getBloomingRadius();
	void setBloomingWeight( float w );
	float getBloomingWeight();

	GCamera* getRenderCamera();
	void setRenderCamera( const GCamera* pCamera );
	GCamera* getInitRenderCamera();
	void setInitRenderCamera( const GCamera* pCamera );

	GTextureManager *getTextureManager();

	/**
	 *	로드한 Scene 데이터를 기반으로 초기화 할것이 있다면
	 *	수행한다.
	 */
	bool initalize();

	/**
	 *	Light 추가. 포인터 변수로 추가해야 한다. 나중에
	 *	scene 에서 해제 시킴.
	 */
	void addLight( GLight *pLight );
	
	/**
	 *	Light 목록 가져오기
	 */
	const vector<GLight*>* getLightList();

	/**
	 *	Camera 추가. 포인터 변수로 추가해야 한다. 나중에
	 *	scene 에서 해제 시킴.
	*/
	void addCamera( GCamera *pCamera );

	/**
	 *	Camera 목록 가져오기
	*/
	const vector<GCamera*>* getCameraList();

	/** 
	 *	Scene 에 물체 추가 
	 */
	virtual GError addObject( GObject* pObject );

	/**
	 *	Scene 에서 물체 삭제
	 */
	virtual void removeObject( GObject* pObject );

	/**
	 *	name 이름을 가진 Scene 에서 물체 삭제
	 */
	virtual void removeObject( const char* name );

	/**
	 *	Scene 에 있는 모든 물체 제거
	 */
	virtual void clearObject();
	virtual void clearDebugObject();

	/**
	 *	index 에 해당하는 물체 가져오기
	 */
	GObject* getObject( int index );

	/**
	 *	현재 물체 갯수 리턴.
	 */
	int getObjectCount();

	/**
	 *	물체 목록 vector 리턴.
	 */
	const vector<GObject*>* getObjectList();

	/**
	 * 모든 물체를 deselect 상태로 만든다.
	 */
	void deselectAllObject();

	/**
	 *	특정 물체를 deselect 상태로 만든다.
	 */
	void deselectObject( GObject* pObject );

	/**
	 *	물체를 select 시킨다.
	 */
	void selectObject( GObject* pObject );

	/**
	 *	select 된 물체 갯수 리턴.
	 */
	int getSelectedGObjectCount();

	/**
	 *	현재 select 된 물체중 첫 번째 물체 리턴.
	 */
	GObject* getSelectedGObject();

	/**
	 *	현재 select 된 모든 물체 리턴.
	 */
	vector<GObject*>* const getSelectedGObjectList();

	/**
	 *	object Number 에 해당하는 물체 리턴.
	 */
	GObject* getObjectByNumber( int objectNumber );

	/**
	 *  center of selected objects
	*/
	GVector getCenterOfSelectedGObjects();

	/**
	 *  bounding box of selected objects
	*/
	GBoundingBox getBoundingBoxOfSelectedGObjects();

	/**
	 *	렌더링을 수행하기 위해 Scene 의 초기화 작업을 수행한다.
	 *	Object 의 matrix 를 모두 적용해서 모든 데이터를 World 좌표계로
	 *	변환해 둔다. Rendering 을 수행하기 전에 항상 먼저 처리해야함.
	 */
	GError convertRenderScene();

	GTriangleWrapperList *createSceneTriangleList( GBoundingBox &bbox );

	bool isSceneTriangleObject( GPolygonObject *pObject );

	void setPhotonMappingOption( GPhotonMappingOption option );
	GPhotonMappingOption* getPhotonMappingOption();
	
	GRayProfilerOption* getOpenGLOption();

	/**
	 *	렌더링을 10회 시행하도록 flag 셋팅
	 */
	bool m_bRun10Times;
	void setRun10Times( bool flag );
	bool IsRun10Times();
	char m_szProfileResultPath[MAX_PATH_LENGTH];
	void setProfileResultPath( const char* path );
	const char* getProfileResultPath();
	bool m_bProfileFlag;
	void setProfileFlag( bool flag );
	bool IsProfileFlag();

	/**
	 *	테스트 용도 flag setting
	 */
	bool m_bTestFlag;
	void setTestFlag( bool flag );
	bool IsTestFlag();
};
