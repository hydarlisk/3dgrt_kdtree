#include "GRenderCommon.h"
#include "GLight.h"
#include "GPointLight.h"
#include "GVirtualLight.h"
#include "GRaySetLight.h"
#include "assert.h"
#include "math.h"

GRenderCommon::GRenderCommon(void)
{
}

GRenderCommon::~GRenderCommon(void)
{
}

/**
 *	Scene 의 광원을 gpu 올릴 정보로 구성한다. ray set data 가 존재한다면
 *	만들어서 cuda 에 올린다. ray set data 는 cuda 에 올릴때 하나의 texture 에
 *	여러개의 light 관련 ray set data 를 담아서 올린다.
 */
cuLight* GRenderCommon::makeCudaLightInfo( GScene *pScene, int *pCount, 
										   cudaRenderPipeline *pCudaRenderPipeline )
{
	cuLightRaySet *pCuRaySetData = NULL;
	int raySetCount = 0, copyRaySetCount = 0;

	const vector<GLight*>* pLightList = pScene->getLightList();
	GLight* pLight = NULL;

	if ( (int) pLightList->size() == 0 ) {
		(*pCount) = 0;
		return NULL;
	}

	cuLight* cuLights = (cuLight*) malloc( sizeof( cuLight ) * pLightList->size() );

	for ( int i = 0; i < (int) pLightList->size(); ++i ) {

		pLight = (*pLightList)[ i ];

		cuLights[ i ].iObjectID = pLight->getObjectNumber();
		cuLights[ i ].iEmitPhoton = 0;
		cuLights[ i ].bUsePhoton = 0;
		cuLights[ i ].bUseDirect = 0;

		/**
		 *	Point Light 를 cuda data 로 변환한다.
		 */
		if ( pLight->getLightType() == typePointLight ) {

			GPointLight *pPointLight = (GPointLight*) pLight;

			cuLights[ i ].bUseDirect = true;
			cuLights[ i ].bUsePhoton = pPointLight->isUsePhoton();

			cuLights[ i ].iEmitPhoton = 0;
			cuLights[ i ].lightType = cuPointLight;
			cuLights[ i ].intensity = pPointLight->getIntensity();
			cuLights[ i ].pos.x = pPointLight->getPosition().x;
			cuLights[ i ].pos.y = pPointLight->getPosition().y;
			cuLights[ i ].pos.z = pPointLight->getPosition().z;
			cuLights[ i ].color.x = pPointLight->getLightColor().r * pPointLight->getLightColor().a;
			cuLights[ i ].color.y = pPointLight->getLightColor().g * pPointLight->getLightColor().a;
			cuLights[ i ].color.z = pPointLight->getLightColor().b * pPointLight->getLightColor().a;

		}

		/**
		 *	Ray Set Light 를 cuda light 로 만든다.
		 */
		if ( pLight->getLightType() == typeVirtualLight ) {

			GVirtualLight *pPolygonLight = (GVirtualLight*) pLight;

			cuLights[ i ].bUseDirect = false;	
			cuLights[ i ].bUsePhoton = false;

			cuLights[ i ].lightType = cuVirtualLight;
			cuLights[ i ].intensity = pPolygonLight->getIntensity();
			cuLights[ i ].iEmitPhoton = 0;

			cuLights[ i ].pos.x = pPolygonLight->getPosition().x;
			cuLights[ i ].pos.y = pPolygonLight->getPosition().y;
			cuLights[ i ].pos.z = pPolygonLight->getPosition().z;

			cuLights[ i ].color.x = pPolygonLight->getLightColor().r * pPolygonLight->getLightColor().a;
			cuLights[ i ].color.y = pPolygonLight->getLightColor().g * pPolygonLight->getLightColor().a;
			cuLights[ i ].color.z = pPolygonLight->getLightColor().b * pPolygonLight->getLightColor().a;

		}

		/**
		 *	Ray Set Light 를 cuda light 로 만든다.
		 */
		if ( pLight->getLightType() == typeRaySetLight ) {

			GRaySetLight *pRaySetLight = (GRaySetLight*) pLight;

			cuLights[ i ].bUseDirect = false;	
			cuLights[ i ].bUsePhoton = pRaySetLight->isUsePhoton();

			cuLights[ i ].lightType = cuRaySetLight;
			cuLights[ i ].intensity = pRaySetLight->getIntensity();
			cuLights[ i ].iEmitPhoton = 0;

			cuLights[ i ].pos.x = pRaySetLight->getPosition().x;
			cuLights[ i ].pos.y = pRaySetLight->getPosition().y;
			cuLights[ i ].pos.z = pRaySetLight->getPosition().z;

			/** 
			 *	photon 생성 모드. light 들중에 rayset light 가 여러개 있을대
			 *	ray set data 는 cuda texture 한개에 다 몰아서 넣을 것이기 때문에
			 *	각 ray set light 가 하나의 linear 한 memory 구조에서 자신의 ray set data
			 *	의 위치를 어떻게 구성할지를 계산.
			 */
			cuLights[ i ].randomMode = pRaySetLight->isRandomMode();
			cuLights[ i ].iStartIndexInRaySetData = raySetCount;
			cuLights[ i ].iRaySetCount = pRaySetLight->getRaySetDataCount();

			cuLights[ i ].color.x = pRaySetLight->getLightColor().r * pRaySetLight->getLightColor().a;
			cuLights[ i ].color.y = pRaySetLight->getLightColor().g * pRaySetLight->getLightColor().a;
			cuLights[ i ].color.z = pRaySetLight->getLightColor().b * pRaySetLight->getLightColor().a;

			raySetCount += pRaySetLight->getRaySetDataCount();

		}

	}

	(*pCount) = (int) pLightList->size();

	/**
	 *	ray set data 가 존재한다면, 위에서 구성한 정보를 기반으로
	 *	다시 한번 돌면서 메모리를 구성을 한다음에
	 *	cuda 에 올린다.
	 */
	if ( raySetCount ) {
	
		int offset = 0;
		pCuRaySetData = (cuLightRaySet*) malloc( sizeof( cuLightRaySet ) * raySetCount );

		cuLight *pCuLight = NULL;

		for ( int i = 0; i < (*pCount); ++i ) {

			pCuLight = ( cuLights + i );

			/**
			 *	Ray Set data 를 linear 한 메모리에 복사한다.
			 */
			if ( pCuLight->lightType == cuRaySetLight ) {

				GRaySetLight *pRaySetLight = (GRaySetLight*)( (*pLightList)[ i ] );
				GRaySet* pRaySetData = pRaySetLight->getRaySetData();

				for ( int index = 0; index < pCuLight->iRaySetCount; ++index ) {
					offset = pCuLight->iStartIndexInRaySetData + index;
					pCuRaySetData[ offset ].pos.x = pRaySetData[ index ].pos[ 0 ];
					pCuRaySetData[ offset ].pos.y = pRaySetData[ index ].pos[ 1 ];
					pCuRaySetData[ offset ].pos.z = pRaySetData[ index ].pos[ 2 ];
					pCuRaySetData[ offset ].dir.x = pRaySetData[ index ].dir[ 0 ];
					pCuRaySetData[ offset ].dir.y = pRaySetData[ index ].dir[ 1 ];
					pCuRaySetData[ offset ].dir.z = pRaySetData[ index ].dir[ 2 ];
					pCuRaySetData[ offset ].power = pRaySetData[ index ].power;
					copyRaySetCount++;
				}
			}
		}

		assert( raySetCount == copyRaySetCount );

		/**
		 *	cuda device memory 로 올리고 메모리를 해제한다. 에러가
		 *	발생한다면 light 자체를 해제하고 NULL 을 리턴시킨다.
		 */
		if ( pCudaRenderPipeline->setLightRaySetData( pCuRaySetData, raySetCount ) != errorNo ) {
			free( pCuRaySetData );
			free( cuLights );
			return NULL;
		} 
		
		free( pCuRaySetData );
	}

	return cuLights;
}
