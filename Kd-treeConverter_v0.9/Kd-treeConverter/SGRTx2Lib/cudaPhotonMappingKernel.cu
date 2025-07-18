#include "cudaPhotonMapping.cuh"

#define VALID_COSINE_VALUE		0.6f
#define E						2.71828183f
#define GAUSSIAN_ALPHA			1.818f
#define GAUSSIAN_BETA			1.953f
#define GAUSSIAN_DIV			0.85816f

#define M_PI					3.14159f
#define HALF_M_PI				1.57079f
#define DOUBLE_M_PI				6.28318f

#define GLOSSY_NU				16.0f
#define GLOSSY_NV				16.0f
#define GLOSSY_COEF				1.0f

#define RECIPROCAL_M_PI			0.31831f	//	1 / M_PI


/**
 *	Photon Tracing 및 Photon Gathering 을 수행할 CUDA Kernel
 *
 *	by graphicsian.
 */

texture<float4, 1, cudaReadModeElementType> photonTexture;
texture<int2, 1, cudaReadModeElementType> photonIndexTexture;
texture<float4, 1, cudaReadModeElementType> areaPhotonTexture;

/**
 *	theta, phi 로 dir 를 구한다.
 *	sgrt 에서 가져옴. thanks to Dr.Cha.
 */	
__device__ void makeSolidAngle( float theta, float phi, float3 normal, float3 *dir )
{
	float3 binormal, tangent;
	float ss, sc, c;
	float absn[3];

	absn[0]	= fabs( normal.x );
	absn[1]	= fabs( normal.y );
	absn[2]	= fabs( normal.z );
	
	float maxval = absn[0]; int maxaxis = 0;
	if( absn[1] > maxval ) { maxval = absn[1]; maxaxis = 1; }
	if( absn[2] > maxval ) { maxval = absn[2]; maxaxis = 2; }

	switch( maxaxis ) {
		case 0:
			binormal.x = 0.0f; binormal.y = 0.0f; binormal.z = 1.0f;
			break;
		case 1:
			binormal.x = 1.0f; binormal.y = 0.0f; binormal.z = 0.0f;
			break;
		case 2:
			binormal.x = 0.0f; binormal.y = 1.0f; binormal.z = 0.0f;
			break;
	}
	
	tangent = normalize( cross( binormal, normal ) );
	binormal = normalize( cross( normal, tangent ) );

	sc = (float)( sinf( theta ) * cosf( phi ) );
	ss = (float)( sinf( theta ) * sinf( phi ) );
	c = (float) cosf( theta );

	*dir = normalize( binormal * sc + tangent * ss + normal * c );
}

/**
 *	glossy 를 위한 aniso photon hemi sphere 샘플링.
 *	sgrt 에서 가져와서 수정. thanks to Dr.차득현.
 */	
__device__ void anisoPhongSampleHemiSphere( float u1, float u2, 
											float3 normal, float3 dir, 
											float3 *nextDir )
{
	float coef = 1.0f;
	float theta, phi;
	float3 h;
	
	/** dir 은 out-going 방향으로 들어왔다고생각. */
	
	// calculate phi of halfway vector h
	if( u1 < 0.25f ) {
	
		u1 = 4.0f * u1;
		phi = atanf( coef * tan( HALF_M_PI * u1 ) );
		
	} else if( u1 < 0.5f ) {
	
		u1 = 4.0f * ( u1 - 0.25f );
		phi = atanf( coef * tan( HALF_M_PI * u1 ) );
		phi = M_PI - ( phi );
		
	} else if( u1 < 0.75f) {
	
		u1 = 4.0f * ( u1 - 0.5f );
		phi = atanf( coef * tan( HALF_M_PI * u1 ) );
		phi = M_PI + ( phi );
		
	} else {
	
		u1 = 4.0f * ( u1 - 0.75f );
		phi = atanf( coef * tanf( HALF_M_PI * u1 ) );
		phi = DOUBLE_M_PI - ( phi );
		
	}

	float cosPhiSquare = cosf( phi );
	cosPhiSquare = cosPhiSquare * cosPhiSquare;
	float sinPhiSquare = 1.0 - cosPhiSquare;

	theta = acosf( powf( ( 1.0f - u2 ), 1.0f / ( GLOSSY_NU * cosPhiSquare + GLOSSY_NV * sinPhiSquare + 1) ) );

	makeSolidAngle( theta, phi, normal, &h );
	
	(*nextDir) = h * dot( dir, h ) * 2.0f - dir;
	
}


/**
 *	point light 에서 photon 을 emit 한다.
 *	dir.w 에는 mint, pos.w 에는 maxt 를 저장해야 한다.
 *	light 정보는 constant memory 에서 참조한다.
 */
__device__ void generatePointLightEmitPhoton( int randomSeed, cuLight* pLight,
											  float4 *pos, float4 *dir, float3 *power ) 
{
	float k1 = radicalInverse( randomSeed, 3 );
	float k2 = radicalInverse( randomSeed, 5 );

	float r, phi;
	
	(*dir).z = 1.0f - 2.0f * k1;
	
	r = sqrtf( max( 0.0f, 1.0f - (*dir).z * (*dir).z ) );
	phi = 2.0f * M_PI * k2;

	(*dir).x = r * cosf( phi );
	(*dir).y = r * sinf( phi );
	(*dir).w = 0.0f;

	(*pos) = make_float4( pLight->pos, FLT_MAX );
	
	(*power) = pLight->color;
	(*power) *= pLight->fOnePhotonPower;
}

/**
 *	ray set light 에서 photon 을 emit 한다.
 *	이 ligth 가 참조하는 ray set data 범위안에서 ray sample 을 하나
 *	선택해야 한다.
 */
__device__ void generateRaySetLightEmitPhoton( int randomSeed, cuLight* pLight,
											   float4 *pos, float4 *dir, float3 *power ) 
{
	float k1 = radicalInverse( randomSeed, 3 );
	float u1 = radicalInverse( randomSeed, 13 );
	float u2 = radicalInverse( randomSeed, 17 );
	float theta = acosf( sqrtf( u1 ) );
	float phi = u2 * 2.0f * M_PI;
	float3 normal, randdir;

	/** random 하게 ray set data 범위 안의 index 를 생성해낸다. */
	//int randomIndex = ( (int)( ( k1 * (float)pLight->iRaySetCount ) ) % pLight->iRaySetCount ) + 
	//					pLight->iStartIndexInRaySetData;
	int randomIndex = ( randomSeed % pLight->iRaySetCount ) + pLight->iStartIndexInRaySetData;

	/** 
	 *	ray set data texture 에서 이 index 에 해당하는 데이터를 가져온다. 
	 *	ray tracing 을 위해서 dir.w 는 0.0, pos.w 는 FLT_MAX 로 설정해야 한다.
	 */
	float4 temp = tex1Dfetch( inLightRaySetTexture, randomIndex * 2 + 0 );
	(*pos).x = temp.x; (*pos).y = temp.y; (*pos).z = temp.z; (*pos).w = FLT_MAX;
	normal.x = temp.w;

	/** temp z 에는 각 ray 의 power 가 기록되어 있다. */
	temp = tex1Dfetch( inLightRaySetTexture, randomIndex * 2 + 1 );
	normal.y = temp.x; normal.z = temp.y;

	/**
	 *	dir 를 생성한다.
	 */
	makeSolidAngle( theta, phi, normal, &randdir );

	(*dir).x = randdir.x;
	(*dir).y = randdir.y;
	(*dir).z = randdir.z;
	(*dir).w = 0.0f;

	(*power) = pLight->color * pLight->fOnePhotonPower;
}

/**
 *	확률적으로 photon 을 generation 한다. 현재 지점에서의 확률값은 nextBound 값과
 *	currentPhoton 의 pos 를 기준으로 구한다. ( 동일한 장면에서는 동일한 값이 나와야 하므로 )
 *	만약, specular 하고 glossy 한 photon 을 저장하는게 아니라면,
 *	현재 photon 이 bound 되는 성질이 ps 이거나 pg 일때는 photon 을 지운다. ( dir 을 모두 0.0 으로 )
 *
 *	generate 부분은 서강대학교 Dr. 차득현군의 photon source 를 수정했음을 밝히는 바입니다.
 */
__device__ bool photonStorageAndBoundingPhoton( int nextBound, int maxBound, int seed,
												cuIntersectionPoint &hitPoint,
												cuPhoton &currentPhoton, cuObjectMaterial &material,
												cuPhoton &nextPhoton, cuRay &nextRay )
{
	int randomSeed = nextBound + (int) ( fabs( currentPhoton.pos.x ) * 1000000.0f + 
										 fabs( currentPhoton.pos.y ) * 10000.0f + 
										 fabs( currentPhoton.pos.z ) * 100.0f ) % 1000000;

	float prob = radicalInverse( randomSeed, 11 );
	
	float3 dir = currentPhoton.dir;
	float3 tempnormal, texColor;
	
	float pd = ( material.diffuse.x + material.diffuse.y + material.diffuse.z ) / 3.0f;
	float pg = 0.0f;
	float ps = ( material.specular.x + material.specular.y + material.specular.z ) / 3.0f;
	float pt = material.transparency;

	/**
	 *	먼저 현재 상태에 따라서 photon 을 저장할지 말지 결정.
	 *	만약 photon 이 bound 될 값이 ps, pg, pt 라면 storing 하지 않기 위해
	 *	photon 을 초기화 시킨다. bound 가 maxbound 에 다 도달했을때도
	 *	이 부분은 수행되어야 한다.
	 */
	if ( prob > pd && prob <= pd + ps + pt + pg ) {
		currentPhoton.dir.x = 0.0f; 
		currentPhoton.dir.y = 0.0f; 
		currentPhoton.dir.z = 0.0f;
	}


	/**
	 *	만약 nextBound 가 maxBound 보다 같거나 크다면 더 이상 추적하지 않는다.
	 */
	if ( nextBound >= maxBound )
		return false;


	if ( prob <= pd ) {

		/** diffuse 반사.	*/
		/** 뒷면에 맞았을때 처리 */
		tempnormal = ( dot( dir, currentPhoton.normal ) < 0.0f ) ? 
						-1.0f * currentPhoton.normal : currentPhoton.normal;
		
		float u1 = radicalInverse( randomSeed, 23 );
		float u2 = radicalInverse( randomSeed, 29 );
		float theta = acosf( sqrtf( u1 ) );
		float phi = u2 * 2.0f * M_PI;
		
		makeSolidAngle( theta, phi, tempnormal, &dir );
		
		//nextRay.setPrevTriIndex( hitPoint.triIndex );
		nextRay.dir.x = dir.x; nextRay.dir.y = dir.y; 
		nextRay.dir.z = dir.z; 
		nextRay.pos.x = currentPhoton.pos.x + nextRay.dir.x * RAY_START_EPSILON; 
		nextRay.pos.y = currentPhoton.pos.y + nextRay.dir.y * RAY_START_EPSILON; 
		nextRay.pos.z = currentPhoton.pos.z + nextRay.dir.z * RAY_START_EPSILON; 
	
		/** 
		 *	next photon 을 가리키는 global memory 에 
		 *	diffuse power 를 계산해 저장한다. 이미확률적으로 결정되었으므로 kd 는
		 *	곱하지 않는다.
		 */
		int texture = float_as_int( material.textureNumber );

		texture = fetchTexture( texture, hitPoint.u, hitPoint.v, texColor );

		if ( texture )
			nextPhoton.power = currentPhoton.power * texColor;
		else
			nextPhoton.power = currentPhoton.power * material.diffuse;
		
		return true;
		
	} else if ( prob <= pd + ps ) {
	
		/**
		 *	specular 반사.
		 */
		tempnormal = ( dot( dir, currentPhoton.normal ) < 0.0f ) ? 
						-1.0f * currentPhoton.normal : currentPhoton.normal;
		
		dir = reflection( dir, tempnormal );
				
		//nextRay.setPrevTriIndex( hitPoint.triIndex );
		nextRay.dir.x = dir.x; nextRay.dir.y = dir.y; 
		nextRay.dir.z = dir.z;
		nextRay.pos.x = currentPhoton.pos.x + nextRay.dir.x * RAY_START_EPSILON; 
		nextRay.pos.y = currentPhoton.pos.y + nextRay.dir.y * RAY_START_EPSILON; 
		nextRay.pos.z = currentPhoton.pos.z + nextRay.dir.z * RAY_START_EPSILON; 
	
		/** 
		 *	next photon 을 가리키는 global memory 에 
		 *	specular power 를 계산해 저장한다. 이미확률적으로 결정되었으므로 ks 는
		 *	곱하지 않는다.
		 */
		int texture = float_as_int( material.textureNumber );
		texture = fetchTexture( texture, hitPoint.u, hitPoint.v, texColor );
		if ( texture )
			nextPhoton.power = currentPhoton.power * texColor;
		else
			nextPhoton.power = currentPhoton.power * material.specular;
					
		return true;
		
	} else if ( prob <= pd + ps + pt ) {
	
		/**
		 *	refraction 반사.
		 */
		dir = refraction( dir, currentPhoton.normal, material.refractionIndex );

		//nextRay.setPrevTriIndex( hitPoint.triIndex );
		nextRay.dir.x = dir.x; nextRay.dir.y = dir.y; 
		nextRay.dir.z = dir.z; 
		nextRay.pos.x = currentPhoton.pos.x + nextRay.dir.x * RAY_START_EPSILON; 
		nextRay.pos.y = currentPhoton.pos.y + nextRay.dir.y * RAY_START_EPSILON; 
		nextRay.pos.z = currentPhoton.pos.z + nextRay.dir.z * RAY_START_EPSILON; 

		
		/** 
		 *	next photon 을 가리키는 global memory power 를 계산해 저장한다.
		 *	투과이므로 이전 power 를 그대로.
		 */
		nextPhoton.power = currentPhoton.power;
		
		return true;
		
	} else if ( prob <= pd + ps + pt + pg ) {
	
		/**
		 *	glossy 반사.
		 */
		float u1, u2;
		
		u1 = radicalInverse( randomSeed, 23 );
		u2 = radicalInverse( randomSeed, 29 );
		float3 nextDir;
		
		anisoPhongSampleHemiSphere( u1, u2, currentPhoton.normal, dir, &nextDir );
		
		//nextRay.setPrevTriIndex( hitPoint.triIndex );
		nextRay.dir.x = nextDir.x; nextRay.dir.y = nextDir.y; 
		nextRay.dir.z = nextDir.z; 
		nextRay.pos.x = currentPhoton.pos.x + nextRay.dir.x * RAY_START_EPSILON; 
		nextRay.pos.y = currentPhoton.pos.y + nextRay.dir.y * RAY_START_EPSILON; 
		nextRay.pos.z = currentPhoton.pos.z + nextRay.dir.z * RAY_START_EPSILON; 

		
		/** 
		 *	next photon 을 가리키는 global memory 에 
		 *	glossy power 를 계산해 저장한다. 이미확률적으로 결정되었으므로 kg 는
		 *	곱하지 않는다.
		 */
		int texture = float_as_int( material.textureNumber );
		texture = fetchTexture( texture, hitPoint.u, hitPoint.v, texColor );
		if ( texture )
			nextPhoton.power = currentPhoton.power * texColor;
		else
			nextPhoton.power = currentPhoton.power * material.specular;
					
		return true;
		
	}

	/** 
	 *	photon 이 죽어서 추적할 필요가 없는 경우는 
	 *  nextPhoton 의 mint 를 MAX_FLT 로 하고,
	 *	photon 정보의 dir 은 모두 0.0 으로 채운다.
	 */
	nextPhoton.dir.x = 0.0f; nextPhoton.dir.y = 0.0f; nextPhoton.dir.z = 0.0f;
	
	return false;
}

/**
 *	Photon 을 emit 할 방향을 계산해서 저장한다.
 *	rayTracing kernel 로 보내기 위해서 cuRay global memory 에 ray 를 기록한다.
 *	생성된 photon 정보는 global photon memory 에 저장한다.
 */
__global__ void cuPhotonEmitKernel( int iEmitPhoton, int iRandomBase,
									cuPhoton *pDevicePhotonMem,
									cuRay *pDeviceRayBuffer )
{
	float4 pos, dir;
	float3 power;
	
	/** 
	 *	각 thread 가 생성할 photon 의 고유 index 계산. BLOCK 은 (block,1)
	 *	thread 도 (thread개수,1) 형태로 kernel 이시작되었으므로.
	 */
	int photonIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	/** 
	 *	photon 개수이내의 thread 만 수행됨. 
	 *	kernel 을 한번만 호출하기 위해서 실제 photon 을 처리하지 않는 thread 도
	 *	몇개 정도 더 실행되는 경우가 있기때문에 체크해야 한다.
	 */
	if ( photonIndex < iEmitPhoton ) {
	
		/**
		 *	photonIndex 에 해당하는 photon 을 뿌릴 light 를 찾아서, emit photon 정보를
		 *	생성해낸다. light 정보는 common kernel 모듈에 의해서 constant memory 에 올라가 있다.
		 */
	
		#ifdef __DEVICE_EMULATION__
			if ( photonIndex == 0 ) {
				char temp[1024];
				OutputDebugString( "------------------------------------------------\n" );
				sprintf( temp, "Constant Light Count = %d\n", constantLightCount );
				OutputDebugString( temp );
				OutputDebugString( "------------------------------------------------\n" );
			}
		#endif

		// 초기화.
		dir = make_float4( 0.0f, 0.0f, 0.0f, FLT_MAX );
		pos = make_float4( 0.0f, 0.0f, 0.0f, FLT_MAX );

		for ( int i = 0; i < constantLightCount; ++i ) {
		
			/** photon emit 하지 않는 light 일때는 continue */
			if ( constantLightInfo[ i ].bUsePhoton == 0 )
				continue;

			if ( photonIndex >= constantLightInfo[ i ].iPhotonStartIndex &&
				 photonIndex < constantLightInfo[ i ].iPhotonStartIndex + constantLightInfo[ i ].iEmitPhoton ) {
					
				/** Point Light 일때 */
				if ( constantLightInfo[ i ].lightType == cuPointLight ) {
				
					/**
					 *	photon generation.
					 */
					generatePointLightEmitPhoton( iRandomBase + photonIndex, 
								&constantLightInfo[ i ], &pos, &dir, &power );
				}

				/**
				 *	Ray Set 으로 이루어진 Light 일때.
				 */
				if ( constantLightInfo[ i ].lightType == cuRaySetLight ) {
					/** 총 Ray Set Data 안에서 random 하게 ray sample 을 선택한다. */
					generateRaySetLightEmitPhoton( iRandomBase + photonIndex, 
								&constantLightInfo[ i ], &pos, &dir, &power );
				}
					

				/** 
				 *	ray tracing kernel 에 넘겨서 intersection 을 구하기 위한 설정 
				 *	mint, maxt 는 각각 dir.z, pos.z 에 저장된다.
				 */
				//pDeviceRayBuffer[ photonIndex ].init();
				pDeviceRayBuffer[ photonIndex ].dir = make_float3(dir.x, dir.y, dir.z);
				pDeviceRayBuffer[ photonIndex ].pos = make_float3(pos.x, pos.y, pos.z);
					
				/** 
				 *	현재 photon 에 대한 정보는 임시적으로 photon map 에 저장한다
				 *	photon 이 출발한 위치이기 때문에 나중에 hit 된 정보로 덮어쓰게
				 *	될것이다. 만약 hit 를 안하면 초기화 시켜야 한다. dir 을 모두 0.0 으로.
				 */
				cuPhoton photon;
				photon.dir.x = dir.x; photon.dir.y = dir.y; photon.dir.z = dir.z; 
				photon.pos.x = pos.x; photon.pos.y = pos.y; photon.pos.z = pos.z;
				photon.power = power;
					 
				pDevicePhotonMem[ photonIndex ] = photon;
				
				break;

			}
		}
		
	}
	
}

/**
 *	이전에 추적한 ray 에 대한 hit 결과를 가지고, photon map 을 구성한다.
 *
 *	각 photon 을 생성할때 이미 photon map 에 해당 photon 의 정보를 기록해 두는데 ( power 값때문에 )
 *	이는 photon 의 시작점 정보이기 때문에 ray hit 를 체크한뒤에 photon map 정보중에
 *	photon.pos, photon.dir, photon.normal 은 hit 한 지점값으로 업데이트 해야 한다.
 *
 *	만약 hit 되지 않았다면 photon.dir 을 모두 0.0 으로 세팅한다.
 * 
 *	이렇게 global photon map 을 구성한뒤에, photon 이 bound 되어야 하는지를 체크해서
 *	새로운 ray 를 생성한다음에 다시 ray memory block 에 기록한다. ( offset 0 ~ iEmitPhoton - 1 )
 *	만약 bound 되지 않는 photon 이라면 mint 를 FLT_MAT 로 해서 저장한다. 
 *	 
 *	 context 안의 pDeviceIntResult 는 bound 되는 photon 이 최소한
 *	하나라도 있는지를 체크하기 위한 global memory 값이다. 만약 하나라도
 *	photon 이 bound 된다면 pDeviceIntResult 을 1 로 한다. 수많은 thread 가
 *	동시에 이 변수에 접근하겠지만 이 변수는 photon 이 bound 될때만 1 로 각 thread 가
 *	세팅하기 때문에 동기화 문제는 발생하지 않는다. 하나의 thread 라도 1 로 세팅하면 1이 되니.
 *
 *	bound 되는 photon 에 대한 정보는 global photon map 의 해당 bound 정보를
 *	저장할 block 에 기록된다. 
 *	
 *	global photon map 메모리 구조는 아래와 같다.
 *	
 *	메모리는 최소한 iMaxPhotonSize ( iEmitPhoton * iMaxBound ) 만큼 잡혀있고, 
 *	각 메모리는 iMaxBound 개만큼의 block 으로 구분한다.
 *	 현재 bound 가 currentBound 라고 하면 photon memory 공간에서 
 *	currentBound * iEmitPhoton 부터 iEmitPhoton 개 안에 저장되는 것이다.
 *
 *	[ 중요 ] ray 와 intersection 은 한 block 으로 계속 재사용하고
 *	global photon memory 는 bound 개수만큼의 block 있다.
 *
 *	따라서 이 메모리를 참조하는 index 는 두개가 서로 다르다.
 *	ray, intersection 은 bound 와 상관없이 index 를 해야하고,
 *	global photon memory 는 현재 bound 를 계산해서 bound * iEmitPhoton + index 
 *	를 해야한다.
 *
 *	direct photon 을 저장하지 않는다면 bound 가 0 인것은 photon map 정보에서 
 *	dir 을 모두 0.0 으로 만든다.
 *
 */
__global__ void cuMakePhotonMapAndBoundingKernel( int iEmitPhoton,
												  int iRandomSeed,
												  int currentBound,
												  int maxBound,
												  bool bSaveDirectPhoton,
												  cuPhoton *pDevicePhotonMem,
												  cuRay *pDeviceRayBuffer,
												  cuIntersectionPoint *pDeviceIntersectionBuffer,
												  int *pDeviceIntResult )
{
	/** 
	 *	각 thread 가 생성할 photon 의 고유 index 계산. BLOCK 은 (block,1)
	 *	thread 도 (thread개수,1) 형태로 kernel 이시작되었으므로.
	 */
	int photonOffset = ( currentBound * iEmitPhoton );
	
	/**
	 *	intersection index 와 ray 를 위한 index
	 */
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int photonIndex = photonOffset + index;
	
	/** 
	 *	photon 개수이내의 thread 만 수행됨. 
	 *	kernel 을 한번만 호출하기 위해서 실제 photon 을 처리하지 않는 thread 도
	 *	몇개 정도 더 실행되는 경우가 있기때문에 체크해야 한다.
	 */
	if ( index < iEmitPhoton ) {
	
		/** 
		 *	hit 한경우 삼각형과 물체정보를 가져와서 global photon map 에 hit 지점정보를 업데이트
		 *	한다. power 는 photon 을 생성할 당시에 이미 기록해 두었으므로 업데이트 하면 안된다.
		 *	intersection result 는 rayIndex 로 접근.
		 */
		if ( pDeviceIntersectionBuffer[ index ].isHit() ) {

			/** 
			 *	photon map 을 hit 정보로 세팅한다.
			 */
			pDevicePhotonMem[ photonIndex ].dir = pDeviceIntersectionBuffer[ index ].dir;
			pDevicePhotonMem[ photonIndex ].pos = pDeviceIntersectionBuffer[ index ].pos;
			pDevicePhotonMem[ photonIndex ].normal = pDeviceIntersectionBuffer[ index ].normal;
			
			// power 는 이미 이전 출발지에서 세팅되어 왔으므로 세팅안해야 한다.
			// pDevicePhotonMem[ photonIndex ].power;

			/** 
			*	max bound 를 초과하지 않았다면
			*	현재 hit 지점의 material 을 바탕으로 bounding photon 을 생성하고
			*	bounding 될때의 power 정보를 구성해서, global photon memory 의
			*	해당 bound 위치에 기록한다.
			*	max bound 체크는 반드시 generateBoundingPhoton 안에서 현재 photon 을 저장할지
			*	말지와 같이 결정해야 한다.
			*/
			cuObjectMaterial material;
			getObjectMaterial( pDeviceIntersectionBuffer[ index ].objectIndex, material );
			 
			/** 
			*	bounding photon generation 해야 하는 경우라면 generation 한다. 
			*	이 함수가 수행된 이후에는 현재 ray 가 새로 bounding 되는 
			*	photon 의 ray 정보로 덮어써짐을 주의하라.
			*/
			if ( photonStorageAndBoundingPhoton( currentBound + 1, maxBound, iRandomSeed + index, 
										pDeviceIntersectionBuffer[ index ],
										pDevicePhotonMem[ photonIndex ], material,
										pDevicePhotonMem[ iEmitPhoton + photonIndex ],
										pDeviceRayBuffer[ index ] ) ) {
				(*pDeviceIntResult) = 1;
			}
			
			// direct photon 저장옵션에따라 현재 photon 저장결정.
			if ( !bSaveDirectPhoton && currentBound == 0 ) {
				pDevicePhotonMem[ photonIndex ].dir = make_float3( 0.0f, 0.0f, 0.0f );
			}
			
		} else {
			/** 
			 *	[ 중요 ]
			 *
			 *	hit 되지 않은 photon 은 photon global memory 의 데이터중 dir 을 모두 0.0f 으로 세팅.
			 *	또한 더이상 추적도 하지 말아야 하므로 ray 의 mint 를 MAX_FLT 로 채운다.
			 */
			pDevicePhotonMem[ photonIndex ].dir = make_float3( 0.0f, 0.0f, 0.0f );
			
//			pDeviceRayBuffer[ index ].dir.w = FLT_MAX;
//			pDeviceRayBuffer[ index ].pos.w = FLT_MAX;
		}
	}	
}

/*******************************************************************************
 *
 *
 *	PHOTON GATHERING 관련 함수
 *
 *
 *	by graphicsian
 *
 *
 *******************************************************************************/


__device__ float G( float roughness, float v )
{
	return v / ( roughness - roughness * v + v );
}

__device__ float Z( float roughness, float t )
{
	float div = ( 1 + roughness * t * t - t * t );
	return roughness / ( div * div );
}

__device__ float calSquareDist( float3 p1, float3 p2 )
{
	return ( p1.x - p2.x ) * ( p1.x - p2.x ) +
		   ( p1.y - p2.y ) * ( p1.y - p2.y ) +
		   ( p1.z - p2.z ) * ( p1.z - p2.z );
}

__device__ float dotProduct( float3 p1, float3 p2 )
{
	return ( p1.x * p2.x ) + ( p1.y * p2.y ) + ( p1.z * p2.z );
}

__device__ float gaussianWeight( float squareDist, float sqaureR )
{
	return GAUSSIAN_ALPHA * ( 1.0f - 
		( ( 1.0f - 1.0f / pow( E, GAUSSIAN_BETA * ( squareDist / ( 2.0f * sqaureR ) ) ) ) / GAUSSIAN_DIV ) );
}

__device__ float3 isotropicGaussianModel( float3 rayDir, 
										  float3 photonDir, 
										  float3 photonPower,
										  float3 rayNormal, 
										  float kd, float kg, 
										  float roughness,
										  float area )
{
	float3 power = make_float3( 0.0f, 0.0f, 0.0f );
	float3 h = normalize( ( rayDir + photonDir ) );
	float cos = dot( h, rayNormal );
	float temp;

	power += kd * RECIPROCAL_M_PI * photonPower;

	if ( kg > 0.0f && cos > 0.2f ) {
		temp = tan( acos( cos ) );
		power += kg * photonPower * ( 1.0f / max( 0.0001f, sqrt( dot( rayDir, rayNormal ) * dot( photonDir, rayNormal ) ) ) ) *
				exp( ( -1.0f * temp * temp ) / ( roughness * roughness ) ) / ( 4.0f * M_PI * roughness * roughness );
	}

	return power;
}

/**
 *	intersection point 의 density area 를
 *	area photon 을 이용해서 측정한다. 주의해야할 것은 intersection point 는
 *	고정된 상태에서 area photon 을 바꾸어 가면서 누적시키는 것이므로 반드시
 *	pPMIsectPoint 에 저장되어 있는 area 에 합을 누적시켜야 한다.  
 */
__global__ void cuCalDensityAreaKernel( cuIntersectionPoint *pIsectPoint, 
										cuPMIntersectionPoint *pPMIsectPoint, 
										int iPointTotalCount, int startOffset, float squareRadius )
{
	int photonIndexOffset;
	int photonIndexCount;
	int2 photonIndex;			//	PhotonIndex 와 매치
	int index = 0, index2 = 0;

	/** intersection point 의 pos, normal */
	float3 iPointPos;
	float3 iPointNormal;

	/** are photon info */
	float3 photonPos;
	float3 photonNormal;
	float photonArea;

	float4 temp;

	/** result 를 위한 임시변수 */
	float densityArea = 0.0f;
	int usedPhotonCount = 0, totalPhotonCount = 0;

	/**
	 *	현재 thread 가 처리해야할 ray 의 offset 을 가져온다.
	 */
	int iPointIndex = startOffset + ( blockIdx.y * gridDim.x + blockIdx.x ) * 
		blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	
	if ( iPointIndex < iPointTotalCount ) {
		
		/** 
		 * intersection point 주변의 photon 을 위한 index 정보 가져옴 
		 */
		photonIndexOffset = pPMIsectPoint[ iPointIndex ].photonIndexOffset;
		photonIndexCount = pPMIsectPoint[ iPointIndex ].photonIndexCount;
		iPointPos = pIsectPoint[ iPointIndex ].pos;
		iPointNormal = pIsectPoint[ iPointIndex ].normal;

		/**
		 *	현재 ray 주변의 photon 정보를 가리키고 있는 photonIndex 데이터를
		 *	이용해서 area photon 에 접근한다.
		 */

		totalPhotonCount = 0;
		usedPhotonCount = 0;

		for ( index = 0; index < photonIndexCount; ++index ) {

			photonIndex = tex1Dfetch( photonIndexTexture, photonIndexOffset + index );

			/** 
			 *	photon cell 안의 각 photon 을 하나씩 처리한다. 
			 *	photonIndex.x = photon start offset.
			 *	photonIndex.y = photon count.
			 */

			for ( index2 = 0; index2 < photonIndex.y; ++index2 ) {

				/** 
				 *	float type 12개가 하나의 photon 정보인데 texture type 이 float4 이므로
				 *	총 3개의 texture 값을 가져와서 data 를 구성한다.
				 *	area photon 의 경우는 power.x 가 area 값을 의미한다. area 를 계산할때는
				 *	pos, normal, area 값만 필요하므로 이 값만 쏙 빼낸다.
				 */
				temp = tex1Dfetch( areaPhotonTexture, ( photonIndex.x + index2 ) * 3 + 0 );
				photonPos.x = temp.x; photonPos.y = temp.y; photonPos.z = temp.z; 
				photonNormal.x = temp.w;

				temp = tex1Dfetch( areaPhotonTexture, ( photonIndex.x + index2 ) * 3 + 1 );
				photonNormal.y = temp.x; photonNormal.z = temp.y; photonArea = temp.z;
				
				/**
				 *	search dist 거리안에 들어오고 ray 삼각형 normal 과 photon 이 뭍은
				 *	삼각형 normal 이 지정된 각도 미만일때만 area 를 누적한다. 이 조건은
				 *	반드시 아래 caIPointRadianceKernel() 에서 photon 선택 기준과 같아야 한다.
				 */
				if ( calSquareDist( iPointPos, photonPos ) <= squareRadius &&
					 dotProduct( iPointNormal, photonNormal ) > VALID_COSINE_VALUE ) {
					densityArea += photonArea;
					usedPhotonCount++;
				}
			}

			totalPhotonCount += photonIndex.y;

		}
		
		pPMIsectPoint[ iPointIndex ].area += densityArea;
		pPMIsectPoint[ iPointIndex ].usedPhotonCount = usedPhotonCount;
		pPMIsectPoint[ iPointIndex ].totalPhotonCount = totalPhotonCount;

	}

}

/**
 *	radiance 계산.
 *	iteration 이 여러번 되므로, ipoint 의 radiance 는 누적해야 한다.
 */
__global__ void caIPointRadianceKernel( cuIntersectionPoint *pIsectPoint, 
										cuPMIntersectionPoint *pPMIsectPoint, 
										int iPointTotalCount, 
										int startOffset, float squareRadius )
{
	int photonIndexOffset;
	int photonIndexCount;
	
	int2 photonIndex;			//	PhotonIndex 와 매치
	int index = 0, index2 = 0;
	
	/** ray geometry info */
	float3 iPointPos;
	float3 iPointDir;
	float3 iPointNormal;
	bool isLight = false;

	float densityArea;
	float3 diffuse, specular;
	float roughness;

	/** photon info */
	float3 photonPos;
	float3 photonDir;
	float3 photonNormal;
	float3 photonPower;
	float4 temp;
	
	/** result 를 위한 임시변수 */
	float3 powerDiffuse, powerSpecular;
	int usedPhotonCount = 0, totalPhotonCount = 0;
	float squareDist;
	
	cuObjectMaterial material;
	int texture;
	float3 texColor;
	bool isTransmission;

	powerDiffuse = make_float3( 0.0f, 0.0f, 0.0f );
	powerSpecular = make_float3( 0.0f, 0.0f, 0.0f );

	/**
	 *	현재 thread 가 처리해야할 ray 의 offset 을 가져온다.
	 */
	int iPointIndex = startOffset + ( blockIdx.y * gridDim.x + blockIdx.x ) * 
		blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	
	if ( iPointIndex < iPointTotalCount ) {

		/** 
		 * intersection point 주변의 photon 을 위한 index 정보 가져옴 
		 */
		photonIndexOffset = pPMIsectPoint[ iPointIndex ].photonIndexOffset;
		photonIndexCount = pPMIsectPoint[ iPointIndex ].photonIndexCount;
		densityArea = pPMIsectPoint[ iPointIndex ].area;
		
		iPointPos = pIsectPoint[ iPointIndex ].pos;
		iPointNormal = pIsectPoint[ iPointIndex ].normal;
		iPointDir = pIsectPoint[ iPointIndex ].dir;

		getObjectMaterial( pIsectPoint[ iPointIndex ].objectIndex, material );
		
		diffuse = material.diffuse;
		specular = material.specular;
		roughness = material.roughness;
		isTransmission = ( material.transparency > 0.0f );
		isLight = ( material.light == 0.0f ) ? 0 : 1;

		texture = float_as_int( material.textureNumber );
		texture = fetchTexture( texture, 
								pIsectPoint[ iPointIndex ].u, pIsectPoint[ iPointIndex ].v, 
								texColor );
		float3 R = reflection( iPointDir, iPointNormal );

		/**
		 *	현재 ray 주변의 photon 정보를 가리키고 있는 photonIndex 데이터를
		 *	이용해서 photon 에 접근한다.
		 *	rayAttribute.y = photonIndex Offset.
		 *	rayAttribute.z = photonIndex Count.
		 */
		totalPhotonCount = 0;
		usedPhotonCount = 0;

		for ( index = 0; index < photonIndexCount; ++index ) {

			photonIndex = tex1Dfetch( photonIndexTexture, photonIndexOffset + index );

			/** 
			 *	photon cell 안의 각 photon 을 하나씩 처리한다. 
			 *	photonIndex.x = photon start offset.
			 *	photonIndex.y = photon count.
			 */

			for ( index2 = 0; index2 < photonIndex.y; ++index2 ) {

				/** 
				 *	float type 12개가 하나의 photon 정보인데 texture type 이 float4 이므로
				 *	총 3개의 texture 값을 가져와서 data 를 구성한다.
				 */
				temp = tex1Dfetch( photonTexture, ( photonIndex.x + index2 ) * 3 + 0 );
				photonPos.x = temp.x; photonPos.y = temp.y; photonPos.z = temp.z; 
				photonNormal.x = temp.w;

				temp = tex1Dfetch( photonTexture, ( photonIndex.x + index2 ) * 3 + 1 );
				photonNormal.y = temp.x; photonNormal.z = temp.y; photonPower.x = temp.z;
				photonPower.y = temp.w;

				temp = tex1Dfetch( photonTexture, ( photonIndex.x + index2 ) * 3 + 2 );
				photonPower.z = temp.x; photonDir.x = temp.y; photonDir.y = temp.z;
				photonDir.z = temp.w;

				/**
				 *	dist 거리안에 들어오고 ray 삼각형 normal 과 photon 이 뭍은
				 *	삼각형 normal 이 45' 미만일때만.
				 *	반드시 calDensityArea() 함수의 photon 선택기준과 같아야 한다.
				 */
				squareDist = calSquareDist( iPointPos, photonPos );

				/**
				 *	반경안에 들어오고, 기하적으로 너무 많은 차이가 나지 않는 지점.
				 *	그리고 광원이거나, 투명물체가 아니면, 삼각형 뒷면에 뭍은건 제외. 
				 *	광원이나 투명한 물체는 뒷면에 뭍어 있는 photon 의 dir 을 바꾸어서
				 *	사용한다.
				 */
				if ( squareDist <= squareRadius && 
					 dotProduct( iPointNormal, photonNormal ) > VALID_COSINE_VALUE &&
					( isTransmission || isLight || dotProduct( photonDir, iPointNormal ) >= 0.0f ) ) {
					
					/**
					 *	diffuse term 계산. dot 값이 음수라면 양수로 바꾸어 준다.
					 */
					//power += photonPower * kd_brdf;
					float pdoti = dot( photonDir, iPointNormal );
					if ( pdoti < 0.0f )
						pdoti = -1.0f * pdoti;
					float Rdotp = dot( R, photonDir );
					if ( Rdotp < 0.0f )
						Rdotp = -1.0f * Rdotp;

					/** 광원일때와 일반 물체일때 구분. */
					if ( isLight ) {
						powerSpecular += photonPower * max( 0.0f, pdoti );
					} else {
						powerDiffuse += photonPower * ( max( 0.0f, pdoti ) * diffuse );
						powerSpecular += photonPower * pow( max( 0.0f, Rdotp ), roughness ) * ( specular );
					}

					usedPhotonCount++;
				}
			}

			totalPhotonCount += photonIndex.y;

		}

		/** 
		 *	texture 존재 여부에 따라서 material 색깔 선택 
		 */
		if ( texture == 1 ) {
			powerDiffuse = powerDiffuse * texColor;
			powerSpecular = powerSpecular * texColor;
		} else {
			powerDiffuse = powerDiffuse * diffuse;
			powerSpecular = powerSpecular * specular;
		}

		/**	
		 *	이 kernel 을 호출하는 쪽에서 iteration 시작 전에 반드시
		 *	power 를 초기화	해야 한다.
		 *	iteration 이 여러번 되므로, ipoint 의 radiance 는 누적해야 한다. 
		 */
		pPMIsectPoint[ iPointIndex ].power[ 0 ] += ( ( powerDiffuse.x + powerSpecular.x ) / ( densityArea ) );
		pPMIsectPoint[ iPointIndex ].power[ 1 ] += ( ( powerDiffuse.y + powerSpecular.y ) / ( densityArea ) );
		pPMIsectPoint[ iPointIndex ].power[ 2 ] += ( ( powerDiffuse.z + powerSpecular.z ) / ( densityArea ) );
		pPMIsectPoint[ iPointIndex ].area = densityArea;

		pPMIsectPoint[ iPointIndex ].usedPhotonCount = usedPhotonCount;
		pPMIsectPoint[ iPointIndex ].totalPhotonCount = totalPhotonCount;

	}

}

 