#pragma once

#include "GBase.h"

/**
 *	ray 와 Object 의 intersection 을
 *	위한 공간자료구조 클래스.
 *	
 *	by graphicsian.
 */
class GSpatialStructure
{
public:
	virtual ~GSpatialStructure() {}

	virtual GError initialize() = 0;

	/**
	 *	rayCount 개수만큼의 ray 와 공간상의 Object 와의 교점을 체크해서
	 *	pResult 에 담아돌려준다. pResult 는 rayCount 만큼의 공간을 잡아서
	 *	인자로 넣어주어야 한다.
	 */
	//virtual GError intersect( GRayInfo *pRay, GIntersection *pResult, int rayCount ) = 0;

	virtual GError uninitialize() = 0;
	virtual int getTriangleCount() = 0;

	virtual bool loadStructureFromFile( const char *filename ) = 0;
	virtual bool saveStructureToFile( const char *filename ) = 0;
};
