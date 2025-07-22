#pragma once

#include "GBase.h"
#include "GScene.h"
#include "GTriangleWrapper.h"
#include "GTriangleWrapperList.h"
//#include "SSERenderPipeline.h"
#include "SSERenderData.h"

#include "GBoundingBox.h"

#define FLOAT_MAX 1.0E10f
#define FM_EPSILON ((float)0.00001)
//#define FM_EPSILON ((float)0.0002)
#define LOCAL_EPSILON ((float)0.001)

typedef std::vector<unsigned int> SVPOLYINDEX;

class  GBVHNode
{
public:
	//! C-tor
	GBVHNode(const GBoundingBox& aabb)
		: m_aabb(aabb), m_pnodeFirstChild(0), m_psvPolyIndex(0), m_axis(3), firstActive(0), flag(false)
	{ /* NOP */ }

	//! D-tor
	~GBVHNode()
	{
		if(isLeaf())
		{
			delete m_psvPolyIndex;
		}
		else
		{
			// see createChilds() for why
			(m_pnodeFirstChild    )->~GBVHNode();
			(m_pnodeFirstChild + 1)->~GBVHNode();

			free(m_pnodeFirstChild);
		}
	}

	//! make this leaf node
	SVPOLYINDEX& leafNode(unsigned int nPoly)
	{
		m_psvPolyIndex = new SVPOLYINDEX(nPoly);
		return *m_psvPolyIndex;
	}

	//! create two child nodes
	GBVHNode* createChilds(const unsigned char axis, const GBoundingBox& aabbA, const GBoundingBox& aabbB)
	{
		m_axis = axis;
		m_pnodeFirstChild = static_cast<GBVHNode*>(malloc(sizeof(GBVHNode) * 2));

		new(m_pnodeFirstChild  ) GBVHNode(aabbA);
		new(m_pnodeFirstChild+1) GBVHNode(aabbB);

		return m_pnodeFirstChild;
	}

	const GBoundingBox getAABB() const
	{
		return m_aabb;
	}

	// just same with getAABB()... only no 'const'
	GBoundingBox* thisAABB() 
	{
		return &m_aabb;
	}

	bool isLeaf() const
	{
		return (m_pnodeFirstChild == 0);
	}

	void getOrderedChilds(const GBVHNode** ppFirstChild, const GBVHNode** ppSecondChild, GVector& vRayDir) const
	{
		unsigned int first = (unsigned int)( !(vRayDir[m_axis] > 0) );
		*ppFirstChild  = m_pnodeFirstChild + first;
		*ppSecondChild = m_pnodeFirstChild + (1-first);
	}

	void getOrderedChilds(GBVHNode** ppFirstChild, GBVHNode** ppSecondChild, GVector& vRayDir) const
	{
		unsigned int first = (unsigned int)( !(vRayDir[m_axis] > 0) );
		*ppFirstChild  = m_pnodeFirstChild + first;
		*ppSecondChild = m_pnodeFirstChild + (1-first);
	}	

	// --------------------------------------------
	// BVH UPDATE
	// --------------------------------------------
	// get left & right child. 
	// for updating bouding boxes
	void getChildforUpdate( GBVHNode** ppLeftChild, GBVHNode** ppRightChild ) const
	{
		*ppLeftChild  = m_pnodeFirstChild ;
		*ppRightChild = m_pnodeFirstChild + 1;
	}

	// inner node 의 AABB를 update한다.
	void in_nodeUpdateAABB( GBoundingBox* bbox )
	{
		GBoundingBox* first = m_pnodeFirstChild->thisAABB();
		GBoundingBox* second = (m_pnodeFirstChild+1)->thisAABB();
		GVector temp_max = first->getMax();
		GVector temp_min = first->getMin();
		bbox->tryupdateMax(temp_max);
		bbox->tryupdateMin(temp_min);
		temp_max = second->getMax();
		temp_min = second->getMin();
		bbox->tryupdateMax(temp_max);
		bbox->tryupdateMin(temp_min);
	}

	// 두 자식의 flag를 return 한다.
	// 한 자식이라도 update되어 flag 셋팅 되어 있으면 return true
	bool getChildrenFlag()
	{
		return ( m_pnodeFirstChild->flag || (m_pnodeFirstChild+1)->flag );
	}

	// 두 자식의 flag를 다시 false로 만든다.
	void setChildrenFlagZero()
	{
		m_pnodeFirstChild->flag = false;
		(m_pnodeFirstChild+1)->flag = false;
	}

	void updateAABB ( GBoundingBox& bbox )
	{
		m_aabb.m_Min = bbox.m_Min;
		m_aabb.m_Max = bbox.m_Max;
	}
	// END BVH UPDATE
	// --------------------------------------------

	const SVPOLYINDEX& getPolyIndex() const
	{
		return *m_psvPolyIndex;
	}

	int getFirstActive() 
	{
		return firstActive;
	}

	void setFirstActive(unsigned int first) 
	{
		firstActive = first;
	}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

	bool flag;// BVH UPDATE에 쓰이는 flag. 나중에 이 변수 없앨 수 있도록???
private:
	// AABB of this node
	GBoundingBox m_aabb;
	
	//! pointer to the first child node	
	//	@note
	//		Second child node is located just after the first child node.
	//		If this node is a leaf node, this pointer will be null.
	GBVHNode* m_pnodeFirstChild;
	
	//! child polys (!=0 only when this node is a leaf node)
	SVPOLYINDEX* m_psvPolyIndex;	
	
	//! split axis (for ordered traversal)
	unsigned char m_axis;

	// for Ranged Traversal (packet)
	// index of first hit ray
	unsigned int firstActive;
};


class GBVHStructure : public GSpatialStructure
{
public:
	GBVHStructure(const GBoundingBox& aabbScene);
	GBVHStructure( GScene *pScene);
	virtual ~GBVHStructure();

	virtual GError initialize() ;
	virtual GError makeSSERenderStructureInfo( SSESceneData *pSSEData );

	//-------------------------------------------------------
	// unused.
	virtual GError uninitialize() { return errorNo; }
	virtual int getTriangleCount() { return 0; }

	virtual bool loadStructureFromFile( const char *filename ) { return false; }
	virtual bool saveStructureToFile( const char *filename ) {return false; }
	// end unused -------------------------------------------

private:
	//! precalculate triangle polys' centroid and AABB
	/*!
		@param aryvCentroid
			calculated centroid will be stored here
		@param aryAABB
			calculated AABB will be stored here
		@pre
			called from construct()
	 */
	void pre_calculate(GVector* aryvCentroid, GBoundingBox* aryAABB);   // 전처리	

	// functors for sort based on centroids
	class compareCentroidX;
	class compareCentroidY;
	class compareCentroidZ;

	//! construct BVH
	/*!
		@param aryvCentroid
			precalculated centroid
		@param aryAABB
			precalculated poly AABB
		@pre
			called from construct()
	 */
	void constructBVH(const GVector* aryvCentroid, const GBoundingBox* aryAABB);

	//! compute cost using SAH
	/*!
		compute cost of partitioning at specific position using surface area heuristics

		@param nLeft
			number of polys in the left of the partition
		@param nLeftArea
			sum of surface area of polys in the left of the partition
		@param nRight
			number of polys in the right of the partition
		@param nRightArea
			sum of surface area of polys in the right of the partition
		@param fAABBArea
			surface area of AABB of the parent node
	 */
	static float SAH ( const unsigned int nLeft,  const float nLeftArea, 
		               const unsigned int nRight, const float nRightArea, const float fAABBArea );

	//! pointer to the root node
	GBVHNode* m_pnodeRoot;

	// aabb about whole scene
	GBoundingBox m_aabbScene;

	GScene *m_pScene;

	// total triangle count.
	unsigned int m_totTriCount;

	// Triangle list.
	GTriangleWrapperList *m_pSceneTriangleList;
};
