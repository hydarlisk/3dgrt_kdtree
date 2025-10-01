/*
 * Sogang Univ, Graphics Lab, 2024
 * 
 * Abura Soba, 2025
 */

#pragma once

#include "VulkanUtils.h"

#include <vector>

using namespace std;

enum BLASMode {
	Default,
	UniformGrid,
	AdaptiveGrid,
	KDTree,
	LoadJson
};

inline const char* BLASModeToString(BLASMode mode) {
	switch (mode) {
	case Default:		return "Default";
	case UniformGrid:   return "UniformGrid";
	case AdaptiveGrid:	return "AdaptiveGrid";
	case KDTree:		return "KDTree";
	case LoadJson:		return "LoadJson";
	default:			return "Unknown";
	}
}

class BVHBuilder {
private:
	vkglTF::Model* scene;

	//BLASSplitter* blasSplitter;
	vks::VulkanDevice& vulkanDevice;
	VkDevice& device;
	VkQueue& queue;

	glm::vec3 minPos{ FLT_MAX ,FLT_MAX ,FLT_MAX };
	glm::vec3 maxPos{ -FLT_MAX,-FLT_MAX,-FLT_MAX };

	//vector<vkglTF::Vertex>* vertices;
	//vector<uint32_t>* indices;

	/* createBLAS */
	PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
	PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
	PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
	PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
	PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
	PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR;
	PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
	PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp;

	struct AccelerationStructure : vks::SimpleVkBuffer {
		VkAccelerationStructureKHR handle;
		uint64_t deviceAddress = 0;
	};

	struct ScratchBuffer
	{
		uint64_t deviceAddress = 0;
		VkBuffer handle = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
	};

	VkTransformMatrixKHR tMat{};
	vks::Buffer tMatBuffer;

	float cellWeight = INIT_CELL_WEIGHT;
	float travelCost = INIT_TRAVEL_COST;

public:
	uint32_t cellsPerLongestAxis;
	vector<AccelerationStructure> splittedBLAS;
	AccelerationStructure tlas;

	VkQueryPool ASBuildTimeStampQueryPool;
	std::vector<uint64_t> ASBuildTimeStamps;
	VkDeviceSize blasSize = 0;
	VkDeviceSize tlasSize = 0;

private:
	/* split BLAS */
	void clipCell_1VB(std::vector<AABB_Triangle_Clipping::_AABB> gridAabb, int cellCnt, std::vector<vkglTF::Vertex>& clippedVertices, std::vector<std::vector<uint32_t>>& splittedIdx);
	void saveGeometries();
	void splitGeometry(BLASMode splitMode);

	/* create AS */
	void initASBuildTimestamp();
	void createAccelerationStructureBuffer(AccelerationStructure& accelerationStructure, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo);
	uint64_t getBufferDeviceAddress(VkBuffer buffer);
	ScratchBuffer createScratchBuffer(VkDeviceSize size);
	void deleteScratchBuffer(ScratchBuffer& scratchBuffer);
	void createBLAS(int cellIdx);
	void createBLASes();
	void createTLAS();

	void deleteAccelerationStructure(AccelerationStructure& accelerationStructure);
	void destroyAS();

public:
	BVHBuilder(vks::VulkanDevice& device, VkQueue& queue);
	~BVHBuilder();

	void setGeometry(vkglTF::Model& scene);
	void createAS(BLASMode splitMode);
	void rebuild(BLASMode splitMode, float cellWeight, float travelCost);
	
	void printASBuildInfo(VkPhysicalDeviceProperties deviceProperties);
};