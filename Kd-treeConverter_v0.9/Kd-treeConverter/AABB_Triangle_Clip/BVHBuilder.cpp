/*
 * Sogang Univ, Graphics Lab, 2024
 * 
 * Abura Soba, 2025
 */

#include "BVHBuilder.hpp"

#include "VulkanRTCommon.h"
#include "VulkanUtils.h"

#include "AABB_Clipping.h"
#include "Kd-treeConstructor.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <vector>
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

#if defined(__ANDROID__)
constexpr std::string_view box_path = "boxes/";
#else
constexpr std::string_view box_path = "./../boxes/";
#endif

struct AccelerationStructure {
	VkAccelerationStructureKHR handle;
	uint64_t deviceAddress = 0;
	VkDeviceMemory memory;
	VkBuffer buffer;

	void destroy(VkDevice device) {
		if (buffer)
		{
			vkDestroyBuffer(device, buffer, nullptr);
		}
		if (memory)
		{
			vkFreeMemory(device, memory, nullptr);
		}
	}
};

std::vector<AABB_Triangle_Clipping::_AABB> loadBoxesFromJson(const std::string& filePath) {
	json j;
	std::vector<AABB_Triangle_Clipping::_AABB> boxes;

#if defined(__ANDROID__)
	AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filePath.c_str(), AASSET_MODE_BUFFER);
	if (!asset) {
		vks::tools::exitFatal("Could not load json from " + filePath + "\n\nMake sure the assets submodule has been checked out and is up-to-date.", -1);
		return boxes;
	}
	size_t size = AAsset_getLength(asset);
	assert(size > 0);
	std::vector<char> buffer(size + 1);
	AAsset_read(asset, buffer.data(), size);
	buffer[size] = '\0';
	AAsset_close(asset);
	j = json::parse(buffer.data());
#else
	std::ifstream inFile(filePath);
	if (!inFile.is_open()) {
		std::cerr << "Error: Could not open file for reading: " << filePath << "\n";
		return boxes;
	}

	inFile >> j;
#endif

	for (const auto& boxJson : j) {
		AABB_Triangle_Clipping::_AABB box;
		std::vector<float> minVec = boxJson["min"].get<std::vector<float>>();
		std::vector<float> maxVec = boxJson["max"].get<std::vector<float>>();

		if (minVec.size() == 3 && maxVec.size() == 3) {
			box.xmin = minVec[0]; box.ymin = minVec[1]; box.zmin = minVec[2];
			box.xmax = maxVec[0]; box.ymax = maxVec[1]; box.zmax = maxVec[2];
			
			boxes.push_back(box);
		}
	}

	return boxes;
}

BVHBuilder::BVHBuilder(vks::VulkanDevice& device, VkQueue& queue) : vulkanDevice(device), device(device.logicalDevice), queue(queue){
	//blasSplitter = new BLASSplitter(device, queue);

	vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetDeviceProcAddr(device.logicalDevice, "vkGetBufferDeviceAddressKHR"));
	vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device.logicalDevice, "vkCmdBuildAccelerationStructuresKHR"));
	vkBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device.logicalDevice, "vkBuildAccelerationStructuresKHR"));
	vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device.logicalDevice, "vkCreateAccelerationStructureKHR"));
	vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device.logicalDevice, "vkDestroyAccelerationStructureKHR"));
	vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device.logicalDevice, "vkGetAccelerationStructureBuildSizesKHR"));
	vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device.logicalDevice, "vkGetAccelerationStructureDeviceAddressKHR"));
	vkCmdWriteTimestamp = reinterpret_cast<PFN_vkCmdWriteTimestamp>(vkGetDeviceProcAddr(device, "vkCmdWriteTimestamp"));

	glm::mat3x4 m = glm::dmat3x4(glm::mat4(1.0f));
	memcpy(&tMat, (void*)&m, sizeof(glm::mat3x4));
	VK_CHECK_RESULT(vulkanDevice.createBuffer(
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&tMatBuffer,
		sizeof(VkTransformMatrixKHR),
		&tMat));
}

BVHBuilder::~BVHBuilder() {
	tMatBuffer.destroy();
	destroyAS();
}


/*** private functions ***/
/**
* vertexBuffer, indexBuffer : triangle info
* uniqueVertices : unordered_map for deduplication
* gridAabb : aabbs of clipping space
* cellCnt  : aabb count of clipping space
* splittedIdx : return of function
*/
void BVHBuilder::deleteAccelerationStructure(AccelerationStructure& accelerationStructure){
	vkFreeMemory(device, accelerationStructure.memory, nullptr);
	vkDestroyBuffer(device, accelerationStructure.buffer, nullptr);
	vkDestroyAccelerationStructureKHR(device, accelerationStructure.handle, nullptr);
}

void printGeometryInfos(int totalVert, int totalIndex, int totalVertSize, int totalIndexSize) {
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
	LOGD("Total Tri Count : %d", totalIndex / 3);
	LOGD("Total Vert Count : %d", totalVert);
	LOGD("Total Index Count : %d", totalIndex);
#else
	std::cout << "Total Vert : " << totalVert << "\n";
	std::cout << "Total Vert Size : " << totalVertSize << " (bytes)\n";
	std::cout << "Total Index : " << totalIndex << "\n";
	std::cout << "Total Index Size : " << totalIndexSize << " (bytes)\n";
	std::cout << "Total Tri Count : " << totalIndex / 3 << "\n";
#endif
}

void BVHBuilder::clipCell_1VB(std::vector<AABB_Triangle_Clipping::_AABB> gridAabb, int cellCnt, std::vector<vkglTF::Vertex>& clippedVertices, std::vector<std::vector<uint32_t>>& splittedIdx) {
	std::vector<int> errors;
	std::unordered_map<vkglTF::Vertex, uint32_t> uniqueVertices;

	for (int i = 0; i < scene->indicesVec.size(); i += 3) {

		vkglTF::Vertex tri[3] = { scene->verticesVec[scene->indicesVec[i]], scene->verticesVec[scene->indicesVec[i + 1]], scene->verticesVec[scene->indicesVec[i + 2]] };//기존 삼각형
		glm::vec3 triMinPos = glm::min(glm::min(tri[0].pos, tri[1].pos), tri[2].pos);
		glm::vec3 triMaxPos = glm::max(glm::max(tri[0].pos, tri[1].pos), tri[2].pos);

		for (int cellIdx = 0; cellIdx < cellCnt; ++cellIdx) {
			if (gridAabb[cellIdx].xmax < triMinPos.x || gridAabb[cellIdx].xmin >= triMaxPos.x) continue;
			if (gridAabb[cellIdx].ymax < triMinPos.y || gridAabb[cellIdx].ymin >= triMaxPos.y) continue;
			if (gridAabb[cellIdx].zmax < triMinPos.z || gridAabb[cellIdx].zmin >= triMaxPos.z) continue;

			//TODO : test aabb-triangle intersection before clipping
			std::vector<vkglTF::Vertex> polygon;
			AABB_Triangle_Clipping::_clip_triangle_against_AABB_np(tri, gridAabb[cellIdx], polygon);//여기서 잘라

			if (polygon.size() < 3) continue;
			for (int k = 1; k < polygon.size() - 1; ++k) {//그걸 삼각형으로 만들어
				if (uniqueVertices.count(polygon[0]) == 0) {
					uniqueVertices[polygon[0]] = static_cast<uint32_t>(clippedVertices.size());
					clippedVertices.push_back(polygon[0]);
				}
				splittedIdx[cellIdx].push_back(uniqueVertices[polygon[0]]);

				if (uniqueVertices.count(polygon[k]) == 0) {
					uniqueVertices[polygon[k]] = static_cast<uint32_t>(clippedVertices.size());
					clippedVertices.push_back(polygon[k]);
				}
				splittedIdx[cellIdx].push_back(uniqueVertices[polygon[k]]);

				if (uniqueVertices.count(polygon[k + 1]) == 0) {
					uniqueVertices[polygon[k + 1]] = static_cast<uint32_t>(clippedVertices.size());
					clippedVertices.push_back(polygon[k + 1]);
				}
				splittedIdx[cellIdx].push_back(uniqueVertices[polygon[k + 1]]);
				//여기서 면적 누적
			}
			//여기서 하나의 cell에 대한 면적이 나옴
		}
	}
}

void BVHBuilder::saveGeometries() {
	scene->splittedIndicesBuffers.resize(1);
	size_t vertexBufferSize = scene->verticesVec.size() * sizeof(vkglTF::Vertex);
	size_t indexBufferSize = scene->indicesVec.size() * sizeof(uint32_t);
	scene->splittedIndicesBuffers[0].count = static_cast<uint32_t>(scene->indicesVec.size());
	scene->vertices.count = static_cast<uint32_t>(scene->verticesVec.size());

	assert((vertexBufferSize > 0) && (indexBufferSize > 0));
	/* host scene->verticesVec -> device scene->vertices */
	/* host scene->indicesVec -> device splittedIndices[0] */
	vulkanDevice.createAndCopyToDeviceBuffer(scene->verticesVec.data(), scene->vertices.buffer, scene->vertices.memory, vertexBufferSize, queue, vkglTF::bufferUsageFlags);
	vulkanDevice.createAndCopyToDeviceBuffer(scene->indicesVec.data(), scene->splittedIndicesBuffers[0].buffer, scene->splittedIndicesBuffers[0].memory, indexBufferSize, queue, vkglTF::bufferUsageFlags);

	printGeometryInfos(scene->vertices.count, scene->splittedIndicesBuffers[0].count, vertexBufferSize, indexBufferSize);
}

void BVHBuilder::splitGeometry(BLASMode splitMode) {
	std::vector<AABB_Triangle_Clipping::_AABB> gridAabb;

	switch (splitMode) {
	case UniformGrid: {
#if defined(__ANDROID__)
		LOGD("BLAS mode : Uniform Grid, Param: %f\n", cellWeight);
#else
		cout << "BLAS mode : Uniform Grid, Param: " << cellWeight << "\n";
#endif
		const glm::vec3 sceneSize = maxPos - minPos;
		const float sceneVolume = sceneSize.x * sceneSize.y * sceneSize.z;
		const float finalCellWeight = std::cbrt(scene->indicesVec.size() / 3 / sceneVolume) * cellWeight;

		const glm::ivec3 numCells = glm::ivec3(sceneSize * finalCellWeight + 0.5f);
		const glm::vec3 cellSize = sceneSize / glm::vec3(numCells);

		gridAabb.resize(numCells.x * numCells.y * numCells.z);

		auto calcIdx = [numCells](glm::ivec3 idx)->int {return numCells.x * numCells.y * idx.z + numCells.x * idx.y + idx.x; };
		//split space by square cells
		for (int i = 0; i < numCells.x; ++i) {
			for (int j = 0; j < numCells.y; ++j) {
				for (int k = 0; k < numCells.z; ++k) {
					gridAabb[calcIdx({ i, j, k })] = {
						minPos.x + i * cellSize.x, minPos.y + j * cellSize.y, minPos.z + k * cellSize.z,
						minPos.x + (i + 1) * cellSize.x, minPos.y + (j + 1) * cellSize.y, minPos.z + (k + 1) * cellSize.z
					};
				}
			}
		}

#if defined(__ANDROID__)
		LOGD("width, height, depth: %d, %d, %d\n", numCells.x, numCells.y, numCells.z);
#else
		std::cout << "width, height, depth: " << numCells.x << ", " << numCells.y << ", " << numCells.z << "\n";
#endif
		break;
	}
	case AdaptiveGrid: {
#if defined(__ANDROID__)
		LOGD("BLAS mode : Adaptive Grid, Param: %f\n", travelCost);
#else
		cout << "BLAS mode : Adaptive Grid, Param: " << travelCost << "\n";
#endif
		kdTreeTravelCost = this->travelCost;
		initialize_kd_tree(scene->verticesVec, scene->indicesVec, minPos, maxPos);

		std::vector<float> splitsX;
		splitsX.push_back(minPos.x);
		splitAlongAxis(0, g_bEdge, g_pTriangleInfos, g_iTriangleSize, g_root_AABB, splitsX);
		std::sort(splitsX.begin(), splitsX.end());
		splitsX.push_back(maxPos.x);

		std::vector<float> splitsY;
		splitsY.push_back(minPos.y);
		splitAlongAxis(1, g_bEdge, g_pTriangleInfos, g_iTriangleSize, g_root_AABB, splitsY);
		std::sort(splitsY.begin(), splitsY.end());
		splitsY.push_back(maxPos.y);

		std::vector<float> splitsZ;
		splitsZ.push_back(minPos.z);
		splitAlongAxis(2, g_bEdge, g_pTriangleInfos, g_iTriangleSize, g_root_AABB, splitsZ);
		std::sort(splitsZ.begin(), splitsZ.end());
		splitsZ.push_back(maxPos.z);

		const int width = splitsX.size() - 1, height = splitsY.size() - 1, depth = splitsZ.size() - 1;
		auto calcIdx = [width, height](glm::ivec3 idx)->int {return width * (height * idx.z + idx.y) + idx.x; };
		gridAabb.resize(width * height * depth);

		for (int i = 0; i < width; ++i) {
			for (int j = 0; j < height; ++j) {
				for (int k = 0; k < depth; ++k) {
					gridAabb[calcIdx({ i, j, k })] = {
						splitsX[i], splitsY[j], splitsZ[k],
						splitsX[i + 1], splitsY[j + 1], splitsZ[k + 1]
					};
				}
			}
		}

#if defined(__ANDROID__)
		LOGD("width, height, depth: %d, %d, %d\n", width, height, depth);
#else
		std::cout << "width, height, depth: " << width << ", " << height << ", " << depth << "\n";
#endif
		uninitialize_kd_tree();
		break;
	}
	case KDTree: {
#if defined(__ANDROID__)
		LOGD("BLAS mode : Kd Tree, Param: %f\n", travelCost);
#else
		cout << "BLAS mode : Kd Tree, Param: " << travelCost << "\n";
#endif
		kdTreeTravelCost = this->travelCost;
		if (!initialize_kd_tree(scene->verticesVec, scene->indicesVec, minPos, maxPos)) {
			vks::tools::exitFatal("Could not initialize Kd-tree.", -1);
			return;
		}
		std::cout << "\n  - Building a Kd-tree\n";
		build_kd_tree_recursive(g_bEdge, g_pTriangleInfos, g_iTriangleSize, g_root_AABB, 0, &(g_pKdTree_Node_Array[0]));
		std::cout << "  - Done!\n\n";

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
		LOGD("Tree Level : %d", g_iKdTree_Level);
		LOGD("Node Count (All, Leaf, Empty) : %d, %d, %d(%f%%)", g_iKdTree_Node_Count, g_iKdTree_LeafNode_Count, g_iKdTree_EmptyNode_Count, 100.0f * g_iKdTree_EmptyNode_Count / g_iKdTree_Node_Count);
		LOGD("Maximum Tri# in LeafNode: %d", g_iKdTree_MaxTriInLeafNode_Count);
#else
		std::cout << "   * Tree Level: " << g_iKdTree_Level << "\n";
		std::cout << "   * Node Count (All,Leaf,Empty) : "
			<< g_iKdTree_Node_Count << "," << g_iKdTree_LeafNode_Count << "," << g_iKdTree_EmptyNode_Count
			<< "(" << 100.0f * g_iKdTree_EmptyNode_Count / g_iKdTree_Node_Count << "%%)\n";
		std::cout << "   * Maximum Tri# in LeafNode: " << g_iKdTree_MaxTriInLeafNode_Count << "\n\n";
#endif
		std::vector<BoundingBox> leafBoxes = extract_leaves_from_kd_tree();

		const uint32_t numCellsTotal = leafBoxes.size();

		gridAabb.resize(numCellsTotal);

		for (int i = 0; i < numCellsTotal; i++) {
			gridAabb[i].xmin = leafBoxes[i].min[0]; gridAabb[i].ymin = leafBoxes[i].min[1]; gridAabb[i].zmin = leafBoxes[i].min[2];
			gridAabb[i].xmax = leafBoxes[i].max[0]; gridAabb[i].ymax = leafBoxes[i].max[1]; gridAabb[i].zmax = leafBoxes[i].max[2];
		}
		uninitialize_kd_tree();
		break;
	}
	case LoadJson: {
#if defined(__ANDROID__)
		LOGD("BLAS mode : Load Json");
#else
		cout << "BLAS mode : Load Json\n";
#endif

		std::string sceneName = std::string(ASSET_PATH);
		size_t lastSlash = sceneName.find_last_of("/");
		size_t lastDot = sceneName.find_last_of(".");
		sceneName = sceneName.substr(lastSlash + 1, lastDot - lastSlash - 1);		

		gridAabb = loadBoxesFromJson(std::string(box_path) + sceneName + ".json");
#if defined(__ANDROID__)
		LOGD("Number of loaded boxes: %d", gridAabb.size());
#else
		std::cout << "Number of loaded boxes: " << gridAabb.size() << "\n";
#endif
		break;
	}
	}

	std::vector<vkglTF::Vertex> clippedVertices;
	std::vector<std::vector<uint32_t>> splittedIdx(gridAabb.size());
	// split vertexes
	clipCell_1VB(gridAabb, gridAabb.size(), clippedVertices, splittedIdx);

	int totalVert = 0;
	int totalVertSize = 0;
	int totalIndex = 0;
	int totalIndexSize = 0;

	//Vertices
	{
		size_t vertexBufferSize = clippedVertices.size() * sizeof(vkglTF::Vertex);
		scene->vertices.count = static_cast<uint32_t>(clippedVertices.size());
		totalVert = scene->vertices.count;
		totalVertSize = vertexBufferSize;
		vulkanDevice.createAndCopyToDeviceBuffer(clippedVertices.data(), scene->vertices.buffer, scene->vertices.memory, vertexBufferSize, queue, vkglTF::bufferUsageFlags);
	}

	// Splitted Indices
	int emptyCnt = 0;
	for (int i = 0; i < splittedIdx.size(); i++) {
		if (splittedIdx[i].size() == 0) {
			emptyCnt++;
			continue;
		}
		size_t indexBufferSize = splittedIdx[i].size() * sizeof(uint32_t);
		vks::SimpleVkBuffer cellIndices{};
		cellIndices.count = static_cast<uint32_t>(splittedIdx[i].size());
		totalIndex += cellIndices.count;
		totalIndexSize += indexBufferSize;
		assert(indexBufferSize > 0);

		vulkanDevice.createAndCopyToDeviceBuffer(splittedIdx[i].data(), cellIndices.buffer, cellIndices.memory, indexBufferSize, queue, vkglTF::bufferUsageFlags);

		scene->splittedIndicesBuffers.push_back(cellIndices);
	}

	printGeometryInfos(totalVert, totalIndex, totalVertSize, totalIndexSize);
}


/* create BLAS */
void BVHBuilder::createAccelerationStructureBuffer(AccelerationStructure& accelerationStructure, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo){
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = buildSizeInfo.accelerationStructureSize;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	VK_CHECK_RESULT(vkCreateBuffer(vulkanDevice.logicalDevice, &bufferCreateInfo, nullptr, &accelerationStructure.buffer));
	VkMemoryRequirements memoryRequirements{};
	vkGetBufferMemoryRequirements(vulkanDevice.logicalDevice, accelerationStructure.buffer, &memoryRequirements);
	VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
	memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
	VkMemoryAllocateInfo memoryAllocateInfo{};
	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = vulkanDevice.getMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VK_CHECK_RESULT(vkAllocateMemory(vulkanDevice.logicalDevice, &memoryAllocateInfo, nullptr, &accelerationStructure.memory));
	VK_CHECK_RESULT(vkBindBufferMemory(vulkanDevice.logicalDevice, accelerationStructure.buffer, accelerationStructure.memory, 0));
}

uint64_t BVHBuilder::getBufferDeviceAddress(VkBuffer buffer)
{
	VkBufferDeviceAddressInfoKHR bufferDeviceAI{};
	bufferDeviceAI.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bufferDeviceAI.buffer = buffer;
	return vkGetBufferDeviceAddressKHR(vulkanDevice.logicalDevice, &bufferDeviceAI);
}

BVHBuilder::ScratchBuffer BVHBuilder::createScratchBuffer(VkDeviceSize size)
{
	ScratchBuffer scratchBuffer{};
	// Buffer and memory
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	VK_CHECK_RESULT(vkCreateBuffer(vulkanDevice.logicalDevice, &bufferCreateInfo, nullptr, &scratchBuffer.handle));
	VkMemoryRequirements memoryRequirements{};
	vkGetBufferMemoryRequirements(vulkanDevice.logicalDevice, scratchBuffer.handle, &memoryRequirements);
	VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
	memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
	VkMemoryAllocateInfo memoryAllocateInfo = {};
	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = vulkanDevice.getMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VK_CHECK_RESULT(vkAllocateMemory(vulkanDevice.logicalDevice, &memoryAllocateInfo, nullptr, &scratchBuffer.memory));
	VK_CHECK_RESULT(vkBindBufferMemory(vulkanDevice.logicalDevice, scratchBuffer.handle, scratchBuffer.memory, 0));
	// Buffer device address
	VkBufferDeviceAddressInfoKHR bufferDeviceAddresInfo{};
	bufferDeviceAddresInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bufferDeviceAddresInfo.buffer = scratchBuffer.handle;
	scratchBuffer.deviceAddress = vkGetBufferDeviceAddressKHR(vulkanDevice.logicalDevice, &bufferDeviceAddresInfo);
	return scratchBuffer;
}

void BVHBuilder::deleteScratchBuffer(BVHBuilder::ScratchBuffer& scratchBuffer)
{
	if (scratchBuffer.memory != VK_NULL_HANDLE) {
		vkFreeMemory(vulkanDevice.logicalDevice, scratchBuffer.memory, nullptr);
	}
	if (scratchBuffer.handle != VK_NULL_HANDLE) {
		vkDestroyBuffer(vulkanDevice.logicalDevice, scratchBuffer.handle, nullptr);
	}
}

void BVHBuilder::initASBuildTimestamp()
{
	ASBuildTimeStamps.resize(scene->splittedIndicesBuffers.size() * 2 + 2);

	VkQueryPoolCreateInfo queryPoolInfo{};
	queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
	queryPoolInfo.queryCount = static_cast<uint32_t>(ASBuildTimeStamps.size());
	VK_CHECK_RESULT(vkCreateQueryPool(vulkanDevice.logicalDevice, &queryPoolInfo, nullptr, &ASBuildTimeStampQueryPool));

	VkCommandBuffer commandBuffer = vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
	vkCmdResetQueryPool(commandBuffer, ASBuildTimeStampQueryPool, 0, static_cast<uint32_t>(ASBuildTimeStamps.size()));
	vulkanDevice.flushCommandBuffer(commandBuffer, queue);

}

void BVHBuilder::createBLAS(int cellIdx) {
	VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo;
	VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo;

	uint32_t primitiveCount = scene->splittedIndicesBuffers[cellIdx].count / 3;
	VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
	VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};
	VkDeviceOrHostAddressConstKHR transformBufferDeviceAddress{};

	vertexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene->vertices.buffer);
	indexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene->splittedIndicesBuffers[cellIdx].buffer);
	transformBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(tMatBuffer.buffer);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	geometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
	geometry.geometry.triangles.maxVertex = scene->vertices.count;
	geometry.geometry.triangles.vertexStride = sizeof(vkglTF::Vertex);
	geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
	geometry.geometry.triangles.indexData = indexBufferDeviceAddress;
	geometry.geometry.triangles.transformData = transformBufferDeviceAddress;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

	buildRangeInfo.firstVertex = 0;
	buildRangeInfo.primitiveOffset = 0;
	buildRangeInfo.primitiveCount = scene->splittedIndicesBuffers[cellIdx].count / 3;
	buildRangeInfo.transformOffset = 0;
	pBuildRangeInfo = &buildRangeInfo;

	VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
	accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationStructureBuildGeometryInfo.geometryCount = 1;
	accelerationStructureBuildGeometryInfo.pGeometries = &geometry;

	VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
	accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(
		vulkanDevice.logicalDevice,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&accelerationStructureBuildGeometryInfo,
		&primitiveCount,
		&accelerationStructureBuildSizesInfo);

	createAccelerationStructureBuffer(splittedBLAS[cellIdx], accelerationStructureBuildSizesInfo);

	VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
	accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	accelerationStructureCreateInfo.buffer = splittedBLAS[cellIdx].buffer;
	accelerationStructureCreateInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
	accelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	vkCreateAccelerationStructureKHR(device, &accelerationStructureCreateInfo, nullptr, &splittedBLAS[cellIdx].handle);

	ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

	accelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	accelerationStructureBuildGeometryInfo.dstAccelerationStructure = splittedBLAS[cellIdx].handle;
	accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

	// BLAS Size
	blasSize += accelerationStructureBuildSizesInfo.accelerationStructureSize;

	VkCommandBuffer commandBuffer = vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	// Timestamp: BLAS build start
	//vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ASBuildTimeStampQueryPool, 2 * cellIdx);
	vkCmdBuildAccelerationStructuresKHR(
		commandBuffer,
		1,
		&accelerationStructureBuildGeometryInfo,
		&pBuildRangeInfo);
	// Timestamp: BLAS build end
	//vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ASBuildTimeStampQueryPool, 2 * cellIdx + 1);
	vulkanDevice.flushCommandBuffer(commandBuffer, queue);

	VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
	accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	accelerationDeviceAddressInfo.accelerationStructure = splittedBLAS[cellIdx].handle;
	splittedBLAS[cellIdx].deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &accelerationDeviceAddressInfo);

	deleteScratchBuffer(scratchBuffer);
}


void BVHBuilder::createBLASes() {
	splittedBLAS.resize(scene->splittedIndicesBuffers.size());
	for (int cellIdx = 0; cellIdx < scene->splittedIndicesBuffers.size(); cellIdx++) {
		createBLAS(cellIdx);
	}
}

void BVHBuilder::createTLAS() {
	vector<VkAccelerationStructureInstanceKHR> instances;
	for (int i = 0; i < scene->splittedIndicesBuffers.size(); i++) {
		VkAccelerationStructureInstanceKHR instance{};
		instance.transform = tMat;
		instance.instanceCustomIndex = instances.size();
		instance.mask = 0xFF;
		instance.instanceShaderBindingTableRecordOffset = 0;
		instance.flags = VK_FLAGS_NONE;
		//instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		instance.accelerationStructureReference = splittedBLAS[i].deviceAddress;
		instances.push_back(instance);
	}

	vks::Buffer instancesBuffer;
	VK_CHECK_RESULT(vulkanDevice.createBuffer(
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&instancesBuffer,
		instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
		instances.data())
	);

	VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
	instanceDataDeviceAddress.deviceAddress = getBufferDeviceAddress(instancesBuffer.buffer);

	VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
	accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
	accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

	VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
	accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationStructureBuildGeometryInfo.geometryCount = 1;
	accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

	uint32_t primitive_count = instances.size();
	VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
	accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(
		vulkanDevice.logicalDevice,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&accelerationStructureBuildGeometryInfo,
		&primitive_count,
		&accelerationStructureBuildSizesInfo);

	createAccelerationStructureBuffer(tlas, accelerationStructureBuildSizesInfo);

	VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
	accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	accelerationStructureCreateInfo.buffer = tlas.buffer;
	accelerationStructureCreateInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
	accelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	vkCreateAccelerationStructureKHR(vulkanDevice.logicalDevice, &accelerationStructureCreateInfo, nullptr, &tlas.handle);

	ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

	VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{};
	accelerationBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	accelerationBuildGeometryInfo.dstAccelerationStructure = tlas.handle;
	accelerationBuildGeometryInfo.geometryCount = 1;
	accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
	accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

	VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
	accelerationStructureBuildRangeInfo.primitiveCount = primitive_count;
	accelerationStructureBuildRangeInfo.primitiveOffset = 0;
	accelerationStructureBuildRangeInfo.firstVertex = 0;
	accelerationStructureBuildRangeInfo.transformOffset = 0;
	vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

	// TLAS size
	tlasSize = accelerationStructureBuildSizesInfo.accelerationStructureSize;

	VkCommandBuffer commandBuffer = vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

	// Timestamp: TLAS build start
	//vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ASBuildTimeStampQueryPool, ASBuildTimeStamps.size() - 2);
	vkCmdBuildAccelerationStructuresKHR(
		commandBuffer,
		1,
		&accelerationBuildGeometryInfo,
		accelerationBuildStructureRangeInfos.data());
	// Timestamp: TLAS build end
	//vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ASBuildTimeStampQueryPool, ASBuildTimeStamps.size() - 1);
	vulkanDevice.flushCommandBuffer(commandBuffer, queue);

	VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
	accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	accelerationDeviceAddressInfo.accelerationStructure = tlas.handle;
	deleteScratchBuffer(scratchBuffer);
	instancesBuffer.destroy();
}

void BVHBuilder::destroyAS() {
	//vertices
	if (scene->vertices.buffer) {
		vkDestroyBuffer(device, scene->vertices.buffer, nullptr);
		vkFreeMemory(device, scene->vertices.memory, nullptr);
	}
	for (int i = 0; i < scene->splittedVerticesBuffers.size(); i++) {
		scene->splittedVerticesBuffers[i].destroy(device);
	}
	scene->splittedVerticesBuffers.clear();
	//indices
	for (int i = 0; i < scene->splittedIndicesBuffers.size(); i++) {
		scene->splittedIndicesBuffers[i].destroy(device);
	}
	scene->splittedIndicesBuffers.clear();
	//timestamp
	vkDestroyQueryPool(device, ASBuildTimeStampQueryPool, nullptr);
	ASBuildTimeStamps.clear();
	//ASes
	for (int i = 0; i < splittedBLAS.size(); i++) {
		deleteAccelerationStructure(splittedBLAS[i]);
	}
	splittedBLAS.clear();
	deleteAccelerationStructure(tlas);
}

uint32_t cellsPerLongestAxis;

vks::Buffer d_splittedPrimitiveIdsDeviceAddress;
vector<AccelerationStructure> splittedBLAS;
AccelerationStructure tlas;

/*** public functions ***/

void BVHBuilder::setGeometry(vkglTF::Model& scene) {
	this->scene = &scene;
	//this->vertices = &scene.verticesVec;
	//this->indices = &scene.indicesVec;

	for (int i = 0; i < scene.verticesVec.size(); i++) {
		if (scene.verticesVec[i].pos.x > maxPos.x) maxPos.x = scene.verticesVec[i].pos.x;
		if (scene.verticesVec[i].pos.x < minPos.x) minPos.x = scene.verticesVec[i].pos.x;
		if (scene.verticesVec[i].pos.y > maxPos.y) maxPos.y = scene.verticesVec[i].pos.y;
		if (scene.verticesVec[i].pos.y < minPos.y) minPos.y = scene.verticesVec[i].pos.y;
		if (scene.verticesVec[i].pos.z > maxPos.z) maxPos.z = scene.verticesVec[i].pos.z;
		if (scene.verticesVec[i].pos.z < minPos.z) minPos.z = scene.verticesVec[i].pos.z;
	}
	maxPos += SCENE_EPSILON;
	minPos -= SCENE_EPSILON;
	//blasSplitter.setSceneMinMax();
}

void BVHBuilder::createAS(BLASMode splitMode) {
	if (splitMode == Default) {
		saveGeometries();
	}
	else {
		splitGeometry(splitMode);
	}
	initASBuildTimestamp();
	createBLASes();
	createTLAS();
}

void BVHBuilder::rebuild(BLASMode splitMode, float cellWeight, float travelCost) {
	this->cellWeight = cellWeight;
	this->travelCost = travelCost;
	vkDeviceWaitIdle(device);
	destroyAS();
	createAS(splitMode);
}

void BVHBuilder::printASBuildInfo(VkPhysicalDeviceProperties deviceProperties)
{
	vkGetQueryPoolResults(vulkanDevice.logicalDevice, ASBuildTimeStampQueryPool, 0, ASBuildTimeStamps.size(), ASBuildTimeStamps.size() * sizeof(uint64_t), ASBuildTimeStamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

	VkPhysicalDeviceLimits device_limits = deviceProperties.limits;
	float delta_in_ms_BLAS = 0.0f;
	for (int i = 0; i < scene->splittedIndicesBuffers.size(); i++)
		delta_in_ms_BLAS += float(ASBuildTimeStamps[i * 2 + 1] - ASBuildTimeStamps[i * 2]);
	delta_in_ms_BLAS *= device_limits.timestampPeriod / 1000000.0f;
	float delta_in_ms_TLAS = float(ASBuildTimeStamps[ASBuildTimeStamps.size() - 1] - ASBuildTimeStamps[ASBuildTimeStamps.size() - 2]) * device_limits.timestampPeriod / 1000000.0f;
#if defined(_WIN32)
	std::cout << "\n*** AS Build Info BEGIN ***\n";
	std::cout << "BLAS build time: " << delta_in_ms_BLAS << " (ms)\n";
	std::cout << "TLAS build time: " << delta_in_ms_TLAS << " (ms)\n";
	std::cout << "BLAS size: " << blasSize << " (Bytes)\n";
	std::cout << "TLAS size: " << tlasSize << " (Bytes)\n";
	std::cout << "*** AS Build Info END ***\n";
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
	LOGD("\n*** AS Build Info BEGIN ***\n");
	LOGD("BLAS build time: %f (ms))\n", delta_in_ms_BLAS);
	LOGD("TLAS build time: %f (ms))\n", delta_in_ms_TLAS);
	LOGD("BLAS size: %llu (Bytes))\n", blasSize);
	LOGD("TLAS size: %llu (Bytes))\n", tlasSize);
	LOGD("*** AS Build Info END ***\n");
#endif
}