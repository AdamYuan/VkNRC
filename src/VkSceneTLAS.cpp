//
// Created by adamyuan on 2/20/24.
//

#include "VkSceneTLAS.hpp"

#include <spdlog/spdlog.h>

VkSceneTLAS::VkSceneTLAS(const myvk::Ptr<VkSceneBLAS> &scene_blas_ptr)
    : m_scene_blas_ptr{scene_blas_ptr}, m_scene_ptr{scene_blas_ptr->GetScenePtr()} {
	create_instance_buffer();
	create_tlas();
}

void VkSceneTLAS::create_instance_buffer() {
	m_instance_buffer = myvk::Buffer::Create(
	    GetDevicePtr(), m_scene_ptr->GetInstanceCount() * sizeof(VkAccelerationStructureInstanceKHR),
	    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
	        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
	m_p_instances = static_cast<VkAccelerationStructureInstanceKHR *>(m_instance_buffer->GetMappedData());
}

void VkSceneTLAS::create_tlas() {
	const auto &device = m_scene_ptr->GetDevicePtr();

	for (uint32_t instance_id : m_scene_ptr->GetInstanceRange())
		m_p_instances[instance_id] = VkAccelerationStructureInstanceKHR{
		    .transform = {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                   0.0f}}, // TODO: Read from dynamic
		    .instanceCustomIndex = instance_id,
		    .mask = 0xFFu,
		    .instanceShaderBindingTableRecordOffset = 0,
		    .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
		    .accelerationStructureReference = m_scene_blas_ptr->GetBLASs()[instance_id]->GetDeviceAddress()};

	VkAccelerationStructureGeometryKHR geometry{
	    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
	    .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
	    .geometry = {.instances = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
	                               .data = {.deviceAddress = m_instance_buffer->GetDeviceAddress()}}}};

	VkAccelerationStructureBuildGeometryInfoKHR build_geom{
	    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
	    .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
	    .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
	             VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
	    .mode =
	        m_tlas ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
	    .srcAccelerationStructure = m_tlas ? m_tlas->GetHandle() : VK_NULL_HANDLE,
	    .geometryCount = 1,
	    .pGeometries = &geometry,
	};

	VkAccelerationStructureBuildSizesInfoKHR build_size{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
	{
		uint32_t instance_count = m_scene_ptr->GetInstanceCount();
		vkGetAccelerationStructureBuildSizesKHR(device->GetHandle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		                                        &build_geom, &instance_count, &build_size);
	}

	auto tlas = myvk::AccelerationStructure::Create(device, build_size.accelerationStructureSize,
	                                                VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
	auto scratch_buffer =
	    myvk::Buffer::Create(device, build_size.buildScratchSize, 0,
	                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	build_geom.scratchData.deviceAddress = scratch_buffer->GetDeviceAddress();
	build_geom.dstAccelerationStructure = tlas->GetHandle();

	VkAccelerationStructureBuildRangeInfoKHR build_range{.primitiveCount = m_scene_ptr->GetInstanceCount()};
	VkAccelerationStructureBuildRangeInfoKHR *p_build_range = &build_range;

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_scene_ptr->GetQueuePtr()));

	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	vkCmdBuildAccelerationStructuresKHR(command_buffer->GetHandle(), 1, &build_geom, &p_build_range);
	command_buffer->End();

	auto fence = myvk::Fence::Create(device);
	command_buffer->Submit(fence);
	fence->Wait();

	m_tlas = tlas;
}
