//
// Created by adamyuan on 2/19/24.
//

#include "VkAccelScene.hpp"
#include <myvk/CommandBuffer.hpp>
#include <myvk/QueryPool.hpp>

constexpr VkDeviceSize kASBuildBatchLimit = 256 * 1024 * 1024;

VkAccelScene::VkAccelScene(const myvk::Ptr<VkScene> &scene_ptr) : m_scene_ptr(scene_ptr) {
	create_blas_s();
	create_tlas();
}

void VkAccelScene::create_blas_s() {
	m_blas_s.resize(m_scene_ptr->GetInstanceCount());

	const auto &device = m_scene_ptr->GetDevicePtr();

	VkAccelerationStructureGeometryKHR geometry = m_scene_ptr->GetBLASGeometry();
	std::vector<VkAccelerationStructureBuildGeometryInfoKHR> blas_build_geoms(m_scene_ptr->GetInstanceCount());
	std::vector<VkAccelerationStructureBuildRangeInfoKHR> blas_build_ranges(m_scene_ptr->GetInstanceCount());
	std::vector<VkAccelerationStructureBuildSizesInfoKHR> blas_build_sizes(m_scene_ptr->GetInstanceCount());

	VkDeviceSize max_scratch_size = 0;
	for (uint32_t instance_id : m_scene_ptr->GetInstanceRange()) {
		auto &build_geom = blas_build_geoms[instance_id];
		auto &build_range = blas_build_ranges[instance_id];
		auto &build_size = blas_build_sizes[instance_id];
		build_geom = {
		    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		    .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
		    .flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
		    .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		    .geometryCount = 1,
		    .pGeometries = &geometry,
		};
		build_range = m_scene_ptr->GetInstanceBLASBuildRange(instance_id);
		build_size = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
		vkGetAccelerationStructureBuildSizesKHR(device->GetHandle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		                                        &build_geom, &build_range.primitiveCount, &build_size);
		max_scratch_size = std::max(max_scratch_size, build_size.buildScratchSize);
	}

	std::vector<VkAccelerationStructureBuildRangeInfoKHR *> blas_pp_build_ranges(m_scene_ptr->GetInstanceCount());
	for (uint32_t instance_id : m_scene_ptr->GetInstanceRange())
		blas_pp_build_ranges[instance_id] = blas_build_ranges.data() + instance_id;

	// Objects for Construction
	auto scratch_buffer = myvk::Buffer::Create(
	    device, max_scratch_size, 0, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	auto query_pool = myvk::QueryPool::Create(device, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
	                                          m_scene_ptr->GetInstanceCount());
	auto command_pool = myvk::CommandPool::Create(m_scene_ptr->GetQueuePtr());

	std::vector<VkDeviceSize> compact_sizes(m_scene_ptr->GetInstanceCount());

	const auto create_blas_batch = [&](std::ranges::input_range auto instance_range) {
		auto command_buffer = myvk::CommandBuffer::Create(command_pool);
		command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

		vkResetQueryPool(device->GetHandle(), query_pool->GetHandle(), instance_range.front(), instance_range.size());

		std::vector<VkAccelerationStructureKHR> query_as_handles;
		query_as_handles.reserve(instance_range.size());

		for (uint32_t instance_id : instance_range) {
			auto &build_geom = blas_build_geoms[instance_id];
			auto &build_range = blas_build_ranges[instance_id];
			auto &build_size = blas_build_sizes[instance_id];
			// Actual allocation of buffer and acceleration structure.
			auto accel_struct = myvk::AccelerationStructure::Create(device, build_size.accelerationStructureSize,
			                                                        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
			m_blas_s[instance_id] = accel_struct;

			build_geom.dstAccelerationStructure = accel_struct->GetHandle();
			build_geom.scratchData.deviceAddress = scratch_buffer->GetDeviceAddress();

			VkAccelerationStructureBuildRangeInfoKHR *p_build_range = &build_range;

			// Building the bottom-level-acceleration-structure
			vkCmdBuildAccelerationStructuresKHR(command_buffer->GetHandle(), 1, &build_geom, &p_build_range);
			command_buffer->CmdPipelineBarrier(
			    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
			    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, {},
			    {scratch_buffer->GetMemoryBarrier(VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
			                                      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
			                                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR)},
			    {});

			query_as_handles.push_back(accel_struct->GetHandle());
		}
		vkCmdWriteAccelerationStructuresPropertiesKHR(
		    command_buffer->GetHandle(), query_as_handles.size(), query_as_handles.data(),
		    VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, query_pool->GetHandle(), instance_range.front());

		command_buffer->End();

		auto fence = myvk::Fence::Create(device);
		command_buffer->Submit(fence);
		fence->Wait();

		// Fetch sizes for compaction
		vkGetQueryPoolResults(device->GetHandle(), query_pool->GetHandle(), instance_range.front(),
		                      instance_range.size(), instance_range.size() * sizeof(VkDeviceSize),
		                      compact_sizes.data() + instance_range.front(), sizeof(VkDeviceSize),
		                      VK_QUERY_RESULT_WAIT_BIT);
	};

	const auto compact_blas_batch = [&](std::ranges::input_range auto instance_range) {
		std::vector<myvk::Ptr<myvk::AccelerationStructure>> destroy_as_s;
		destroy_as_s.reserve(instance_range.size());

		auto command_buffer = myvk::CommandBuffer::Create(command_pool);
		command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

		for (uint32_t instance_id : instance_range) {
			auto accel_struct = m_blas_s[instance_id];
			auto compact_accel_struct = myvk::AccelerationStructure::Create(
			    device, compact_sizes[instance_id], VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);

			VkCopyAccelerationStructureInfoKHR copy_info{.sType =
			                                                 VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
			                                             .src = accel_struct->GetHandle(),
			                                             .dst = compact_accel_struct->GetHandle(),
			                                             .mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR};
			vkCmdCopyAccelerationStructureKHR(command_buffer->GetHandle(), &copy_info);

			destroy_as_s.push_back(accel_struct);
			m_blas_s[instance_id] = compact_accel_struct;
		}

		command_buffer->End();

		auto fence = myvk::Fence::Create(device);
		command_buffer->Submit(fence);
		fence->Wait();
	};

	for (VkDeviceSize batch_memory_size = 0, batch_first_instance = 0, batch_instance_count = 0;
	     uint32_t instance_id : m_scene_ptr->GetInstanceRange()) {
		batch_memory_size += blas_build_sizes[instance_id].accelerationStructureSize;
		++batch_instance_count;

		if (batch_memory_size >= kASBuildBatchLimit || instance_id == m_scene_ptr->GetInstanceRange().back()) {
			create_blas_batch(std::views::iota(batch_first_instance, batch_first_instance + batch_instance_count));
			compact_blas_batch(std::views::iota(batch_first_instance, batch_first_instance + batch_instance_count));

			batch_memory_size = 0;
			batch_first_instance += batch_instance_count;
			batch_instance_count = 0;
		}
	}
}
void VkAccelScene::create_tlas() {}
