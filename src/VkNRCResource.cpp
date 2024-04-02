//
// Created by adamyuan on 4/2/24.
//

#include "VkNRCResource.hpp"

#include "CuNRCNetwork.hpp"
#include <myvk/CommandBuffer.hpp>

void VkNRCResource::create_fixed() {
	auto create_count_buffer = [this]() {
		return myvk::Buffer::Create(GetDevicePtr(), sizeof(uint32_t),
		                            VMA_ALLOCATION_CREATE_MAPPED_BIT |
		                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	};
	for (auto &train_inputs : m_batch_train_inputs)
		train_inputs = std::make_unique<CuVkBuffer>(
		    myvk::ExportBuffer::Create(GetDevicePtr(), NRCState::GetTrainBatchSize() * kCuNRCInputDims * sizeof(float),
		                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
	for (auto &train_targets : m_batch_train_targets)
		train_targets = std::make_unique<CuVkBuffer>(
		    myvk::ExportBuffer::Create(GetDevicePtr(), NRCState::GetTrainBatchSize() * kCuNRCOutputDims * sizeof(float),
		                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

	for (auto &frame : m_frames) {
		frame.inference_count = create_count_buffer();
		for (auto &train_count : frame.batch_train_counts)
			train_count = create_count_buffer();
		for (auto &train_factors : frame.batch_train_factors)
			train_factors =
			    myvk::Buffer::Create(GetDevicePtr(), NRCState::GetTrainBatchSize() * kCuNRCOutputDims * sizeof(float),
			                         0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}
}

void VkNRCResource::Resize(VkExtent2D extent) {
	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	m_inference_inputs = std::make_unique<CuVkBuffer>(myvk::ExportBuffer::Create(
	    GetDevicePtr(), NRCState::GetInferenceCount(extent) * kCuNRCInputDims * sizeof(float),
	    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
	m_inference_outputs = std::make_unique<CuVkBuffer>(myvk::ExportBuffer::Create(
	    GetDevicePtr(), NRCState::GetInferenceCount(extent) * kCuNRCOutputDims * sizeof(float),
	    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

	for (auto &frame : m_frames) {
		auto bias_factor_r_image = myvk::Image::CreateTexture2D(
		    GetDevicePtr(), extent, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
		frame.bias_factor_r = myvk::ImageView::Create(bias_factor_r_image, VK_IMAGE_VIEW_TYPE_2D);

		command_buffer->CmdPipelineBarrier(
		    VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, {},
		    {bias_factor_r_image->GetMemoryBarrier(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED,
		                                           VK_IMAGE_LAYOUT_GENERAL)});

		auto factor_gb_image = myvk::Image::CreateTexture2D(GetDevicePtr(), extent, 1, VK_FORMAT_R32G32_SFLOAT,
		                                                    VK_IMAGE_USAGE_STORAGE_BIT);
		frame.factor_gb = myvk::ImageView::Create(factor_gb_image, VK_IMAGE_VIEW_TYPE_2D);

		frame.inference_dst =
		    myvk::Buffer::Create(GetDevicePtr(), NRCState::GetInferenceCount(extent) * sizeof(uint32_t), 0,
		                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		command_buffer->CmdPipelineBarrier(
		    VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, {},
		    {factor_gb_image->GetMemoryBarrier(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED,
		                                       VK_IMAGE_LAYOUT_GENERAL)});
	}

	auto accumulate_image = myvk::Image::CreateTexture2D(GetDevicePtr(), extent, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
	                                                     VK_IMAGE_USAGE_STORAGE_BIT);
	m_accumulate_view = myvk::ImageView::Create(accumulate_image, VK_IMAGE_VIEW_TYPE_2D);
	command_buffer->CmdPipelineBarrier(
	    VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, {},
	    {accumulate_image->GetMemoryBarrier(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED,
	                                        VK_IMAGE_LAYOUT_GENERAL)});
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
}
