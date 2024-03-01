//
// Created by adamyuan on 2/23/24.
//

#include "VkNRCState.hpp"

#include <myvk/CommandBuffer.hpp>

namespace nrc {
inline static constexpr uint32_t kNNHiddenLayers = 5, kNNWidth = 64, kPaddedNNOutWidth = 16, kTrainBatchSize = 16384,
                                 kTrainBatchCount = 4;
struct PackedNRCInput {
	uint32_t primitive_id, flip_bit_instance_id;
	uint32_t barycentric_2x16U;
	uint32_t scattered_dir_2x16U;
};
struct NRCEvalRecord {
	uint32_t pixel_x_y;
	PackedNRCInput packed_input;
};
struct NRCTrainRecord {
	uint32_t radiance_RG, radiance_B;
	PackedNRCInput packed_input;
};
} // namespace nrc

VkDeviceSize VkNRCState::GetEvalRecordBufferSize(VkExtent2D extent) {
	return VkDeviceSize{extent.width} * VkDeviceSize{extent.height} * sizeof(nrc::NRCEvalRecord);
}
VkDeviceSize VkNRCState::GetTrainBatchRecordBufferSize() {
	return VkDeviceSize{nrc::kTrainBatchSize * nrc::kTrainBatchCount} * sizeof(nrc::NRCTrainRecord);
}
uint32_t VkNRCState::GetTrainBatchCount() { return nrc::kTrainBatchCount; }

void VkNRCState::create_weight_buffer() {
	constexpr uint32_t kWeightCount =
	    nrc::kNNWidth * nrc::kNNWidth * nrc::kNNHiddenLayers + nrc::kNNWidth * nrc::kPaddedNNOutWidth;
	m_weights =
	    myvk::Buffer::Create(GetDevicePtr(), kWeightCount * sizeof(uint16_t), 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void VkNRCState::create_result_image() {
	auto image = myvk::Image::CreateTexture2D(GetDevicePtr(), m_extent, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
	                                          VK_IMAGE_USAGE_STORAGE_BIT);
	m_result_view = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	command_buffer->CmdPipelineBarrier(
	    VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, {},
	    {image->GetMemoryBarrier(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL)});
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
}
