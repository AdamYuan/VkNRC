//
// Created by adamyuan on 2/23/24.
//

#include "VkNRCState.hpp"

#include <myvk/CommandBuffer.hpp>

namespace nrc {

struct PackedNRCInput {
	uint32_t primitive_id, flip_bit_instance_id;
	uint32_t barycentric_2x16U;
	uint32_t scattered_dir_2x16U;
};
struct NRCEvalRecord {
	uint32_t screen_x16_y16, train_b4_l14_r14;
	PackedNRCInput packed_input;
};
struct NRCTrainRecord {
	alignas(glm::vec4) glm::vec4 base_extra_r;
	alignas(glm::vec2) glm::vec2 extra_gb;
	PackedNRCInput packed_input;
};
} // namespace nrc

VkDeviceSize VkNRCState::GetEvalRecordBufferSize(VkExtent2D extent) {
	return VkDeviceSize{extent.width} * VkDeviceSize{extent.height} * sizeof(nrc::NRCEvalRecord);
}
VkDeviceSize VkNRCState::GetBatchTrainRecordBufferSize() {
	return VkDeviceSize{kTrainBatchSize} * sizeof(nrc::NRCTrainRecord);
}

void VkNRCState::initialize_weights(std::span<half_float::half, kNNWeighCount> weights) {
	// (He) Kaiming Initialization
	std::normal_distribution<float> norm{0, std::sqrt(2.0f / float(kNNWidth))};
	for (auto &w : weights)
		w = norm(m_rng);
}

void VkNRCState::create_weight_buffer() {
	m_weights = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(uint16_t), 0,
	                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	std::array<half_float::half, GetWeightCount()> initial_weights;
	initialize_weights(initial_weights);

	auto weights_staging = myvk::Buffer::CreateStaging(GetDevicePtr(), initial_weights.begin(), initial_weights.end());

	// Zero Initialize
	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	command_buffer->CmdCopy(weights_staging, m_weights, {VkBufferCopy{.size = GetWeightCount() * sizeof(uint16_t)}});
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
}
void VkNRCState::create_adam_buffer() {
	m_adam_mv = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(float) * 2, 0,
	                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	// Zero Initialize
	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	vkCmdFillBuffer(command_buffer->GetHandle(), m_adam_mv->GetHandle(), 0, m_adam_mv->GetSize(), 0);
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
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
