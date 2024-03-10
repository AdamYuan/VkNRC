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
	float bias_r, bias_g, bias_b, factor_r, factor_g, factor_b;
	PackedNRCInput packed_input;
};
struct OptimizerState {
	uint32_t t;
	float beta1_t, beta2_t, alpha_t, alpha_t_1;
};
struct OptimizerEntry {
	float m, v, weight, ema_weight;
};
} // namespace nrc

VkDeviceSize VkNRCState::GetEvalRecordBufferSize(VkExtent2D extent) {
	return (extent.width * extent.height + kTrainBatchSize * kTrainBatchCount) * sizeof(nrc::NRCEvalRecord);
}
VkDeviceSize VkNRCState::GetBatchTrainRecordBufferSize() { return kTrainBatchSize * sizeof(nrc::NRCTrainRecord); }

void VkNRCState::initialize_weights(std::span<float, kNNWeighCount> weights) {
	// (He) Kaiming Initialization
	std::normal_distribution<float> norm{0, std::sqrt(2.0f / float(kNNWidth))};
	for (auto &w : weights)
		w = norm(m_rng);
}

void VkNRCState::create_mlp_buffer() {
	std::array<float, GetWeightCount()> fp32_weights{};
	initialize_weights(fp32_weights);
	std::array<half_float::half, GetWeightCount()> fp16_weights{};
	nrc::OptimizerState optimizer_state = {.t = 0, .beta1_t = 1.0f, .beta2_t = 1.0f, .alpha_t = 1.0f};
	std::array<nrc::OptimizerEntry, GetWeightCount()> optimizer_entries{};
	for (uint32_t i = 0; i < GetWeightCount(); ++i) {
		fp16_weights[i] = fp32_weights[i];
		optimizer_entries[i] = {
		    .weight = fp32_weights[i],
		    .ema_weight = fp32_weights[i],
		};
	}

	m_weights = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(uint16_t), 0,
	                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_use_weights = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(uint16_t), 0,
	                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	m_optimizer_state = myvk::Buffer::Create(GetDevicePtr(), sizeof(nrc::OptimizerState), 0,
	                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	m_optimizer_entries = myvk::Buffer::Create(GetDevicePtr(), GetWeightCount() * sizeof(nrc::OptimizerEntry), 0,
	                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	auto weights_staging = myvk::Buffer::CreateStaging(GetDevicePtr(), fp16_weights.begin(), fp16_weights.end());
	auto optimizer_state_staging = myvk::Buffer::CreateStaging(GetDevicePtr(), optimizer_state);
	auto optimizer_entries_staging =
	    myvk::Buffer::CreateStaging(GetDevicePtr(), optimizer_entries.begin(), optimizer_entries.end());

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	command_buffer->CmdCopy(weights_staging, m_weights, {VkBufferCopy{.size = weights_staging->GetSize()}});
	command_buffer->CmdCopy(weights_staging, m_use_weights, {VkBufferCopy{.size = weights_staging->GetSize()}});
	command_buffer->CmdCopy(optimizer_state_staging, m_optimizer_state,
	                        {VkBufferCopy{.size = optimizer_state_staging->GetSize()}});
	command_buffer->CmdCopy(optimizer_entries_staging, m_optimizer_entries,
	                        {VkBufferCopy{.size = optimizer_entries_staging->GetSize()}});
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
}

void VkNRCState::create_accumulate_image(VkExtent2D extent) {
	auto image = myvk::Image::CreateTexture2D(GetDevicePtr(), extent, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
	                                          VK_IMAGE_USAGE_STORAGE_BIT);
	m_accumulate_view = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);

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
