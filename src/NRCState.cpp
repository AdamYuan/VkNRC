//
// Created by adamyuan on 2/23/24.
//

#include "NRCState.hpp"

void NRCState::ResetMLPBuffers(VkExtent2D extent) {
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

void NRCState::Resize(VkExtent2D extent) {
}
