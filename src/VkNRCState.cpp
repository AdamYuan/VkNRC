//
// Created by adamyuan on 2/23/24.
//

#include "VkNRCState.hpp"

#include "BlueNoise.hpp"

#include <algorithm>
#include <myvk/CommandBuffer.hpp>

void VkNRCState::create_result_image(VkExtent2D extent) {
	auto image = myvk::Image::CreateTexture2D(GetDevicePtr(), extent, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
	                                          VK_IMAGE_USAGE_STORAGE_BIT);
	m_result_view = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	command_buffer->CmdPipelineBarrier(
	    VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT, {}, {},
	    {image->GetMemoryBarrier(VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL)});
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
}

void VkNRCState::create_noise_image() {
	constexpr uint32_t kSize = 256;
	const auto &kData = BlueNoise::Get256RG();
	auto image = myvk::Image::CreateTexture2D(GetDevicePtr(), {kSize, kSize}, 1, VK_FORMAT_R8G8_UNORM,
	                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
	m_noise_view = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);

	auto staging_buffer = myvk::Buffer::CreateStaging(GetDevicePtr(), kData.begin(), kData.end());

	VkBufferImageCopy region = {.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
	                            .imageExtent = {kSize, kSize, 1}};

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	command_buffer->CmdPipelineBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT, {}, {},
	                                   image->GetDstMemoryBarriers({region}, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                                                               VK_IMAGE_LAYOUT_UNDEFINED,
	                                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
	command_buffer->CmdCopy(staging_buffer, image, {region});
	command_buffer->CmdPipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_NONE, {}, {},
	                                   image->GetDstMemoryBarriers({region}, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
	                                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	command_buffer->End();

	auto fence = myvk::Fence::Create(GetDevicePtr());
	command_buffer->Submit(fence);
	fence->Wait();
}

myvk::Ptr<myvk::Buffer> VkNRCState::MakeSobolBuffer(VkBufferUsageFlags usages) const {
	return myvk::Buffer::Create(
	    GetDevicePtr(), Sobol::kDimension * sizeof(float),
	    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, usages);
}
void VkNRCState::UpdateSobolBuffer(const myvk::Ptr<myvk::Buffer> &sobol_buffer) const {
	std::ranges::copy(m_sobol.GetFloat(), static_cast<float *>(sobol_buffer->GetMappedData()));
}
