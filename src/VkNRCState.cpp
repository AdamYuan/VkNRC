//
// Created by adamyuan on 2/23/24.
//

#include "VkNRCState.hpp"

#include "BlueNoise.hpp"

#include <algorithm>
#include <myvk/CommandBuffer.hpp>

void VkNRCState::create_weight_buffer() {

}

void VkNRCState::create_result_image() {
	auto image = myvk::Image::CreateTexture2D(GetDevicePtr(), m_extent, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
	                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
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
