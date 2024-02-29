//
// Created by adamyuan on 2/29/24.
//

#include "ScreenPass.hpp"

namespace rg {

namespace screen_pass {
struct PushConstant_Data {
	uint32_t samples;
};
} // namespace screen_pass
using screen_pass::PushConstant_Data;

ScreenPass::ScreenPass(myvk_rg::Parent parent, const Args &args)
    : myvk_rg::GraphicsPassBase(parent), m_nrc_state_ptr(args.nrc_state_ptr) {
	AddColorAttachmentInput<myvk_rg::Usage::kColorAttachmentW>(0, {"screen"}, args.screen_image);
	AddInputAttachmentInput(0, {0}, {"color"}, args.color_image);
	AddDescriptorInput<myvk_rg::Usage::kStorageImageRW, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>({1}, {"accumulate"},
	                                                                                             args.accumulate_image);
}

void ScreenPass::CreatePipeline() {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();

	auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()},
	                                                    {{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant_Data)}});

	constexpr uint32_t kVertSpv[] = {
#include <shader/screen.vert.u32>
	};
	constexpr uint32_t kFragSpv[] = {
#include <shader/screen.frag.u32>
	};

	std::shared_ptr<myvk::ShaderModule> vert_shader_module, frag_shader_module;
	vert_shader_module = myvk::ShaderModule::Create(device, kVertSpv, sizeof(kVertSpv));
	frag_shader_module = myvk::ShaderModule::Create(device, kFragSpv, sizeof(kFragSpv));

	std::vector<VkPipelineShaderStageCreateInfo> shader_stages = {
	    vert_shader_module->GetPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT),
	    frag_shader_module->GetPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT)};

	myvk::GraphicsPipelineState pipeline_state = {};
	auto extent = GetRenderGraphPtr()->GetCanvasSize();
	pipeline_state.m_viewport_state.Enable(std::vector<VkViewport>{{0, 0, (float)extent.width, (float)extent.height}},
	                                       std::vector<VkRect2D>{{{0, 0}, extent}});
	pipeline_state.m_vertex_input_state.Enable();
	pipeline_state.m_input_assembly_state.Enable(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pipeline_state.m_rasterization_state.Initialize(VK_POLYGON_MODE_FILL, VK_FRONT_FACE_COUNTER_CLOCKWISE,
	                                                VK_CULL_MODE_FRONT_BIT);
	pipeline_state.m_multisample_state.Enable(VK_SAMPLE_COUNT_1_BIT);
	pipeline_state.m_color_blend_state.Enable(1, VK_FALSE);

	m_pipeline =
	    myvk::GraphicsPipeline::Create(pipeline_layout, GetVkRenderPass(), shader_stages, pipeline_state, GetSubpass());
}

void ScreenPass::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	PushConstant_Data pc_data{.samples = m_nrc_state_ptr->GetSampleCount()};
	command_buffer->CmdBindPipeline(m_pipeline);
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, m_pipeline);
	command_buffer->CmdPushConstants(m_pipeline->GetPipelineLayoutPtr(), VK_SHADER_STAGE_FRAGMENT_BIT, 0,
	                                 sizeof(pc_data), &pc_data);
	command_buffer->CmdDraw(3, 1, 0, 0);
}

} // namespace rg
