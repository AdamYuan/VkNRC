//
// Created by adamyuan on 2/22/24.
//

#include "VBufferPass.hpp"

namespace rg {

struct PushConstant_Data {
	glm::mat4 view_proj;
	uint32_t primitive_base;
};

VBufferPass::VBufferPass(myvk_rg::Parent parent, const Args &args)
    : myvk_rg::GraphicsPassBase(parent), m_scene_ptr(args.scene_ptr), m_camera_ptr(args.camera_ptr) {
	AddInput<myvk_rg::Usage::kVertexBuffer>({"vertices"}, args.scene_resources.vertices);
	AddInput<myvk_rg::Usage::kVertexBuffer>({"transforms"}, args.scene_resources.transforms);
	AddInput<myvk_rg::Usage::kIndexBuffer>({"vertex_indices"}, args.scene_resources.vertex_indices);

	// Primitive ID + Instance ID
	auto v_buffer = CreateResource<myvk_rg::ManagedImage>({"v_buffer"}, VK_FORMAT_R32G32_UINT);
	v_buffer->SetLoadOp(VK_ATTACHMENT_LOAD_OP_CLEAR);
	v_buffer->SetClearColorValue({.uint32 = {(uint32_t)-1, (uint32_t)-1}});
	auto depth = CreateResource<myvk_rg::ManagedImage>({"depth"}, VK_FORMAT_D24_UNORM_S8_UINT);
	depth->SetLoadOp(VK_ATTACHMENT_LOAD_OP_CLEAR);
	depth->SetClearDepthStencilValue({.depth = 1.0f});

	AddColorAttachmentInput<myvk_rg::Usage::kColorAttachmentW>(0, {"v_buffer_in"}, v_buffer->Alias());
	AddDepthAttachmentInput<myvk_rg::Usage::kDepthAttachmentRW>({"depth_in"}, depth->Alias());
}

void VBufferPass::CreatePipeline() {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();

	auto pipeline_layout = myvk::PipelineLayout::Create(
	    device, {}, {{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant_Data)}});

	constexpr uint32_t kVertSpv[] = {
#include <shader/vbuffer.vert.u32>
	};
	constexpr uint32_t kFragSpv[] = {
#include <shader/vbuffer.frag.u32>
	};

	std::shared_ptr<myvk::ShaderModule> vert_shader_module, frag_shader_module;
	vert_shader_module = myvk::ShaderModule::Create(device, kVertSpv, sizeof(kVertSpv));
	frag_shader_module = myvk::ShaderModule::Create(device, kFragSpv, sizeof(kFragSpv));

	std::vector<VkPipelineShaderStageCreateInfo> shader_stages = {
	    vert_shader_module->GetPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT),
	    frag_shader_module->GetPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT)};

	myvk::GraphicsPipelineState pipeline_state = {};
	pipeline_state.m_vertex_input_state.Enable(
	    {{.binding = 0, .stride = sizeof(glm::vec3), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX},
	     {.binding = 1, .stride = sizeof(glm::mat3x4), .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE}},
	    {{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
	     {.location = 1, .binding = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 0},
	     {.location = 2, .binding = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 4 * sizeof(float)},
	     {.location = 3, .binding = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 8 * sizeof(float)}});
	pipeline_state.m_input_assembly_state.Enable(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pipeline_state.m_rasterization_state.Initialize(VK_POLYGON_MODE_FILL, VK_FRONT_FACE_COUNTER_CLOCKWISE,
	                                                VK_CULL_MODE_NONE);
	pipeline_state.m_depth_stencil_state.Enable(VK_TRUE, VK_TRUE);
	pipeline_state.m_multisample_state.Enable(VK_SAMPLE_COUNT_1_BIT);
	pipeline_state.m_color_blend_state.Enable(1, VK_FALSE);
	auto extent = GetRenderGraphPtr()->GetCanvasSize();
	pipeline_state.m_viewport_state.Enable(
	    std::vector<VkViewport>{{0, 0, (float)extent.width, (float)extent.height, 0.0f, 1.0f}},
	    std::vector<VkRect2D>{{{0, 0}, extent}});

	m_pipeline =
	    myvk::GraphicsPipeline::Create(pipeline_layout, GetVkRenderPass(), shader_stages, pipeline_state, GetSubpass());
}

void VBufferPass::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	PushConstant_Data pc_data{
	    .view_proj = m_camera_ptr->GetVkViewProjection(GetRenderGraphPtr()->GetCanvasAspectRatio(), 0.01f, 2.0f)};
	std::array<VkBuffer, 2> vertex_buffers = {GetInputBuffer({"vertices"})->GetBufferView().buffer->GetHandle(),
	                                          GetInputBuffer({"transforms"})->GetBufferView().buffer->GetHandle()};
	std::array<VkDeviceSize, 2> vertex_buffer_offsets = {};

	command_buffer->CmdBindPipeline(m_pipeline);
	command_buffer->CmdPushConstants(
	    m_pipeline->GetPipelineLayoutPtr(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	    offsetof(PushConstant_Data, view_proj), sizeof(pc_data.view_proj), &pc_data.view_proj);
	vkCmdBindVertexBuffers(command_buffer->GetHandle(), 0, 2, vertex_buffers.data(), vertex_buffer_offsets.data());
	command_buffer->CmdBindIndexBuffer(GetInputBuffer({"vertex_indices"})->GetBufferView().buffer, 0,
	                                   VK_INDEX_TYPE_UINT32);
	for (uint32_t instance_id : m_scene_ptr->GetInstanceRange()) {
		auto instance = m_scene_ptr->GetInstance(instance_id);
		pc_data.primitive_base = instance.first_index / 3;
		command_buffer->CmdPushConstants(
		    m_pipeline->GetPipelineLayoutPtr(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		    offsetof(PushConstant_Data, primitive_base), sizeof(pc_data.primitive_base), &pc_data.primitive_base);
		command_buffer->CmdDrawIndexed(instance.index_count, 1, instance.first_index, 0, instance_id);
	}
}

} // namespace rg
