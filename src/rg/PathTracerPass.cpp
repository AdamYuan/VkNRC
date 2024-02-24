//
// Created by adamyuan on 2/22/24.
//

#include "PathTracerPass.hpp"

#include <spdlog/spdlog.h>

namespace rg {

namespace path_tracer_pass {
struct PushConstant_Data {
	alignas(sizeof(glm::vec4)) glm::vec3 position;
	alignas(sizeof(glm::vec4)) glm::vec3 look;
	alignas(sizeof(glm::vec4)) glm::vec3 side;
	alignas(sizeof(glm::vec4)) glm::vec3 up;
	uint32_t samples;
};
} // namespace path_tracer_pass
using path_tracer_pass::PushConstant_Data;

PathTracerPass::PathTracerPass(myvk_rg::Parent parent, const PathTracerPass::Args &args)
    : myvk_rg::GraphicsPassBase(parent), m_camera_ptr(args.camera_ptr), m_scene_ptr(args.scene_ptr),
      m_nrc_state_ptr(args.nrc_state_ptr) {
	// Scene
	AddDescriptorInput<myvk_rg::Usage::kAccelerationStructureR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {0}, {"tlas"}, args.scene_resources.tlas);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {1}, {"vertices"}, args.scene_resources.vertices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {2}, {"vertex_indices"}, args.scene_resources.vertex_indices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {3}, {"texcoords"}, args.scene_resources.texcoords);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {4}, {"texcoord_indices"}, args.scene_resources.texcoord_indices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {5}, {"materials"}, args.scene_resources.materials);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {6}, {"material_ids"}, args.scene_resources.material_ids);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {7}, {"transforms"}, args.scene_resources.transforms);
	for (uint32_t texture_id = 0; const auto &texture : args.scene_resources.textures) {
		AddDescriptorInput<myvk_rg::Usage::kSampledImage, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
		    {8, texture_id}, {"textures", texture_id}, texture, args.scene_resources.texture_sampler);
		++texture_id;
	}
	// V-Buffer
	AddInputAttachmentInput(0, {9}, {"v_buffer"}, args.vbuffer_image);
	// NRC
	AddDescriptorInput<myvk_rg::Usage::kUniformBuffer, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {10}, {"sobol"}, args.nrc_resources.sobol);
	AddDescriptorInput<myvk_rg::Usage::kSampledImage, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {11}, {"noise"}, args.nrc_resources.noise, args.nrc_resources.noise_sampler);
	AddDescriptorInput<myvk_rg::Usage::kStorageImageRW, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT>(
	    {12}, {"result"}, args.nrc_resources.result);

	AddColorAttachmentInput<myvk_rg::Usage::kColorAttachmentW>(0, {"out_in"}, args.out_image);
}

void PathTracerPass::CreatePipeline() {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();

	auto pipeline_layout = myvk::PipelineLayout::Create(
	    device, {GetVkDescriptorSetLayout()},
	    {{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant_Data)}});

	constexpr uint32_t kVertSpv[] = {
#include <shader/path_tracer.vert.u32>
	};
	constexpr uint32_t kFragSpv[] = {
#include <shader/path_tracer.frag.u32>
	};

	std::shared_ptr<myvk::ShaderModule> vert_shader_module, frag_shader_module;
	vert_shader_module = myvk::ShaderModule::Create(device, kVertSpv, sizeof(kVertSpv));
	frag_shader_module = myvk::ShaderModule::Create(device, kFragSpv, sizeof(kFragSpv));
	frag_shader_module->AddSpecialization(0, (uint32_t)m_scene_ptr->GetTextures().size());

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

void PathTracerPass::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	PushConstant_Data pc_data{};
	{
		auto look_side_up = m_camera_ptr->GetVkLookSideUp(GetRenderGraphPtr()->GetCanvasAspectRatio());
		pc_data = {.position = m_camera_ptr->position,
		           .look = look_side_up.look,
		           .side = look_side_up.side,
		           .up = look_side_up.up,
		           .samples = m_nrc_state_ptr->GetSampleCount()};
	}
	command_buffer->CmdBindPipeline(m_pipeline);
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, m_pipeline);
	command_buffer->CmdPushConstants(m_pipeline->GetPipelineLayoutPtr(),
	                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc_data),
	                                 &pc_data);
	command_buffer->CmdDraw(3, 1, 0, 0);
}

} // namespace rg
