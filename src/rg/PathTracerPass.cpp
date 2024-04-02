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
	uint32_t seed;
	alignas(sizeof(VkExtent2D)) VkExtent2D extent;
	uint32_t left_method, right_method;
	float train_probability;
};
} // namespace path_tracer_pass
using path_tracer_pass::PushConstant_Data;

PathTracerPass::PathTracerPass(myvk_rg::Parent parent, const PathTracerPass::Args &args)
    : myvk_rg::ComputePassBase(parent), m_camera_ptr(args.camera_ptr), m_scene_ptr(args.scene_ptr),
      m_nrc_state_ptr(args.nrc_state_ptr) {
	// Scene
	AddDescriptorInput<myvk_rg::Usage::kAccelerationStructureR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {0}, {"tlas"}, args.scene_resources.tlas);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {1}, {"vertices"}, args.scene_resources.vertices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {2}, {"vertex_indices"}, args.scene_resources.vertex_indices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {3}, {"texcoords"}, args.scene_resources.texcoords);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {4}, {"texcoord_indices"}, args.scene_resources.texcoord_indices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {5}, {"materials"}, args.scene_resources.materials);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {6}, {"material_ids"}, args.scene_resources.material_ids);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {7}, {"transforms"}, args.scene_resources.transforms);
	for (uint32_t texture_id = 0; const auto &texture : args.scene_resources.textures) {
		AddDescriptorInput<myvk_rg::Usage::kSampledImage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {8, texture_id}, {"textures", texture_id}, texture, args.scene_resources.texture_sampler);
		++texture_id;
	}
	// V-Buffer
	AddDescriptorInput<myvk_rg::Usage::kSampledImage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {9}, {"v_buffer"}, args.vbuffer_image,
	    myvk::Sampler::Create(GetRenderGraphPtr()->GetDevicePtr(), VK_FILTER_NEAREST,
	                          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST, 0.0f));
	// NRC
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({12}, {"eval_count"},
	                                                                                             args.eval_count);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({13}, {"eval_inputs"},
	                                                                                            args.eval_inputs);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({14}, {"eval_dests"},
	                                                                                            args.eval_dests);
	for (uint32_t b = 0; b < NRCState::GetTrainBatchCount(); ++b) {
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {15, b}, {"batch_train_count", b}, args.batch_train_count[b]);
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {16, b}, {"batch_train_inputs", b}, args.batch_train_inputs[b]);
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {17, b}, {"batch_train_biases", b}, args.batch_train_biases[b]);
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {18, b}, {"batch_train_factors", b}, args.batch_train_factors[b]);
	}
	AddDescriptorInput<myvk_rg::Usage::kStorageImageW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({19}, {"bias_factor_r"},
	                                                                                           args.bias_factor_r);
	AddDescriptorInput<myvk_rg::Usage::kStorageImageW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({20}, {"factor_gb"},
	                                                                                           args.factor_gb);
}

myvk::Ptr<myvk::ComputePipeline> PathTracerPass::CreatePipeline() const {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();
	auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()},
	                                                    {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant_Data)}});
	constexpr uint32_t kCompSpv[] = {
#include <shader/path_tracer.comp.u32>
	};
	auto shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	shader_module->AddSpecialization(0, (uint32_t)m_scene_ptr->GetTextures().size());
	return myvk::ComputePipeline::Create(pipeline_layout, shader_module);
}

void PathTracerPass::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	auto extent = GetRenderGraphPtr()->GetCanvasSize();
	PushConstant_Data pc_data{};
	{
		auto look_side_up = m_camera_ptr->GetVkLookSideUp(GetRenderGraphPtr()->GetCanvasAspectRatio());
		pc_data = {
		    .position = m_camera_ptr->position,
		    .look = look_side_up.look,
		    .side = look_side_up.side,
		    .up = look_side_up.up,
		    .seed = m_nrc_state_ptr->GetSeed(),
		    .extent = extent,
		    .left_method = static_cast<uint32_t>(m_nrc_state_ptr->GetLeftMethod()),
		    .right_method = static_cast<uint32_t>(m_nrc_state_ptr->GetRightMethod()),
		    .train_probability = m_nrc_state_ptr->GetTrainProbability(),
		};
	}
	command_buffer->CmdBindPipeline(GetVkPipeline());
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, GetVkPipeline());
	command_buffer->CmdPushConstants(GetVkPipeline()->GetPipelineLayoutPtr(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                                 sizeof(pc_data), &pc_data);
	command_buffer->CmdDispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
}

} // namespace rg
