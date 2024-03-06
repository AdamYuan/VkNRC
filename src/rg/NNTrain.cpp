//
// Created by adamyuan on 3/5/24.
//

#include "NNTrain.hpp"

#include "NNGradientShader.hpp"

namespace rg {

NNTrain::NNGradient::NNGradient(myvk_rg::Parent parent, const myvk_rg::Buffer &cmd, const myvk_rg::Buffer &gradients,
                                const NNTrain::Args &args)
    : myvk_rg::ComputePassBase(parent), m_scene_ptr(args.scene_ptr) {
	AddInput<myvk_rg::Usage::kDrawIndirectBuffer>({"cmd"}, cmd);
	// Scene
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {0}, {"vertices"}, args.scene_resources.vertices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {1}, {"vertex_indices"}, args.scene_resources.vertex_indices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {2}, {"texcoords"}, args.scene_resources.texcoords);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {3}, {"texcoord_indices"}, args.scene_resources.texcoord_indices);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {4}, {"materials"}, args.scene_resources.materials);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {5}, {"material_ids"}, args.scene_resources.material_ids);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {6}, {"transforms"}, args.scene_resources.transforms);
	for (uint32_t texture_id = 0; const auto &texture : args.scene_resources.textures) {
		AddDescriptorInput<myvk_rg::Usage::kSampledImage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {7, texture_id}, {"textures", texture_id}, texture, args.scene_resources.texture_sampler);
		++texture_id;
	}
	// NRC
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {8}, {"batch_train_records"}, args.batch_train_records);
	AddDescriptorInput<myvk_rg::Usage::kUniformBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {9}, {"batch_train_count"}, args.batch_train_count);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({10}, {"weights"},
	                                                                                            args.weights);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({11}, {"gradients"},
	                                                                                             gradients);
}
void NNTrain::NNGradient::CreatePipeline() {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();
	auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()}, {});
	auto [shader_module, required_subgroup_info] = NNGradientShader::Create(device);
	shader_module->AddSpecialization(0, (uint32_t)m_scene_ptr->GetTextures().size());
	VkPipelineShaderStageCreateInfo shader_stage =
	    shader_module->GetPipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT);
	shader_stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
	shader_stage.pNext = &required_subgroup_info;
	VkComputePipelineCreateInfo create_info{
	    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
	    .stage = shader_stage,
	};
	m_pipeline = myvk::ComputePipeline::Create(pipeline_layout, create_info);
}
void NNTrain::NNGradient::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	command_buffer->CmdBindPipeline(m_pipeline);
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, m_pipeline);
	command_buffer->CmdDispatchIndirect(GetInputBuffer({"cmd"})->GetBufferView().buffer);
}

NNTrain::NNAdam::NNAdam(myvk_rg::Parent parent, const myvk_rg::Buffer &gradients, const NNTrain::Args &args)
    : myvk_rg::ComputePassBase(parent), m_nrc_state_ptr(args.nrc_state_ptr), m_batch_index{args.batch_index} {
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"weights"},
	                                                                                             args.weights);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({1}, {"gradients"},
	                                                                                            gradients);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({2}, {"adam_mv"},
	                                                                                             args.adam_mv);
	AddDescriptorInput<myvk_rg::Usage::kUniformBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {3}, {"batch_train_count"}, args.batch_train_count);
}
namespace nn_adam {
struct PushConstant_Data {
	glm::vec2 adam_beta_t;
};
} // namespace nn_adam
void NNTrain::NNAdam::CreatePipeline() {
	using nn_adam::PushConstant_Data;

	auto &device = GetRenderGraphPtr()->GetDevicePtr();
	auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()},
	                                                    {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstant_Data)}});
	constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_adam.comp.u32>
	};
	auto shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	shader_module->AddSpecialization(0, VkNRCState::GetAdamBeta().x);
	shader_module->AddSpecialization(1, VkNRCState::GetAdamBeta().y);
	m_pipeline = myvk::ComputePipeline::Create(pipeline_layout, shader_module);
}
void NNTrain::NNAdam::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	using nn_adam::PushConstant_Data;
	PushConstant_Data pc_data{.adam_beta_t = m_nrc_state_ptr->GetAdamBetaT(m_batch_index)};

	command_buffer->CmdBindPipeline(m_pipeline);
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, m_pipeline);
	command_buffer->CmdPushConstants(m_pipeline->GetPipelineLayoutPtr(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                                 sizeof(pc_data), &pc_data);
	command_buffer->CmdDispatch(VkNRCState::GetWeightCount() / 64, 1, 1);
}

} // namespace rg