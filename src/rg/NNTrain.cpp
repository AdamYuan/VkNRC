//
// Created by adamyuan on 3/5/24.
//

#include "NNTrain.hpp"

#include "NNGradientShader.hpp"

namespace rg {

NNTrain::NNPreparePass::NNPreparePass(myvk_rg::Parent parent, const Args &args) : myvk_rg::ComputePassBase(parent) {
	auto indirect_cmd = CreateResource<myvk_rg::ManagedBuffer>({"indirect_cmd"}, sizeof(VkDispatchIndirectCommand));
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"count"},
	                                                                                             args.count);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({1}, {"indirect_cmd"},
	                                                                                            indirect_cmd->Alias());
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {2}, {"optimizer_state"}, args.optimizer_state);
}
myvk::Ptr<myvk::ComputePipeline> NNTrain::NNPreparePass::CreatePipeline() const {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();
	auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()}, {});
	constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_train_prepare.comp.u32>
	};
	auto shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	return myvk::ComputePipeline::Create(pipeline_layout, shader_module);
}
void NNTrain::NNPreparePass::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	command_buffer->CmdBindPipeline(GetVkPipeline());
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, GetVkPipeline());
	command_buffer->CmdDispatch(1, 1, 1);
}

NNTrain::NNGradient::NNGradient(myvk_rg::Parent parent, const Args &args)
    : myvk_rg::ComputePassBase(parent), m_scene_ptr(args.scene_ptr) {
	AddInput<myvk_rg::Usage::kDrawIndirectBuffer>({"cmd"}, args.cmd);
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
	    {8}, {"batch_train_records"}, args.records);
	AddDescriptorInput<myvk_rg::Usage::kUniformBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {9}, {"batch_train_count"}, args.count);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({10}, {"weights"},
	                                                                                            args.weights);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({11}, {"gradients"},
	                                                                                             args.gradients);
}
myvk::Ptr<myvk::ComputePipeline> NNTrain::NNGradient::CreatePipeline() const {
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
	return myvk::ComputePipeline::Create(pipeline_layout, create_info);
}
void NNTrain::NNGradient::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	command_buffer->CmdBindPipeline(GetVkPipeline());
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, GetVkPipeline());
	command_buffer->CmdDispatchIndirect(GetInputBuffer({"cmd"})->GetBufferView().buffer);
}

NNTrain::NNOptimizer::NNOptimizer(myvk_rg::Parent parent, const Args &args)
    : myvk_rg::ComputePassBase(parent), m_write_use{args.opt_use_weights}, m_nrc_state_ptr{args.nrc_state_ptr} {
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"weights"},
	                                                                                            args.weights);
	if (args.opt_use_weights) {
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {1}, {"use_weights"}, *args.opt_use_weights);
	}
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({2}, {"gradients"},
	                                                                                            args.gradients);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {3}, {"optimizer_entries"}, args.optimizer_entries);
	AddDescriptorInput<myvk_rg::Usage::kUniformBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
	    {4}, {"batch_train_count"}, args.count);
	AddDescriptorInput<myvk_rg::Usage::kUniformBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({5}, {"optimizer_state"},
	                                                                                           args.optimizer_state);
}
myvk::Ptr<myvk::ComputePipeline> NNTrain::NNOptimizer::CreatePipeline() const {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();
	auto pipeline_layout = myvk::PipelineLayout::Create(
	    device, {GetVkDescriptorSetLayout()}, {{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(uint32_t)}});
	myvk::Ptr<myvk::ShaderModule> shader_module;
	if (m_write_use) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_optimize_use.comp.u32>
		};
		shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	} else {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_optimize.comp.u32>
		};
		shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	}
	return myvk::ComputePipeline::Create(pipeline_layout, shader_module);
}
void NNTrain::NNOptimizer::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	command_buffer->CmdBindPipeline(GetVkPipeline());
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, GetVkPipeline());
	if (m_write_use) {
		uint32_t use_ema_weights = m_nrc_state_ptr->IsUseEMAWeights();
		command_buffer->CmdPushConstants(GetVkPipeline()->GetPipelineLayoutPtr(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
		                                 sizeof(use_ema_weights), &use_ema_weights);
	}
	command_buffer->CmdDispatch(VkNRCState::GetWeightCount() / 64, 1, 1);
}

} // namespace rg