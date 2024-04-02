//
// Created by adamyuan on 4/2/24.
//

#include "ReconstructPass.hpp"

namespace rg {

ReconstructPass::ReconstructPass(myvk_rg::Parent parent, const ReconstructPass::Args &args)
    : myvk_rg::ComputePassBase(parent) {
	// NRC
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"eval_dests"},
	                                                                                            args.eval_dests);
	AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({1}, {"eval_outputs"},
	                                                                                            args.eval_outputs);
	AddDescriptorInput<myvk_rg::Usage::kStorageImageRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({2}, {"bias_factor_r"},
	                                                                                            args.bias_factor_r);
	AddDescriptorInput<myvk_rg::Usage::kStorageImageR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({3}, {"factor_gb"},
	                                                                                           args.factor_gb);
	for (uint32_t b = 0; b < NRCState::GetTrainBatchCount(); ++b) {
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {4, b}, {"batch_train_biases", b}, args.batch_train_biases[b]);
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {5, b}, {"batch_train_factors", b}, args.batch_train_factors[b]);
	}
}

myvk::Ptr<myvk::ComputePipeline> ReconstructPass::CreatePipeline() const {
	auto &device = GetRenderGraphPtr()->GetDevicePtr();
	auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()},
	                                                    {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)}});
	constexpr uint32_t kCompSpv[] = {
#include <shader/reconstruct.comp.u32>
	};
	auto shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	return myvk::ComputePipeline::Create(pipeline_layout, shader_module);
}

void ReconstructPass::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
	command_buffer->CmdBindPipeline(GetVkPipeline());
	command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, GetVkPipeline());
	command_buffer->CmdPushConstants(GetVkPipeline()->GetPipelineLayoutPtr(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                                 sizeof(m_count), &m_count);
	command_buffer->CmdDispatch((m_count + 127u) / 128u, 1, 1);
}

} // namespace rg
