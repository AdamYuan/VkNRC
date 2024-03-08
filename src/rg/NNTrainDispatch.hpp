//
// Created by adamyuan on 3/7/24.
//

#pragma once
#ifndef VKNRC_NNTRAINDISPATCH_HPP
#define VKNRC_NNTRAINDISPATCH_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

template <typename NNPass> class NNTrainDispatch final : public myvk_rg::PassGroupBase {
private:
	class NNGenIndirect final : public myvk_rg::ComputePassBase {
	private:
		myvk::Ptr<myvk::ComputePipeline> m_pipeline;
		struct VkDispatchIndirectCommand {
			uint32_t x, y, z;
		};

	public:
		NNGenIndirect(myvk_rg::Parent parent, const myvk_rg::Buffer &count) : myvk_rg::ComputePassBase(parent) {
			auto indirect_cmd =
			    CreateResource<myvk_rg::ManagedBuffer>({"indirect_cmd"}, sizeof(VkDispatchIndirectCommand));
			AddDescriptorInput<myvk_rg::Usage::kUniformBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"count"},
			                                                                                           count);
			AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
			    {1}, {"indirect_cmd"}, indirect_cmd->Alias());
		}
		void CreatePipeline() final {
			auto &device = GetRenderGraphPtr()->GetDevicePtr();
			auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()}, {});
			constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_train_indirect.comp.u32>
			};
			auto shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
			m_pipeline = myvk::ComputePipeline::Create(pipeline_layout, shader_module);
		}
		void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final {
			command_buffer->CmdBindPipeline(m_pipeline);
			command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, m_pipeline);
			command_buffer->CmdDispatch(1, 1, 1);
		}
		auto GetIndirectCmdOutput() { return MakeBufferOutput({"indirect_cmd"}); }
		inline ~NNGenIndirect() final = default;
	};

public:
	template <typename... Args>
	NNTrainDispatch(myvk_rg::Parent parent, const myvk_rg::Buffer &count, Args &&...args)
	    : myvk_rg::PassGroupBase(parent) {
		auto gen_indirect_pass = CreatePass<NNGenIndirect>({"gen_pass"}, count);
		CreatePass<NNPass>({"pass"}, gen_indirect_pass->GetIndirectCmdOutput(), std::forward<Args>(args)...);
	}
	inline NNPass *Get() const { return GetPass<NNPass>({"pass"}); }
	inline ~NNTrainDispatch() final = default;
};

} // namespace rg

#endif // VKNRC_NNTRAINDISPATCH_HPP