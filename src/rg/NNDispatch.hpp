//
// Created by adamyuan on 3/2/24.
//

#pragma once
#ifndef VKNRC_RG_NNDISPATCH_HPP
#define VKNRC_RG_NNDISPATCH_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

template <typename NNPass> class NNDispatch final : public myvk_rg::PassGroupBase {
private:
	class NNGenIndirect final : public myvk_rg::ComputePassBase {
	private:
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
		myvk::Ptr<myvk::ComputePipeline> CreatePipeline() const final {
			auto &device = GetRenderGraphPtr()->GetDevicePtr();
			auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()}, {});
			constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_indirect.comp.u32>
			};
			auto shader_module = myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
			return myvk::ComputePipeline::Create(pipeline_layout, shader_module);
		}
		void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final {
			command_buffer->CmdBindPipeline(GetVkPipeline());
			command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, GetVkPipeline());
			command_buffer->CmdDispatch(1, 1, 1);
		}
		auto GetIndirectCmdOutput() { return MakeBufferOutput({"indirect_cmd"}); }
		inline ~NNGenIndirect() final = default;
	};

public:
	template <typename... Args>
	NNDispatch(myvk_rg::Parent parent, const myvk_rg::Buffer &count, Args &&...args) : myvk_rg::PassGroupBase(parent) {
		auto gen_indirect_pass = CreatePass<NNGenIndirect>({"gen_pass"}, count);
		CreatePass<NNPass>({"pass"}, gen_indirect_pass->GetIndirectCmdOutput(), std::forward<Args>(args)...);
	}
	inline NNPass *Get() const { return GetPass<NNPass>({"pass"}); }
	inline ~NNDispatch() final = default;
};

} // namespace rg

#endif
