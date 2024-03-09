//
// Created by adamyuan on 3/5/24.
//

#pragma once
#ifndef VKNRC_NNTRAIN_HPP
#define VKNRC_NNTRAIN_HPP

#include "../VkNRCState.hpp"
#include "../VkScene.hpp"
#include "SceneResources.hpp"
#include <myvk_rg/RenderGraph.hpp>
#include <myvk_rg/pass/BufferFillPass.hpp>

namespace rg {

class NNTrain final : public myvk_rg::PassGroupBase {
public:
	struct Args {
		const myvk::Ptr<VkNRCState> &nrc_state_ptr;
		const myvk_rg::Buffer &weights;
		const std::optional<myvk_rg::Buffer> &opt_use_weights;
		const myvk_rg::Buffer &optimizer_state, &optimizer_entries;

		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk_rg::Buffer &batch_train_count, &batch_train_records;
	};

private:
	class NNPreparePass final : public myvk_rg::ComputePassBase {
	public:
		struct Args {
			const myvk_rg::Buffer &count, &optimizer_state;
		};

	private:
		myvk::Ptr<myvk::ComputePipeline> m_pipeline;
		struct VkDispatchIndirectCommand {
			uint32_t x, y, z;
		};

	public:
		NNPreparePass(myvk_rg::Parent parent, const Args &args);
		void CreatePipeline() final;
		void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
		auto GetIndirectCmdOutput() const { return MakeBufferOutput({"indirect_cmd"}); }
		auto GetCountOutput() const { return MakeBufferOutput({"count"}); }
		auto GetOptimizerStateOutput() const { return MakeBufferOutput({"optimizer_state"}); }
		inline ~NNPreparePass() final = default;
	};

	class NNGradient final : public myvk_rg::ComputePassBase {
	public:
		struct Args {
			const myvk::Ptr<VkScene> &scene_ptr;
			const SceneResources &scene_resources;
			const myvk_rg::Buffer &cmd, &gradients, &count, &records, &weights;
		};

	private:
		myvk::Ptr<myvk::ComputePipeline> m_pipeline;
		myvk::Ptr<VkScene> m_scene_ptr;

	public:
		NNGradient(myvk_rg::Parent parent, const Args &args);
		inline ~NNGradient() final = default;
		void CreatePipeline() final;
		void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
		inline auto GetGradientOutput() const { return MakeBufferOutput({"gradients"}); }
	};

	class NNOptimizer final : public myvk_rg::ComputePassBase {
	public:
		struct Args {
			const myvk::Ptr<VkNRCState> &nrc_state_ptr;
			const myvk_rg::Buffer &gradients, &count, &weights;
			const std::optional<myvk_rg::Buffer> &opt_use_weights;
			const myvk_rg::Buffer &optimizer_state, &optimizer_entries;
		};

	private:
		myvk::Ptr<VkNRCState> m_nrc_state_ptr;
		myvk::Ptr<myvk::ComputePipeline> m_pipeline;
		bool m_write_use;

	public:
		NNOptimizer(myvk_rg::Parent parent, const Args &args);
		inline ~NNOptimizer() final = default;
		void CreatePipeline() final;
		void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
		inline auto GetWeightOutput() const { return MakeBufferOutput({"weights"}); }
		inline auto GetEMAWeightOutput() const { return MakeBufferOutput({"use_weights"}); }
		inline auto GetOptimizerEntriesOutput() const { return MakeBufferOutput({"optimizer_entries"}); }
	};

public:
	inline NNTrain(myvk_rg::Parent parent, const Args &args) : myvk_rg::PassGroupBase(parent) {
		auto gradients =
		    CreateResource<myvk_rg::ManagedBuffer>({"gradients"}, VkNRCState::GetWeightCount() * sizeof(float));
		auto clear_pass = CreatePass<myvk_rg::BufferFillPass>({"clear_pass"}, gradients->Alias(), 0);
		auto prepare_pass =
		    CreatePass<NNPreparePass>({"prepare_pass"}, NNPreparePass::Args{.count = args.batch_train_count,
		                                                                    .optimizer_state = args.optimizer_state});
		auto gradient_pass =
		    CreatePass<NNGradient>({"gradient_pass"}, NNGradient::Args{.scene_ptr = args.scene_ptr,
		                                                               .scene_resources = args.scene_resources,
		                                                               .cmd = prepare_pass->GetIndirectCmdOutput(),
		                                                               .gradients = clear_pass->GetDstOutput(),
		                                                               .count = prepare_pass->GetCountOutput(),
		                                                               .records = args.batch_train_records,
		                                                               .weights = args.weights});
		CreatePass<NNOptimizer>({"optimizer_pass"},
		                        NNOptimizer::Args{.nrc_state_ptr = args.nrc_state_ptr,
		                                          .gradients = gradient_pass->GetGradientOutput(),
		                                          .count = prepare_pass->GetCountOutput(),
		                                          .weights = args.weights,
		                                          .opt_use_weights = args.opt_use_weights,
		                                          .optimizer_state = prepare_pass->GetOptimizerStateOutput(),
		                                          .optimizer_entries = args.optimizer_entries});
	}
	inline ~NNTrain() final = default;
	inline auto GetWeightOutput() const { return GetPass<NNOptimizer>({"optimizer_pass"})->GetWeightOutput(); }
	inline auto GetEMAWeightOutput() const { return GetPass<NNOptimizer>({"optimizer_pass"})->GetEMAWeightOutput(); }
	inline auto GetOptimizerEntriesOutput() const {
		return GetPass<NNOptimizer>({"optimizer_pass"})->GetOptimizerEntriesOutput();
	}
	inline auto GetOptimizerStateOutput() const {
		return GetPass<NNPreparePass>({"prepare_pass"})->GetOptimizerStateOutput();
	}
};

} // namespace rg

#endif // VKNRC_NNTRAIN_HPP
