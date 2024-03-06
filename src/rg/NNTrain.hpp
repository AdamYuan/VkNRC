//
// Created by adamyuan on 3/5/24.
//

#pragma once
#ifndef VKNRC_NNTRAIN_HPP
#define VKNRC_NNTRAIN_HPP

#include "../VkNRCState.hpp"
#include "../VkScene.hpp"
#include "NNDispatch.hpp"
#include "SceneResources.hpp"
#include <myvk_rg/RenderGraph.hpp>
#include <myvk_rg/pass/BufferFillPass.hpp>

namespace rg {

class NNTrain final : public myvk_rg::PassGroupBase {
public:
	struct Args {
		const myvk_rg::Buffer &weights, &adam_mv;

		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk::Ptr<VkNRCState> &nrc_state_ptr;
		const myvk_rg::Buffer &batch_train_count, &batch_train_records;
		uint32_t batch_index;
	};

private:
	class NNGradient final : public myvk_rg::ComputePassBase {
	private:
		myvk::Ptr<myvk::ComputePipeline> m_pipeline;
		myvk::Ptr<VkScene> m_scene_ptr;

	public:
		NNGradient(myvk_rg::Parent parent, const myvk_rg::Buffer &cmd, const myvk_rg::Buffer &gradients,
		           const Args &args);
		inline ~NNGradient() final = default;
		void CreatePipeline() final;
		void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
		inline auto GetGradientOutput() const { return MakeBufferOutput({"gradients"}); }
	};

	class NNAdam final : public myvk_rg::ComputePassBase {
	private:
		myvk::Ptr<myvk::ComputePipeline> m_pipeline;
		myvk::Ptr<VkNRCState> m_nrc_state_ptr;
		uint32_t m_batch_index;

	public:
		NNAdam(myvk_rg::Parent parent, const myvk_rg::Buffer &gradients, const Args &args);
		inline ~NNAdam() final = default;
		void CreatePipeline() final;
		void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
		inline auto GetWeightOutput() const { return MakeBufferOutput({"weights"}); }
		inline auto GetAdamMVOutput() const { return MakeBufferOutput({"adam_mv"}); }
	};

public:
	inline NNTrain(myvk_rg::Parent parent, const Args &args) : myvk_rg::PassGroupBase(parent) {
		auto gradients =
		    CreateResource<myvk_rg::ManagedBuffer>({"gradients"}, VkNRCState::GetWeightCount() * sizeof(float));
		auto clear_pass = CreatePass<myvk_rg::BufferFillPass>({"clear_pass"}, gradients->Alias(), 0);
		auto gradient_pass = CreatePass<NNDispatch<NNGradient>>({"gradient_pass"}, args.batch_train_count,
		                                                        clear_pass->GetDstOutput(), args);
		CreatePass<NNAdam>({"adam_pass"}, gradient_pass->Get()->GetGradientOutput(), args);
	}
	inline ~NNTrain() final = default;
	inline auto GetWeightOutput() const { return GetPass<NNAdam>({"adam_pass"})->GetWeightOutput(); }
	inline auto GetAdamMVOutput() const { return GetPass<NNAdam>({"adam_pass"})->GetAdamMVOutput(); }
};

} // namespace rg

#endif // VKNRC_NNTRAIN_HPP
