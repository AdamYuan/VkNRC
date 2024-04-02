//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_PATHTRACERPASS_HPP
#define VKNRC_PATHTRACERPASS_HPP

#include "../Camera.hpp"
#include "../NRCState.hpp"
#include "../VkScene.hpp"
#include "SceneResources.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class PathTracerPass final : public myvk_rg::ComputePassBase {
public:
	struct Args {
		const myvk_rg::Image &vbuffer_image;
		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk::Ptr<NRCState> &nrc_state_ptr;
		const myvk::Ptr<Camera> &camera_ptr;
		const myvk_rg::Buffer &eval_count, &eval_inputs, &eval_dests;
		std::span<const myvk_rg::Buffer, NRCState::GetTrainBatchCount()> batch_train_count, batch_train_inputs,
		    batch_train_biases, batch_train_factors;
		const myvk_rg::Image &bias_factor_r, &factor_gb;
	};

private:
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<NRCState> m_nrc_state_ptr;
	myvk::Ptr<Camera> m_camera_ptr;

public:
	PathTracerPass(myvk_rg::Parent parent, const Args &args);
	inline ~PathTracerPass() final = default;
	myvk::Ptr<myvk::ComputePipeline> CreatePipeline() const final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	inline auto GetBiasFactorROutput() const { return MakeImageOutput({"bias_factor_r"}); }
	inline auto GetFactorGBOutput() const { return MakeImageOutput({"factor_gb"}); }
	inline auto GetEvalCountOutput() const { return MakeBufferOutput({"eval_count"}); }
	inline auto GetEvalInputsOutput() const { return MakeBufferOutput({"eval_inputs"}); }
	inline auto GetEvalOutputsOutput() const { return MakeBufferOutput({"eval_outputs"}); }
	inline auto GetBatchTrainCountOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_count", batch_index});
	}
	inline auto GetBatchTrainInputsOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_inputs", batch_index});
	}
	inline auto GetBatchTrainBiasesOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_biases", batch_index});
	}
	inline auto GetBatchTrainFactorsOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_factors", batch_index});
	}
};

} // namespace rg

#endif // VKNRC_PATHTRACERPASS_HPP
