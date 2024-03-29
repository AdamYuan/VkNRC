//
// Created by adamyuan on 3/1/24.
//

#pragma once
#ifndef VKNRC_RG_NNINFERENCE_HPP
#define VKNRC_RG_NNINFERENCE_HPP

#include "../VkNRCState.hpp"
#include "../VkScene.hpp"
#include "SceneResources.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class NNInference final : public myvk_rg::ComputePassBase {
public:
	struct Args {
		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk_rg::Image &bias_factor_r, &factor_gb;
		const myvk_rg::Buffer &weights, &eval_count, &eval_records;
		std::span<const myvk_rg::Buffer, VkNRCState::GetTrainBatchCount()> batch_train_records;
	};

private:
	myvk::Ptr<VkScene> m_scene_ptr;

public:
	NNInference(myvk_rg::Parent parent, const myvk_rg::Buffer &cmd, const Args &args);
	inline ~NNInference() final = default;
	myvk::Ptr<myvk::ComputePipeline> CreatePipeline() const final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	inline auto GetColorOutput() const { return MakeImageOutput({"base_extra_r"}); }
	inline auto GetBatchTrainRecordsOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_records", batch_index});
	}
};

} // namespace rg

#endif
