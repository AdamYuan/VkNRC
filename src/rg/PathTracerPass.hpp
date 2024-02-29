//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_PATHTRACERPASS_HPP
#define VKNRC_PATHTRACERPASS_HPP

#include "../Camera.hpp"
#include "../VkNRCState.hpp"
#include "../VkScene.hpp"
#include "NRCResources.hpp"
#include "SceneResources.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class PathTracerPass final : public myvk_rg::ComputePassBase {
public:
	struct Args {
		const myvk_rg::Image &vbuffer_image;
		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk::Ptr<VkNRCState> &nrc_state_ptr;
		const myvk_rg::Image &accumulate_image;
		const myvk_rg::Buffer &eval_count, &eval_records, &batch_train_counts, &batch_train_records;
		const myvk::Ptr<Camera> &camera_ptr;
	};

private:
	myvk::Ptr<myvk::ComputePipeline> m_pipeline;
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<VkNRCState> m_nrc_state_ptr;
	myvk::Ptr<Camera> m_camera_ptr;

public:
	PathTracerPass(myvk_rg::Parent parent, const Args &args);
	inline ~PathTracerPass() final = default;
	void CreatePipeline() final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	inline auto GetColorOutput() { return MakeImageOutput({"color"}); }
	inline auto GetAccumulateOutput() { return MakeImageOutput({"accumulate"}); }
	inline auto GetEvalCountOutput() { return MakeBufferOutput({"eval_count"}); }
	inline auto GetEvalRecordsOutput() { return MakeBufferOutput({"eval_records"}); }
	inline auto GetBatchTrainCountsOutput() { return MakeBufferOutput({"batch_train_counts"}); }
	inline auto GetBatchTrainRecordsOutput() { return MakeBufferOutput({"batch_train_records"}); }
};

} // namespace rg

#endif // VKNRC_PATHTRACERPASS_HPP
