//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_PATHTRACERPASS_HPP
#define VKNRC_PATHTRACERPASS_HPP

#include "../Camera.hpp"
#include "../VkNRCState.hpp"
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
		const myvk::Ptr<VkNRCState> &nrc_state_ptr;
		const myvk_rg::Buffer &eval_count, &eval_records;
		std::span<const myvk_rg::Buffer, VkNRCState::GetTrainBatchCount()> batch_train_records, batch_train_counts;
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
	inline auto GetBaseExtraROutput() const { return MakeImageOutput({"base_extra_r"}); }
	inline auto GetExtraGBOutput() const { return MakeImageOutput({"extra_gb"}); }
	inline auto GetEvalCountOutput() const { return MakeBufferOutput({"eval_count"}); }
	inline auto GetEvalRecordsOutput() const { return MakeBufferOutput({"eval_records"}); }
	inline auto GetBatchTrainCountOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_count", batch_index});
	}
	inline auto GetBatchTrainRecordsOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_records", batch_index});
	}
};

} // namespace rg

#endif // VKNRC_PATHTRACERPASS_HPP
