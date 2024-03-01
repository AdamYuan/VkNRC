//
// Created by adamyuan on 3/1/24.
//

#pragma once
#ifndef VKNRC_RG_NNINFERENCE_HPP
#define VKNRC_RG_NNINFERENCE_HPP

#include "../VkScene.hpp"
#include "SceneResources.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class NNInference final : public myvk_rg::ComputePassBase {
public:
	struct Args {
		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk_rg::Image &color;
		const myvk_rg::Buffer &weights, &eval_count, &eval_records;
	};

private:
	myvk::Ptr<myvk::ComputePipeline> m_pipeline;
	myvk::Ptr<VkScene> m_scene_ptr;

public:
	NNInference(myvk_rg::Parent parent, const Args &args);
	inline ~NNInference() final = default;
	void CreatePipeline() final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	inline auto GetColorOutput() { return MakeImageOutput({"color"}); }
};

} // namespace rg

#endif
