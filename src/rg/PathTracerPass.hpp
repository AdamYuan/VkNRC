//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_PATHTRACERPASS_HPP
#define VKNRC_PATHTRACERPASS_HPP

#include "../Camera.hpp"
#include "../VkScene.hpp"
#include "../VkNRCState.hpp"
#include "SceneResources.hpp"
#include "NRCResources.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class PathTracerPass final : public myvk_rg::GraphicsPassBase {
public:
	struct Args {
		const myvk_rg::Image &vbuffer_image;
		const myvk_rg::Image &out_image;
		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk::Ptr<VkNRCState> &nrc_state_ptr;
		const NRCResources &nrc_resources;
		const myvk::Ptr<Camera> &camera_ptr;
	};

private:
	myvk::Ptr<myvk::GraphicsPipeline> m_pipeline;
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<VkNRCState> m_nrc_state_ptr;
	myvk::Ptr<Camera> m_camera_ptr;

public:
	PathTracerPass(myvk_rg::Parent parent, const Args &args);
	inline ~PathTracerPass() final = default;
	void CreatePipeline() final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	inline auto GetImageOutput() { return MakeImageOutput({"out_in"}); }
};

} // namespace rg

#endif // VKNRC_PATHTRACERPASS_HPP
