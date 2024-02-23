//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_GBUFFERPASS_HPP
#define VKNRC_RG_GBUFFERPASS_HPP

#include "../Camera.hpp"
#include "../VkScene.hpp"
#include "SceneResources.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class VBufferPass final : public myvk_rg::GraphicsPassBase {
private:
	myvk::Ptr<myvk::GraphicsPipeline> m_pipeline;
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<Camera> m_camera_ptr;

public:
	struct Args {
		const myvk::Ptr<VkScene> &scene_ptr;
		const SceneResources &scene_resources;
		const myvk::Ptr<Camera> &camera_ptr;
	};
	VBufferPass(myvk_rg::Parent parent, const Args &args);
	inline ~VBufferPass() final = default;
	void CreatePipeline() final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	auto GetVBufferOutput() { return MakeImageOutput({"v_buffer_in"}); }
	auto GetDepthOutput() { return MakeImageOutput({"depth_in"}); }
};

} // namespace rg

#endif
