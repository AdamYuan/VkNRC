//
// Created by adamyuan on 2/29/24.
//

#pragma once
#ifndef VKNRC_RG_SCREENPASS_HPP
#define VKNRC_RG_SCREENPASS_HPP

#include "../VkNRCState.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class ScreenPass final : public myvk_rg::GraphicsPassBase {
public:
	struct Args {
		const myvk::Ptr<VkNRCState> &nrc_state_ptr;
		const myvk_rg::Image &accumulate_image, &color_image, &screen_image;
	};

private:
	myvk::Ptr<VkNRCState> m_nrc_state_ptr;
	myvk::Ptr<myvk::GraphicsPipeline> m_pipeline;

public:
	ScreenPass(myvk_rg::Parent parent, const Args &args);
	inline ~ScreenPass() final = default;
	void CreatePipeline() final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	inline auto GetScreenOutput() { return MakeImageOutput({"screen"}); }
};

} // namespace rg

#endif
