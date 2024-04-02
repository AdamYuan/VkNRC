//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_SCREENRENDERGRAPH_HPP
#define VKNRC_RG_SCREENRENDERGRAPH_HPP

#include "../Camera.hpp"
#include "../NRCState.hpp"
#include "../VkNRCResource.hpp"
#include "../VkSceneTLAS.hpp"
#include "SceneResources.hpp"

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class ScreenRenderGraph final : public myvk_rg::RenderGraphBase {
private:
	myvk::Ptr<NRCState> m_nrc_state_ptr;
	myvk::Ptr<VkNRCResource> m_nrc_resource_ptr;
	uint32_t m_frame_index;

public:
	explicit ScreenRenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager,
	                           const myvk::Ptr<NRCState> &nrc_state_ptr,
	                           const myvk::Ptr<VkNRCResource> &nrc_resource_ptr, uint32_t frame_index);
	~ScreenRenderGraph() final = default;
	void PreExecute() const final;
};

} // namespace rg

#endif
