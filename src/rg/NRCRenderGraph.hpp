//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_NRCRENDERGRAPH_HPP
#define VKNRC_RG_NRCRENDERGRAPH_HPP

#include "../Camera.hpp"
#include "../VkNRCState.hpp"
#include "../VkSceneTLAS.hpp"
#include "SceneResources.hpp"
#include "NRCResources.hpp"

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class NRCRenderGraph final : public myvk_rg::RenderGraphBase {
private:
	myvk::Ptr<VkSceneTLAS> m_scene_tlas_ptr;
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<VkNRCState> m_nrc_state_ptr;

	SceneResources create_scene_resources();
	NRCResources create_nrc_resources();

public:
	explicit NRCRenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager,
	                        const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr, const myvk::Ptr<VkNRCState> &nrc_state_ptr,
	                        const myvk::Ptr<Camera> &camera_ptr);
	~NRCRenderGraph() final = default;
	void Update() const;
};

} // namespace rg

#endif
