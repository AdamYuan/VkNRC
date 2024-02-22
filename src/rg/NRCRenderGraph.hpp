//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_NRCRENDERGRAPH_HPP
#define VKNRC_RG_NRCRENDERGRAPH_HPP

#include "../Camera.hpp"
#include "../VkSceneTLAS.hpp"
#include "SceneResources.hpp"

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class NRCRenderGraph final : public myvk_rg::RenderGraphBase {
private:
	myvk::Ptr<VkSceneTLAS> m_scene_tlas_ptr;
	myvk::Ptr<VkScene> m_scene_ptr;

	SceneResources create_scene_resources();

public:
	explicit NRCRenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager,
	                        const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr, const myvk::Ptr<Camera> &camera_ptr);
	~NRCRenderGraph() final = default;
	void UpdateScene() const;
};

} // namespace rg

#endif
