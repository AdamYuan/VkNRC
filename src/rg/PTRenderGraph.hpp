//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_NRCRENDERGRAPH_HPP
#define VKNRC_RG_NRCRENDERGRAPH_HPP

#include "../Camera.hpp"
#include "../NRCState.hpp"
#include "../VkNRCResource.hpp"
#include "../VkSceneTLAS.hpp"
#include "SceneResources.hpp"

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class PTRenderGraph final : public myvk_rg::RenderGraphBase {
private:
	myvk::Ptr<VkSceneTLAS> m_scene_tlas_ptr;
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<NRCState> m_nrc_state_ptr;
	myvk::Ptr<VkNRCResource> m_nrc_resource_ptr;
	uint32_t m_frame_index;

	SceneResources create_scene_resources();

public:
	explicit PTRenderGraph(
	                       const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr, const myvk::Ptr<Camera> &camera_ptr,
	                       const myvk::Ptr<NRCState> &nrc_state_ptr, const myvk::Ptr<VkNRCResource> &nrc_resource_ptr,
	                       uint32_t frame_index);
	~PTRenderGraph() final = default;
	void PreExecute() const final;
};

} // namespace rg

#endif
