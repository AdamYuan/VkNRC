//
// Created by adamyuan on 2/22/24.
//

#pragma once
#ifndef VKNRC_RG_RECONSTRUCTRENDERGRAPH_HPP
#define VKNRC_RG_RECONSTRUCTRENDERGRAPH_HPP

#include "../VkNRCResource.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class ReconstructRenderGraph final : public myvk_rg::RenderGraphBase {
private:
	myvk::Ptr<VkNRCResource> m_nrc_resource_ptr;
	uint32_t m_frame_index;

public:
	explicit ReconstructRenderGraph(const myvk::Ptr<VkNRCResource> &nrc_resource_ptr, uint32_t frame_index);
	void SetInferenceCount(uint32_t count);
	~ReconstructRenderGraph() final = default;
	void PreExecute() const final;
};

} // namespace rg

#endif
