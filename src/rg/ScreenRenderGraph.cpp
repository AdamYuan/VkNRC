//
// Created by adamyuan on 2/22/24.
//

#include "ScreenRenderGraph.hpp"

#include "PathTracerPass.hpp"
#include "ScreenPass.hpp"
#include "VBufferPass.hpp"

#include <myvk_rg/pass/ImGuiPass.hpp>
#include <myvk_rg/pass/ImageBlitPass.hpp>
#include <myvk_rg/resource/AccelerationStructure.hpp>
#include <myvk_rg/resource/InputBuffer.hpp>
#include <myvk_rg/resource/InputImage.hpp>
#include <myvk_rg/resource/SwapchainImage.hpp>

namespace rg {

inline static void SetNoSync(auto *r) {
	r->SetSyncType(myvk_rg::ExternalSyncType::kCustom);
	r->SetSrcAccessFlags(VK_ACCESS_2_NONE);
	r->SetDstAccessFlags(VK_ACCESS_2_NONE);
	r->SetSrcPipelineStages(VK_PIPELINE_STAGE_2_NONE);
	r->SetDstPipelineStages(VK_PIPELINE_STAGE_2_NONE);
}

ScreenRenderGraph::ScreenRenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager,
                                     const myvk::Ptr<NRCState> &nrc_state_ptr,
                                     const myvk::Ptr<VkNRCResource> &nrc_resource_ptr, uint32_t frame_index)
    : RenderGraphBase(frame_manager->GetDevicePtr()), m_nrc_state_ptr(nrc_state_ptr),
      m_nrc_resource_ptr(nrc_resource_ptr), m_frame_index(frame_index) {
	auto accumulate = CreateResource<myvk_rg::InputImage>({"accumulate"}, m_nrc_resource_ptr->GetAccumulateImageView());
	auto bias_factor_r = CreateResource<myvk_rg::InputImage>(
	    {"bias_factor_r"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).bias_factor_r);
	SetNoSync(bias_factor_r);
	bias_factor_r->SetSrcLayout(VK_IMAGE_LAYOUT_GENERAL);

	auto swapchain_image = CreateResource<myvk_rg::SwapchainImage>({"swapchain_image"}, frame_manager);
	swapchain_image->SetLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);

	auto screen_pass =
	    CreatePass<ScreenPass>({"screen_pass"}, ScreenPass::Args{.nrc_state_ptr = m_nrc_state_ptr,
	                                                             .accumulate_image = accumulate->Alias(),
	                                                             .color_image = bias_factor_r->Alias(),
	                                                             .screen_image = swapchain_image->Alias()});
	auto imgui_pass = CreatePass<myvk_rg::ImGuiPass>({"imgui_pass"}, screen_pass->GetScreenOutput());
	AddResult({"present"}, imgui_pass->GetImageOutput());
}

void ScreenRenderGraph::PreExecute() const {
	// Update Externals
	GetResource<myvk_rg::InputImage>({"bias_factor_r"})
	    ->SetVkImageView(m_nrc_resource_ptr->GetFrameResources(m_frame_index).bias_factor_r);
	GetResource<myvk_rg::InputImage>({"accumulate"})->SetVkImageView(m_nrc_resource_ptr->GetAccumulateImageView());
}

} // namespace rg
