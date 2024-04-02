//
// Created by adamyuan on 2/22/24.
//

#include "ReconstructRenderGraph.hpp"

#include "ReconstructPass.hpp"

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

ReconstructRenderGraph::ReconstructRenderGraph(const myvk::Ptr<VkNRCResource> &nrc_resource_ptr, uint32_t frame_index)
    : RenderGraphBase(nrc_resource_ptr->GetDevicePtr()), m_nrc_resource_ptr(nrc_resource_ptr),
      m_frame_index(frame_index) {
	auto eval_dests = CreateResource<myvk_rg::InputBuffer>(
	    {"eval_dests"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).inference_dst);
	auto eval_outputs = CreateResource<myvk_rg::InputBuffer>(
	    {"eval_outputs"}, m_nrc_resource_ptr->GetInferenceOutputBuffer()->GetVkBuffer());
	SetNoSync(eval_dests);
	SetNoSync(eval_outputs);

	std::array<myvk_rg::Buffer, NRCState::GetTrainBatchCount()> batch_train_biases, batch_train_factors;
	for (uint32_t b = 0; b < NRCState::GetTrainBatchCount(); ++b) {
		auto train_biases = CreateResource<myvk_rg::InputBuffer>(
		    {"batch_train_biases", b}, m_nrc_resource_ptr->GetBatchTrainTargetBufferArray()[b]->GetVkBuffer());
		auto train_factors = CreateResource<myvk_rg::InputBuffer>(
		    {"batch_train_factors", b}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).batch_train_factors[b]);
		SetNoSync(train_biases);
		SetNoSync(train_factors);

		batch_train_biases[b] = train_biases->Alias();
		batch_train_factors[b] = train_factors->Alias();
	}

	auto bias_factor_r = CreateResource<myvk_rg::InputImage>(
	    {"bias_factor_r"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).bias_factor_r);
	auto factor_gb = CreateResource<myvk_rg::InputImage>(
	    {"factor_gb"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).factor_gb);

	SetNoSync(bias_factor_r);
	SetNoSync(factor_gb);

	bias_factor_r->SetSrcLayout(VK_IMAGE_LAYOUT_GENERAL);
	factor_gb->SetSrcLayout(VK_IMAGE_LAYOUT_GENERAL);
	bias_factor_r->SetDstLayout(VK_IMAGE_LAYOUT_GENERAL);
	factor_gb->SetDstLayout(VK_IMAGE_LAYOUT_GENERAL);

	auto reconstruct_pass =
	    CreatePass<ReconstructPass>({"reconstruct_pass"}, ReconstructPass::Args{
	                                                          .eval_dests = eval_dests->Alias(),
	                                                          .eval_outputs = eval_outputs->Alias(),
	                                                          .bias_factor_r = bias_factor_r->Alias(),
	                                                          .factor_gb = factor_gb->Alias(),
	                                                          .batch_train_biases = batch_train_biases,
	                                                          .batch_train_factors = batch_train_factors,
	                                                      });
	AddResult({"color"}, reconstruct_pass->GetColorOutput());
}

void ReconstructRenderGraph::SetInferenceCount(uint32_t count) {
	GetPass<ReconstructPass>({"reconstruct_pass"})->SetCount(count);
}

void ReconstructRenderGraph::PreExecute() const {
	// Update Externals
	GetResource<myvk_rg::InputBuffer>({"eval_outputs"})
	    ->SetBufferView(m_nrc_resource_ptr->GetInferenceOutputBuffer()->GetVkBuffer());
	GetResource<myvk_rg::InputBuffer>({"eval_dests"})
	    ->SetBufferView(m_nrc_resource_ptr->GetFrameResources(m_frame_index).inference_dst);

	GetResource<myvk_rg::InputImage>({"bias_factor_r"})
	    ->SetVkImageView(m_nrc_resource_ptr->GetFrameResources(m_frame_index).bias_factor_r);
	GetResource<myvk_rg::InputImage>({"factor_gb"})
	    ->SetVkImageView(m_nrc_resource_ptr->GetFrameResources(m_frame_index).factor_gb);
}

} // namespace rg
