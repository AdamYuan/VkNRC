//
// Created by adamyuan on 2/22/24.
//

#include "PTRenderGraph.hpp"

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

PTRenderGraph::PTRenderGraph(const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr, const myvk::Ptr<Camera> &camera_ptr,
                             const myvk::Ptr<NRCState> &nrc_state_ptr, const myvk::Ptr<VkNRCResource> &nrc_resource_ptr,
                             uint32_t frame_index)
    : RenderGraphBase(scene_tlas_ptr->GetDevicePtr()), m_scene_tlas_ptr(scene_tlas_ptr),
      m_scene_ptr(scene_tlas_ptr->GetScenePtr()), m_nrc_state_ptr(nrc_state_ptr), m_nrc_resource_ptr(nrc_resource_ptr),
      m_frame_index(frame_index) {
	SceneResources scene_resources = create_scene_resources();

	auto eval_count = CreateResource<myvk_rg::InputBuffer>(
	    {"eval_count"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).inference_count);
	auto eval_inputs = CreateResource<myvk_rg::InputBuffer>(
	    {"eval_inputs"}, m_nrc_resource_ptr->GetInferenceInputBuffer()->GetVkBuffer());
	auto eval_dests = CreateResource<myvk_rg::InputBuffer>(
	    {"eval_dests"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).inference_dst);
	SetNoSync(eval_count);
	SetNoSync(eval_inputs);
	SetNoSync(eval_dests);

	std::array<myvk_rg::Buffer, NRCState::GetTrainBatchCount()> batch_train_count, batch_train_inputs,
	    batch_train_biases, batch_train_factors;
	for (uint32_t b = 0; b < NRCState::GetTrainBatchCount(); ++b) {
		auto train_count = CreateResource<myvk_rg::InputBuffer>(
		    {"batch_train_count", b}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).batch_train_counts[b]);
		auto train_inputs = CreateResource<myvk_rg::InputBuffer>(
		    {"batch_train_inputs", b}, m_nrc_resource_ptr->GetBatchTrainInputBufferArray()[b]->GetVkBuffer());
		auto train_biases = CreateResource<myvk_rg::InputBuffer>(
		    {"batch_train_biases", b}, m_nrc_resource_ptr->GetBatchTrainTargetBufferArray()[b]->GetVkBuffer());
		auto train_factors = CreateResource<myvk_rg::InputBuffer>(
		    {"batch_train_factors", b}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).batch_train_factors[b]);
		SetNoSync(train_count);
		SetNoSync(train_inputs);
		SetNoSync(train_biases);
		SetNoSync(train_factors);

		batch_train_count[b] = train_count->Alias();
		batch_train_inputs[b] = train_inputs->Alias();
		batch_train_biases[b] = train_biases->Alias();
		batch_train_factors[b] = train_factors->Alias();
	}

	auto bias_factor_r = CreateResource<myvk_rg::InputImage>(
	    {"bias_factor_r"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).bias_factor_r);
	auto factor_gb = CreateResource<myvk_rg::InputImage>(
	    {"factor_gb"}, m_nrc_resource_ptr->GetFrameResources(m_frame_index).factor_gb);

	SetNoSync(bias_factor_r);
	SetNoSync(factor_gb);

	bias_factor_r->SetSrcLayout(VK_IMAGE_LAYOUT_UNDEFINED);
	factor_gb->SetSrcLayout(VK_IMAGE_LAYOUT_UNDEFINED);
	bias_factor_r->SetDstLayout(VK_IMAGE_LAYOUT_GENERAL);
	factor_gb->SetDstLayout(VK_IMAGE_LAYOUT_GENERAL);

	auto vbuffer_pass = CreatePass<VBufferPass>(
	    {"vbuffer_pass"},
	    VBufferPass::Args{.scene_ptr = m_scene_ptr, .scene_resources = scene_resources, .camera_ptr = camera_ptr});

	auto path_tracer_pass =
	    CreatePass<PathTracerPass>({"path_tracer_pass"}, PathTracerPass::Args{
	                                                         .vbuffer_image = vbuffer_pass->GetVBufferOutput(),
	                                                         .scene_ptr = m_scene_ptr,
	                                                         .scene_resources = scene_resources,
	                                                         .nrc_state_ptr = m_nrc_state_ptr,
	                                                         .camera_ptr = camera_ptr,
	                                                         .eval_count = eval_count->Alias(),
	                                                         .eval_inputs = eval_inputs->Alias(),
	                                                         .eval_dests = eval_dests->Alias(),
	                                                         .batch_train_count = batch_train_count,
	                                                         .batch_train_inputs = batch_train_inputs,
	                                                         .batch_train_biases = batch_train_biases,
	                                                         .batch_train_factors = batch_train_factors,
	                                                         .bias_factor_r = bias_factor_r->Alias(),
	                                                         .factor_gb = factor_gb->Alias(),
	                                                     });

	AddResult({"bias_factor_r"}, path_tracer_pass->GetBiasFactorROutput());
	// AddResult({"factor_gb"}, path_tracer_pass->GetFactorGBOutput());
}

void PTRenderGraph::PreExecute() const {
	// Update Externals
	GetResource<myvk_rg::AccelerationStructure>({"tlas"})->SetAS(m_scene_tlas_ptr->GetTLAS());

	GetResource<myvk_rg::InputBuffer>({"eval_inputs"})
	    ->SetBufferView(m_nrc_resource_ptr->GetInferenceInputBuffer()->GetVkBuffer());
	GetResource<myvk_rg::InputBuffer>({"eval_dests"})
	    ->SetBufferView(m_nrc_resource_ptr->GetFrameResources(m_frame_index).inference_dst);

	GetResource<myvk_rg::InputImage>({"bias_factor_r"})
	    ->SetVkImageView(m_nrc_resource_ptr->GetFrameResources(m_frame_index).bias_factor_r);
	GetResource<myvk_rg::InputImage>({"factor_gb"})
	    ->SetVkImageView(m_nrc_resource_ptr->GetFrameResources(m_frame_index).factor_gb);

	m_nrc_resource_ptr->ResetCounts(m_frame_index);
	// Update Mapped Internals
	m_scene_ptr->UpdateTransformBuffer(GetResource<myvk_rg::ManagedBuffer>({"transforms"})->GetMappedData());
}

SceneResources PTRenderGraph::create_scene_resources() {
	auto transform_buffer =
	    CreateResource<myvk_rg::ManagedBuffer>({"transforms"}, m_scene_ptr->GetTransformBufferSize());
	transform_buffer->SetMapped(true);

	SceneResources sr = {
	    .tlas = CreateResource<myvk_rg::AccelerationStructure>({"tlas"}, m_scene_tlas_ptr->GetTLAS())->Alias(),
	    .vertices = CreateResource<myvk_rg::InputBuffer>({"vertices"}, m_scene_ptr->GetVertexBuffer())->Alias(),
	    .vertex_indices =
	        CreateResource<myvk_rg::InputBuffer>({"vertex_indices"}, m_scene_ptr->GetVertexIndexBuffer())->Alias(),
	    .texcoords = CreateResource<myvk_rg::InputBuffer>({"texcoords"}, m_scene_ptr->GetTexcoordBuffer())->Alias(),
	    .texcoord_indices =
	        CreateResource<myvk_rg::InputBuffer>({"texcoord_indices"}, m_scene_ptr->GetTexcoordIndexBuffer())->Alias(),
	    .materials = CreateResource<myvk_rg::InputBuffer>({"materials"}, m_scene_ptr->GetMaterialBuffer())->Alias(),
	    .material_ids =
	        CreateResource<myvk_rg::InputBuffer>({"material_ids"}, m_scene_ptr->GetMaterialIDBuffer())->Alias(),
	    .transforms = transform_buffer->Alias(),
	    .texture_sampler = myvk::Sampler::Create(GetDevicePtr(), VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT),
	};
	for (uint32_t tex_id = 0; const auto &texture : m_scene_ptr->GetTextures())
		sr.textures.push_back(CreateResource<myvk_rg::InputImage>({"texture", tex_id++}, texture)->Alias());
	return sr;
}

} // namespace rg
