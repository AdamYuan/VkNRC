//
// Created by adamyuan on 2/22/24.
//

#include "NRCRenderGraph.hpp"

#include "NNDispatch.hpp"
#include "NNInference.hpp"
#include "NNTrain.hpp"
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

NRCRenderGraph::NRCRenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager,
                               const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr, const myvk::Ptr<VkNRCState> &nrc_state_ptr,
                               const myvk::Ptr<Camera> &camera_ptr)
    : RenderGraphBase(frame_manager->GetDevicePtr()), m_scene_tlas_ptr(scene_tlas_ptr),
      m_scene_ptr(scene_tlas_ptr->GetScenePtr()), m_nrc_state_ptr(nrc_state_ptr) {
	SceneResources scene_resources = create_scene_resources();
	NRCResources nrc_resources = create_nrc_resources();

	auto vbuffer_pass = CreatePass<VBufferPass>(
	    {"vbuffer_pass"},
	    VBufferPass::Args{.scene_ptr = m_scene_ptr, .scene_resources = scene_resources, .camera_ptr = camera_ptr});

	auto path_tracer_pass = CreatePass<PathTracerPass>(
	    {"path_tracer_pass"}, PathTracerPass::Args{.vbuffer_image = vbuffer_pass->GetVBufferOutput(),
	                                               .scene_ptr = m_scene_ptr,
	                                               .scene_resources = scene_resources,
	                                               .nrc_state_ptr = m_nrc_state_ptr,
	                                               .eval_count = nrc_resources.eval_count,
	                                               .eval_records = nrc_resources.eval_records,
	                                               .batch_train_records = nrc_resources.batch_train_records,
	                                               .batch_train_counts = nrc_resources.batch_train_counts,
	                                               .camera_ptr = camera_ptr});

	auto nn_inference_pass = CreatePass<NNDispatch<NNInference>>(
	    {"nn_inference_pass"}, path_tracer_pass->GetEvalCountOutput(),
	    NNInference::Args{.scene_ptr = m_scene_ptr,
	                      .scene_resources = scene_resources,
	                      .bias_factor_r = path_tracer_pass->GetBiasFactorROutput(),
	                      .factor_gb = path_tracer_pass->GetFactorGBOutput(),
	                      .weights = nrc_resources.use_weights,
	                      .eval_count = path_tracer_pass->GetEvalCountOutput(),
	                      .eval_records = path_tracer_pass->GetEvalRecordsOutput(),
	                      .batch_train_records = path_tracer_pass->GetBatchTrainRecordsOutputs()});

	for (uint32_t b = 0; b < VkNRCState::GetTrainBatchCount(); ++b) {
		myvk_rg::Buffer weights = nrc_resources.weights, optimizer_entries = nrc_resources.optimizer_entries,
		                optimizer_state = nrc_resources.optimizer_state;
		if (b != 0) {
			auto prev_train_pass = GetPass<NNTrain>({"nn_train_pass", b - 1});
			weights = prev_train_pass->GetWeightOutput();
			optimizer_entries = prev_train_pass->GetOptimizerEntriesOutput();
			optimizer_state = prev_train_pass->GetOptimizerStateOutput();
		}
		std::optional<myvk_rg::Buffer> opt_use_weights = std::nullopt;
		if (b == VkNRCState::GetTrainBatchCount() - 1)
			opt_use_weights = nrc_resources.use_weights;
		CreatePass<NNTrain>(
		    {"nn_train_pass", b},
		    NNTrain::Args{.nrc_state_ptr = m_nrc_state_ptr,
		                  .weights = weights,
		                  .opt_use_weights = opt_use_weights,
		                  .optimizer_state = optimizer_state,
		                  .optimizer_entries = optimizer_entries,
		                  .scene_ptr = m_scene_ptr,
		                  .scene_resources = scene_resources,
		                  .batch_train_count = path_tracer_pass->GetBatchTrainCountOutput(b),
		                  .batch_train_records = nn_inference_pass->Get()->GetBatchTrainRecordsOutput(b)});
	}

	auto swapchain_image = CreateResource<myvk_rg::SwapchainImage>({"swapchain_image"}, frame_manager);
	swapchain_image->SetLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);

	auto screen_pass = CreatePass<ScreenPass>(
	    {"screen_pass"}, ScreenPass::Args{.nrc_state_ptr = m_nrc_state_ptr,
	                                      .accumulate_image = nrc_resources.accumulate,
	                                      .color_image = nn_inference_pass->Get()->GetColorOutput(),
	                                      .screen_image = swapchain_image->Alias()});
	auto imgui_pass = CreatePass<myvk_rg::ImGuiPass>({"imgui_pass"}, screen_pass->GetScreenOutput());
	AddResult({"present"}, imgui_pass->GetImageOutput());

	auto final_train_pass = GetPass<NNTrain>({"nn_train_pass", VkNRCState::GetTrainBatchCount() - 1});
	AddResult({"weights"}, final_train_pass->GetWeightOutput());
	AddResult({"use_weights"}, final_train_pass->GetEMAWeightOutput());
	AddResult({"optimizer_entries"}, final_train_pass->GetOptimizerEntriesOutput());
	AddResult({"optimizer_state"}, final_train_pass->GetOptimizerStateOutput());
}

void NRCRenderGraph::PreExecute() const {
	// Update Externals
	GetResource<myvk_rg::AccelerationStructure>({"tlas"})->SetAS(m_scene_tlas_ptr->GetTLAS());
	GetResource<myvk_rg::InputImage>({"accumulate"})->SetVkImageView(m_nrc_state_ptr->GetAccumulateImageView());
	GetResource<myvk_rg::InputBuffer>({"weights"})->SetBufferView(m_nrc_state_ptr->GetWeightBuffer());
	GetResource<myvk_rg::InputBuffer>({"use_weights"})->SetBufferView(m_nrc_state_ptr->GetUseWeightBuffer());
	GetResource<myvk_rg::InputBuffer>({"optimizer_state"})->SetBufferView(m_nrc_state_ptr->GetOptimizerStateBuffer());
	GetResource<myvk_rg::InputBuffer>({"optimizer_entries"})->SetBufferView(m_nrc_state_ptr->GetOptimizerEntryBuffer());
	// Update Mapped Internals
	m_scene_ptr->UpdateTransformBuffer(GetResource<myvk_rg::ManagedBuffer>({"transforms"})->GetMappedData());
	*GetResource<myvk_rg::ManagedBuffer>({"eval_count"})->GetMappedData<uint32_t>() = 0u;
	for (uint32_t b = 0; b < VkNRCState::GetTrainBatchCount(); ++b)
		*GetResource<myvk_rg::ManagedBuffer>({"batch_train_count", b})->GetMappedData<uint32_t>() = 0u;
}

SceneResources NRCRenderGraph::create_scene_resources() {
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

NRCResources NRCRenderGraph::create_nrc_resources() {
	auto eval_record_buffer = CreateResource<myvk_rg::ManagedBuffer>({"eval_records"});
	eval_record_buffer->SetSizeFunc(
	    [](VkExtent2D extent) -> VkDeviceSize { return VkNRCState::GetEvalRecordBufferSize(extent); });

	auto eval_record_count_buffer = CreateResource<myvk_rg::ManagedBuffer>({"eval_count"}, sizeof(uint32_t));
	eval_record_count_buffer->SetMapped(true);

	std::array<myvk_rg::Buffer, VkNRCState::GetTrainBatchCount()> batch_train_counts, batch_train_records;
	for (uint32_t b = 0; b < VkNRCState::GetTrainBatchCount(); ++b) {
		auto count_buffer = CreateResource<myvk_rg::ManagedBuffer>({"batch_train_count", b}, sizeof(uint32_t));
		count_buffer->SetMapped(true);
		auto record_buffer = CreateResource<myvk_rg::ManagedBuffer>({"batch_train_records", b},
		                                                            VkNRCState::GetBatchTrainRecordBufferSize());
		batch_train_counts[b] = count_buffer->Alias();
		batch_train_records[b] = record_buffer->Alias();
	}

	NRCResources nr = {
	    .accumulate =
	        CreateResource<myvk_rg::InputImage>({"accumulate"}, m_nrc_state_ptr->GetAccumulateImageView())->Alias(),
	    .weights = CreateResource<myvk_rg::InputBuffer>({"weights"}, m_nrc_state_ptr->GetWeightBuffer())->Alias(),
	    .use_weights =
	        CreateResource<myvk_rg::InputBuffer>({"use_weights"}, m_nrc_state_ptr->GetUseWeightBuffer())->Alias(),
	    .optimizer_state =
	        CreateResource<myvk_rg::InputBuffer>({"optimizer_state"}, m_nrc_state_ptr->GetOptimizerStateBuffer())
	            ->Alias(),
	    .optimizer_entries =
	        CreateResource<myvk_rg::InputBuffer>({"optimizer_entries"}, m_nrc_state_ptr->GetOptimizerEntryBuffer())
	            ->Alias(),
	    .eval_records = eval_record_buffer->Alias(),
	    .eval_count = eval_record_count_buffer->Alias(),
	    .batch_train_records = std::move(batch_train_records),
	    .batch_train_counts = std::move(batch_train_counts),
	};
	return nr;
}

} // namespace rg
