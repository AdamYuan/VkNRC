//
// Created by adamyuan on 2/22/24.
//

#include "NRCRenderGraph.hpp"

#include "PathTracerPass.hpp"
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

	auto swapchain_image = CreateResource<myvk_rg::SwapchainImage>({"swapchain_image"}, frame_manager);
	swapchain_image->SetLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE);

	auto path_tracer_pass = CreatePass<PathTracerPass>(
	    {"path_tracer_pass"}, PathTracerPass::Args{.vbuffer_image = vbuffer_pass->GetVBufferOutput(),
	                                               .out_image = swapchain_image->Alias(),
	                                               .scene_ptr = m_scene_ptr,
	                                               .scene_resources = scene_resources,
	                                               .nrc_state_ptr = m_nrc_state_ptr,
	                                               .nrc_resources = nrc_resources,
	                                               .camera_ptr = camera_ptr});

	auto imgui_pass = CreatePass<myvk_rg::ImGuiPass>({"imgui_pass"}, path_tracer_pass->GetImageOutput());
	AddResult({"result"}, imgui_pass->GetImageOutput());
}

void NRCRenderGraph::Update() const {
	GetResource<myvk_rg::AccelerationStructure>({"tlas"})->SetAS(m_scene_tlas_ptr->GetTLAS());
	m_scene_ptr->UpdateTransformBuffer(std::static_pointer_cast<myvk::Buffer>(
	    GetResource<myvk_rg::InputBuffer>({"transforms"})->GetBufferView().buffer));

	GetResource<myvk_rg::InputImage>({"result"})->SetVkImageView(m_nrc_state_ptr->GetResultImageView());
	m_nrc_state_ptr->UpdateSobolBuffer(
	    std::static_pointer_cast<myvk::Buffer>(GetResource<myvk_rg::InputBuffer>({"sobol"})->GetBufferView().buffer));
}

SceneResources NRCRenderGraph::create_scene_resources() {
	auto transform_buffer =
	    m_scene_ptr->MakeTransformBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

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
	    .transforms = CreateResource<myvk_rg::InputBuffer>({"transforms"}, transform_buffer)->Alias(),
	    .texture_sampler = myvk::Sampler::Create(GetDevicePtr(), VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT),
	};
	for (uint32_t tex_id = 0; const auto &texture : m_scene_ptr->GetTextures())
		sr.textures.push_back(CreateResource<myvk_rg::InputImage>({"texture", tex_id++}, texture)->Alias());
	return sr;
}

NRCResources NRCRenderGraph::create_nrc_resources() {
	auto sobol_buffer = m_nrc_state_ptr->MakeSobolBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	NRCResources nr = {
	    .sobol = CreateResource<myvk_rg::InputBuffer>({"sobol"}, sobol_buffer)->Alias(),
	    .noise = CreateResource<myvk_rg::InputImage>({"noise"}, m_nrc_state_ptr->GetNoiseImageView())->Alias(),
	    .result = CreateResource<myvk_rg::InputImage>({"result"}, m_nrc_state_ptr->GetResultImageView())->Alias(),
	    .noise_sampler = myvk::Sampler::Create(GetDevicePtr(), VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT),
	};
	return nr;
}

} // namespace rg
