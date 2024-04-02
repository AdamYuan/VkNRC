#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>

#include <spdlog/spdlog.h>

#include "CuNRCNetwork.hpp"
#include "rg/PTRenderGraph.hpp"
#include "rg/ReconstructRenderGraph.hpp"
#include "rg/ScreenRenderGraph.hpp"

constexpr uint32_t kFrameCount = 3, kWidth = 1280, kHeight = 720;

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc == 0) {
		spdlog::error("No OBJ file");
		return EXIT_FAILURE;
	}
	GLFWwindow *window = myvk::GLFWCreateWindow("VkNRC", kWidth, kHeight, true);

	// Scene scene = Scene::LoadOBJShapeInstanceSAH(argv[0], 7); // at most 128 instances
	Scene scene = Scene::LoadOBJSingleInstance(argv[0]);
	if (scene.Empty())
		return EXIT_FAILURE;
	spdlog::info("Loaded {} Vertices, {} Texcoords, {} Materials, {} Instances", scene.GetVertices().size(),
	             scene.GetTexcoords().size(), scene.GetMaterials().size(), scene.GetInstances().size());

	auto instance = myvk::Instance::CreateWithGlfwExtensions();
	myvk::Ptr<myvk::Queue> generic_queue, compute_queue;
	myvk::Ptr<myvk::PresentQueue> present_queue;
	auto physical_device = myvk::PhysicalDevice::Fetch(instance)[0];
	auto features = physical_device->GetDefaultFeatures();
	features.vk11.storageBuffer16BitAccess = VK_TRUE;
	features.vk12.bufferDeviceAddress = VK_TRUE;
	features.vk12.hostQueryReset = VK_TRUE;
	features.vk12.shaderFloat16 = VK_TRUE;
	features.vk12.vulkanMemoryModel = VK_TRUE;
	features.vk12.vulkanMemoryModelDeviceScope = VK_TRUE;
	features.vk13.computeFullSubgroups = VK_TRUE;
	features.vk13.subgroupSizeControl = VK_TRUE;
	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, .pNext = nullptr, .rayQuery = VK_TRUE};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
	    .pNext = &ray_query_features,
	    .accelerationStructure = VK_TRUE};
	features.vk13.pNext = &accel_features;
	auto device = myvk::Device::Create(
	    physical_device,
	    myvk::GenericPresentQueueSelector{&generic_queue, myvk::Surface::Create(instance, window), &present_queue},
	    features,
	    {
	        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	        VK_KHR_RAY_QUERY_EXTENSION_NAME,
	        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
	        myvk::ExportBuffer::GetExternalMemoryExtensionName(),
	    });
	myvk::ImGuiInit(window, myvk::CommandPool::Create(generic_queue));

	auto camera = myvk::MakePtr<Camera>();
	Camera::Control cam_control{.sensitivity = 0.005f, .speed = 0.5f, .prev_cursor_pos = {}};

	auto vk_scene = myvk::MakePtr<VkScene>(generic_queue, scene);
	auto vk_scene_blas = myvk::MakePtr<VkSceneBLAS>(vk_scene);
	auto vk_scene_tlas = myvk::MakePtr<VkSceneTLAS>(vk_scene_blas);
	auto nrc_state = std::make_shared<NRCState>();
	auto cu_nrc_network = std::make_unique<CuNRCNetwork>();

	cu_nrc_network->Reset();
	cu_nrc_network->Synchronize();

	auto frame_manager = myvk::FrameManager::Create(generic_queue, present_queue, false, kFrameCount);
	auto vk_nrc_resource = myvk::MakePtr<VkNRCResource>(generic_queue, frame_manager->GetExtent(), kFrameCount);
	frame_manager->SetResizeFunc([&](VkExtent2D extent) {
		cu_nrc_network->Synchronize();
		vk_nrc_resource->Resize(extent);
		cu_nrc_network->Synchronize();
	});

	std::array<myvk::Ptr<myvk::CommandBuffer>, kFrameCount> inter_command_buffers;
	std::array<myvk::Ptr<rg::PTRenderGraph>, kFrameCount> pt_render_graphs;
	std::array<myvk::Ptr<rg::ReconstructRenderGraph>, kFrameCount> recon_render_graphs;
	std::array<myvk::Ptr<rg::ScreenRenderGraph>, kFrameCount> screen_render_graphs;
	for (uint32_t i = 0; i < kFrameCount; ++i) {
		inter_command_buffers[i] = myvk::CommandBuffer::Create(myvk::CommandPool::Create(generic_queue));
		pt_render_graphs[i] = myvk::MakePtr<rg::PTRenderGraph>(vk_scene_tlas, camera, nrc_state, vk_nrc_resource, i);
		recon_render_graphs[i] = myvk::MakePtr<rg::ReconstructRenderGraph>(vk_nrc_resource, i);
		screen_render_graphs[i] = myvk::MakePtr<rg::ScreenRenderGraph>(frame_manager, nrc_state, vk_nrc_resource, i);
	}

	bool view_accumulate = nrc_state->IsAccumulate();
	int view_left_method = static_cast<int>(nrc_state->GetLeftMethod());
	int view_right_method = static_cast<int>(nrc_state->GetRightMethod());
	bool nrc_lock = false, nrc_train_one_frame = false;

	auto fence = myvk::Fence::Create(device);

	double prev_time = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		double delta;
		{
			double cur_time = glfwGetTime();
			delta = cur_time - prev_time;
			prev_time = cur_time;
		}
		glfwPollEvents();

		nrc_train_one_frame = false;

		myvk::ImGuiNewFrame();
		ImGui::Begin("Panel");
		ImGui::Text("FPS %.1f", ImGui::GetIO().Framerate);
		if (ImGui::CollapsingHeader("View")) {
			if (ImGui::Checkbox("Accumulate", &view_accumulate))
				nrc_state->SetAccumulate(view_accumulate);
			if (nrc_state->IsAccumulate()) {
				ImGui::SameLine();
				ImGui::Text("SPP %d", nrc_state->GetAccumulateCount());
			}
			constexpr const char *kViewTypeComboStr = "None\0NRC\0Cache\0";
			if (ImGui::Combo("Left", &view_left_method, kViewTypeComboStr)) {
				nrc_state->SetLeftMethod(static_cast<NRCState::Method>(view_left_method));
				nrc_state->ResetAccumulateCount();
			}
			if (ImGui::Combo("Right", &view_right_method, kViewTypeComboStr)) {
				nrc_state->SetRightMethod(static_cast<NRCState::Method>(view_right_method));
				nrc_state->ResetAccumulateCount();
			}
		}
		if (ImGui::CollapsingHeader("NRC")) {
			if (ImGui::Checkbox("Lock", &nrc_lock))
				nrc_state->ResetAccumulateCount();
			if (nrc_lock) {
				ImGui::SameLine();
				if (ImGui::Button("Train 1-Frame")) {
					nrc_state->ResetAccumulateCount();
					nrc_train_one_frame = true;
				}
			}
			if (ImGui::Button("Re-Train")) {
				nrc_state->ResetAccumulateCount();
				cu_nrc_network->Synchronize();
				cu_nrc_network->Reset();
				cu_nrc_network->Synchronize();
			}
		}
		ImGui::End();
		ImGui::Render();

		if (camera->DragControl(window, &cam_control, delta))
			nrc_state->ResetAccumulateCount();

		if (nrc_lock && !nrc_train_one_frame)
			nrc_state->SetTrainProbability(0.0f);
		else
			nrc_state->SetTrainProbability(NRCState::GetDefaultTrainProbability());

		if (frame_manager->NewFrame()) {
			uint32_t frame_index = frame_manager->GetCurrentFrame();
			auto &pt_render_graph = pt_render_graphs[frame_index];
			auto &recon_render_graph = recon_render_graphs[frame_index];
			auto &screen_render_graph = screen_render_graphs[frame_index];

			const auto &inter_command_buffer = inter_command_buffers[frame_index];

			inter_command_buffer->GetCommandPoolPtr()->Reset();
			inter_command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			pt_render_graph->SetCanvasSize(frame_manager->GetExtent());
			pt_render_graph->CmdExecute(inter_command_buffer);
			inter_command_buffer->End();
			fence->Reset();
			inter_command_buffer->Submit(fence);
			fence->Wait();

			uint32_t inference_count = vk_nrc_resource->GetInferenceCount(frame_index);
			cu_nrc_network->Inference(*vk_nrc_resource->GetInferenceInputBuffer(),
			                          *vk_nrc_resource->GetInferenceOutputBuffer(), inference_count);
			cu_nrc_network->Synchronize();

			inter_command_buffer->GetCommandPoolPtr()->Reset();
			inter_command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			recon_render_graph->SetCanvasSize(frame_manager->GetExtent());
			recon_render_graph->SetInferenceCount(inference_count);
			recon_render_graph->CmdExecute(inter_command_buffer);
			inter_command_buffer->End();
			fence->Reset();
			inter_command_buffer->Submit(fence);
			fence->Wait();

			for (uint32_t batch = 0; batch < NRCState::GetTrainBatchCount(); ++batch) {
				uint32_t train_count = vk_nrc_resource->GetBatchTrainCount(frame_index, batch);
				cu_nrc_network->Train(*vk_nrc_resource->GetBatchTrainInputBufferArray()[batch],
				                      *vk_nrc_resource->GetBatchTrainTargetBufferArray()[batch], train_count);
			}

			const auto &command_buffer = frame_manager->GetCurrentCommandBuffer();
			command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			screen_render_graph->SetCanvasSize(frame_manager->GetExtent());
			screen_render_graph->CmdExecute(command_buffer);
			command_buffer->End();

			frame_manager->Render();
		}

		nrc_state->NextFrame();
	}

	frame_manager->WaitIdle();
	glfwTerminate();
	return 0;
}
