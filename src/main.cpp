#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>

#include <spdlog/spdlog.h>

#include "CuNRCNetwork.hpp"
#include "VkNRCResource.hpp"
#include "rg/NRCRenderGraph.hpp"

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
	auto vk_nrc_resource = myvk::MakePtr<VkNRCResource>(generic_queue, VkExtent2D{kWidth, kHeight}, kFrameCount);
	auto cu_nrc_network = std::make_unique<CuNRCNetwork>();

	auto frame_manager = myvk::FrameManager::Create(generic_queue, present_queue, false, kFrameCount);
	frame_manager->SetResizeFunc([&](VkExtent2D extent) { vk_nrc_resource->Resize(extent); });
	std::array<myvk::Ptr<rg::NRCRenderGraph>, kFrameCount> render_graphs;
	for (auto &rg : render_graphs)
		rg = myvk::MakePtr<rg::NRCRenderGraph>(frame_manager, vk_scene_tlas, nrc_state, camera);

	bool view_accumulate = nrc_state->IsAccumulate();
	int view_left_method = static_cast<int>(nrc_state->GetLeftMethod());
	int view_right_method = static_cast<int>(nrc_state->GetRightMethod());
	bool nrc_lock = false, nrc_train_one_frame = false;

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
				// nrc_state->ResetMLPBuffers();
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
			const auto &command_buffer = frame_manager->GetCurrentCommandBuffer();
			auto &render_graph = render_graphs[frame_manager->GetCurrentFrame()];

			command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			render_graph->SetCanvasSize(frame_manager->GetExtent());
			render_graph->CmdExecute(command_buffer);
			command_buffer->End();

			frame_manager->Render();
		}

		nrc_state->NextFrame();
	}

	frame_manager->WaitIdle();
	glfwTerminate();
	return 0;
}
