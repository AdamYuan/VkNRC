#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>

#include <spdlog/spdlog.h>

#include "Camera.hpp"
#include "VkSceneTLAS.hpp"
#include "rg/NRCRenderGraph.hpp"

constexpr uint32_t kFrameCount = 3;

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc == 0) {
		spdlog::error("No OBJ file");
		return EXIT_FAILURE;
	}
	GLFWwindow *window = myvk::GLFWCreateWindow("VkNRC", 640, 480, true);

	Scene scene = Scene::LoadOBJShapeInstanceSAH(argv[0], 7); // at most 128 instances
	if (scene.Empty())
		return EXIT_FAILURE;
	spdlog::info("Loaded {} Vertices, {} Texcoords, {} Materials, {} Instances", scene.GetVertices().size(),
	             scene.GetTexcoords().size(), scene.GetMaterials().size(), scene.GetInstances().size());

	auto instance = myvk::Instance::CreateWithGlfwExtensions();
	myvk::Ptr<myvk::Queue> generic_queue, compute_queue;
	myvk::Ptr<myvk::PresentQueue> present_queue;
	auto physical_device = myvk::PhysicalDevice::Fetch(instance)[0];
	auto features = physical_device->GetDefaultFeatures();
	features.vk12.bufferDeviceAddress = VK_TRUE;
	features.vk12.hostQueryReset = VK_TRUE;
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
	    .accelerationStructure = VK_TRUE,
	};
	features.vk13.pNext = &accel_features;
	auto device = myvk::Device::Create(
	    physical_device,
	    myvk::GenericPresentQueueSelector{&generic_queue, myvk::Surface::Create(instance, window), &present_queue},
	    features,
	    {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	     VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME});
	myvk::ImGuiInit(window, myvk::CommandPool::Create(generic_queue));

	auto camera = myvk::MakePtr<Camera>();

	auto vk_scene = myvk::MakePtr<VkScene>(generic_queue, scene);
	auto vk_scene_blas = myvk::MakePtr<VkSceneBLAS>(vk_scene);
	auto vk_scene_tlas = myvk::MakePtr<VkSceneTLAS>(vk_scene_blas);

	auto frame_manager = myvk::FrameManager::Create(generic_queue, present_queue, false, kFrameCount,
	                                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
	std::array<myvk::Ptr<rg::NRCRenderGraph>, kFrameCount> render_graphs;
	for (auto &rg : render_graphs)
		rg = myvk::MakePtr<rg::NRCRenderGraph>(frame_manager, vk_scene_tlas, camera);

	double prev_time = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		double delta;
		{
			double cur_time = glfwGetTime();
			delta = cur_time - prev_time;
			prev_time = cur_time;
		}
		glfwPollEvents();

		myvk::ImGuiNewFrame();
		ImGui::Begin("Test");
		ImGui::Text("%f", ImGui::GetIO().Framerate);
		ImGui::End();
		ImGui::Render();

		camera->DragControl(window, delta);

		if (frame_manager->NewFrame()) {
			const auto &command_buffer = frame_manager->GetCurrentCommandBuffer();
			auto &render_graph = render_graphs[frame_manager->GetCurrentFrame()];
			render_graph->UpdateScene();

			command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			render_graph->SetCanvasSize(frame_manager->GetExtent());
			render_graph->CmdExecute(command_buffer);
			command_buffer->End();

			frame_manager->Render();
		}
	}

	frame_manager->WaitIdle();
	glfwTerminate();
	return 0;
}
