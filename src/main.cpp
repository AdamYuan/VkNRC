#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>

#include <myvk_rg/RenderGraph.hpp>
#include <myvk_rg/pass/ImGuiPass.hpp>
#include <myvk_rg/resource/SwapchainImage.hpp>

#include "VkRTScene.hpp"

class NRCRenderGraph : public myvk_rg::RenderGraphBase {
public:
	explicit NRCRenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager)
	    : RenderGraphBase(frame_manager->GetDevicePtr()) {
		auto swapchain_image = CreateResource<myvk_rg::SwapchainImage>({"swapchain_image"}, frame_manager);
		swapchain_image->SetLoadOp(VK_ATTACHMENT_LOAD_OP_CLEAR);
		swapchain_image->SetClearColorValue({.float32 = {0.5f, 0.5f, 0.5f, 1.0f}});

		auto imgui_pass = CreatePass<myvk_rg::ImGuiPass>({"imgui_pass"}, swapchain_image->Alias());
		AddResult({"result"}, imgui_pass->GetImageOutput());
	}
};

int main() {
	GLFWwindow *window = myvk::GLFWCreateWindow("Test", 640, 480, true);

	Scene scene = Scene::LoadOBJ("/home/adamyuan/models/sponza/sponza.obj");
	printf("%zu Vertices, %zu Texcoords, %zu Materials, %zu Instances\n", scene.GetVertices().size(),
	       scene.GetTexcoords().size(), scene.GetMaterials().size(), scene.GetInstances().size());

	auto instance = myvk::Instance::CreateWithGlfwExtensions();
	myvk::Ptr<myvk::Queue> generic_queue;
	myvk::Ptr<myvk::PresentQueue> present_queue;
	auto physical_device = myvk::PhysicalDevice::Fetch(instance)[0];
	auto features = physical_device->GetDefaultFeatures();
	features.vk12.bufferDeviceAddress = true;
	auto device = myvk::Device::Create(
	    physical_device,
	    myvk::GenericPresentQueueSelector{&generic_queue, myvk::Surface::Create(instance, window), &present_queue},
	    features,
	    {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	     VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME});
	myvk::ImGuiInit(window, myvk::CommandPool::Create(generic_queue));

	myvk::Ptr<VkRTScene> rt_scene = myvk::MakePtr<VkRTScene>(generic_queue, scene);

	auto frame_manager = myvk::FrameManager::Create(generic_queue, present_queue, false, 1);
	auto render_graph = myvk::MakePtr<NRCRenderGraph>(frame_manager);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		myvk::ImGuiNewFrame();
		ImGui::Begin("Test");
		ImGui::Text("%f", ImGui::GetIO().Framerate);
		ImGui::End();
		ImGui::Render();

		if (frame_manager->NewFrame()) {
			const auto &command_buffer = frame_manager->GetCurrentCommandBuffer();

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
