#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include <half.hpp>
#include <stb_image.h>

#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>

#include <myvk_rg/RenderGraph.hpp>
#include <myvk_rg/pass/BufferFillPass.hpp>
#include <myvk_rg/pass/ImGuiPass.hpp>
#include <myvk_rg/pass/ImageBlitPass.hpp>
#include <myvk_rg/resource/InputBuffer.hpp>
#include <myvk_rg/resource/InputImage.hpp>
#include <myvk_rg/resource/SwapchainImage.hpp>

constexpr uint32_t kFrameCount = 3;
constexpr uint32_t kBatchSize = 16384;
constexpr uint32_t kWindowSize = 640;
constexpr uint32_t kWeightCount = 64 * 64 * 5 + 64 * 3;

inline static std::vector<uint8_t> load_spv(const std::filesystem::path &filename) {
	std::ifstream fin(filename, std::ios::binary);
	return std::vector<unsigned char>(std::istreambuf_iterator<char>(fin), {});
}

inline static myvk::Ptr<myvk::ImageView> load_image(const myvk::Ptr<myvk::Queue> &queue,
                                                    const std::filesystem::path &filename) {
	const auto &device = queue->GetDevicePtr();
	int width, height, channels;
	stbi_uc *data = stbi_load(filename.string().c_str(), &width, &height, &channels, 4);
	if (data == nullptr)
		return nullptr;
	auto staging_buffer = myvk::Buffer::CreateStaging(device, data, data + width * height * 4);
	stbi_image_free(data);

	VkExtent2D extent = {(uint32_t)width, (uint32_t)height};
	auto image = myvk::Image::CreateTexture2D(device, extent, 1, VK_FORMAT_R8G8B8A8_UNORM,
	                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
	auto image_view = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);

	// Copy buffer to image
	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(queue));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	VkBufferImageCopy copy = {.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	                                               .mipLevel = 0,
	                                               .baseArrayLayer = 0,
	                                               .layerCount = 1},
	                          .imageExtent = {extent.width, extent.height, 1}};
	command_buffer->CmdPipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, {}, {},
	                                   image->GetDstMemoryBarriers({copy}, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
	                                                               VK_IMAGE_LAYOUT_UNDEFINED,
	                                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
	command_buffer->CmdCopy(staging_buffer, image, {copy});
	command_buffer->CmdPipelineBarrier(
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, {}, {},
	    image->GetDstMemoryBarriers({copy}, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
	                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	command_buffer->End();

	auto fence = myvk::Fence::Create(device);
	command_buffer->Submit(fence);
	fence->Wait();

	return image_view;
}

struct WeightBuffers {
	myvk::Ptr<myvk::Buffer> weights, fp_weights;
};

std::mt19937 rng{std::random_device{}()};

inline static WeightBuffers load_weights(const myvk::Ptr<myvk::Queue> &queue) {
	const auto &device = queue->GetDevicePtr();

	std::normal_distribution<float> norm{0, std::sqrt(2.0f / 64.0f)};
	std::array<float, kWeightCount> fp_initial_weights;
	for (auto &w : fp_initial_weights)
		w = norm(rng);
	std::array<half_float::half, kWeightCount> initial_weights{};
	for (uint32_t i = 0; i < kWeightCount; ++i)
		initial_weights[i] = fp_initial_weights[i];

	auto weights = myvk::Buffer::Create(device, kWeightCount * sizeof(uint16_t), 0,
	                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	auto fp_weights = myvk::Buffer::Create(device, kWeightCount * sizeof(float), 0,
	                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	auto weights_staging = myvk::Buffer::CreateStaging(device, initial_weights.begin(), initial_weights.end());
	auto fp_weights_staging = myvk::Buffer::CreateStaging(device, fp_initial_weights.begin(), fp_initial_weights.end());

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(queue));
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	command_buffer->CmdCopy(weights_staging, weights, {VkBufferCopy{.size = weights->GetSize()}});
	command_buffer->CmdCopy(fp_weights_staging, fp_weights, {VkBufferCopy{.size = fp_weights->GetSize()}});
	command_buffer->End();

	auto fence = myvk::Fence::Create(device);
	command_buffer->Submit(fence);
	fence->Wait();

	return {.weights = std::move(weights), .fp_weights = std::move(fp_weights)};
}

struct GradientPass final : public myvk_rg::ComputePassBase {
	myvk::Ptr<myvk::ComputePipeline> pipeline;

	inline GradientPass(myvk_rg::Parent parent, const myvk_rg::Image &image, const myvk_rg::Buffer &weights,
	                    const myvk_rg::Buffer &gradients)
	    : myvk_rg::ComputePassBase(parent) {
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"weights"},
		                                                                                            weights);
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({1}, {"gradients"},
		                                                                                             gradients);
		AddDescriptorInput<myvk_rg::Usage::kSampledImage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {2}, {"image"}, image,
		    myvk::Sampler::Create(GetRenderGraphPtr()->GetDevicePtr(), VK_FILTER_LINEAR,
		                          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));
	}
	inline ~GradientPass() final = default;

	void CreatePipeline() {
		auto &device = GetRenderGraphPtr()->GetDevicePtr();
		auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()},
		                                                    {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2}});
		auto spv = load_spv("mlp_learning_an_image/gradient_32.spv");
		auto shader_module = myvk::ShaderModule::Create(device, (const uint32_t *)spv.data(), spv.size());
		pipeline = myvk::ComputePipeline::Create(pipeline_layout, shader_module);
	}
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
		std::uniform_int_distribution<uint32_t> dis(0);
		uint32_t pc_data[2] = {dis(rng), dis(rng)};
		command_buffer->CmdBindPipeline(pipeline);
		command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, pipeline);
		command_buffer->CmdPushConstants(pipeline->GetPipelineLayoutPtr(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
		                                 sizeof(pc_data), &pc_data);
		command_buffer->CmdDispatch(kBatchSize / 128, 1, 1);
	}
	inline auto GetGradientsOutput() const { return MakeBufferOutput({"gradients"}); }
};
struct OptimizePass final : public myvk_rg::ComputePassBase {
	myvk::Ptr<myvk::ComputePipeline> pipeline;

	inline OptimizePass(myvk_rg::Parent parent, const myvk_rg::Buffer &weights, const myvk_rg::Buffer &fp_weights,
	                    const myvk_rg::Buffer &gradients)
	    : myvk_rg::ComputePassBase(parent) {
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"weights"},
		                                                                                            weights);
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferRW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>(
		    {1}, {"fp_weights"}, fp_weights);
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({2}, {"gradients"},
		                                                                                            gradients);
	}
	inline ~OptimizePass() final = default;

	void CreatePipeline() {
		auto &device = GetRenderGraphPtr()->GetDevicePtr();
		auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()}, {});
		auto spv = load_spv("mlp_learning_an_image/optimize_32.spv");
		auto shader_module = myvk::ShaderModule::Create(device, (const uint32_t *)spv.data(), spv.size());
		pipeline = myvk::ComputePipeline::Create(pipeline_layout, shader_module);
	}
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
		command_buffer->CmdBindPipeline(pipeline);
		command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, pipeline);
		command_buffer->CmdDispatch(kWeightCount / 64, 1, 1);
	}
	inline auto GetWeightsOutput() const { return MakeBufferOutput({"weights"}); }
	inline auto GetFPWeightsOutput() const { return MakeBufferOutput({"fp_weights"}); }
};
struct TrainPass final : public myvk_rg::PassGroupBase {
	inline TrainPass(myvk_rg::Parent parent, const myvk_rg::Image &image, const myvk_rg::Buffer &weights,
	                 const myvk_rg::Buffer &fp_weights)
	    : myvk_rg::PassGroupBase(parent) {
		auto gradients = CreateResource<myvk_rg::ManagedBuffer>({"gradients"}, kWeightCount * sizeof(float));
		auto clear_pass = CreatePass<myvk_rg::BufferFillPass>({"clear_pass"}, gradients->Alias());
		auto gradient_pass = CreatePass<GradientPass>({"gradient_pass"}, image, weights, clear_pass->GetDstOutput());
		CreatePass<OptimizePass>({"optimize_pass"}, weights, fp_weights, gradient_pass->GetGradientsOutput());
	}
	inline ~TrainPass() final = default;

	inline auto GetWeightsOutput() const { return GetPass<OptimizePass>({"optimize_pass"})->GetWeightsOutput(); }
	inline auto GetFPWeightsOutput() const { return GetPass<OptimizePass>({"optimize_pass"})->GetFPWeightsOutput(); }
};
struct InferencePass final : public myvk_rg::ComputePassBase {
	myvk::Ptr<myvk::ComputePipeline> pipeline;

	inline InferencePass(myvk_rg::Parent parent, const myvk_rg::Buffer &weights) : myvk_rg::ComputePassBase(parent) {
		auto out = CreateResource<myvk_rg::ManagedImage>({"out"}, VK_FORMAT_R8G8B8A8_UNORM);
		out->SetSize2D({kWindowSize, kWindowSize});
		AddDescriptorInput<myvk_rg::Usage::kStorageBufferR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({0}, {"weights"},
		                                                                                            weights);
		AddDescriptorInput<myvk_rg::Usage::kStorageImageW, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT>({1}, {"out"},
		                                                                                           out->Alias());
	}
	inline ~InferencePass() final = default;

	void CreatePipeline() {
		auto &device = GetRenderGraphPtr()->GetDevicePtr();
		auto pipeline_layout = myvk::PipelineLayout::Create(device, {GetVkDescriptorSetLayout()}, {});
		auto spv = load_spv("mlp_learning_an_image/inference_32.spv");
		auto shader_module = myvk::ShaderModule::Create(device, (const uint32_t *)spv.data(), spv.size());
		pipeline = myvk::ComputePipeline::Create(pipeline_layout, shader_module);
	}
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const {
		command_buffer->CmdBindPipeline(pipeline);
		command_buffer->CmdBindDescriptorSets({GetVkDescriptorSet()}, pipeline);
		command_buffer->CmdDispatch(kWindowSize * kWindowSize / 128, 1, 1);
	}

	inline auto GetImageOutput() const { return MakeImageOutput({"out"}); }
};

struct RenderGraph final : public myvk_rg::RenderGraphBase {
	inline RenderGraph(const myvk::Ptr<myvk::FrameManager> &frame_manager, const myvk::Ptr<myvk::ImageView> &image_view,
	                   const myvk::Ptr<myvk::BufferBase> &weight_buffer,
	                   const myvk::Ptr<myvk::BufferBase> &fp_weight_buffer)
	    : myvk_rg::RenderGraphBase(frame_manager->GetDevicePtr()) {
		auto weights = CreateResource<myvk_rg::InputBuffer>({"weights"}, weight_buffer);
		auto fp_weights = CreateResource<myvk_rg::InputBuffer>({"fp_weights"}, fp_weight_buffer);
		auto image = CreateResource<myvk_rg::InputImage>({"image"}, image_view);
		auto swapchain_image = CreateResource<myvk_rg::SwapchainImage>({"swapchain_image"}, frame_manager);

		auto train_pass = CreatePass<TrainPass>({"train_pass"}, image->Alias(), weights->Alias(), fp_weights->Alias());
		auto inference_pass = CreatePass<InferencePass>({"inference_pass"}, train_pass->GetWeightsOutput());
		auto blit_pass = CreatePass<myvk_rg::ImageBlitPass>({"blit_pass"}, inference_pass->GetImageOutput(),
		                                                    swapchain_image->Alias(), VK_FILTER_NEAREST);

		auto imgui_pass = CreatePass<myvk_rg::ImGuiPass>({"imgui_pass"}, blit_pass->GetDstOutput());
		AddResult({"screen"}, imgui_pass->GetImageOutput());
		AddResult({"weights"}, train_pass->GetWeightsOutput());
		AddResult({"fp_weights"}, train_pass->GetFPWeightsOutput());
	}
	inline ~RenderGraph() final = default;
};

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1) {
		printf("./MLP_Learning_An_Image [filename]");
		return EXIT_FAILURE;
	}

	GLFWwindow *window = myvk::GLFWCreateWindow("MLP Learning An Image", kWindowSize, kWindowSize, false);

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
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
	    .shaderBufferFloat32AtomicAdd = VK_TRUE};
	VkPhysicalDeviceCooperativeMatrixFeaturesNV cooperative_matrix_features = {
	    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
	    .pNext = &atomic_float_features,
	    .cooperativeMatrix = VK_TRUE,
	    .cooperativeMatrixRobustBufferAccess = VK_FALSE};
	features.vk13.pNext = &cooperative_matrix_features;
	auto device = myvk::Device::Create(
	    physical_device,
	    myvk::GenericPresentQueueSelector{&generic_queue, myvk::Surface::Create(instance, window), &present_queue},
	    features,
	    {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	     VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
	     VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME});
	myvk::ImGuiInit(window, myvk::CommandPool::Create(generic_queue));

	auto image_view = load_image(generic_queue, argv[0]);
	auto [weight_buffer, fp_weight_buffer] = load_weights(generic_queue);

	auto frame_manager =
	    myvk::FrameManager::Create(generic_queue, present_queue, false, kFrameCount,
	                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	std::array<myvk::Ptr<RenderGraph>, kFrameCount> render_graphs;
	for (auto &rg : render_graphs)
		rg = myvk::MakePtr<RenderGraph>(frame_manager, image_view, weight_buffer, fp_weight_buffer);

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

		if (frame_manager->NewFrame()) {
			const auto &command_buffer = frame_manager->GetCurrentCommandBuffer();
			auto &render_graph = render_graphs[frame_manager->GetCurrentFrame()];

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
	return 0;
}
