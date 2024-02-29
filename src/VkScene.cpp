//
// Created by adamyuan on 2/18/24.
//

#include "VkScene.hpp"

#include <future>
#include <thread>

#include <glm/gtc/matrix_transform.hpp>
#include <myvk/Image.hpp>
#include <spdlog/spdlog.h>
#include <stb_image.h>

VkScene::VkScene(const myvk::Ptr<myvk::Queue> &queue, const Scene &scene)
    : m_queue_ptr(queue), m_instances(scene.GetInstances()),
      m_transforms(scene.GetInstances().size(), {.rotate = glm::identity<glm::mat3>()}) {
	upload_buffers(scene, make_materials(scene));
}

VkTransformMatrixKHR VkScene::GetVkTransform(uint32_t instance_id) const {
	const auto &transform = m_transforms[instance_id];
	const auto &t = transform.translate;
	const auto &r = transform.rotate;
	return VkTransformMatrixKHR{
	    r[0][0], r[1][0], r[2][0], t[0], //
	    r[0][1], r[1][1], r[2][1], t[1], //
	    r[0][2], r[1][2], r[2][2], t[2], //
	};
}

VkAccelerationStructureGeometryKHR VkScene::GetBLASGeometry() const {
	VkAccelerationStructureGeometryDataKHR geometry_data = {
	    .triangles = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
	                  .pNext = nullptr,
	                  .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
	                  .vertexData = {.deviceAddress = m_vertex_buffer->GetDeviceAddress()},
	                  .vertexStride = sizeof(glm::vec3),
	                  .maxVertex = GetVertexCount() - 1u,
	                  .indexType = VK_INDEX_TYPE_UINT32,
	                  .indexData = {.deviceAddress = m_vertex_index_buffer->GetDeviceAddress()},
	                  .transformData = {.deviceAddress = 0}}};

	VkAccelerationStructureGeometryKHR geometry = {.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
	                                               .pNext = nullptr,
	                                               .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
	                                               .geometry = geometry_data,
	                                               .flags = VK_GEOMETRY_OPAQUE_BIT_KHR};
	return geometry;
}

VkAccelerationStructureBuildRangeInfoKHR VkScene::GetInstanceBLASBuildRange(uint32_t instance_id) const {
	const auto &instance = m_instances[instance_id];
	VkAccelerationStructureBuildRangeInfoKHR build_range = {
	    .primitiveCount = instance.index_count / 3u,
	    .primitiveOffset = instance.first_index * (uint32_t)sizeof(uint32_t),
	    .firstVertex = 0,
	    .transformOffset = 0,
	};
	return build_range;
}

VkDeviceSize VkScene::GetTransformBufferSize() const { return GetInstanceCount() * sizeof(Transform); }
void VkScene::UpdateTransformBuffer(void *p_mapped) const {
	auto *ts = static_cast<glm::mat3x4 *>(p_mapped);
	for (uint32_t i = 0; i < GetInstanceCount(); ++i) {
		auto &t = GetTransform(i);
		// Transpose it to avoid alignment issue
		ts[i] = glm::transpose(glm::mat4x3{t.rotate[0], t.rotate[1], t.rotate[2], t.translate});
	}
}

void VkScene::upload_buffers(const Scene &scene, std::span<const Material> materials) {
	const auto &device = GetDevicePtr();

	auto vertex_staging_buffer =
	    myvk::Buffer::CreateStaging(device, scene.GetVertices().begin(), scene.GetVertices().end());
	auto texcoord_staging_buffer =
	    myvk::Buffer::CreateStaging(device, scene.GetTexcoords().begin(), scene.GetTexcoords().end());
	auto vertex_index_staging_buffer =
	    myvk::Buffer::CreateStaging(device, scene.GetVertexIndices().begin(), scene.GetVertexIndices().end());
	auto texcoord_index_staging_buffer =
	    myvk::Buffer::CreateStaging(device, scene.GetTexcoordIndices().begin(), scene.GetTexcoordIndices().end());
	auto material_id_staging_buffer =
	    myvk::Buffer::CreateStaging(device, scene.GetMaterialIDs().begin(), scene.GetMaterialIDs().end());
	auto material_staging_buffer = myvk::Buffer::CreateStaging(device, materials.begin(), materials.end());

	auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
	const auto cmd_create_copy = [&]<typename Buffer_T = myvk::Buffer>(
	    const myvk::Ptr<myvk::Buffer> &src, myvk::Ptr<Buffer_T> *p_dst, VkBufferUsageFlags base_usages) {
		*p_dst = Buffer_T::Create(device, src->GetSize(), 0,
		                          base_usages | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		command_buffer->CmdCopy(src, *p_dst, {{0, 0, src->GetSize()}});
	};
	command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	constexpr VkBufferUsageFlags kASInputUsages = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
	                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

	cmd_create_copy(vertex_staging_buffer, &m_vertex_buffer, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | kASInputUsages);
	cmd_create_copy(vertex_index_staging_buffer, &m_vertex_index_buffer,
	                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | kASInputUsages);
	cmd_create_copy(texcoord_staging_buffer, &m_texcoord_buffer, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	cmd_create_copy(texcoord_index_staging_buffer, &m_texcoord_index_buffer, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	cmd_create_copy(material_id_staging_buffer, &m_material_id_buffer, 0);
	cmd_create_copy(material_staging_buffer, &m_material_buffer, 0);
	command_buffer->End();

	auto fence = myvk::Fence::Create(device);
	command_buffer->Submit(fence);
	fence->Wait();
}

std::vector<VkScene::Material> VkScene::make_materials(const Scene &scene) {
	std::vector<Material> materials;
	materials.reserve(scene.GetMaterials().size());
	for (const auto &material : scene.GetMaterials())
		materials.push_back({
		    .diffuse = material.diffuse,
		    .diffuse_texture_id = -1u,
		    .specular = material.specular,
		    .specular_texture_id = -1u,
		    .metallic = material.metallic,
		    .roughness = material.roughness,
		    .ior = material.ior,
		});
	load_textures<TexLoad{&Scene::Material::diffuse_texture, &Material::diffuse_texture_id},
	              TexLoad{&Scene::Material::specular_texture, &Material::specular_texture_id}>(
	    scene, [&](uint32_t material_id) -> Material & { return materials[material_id]; });
	return materials;
}

template <VkScene::TexLoad... Loads> void VkScene::load_textures(const Scene &scene, auto &&get_material) {
	const auto &device = GetDevicePtr();

	std::unordered_map<std::filesystem::path, uint32_t> path_id_map;
	std::vector<std::filesystem::path> paths;

	for (uint32_t mat_id = 0; mat_id < scene.GetMaterials().size(); ++mat_id)
		(
		    [&]() {
			    const auto &path = scene.GetMaterials()[mat_id].*Loads.p_path;
			    auto &dst_mat = get_material(mat_id);
			    auto it = path_id_map.find(path);
			    if (it != path_id_map.end()) {
				    dst_mat.*Loads.p_id = it->second;
				    return;
			    }
			    uint32_t path_id = paths.size();
			    path_id_map[path] = path_id;
			    paths.push_back(path);
			    dst_mat.*Loads.p_id = path_id;
		    }(),
		    ...);

	std::vector<uint32_t> path_id_texture_ids(paths.size());

	m_textures.resize(paths.size());

	std::atomic_uint32_t atomic_path_id{0}, atomic_texture_id{0};

	const auto load_texture = [&]() -> void {
		auto command_pool = myvk::CommandPool::Create(m_queue_ptr);
		while (true) {
			uint32_t path_id = atomic_path_id.fetch_add(1, std::memory_order_relaxed);
			if (path_id >= paths.size())
				break;

			const auto &path = paths[path_id];

			// Load texture data from file
			int width, height, channels;
			stbi_uc *data;
			if (path.empty() || (data = stbi_load(path.string().c_str(), &width, &height, &channels, 4)) == nullptr) {
				if (!path.empty())
					spdlog::warn("Texture {} fails to load", path.string());
				path_id_texture_ids[path_id] = -1;
				continue;
			}
			// Create staging buffer
			auto staging_buffer = myvk::Buffer::CreateStaging(device, data, data + width * height * 4);
			// Free texture data
			stbi_image_free(data);

			// Assign TextureID
			uint32_t texture_id = atomic_texture_id.fetch_add(1, std::memory_order_relaxed);
			path_id_texture_ids[path_id] = texture_id;

			// Create Image and ImageView
			VkExtent2D extent = {(uint32_t)width, (uint32_t)height};
			auto image = myvk::Image::CreateTexture2D(device, extent, 1, VK_FORMAT_R8G8B8A8_SRGB,
			                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
			m_textures[texture_id] = myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D);

			// Copy buffer to image
			auto command_buffer = myvk::CommandBuffer::Create(command_pool);
			command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			VkBufferImageCopy copy = {.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			                                               .mipLevel = 0,
			                                               .baseArrayLayer = 0,
			                                               .layerCount = 1},
			                          .imageExtent = {extent.width, extent.height, 1}};
			command_buffer->CmdPipelineBarrier(
			    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, {}, {},
			    image->GetDstMemoryBarriers({copy}, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
			                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
			command_buffer->CmdCopy(staging_buffer, image, {copy});
			command_buffer->CmdPipelineBarrier(
			    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, {}, {},
			    image->GetDstMemoryBarriers({copy}, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
			                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
			command_buffer->End();

			auto fence = myvk::Fence::Create(device);
			command_buffer->Submit(fence);
			fence->Wait();

			spdlog::info("Texture {} loaded", path.string());
		}
	};

	{
		std::vector<std::future<void>> futures(std::thread::hardware_concurrency() - 1u);
		for (auto &future : futures)
			future = std::async(load_texture);
		load_texture();
	}

	// Pop empty textures
	while (!m_textures.empty() && m_textures.back() == nullptr)
		m_textures.pop_back();

	// Set Texture ID
	for (uint32_t mat_id = 0; mat_id < scene.GetMaterials().size(); ++mat_id)
		(
		    [&]() {
			    auto &dst_mat = get_material(mat_id);
			    dst_mat.*Loads.p_id = path_id_texture_ids[dst_mat.*Loads.p_id];
		    }(),
		    ...);

	// If Textures are empty, insert an empty one
	if (m_textures.empty()) {
		auto image =
		    myvk::Image::CreateTexture2D(device, {1, 1}, 1, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT);
		m_textures.push_back(myvk::ImageView::Create(image, VK_IMAGE_VIEW_TYPE_2D));

		// Copy buffer to image
		auto command_buffer = myvk::CommandBuffer::Create(myvk::CommandPool::Create(m_queue_ptr));
		command_buffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		command_buffer->CmdPipelineBarrier(
		    VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, {}, {},
		    {image->GetMemoryBarrier(VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED,
		                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)});
		command_buffer->End();

		auto fence = myvk::Fence::Create(device);
		command_buffer->Submit(fence);
		fence->Wait();
	}
}
