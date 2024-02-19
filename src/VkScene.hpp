//
// Created by adamyuan on 2/18/24.
//

#pragma once
#ifndef VKNRC_VKSCENE_HPP
#define VKNRC_VKSCENE_HPP

#include "Scene.hpp"

#include <ranges>
#include <span>

#include <myvk/AccelerationStructure.hpp>
#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>
#include <myvk/Queue.hpp>

class VkScene final : public myvk::DeviceObjectBase {
public:
	struct Material {
		glm::vec3 albedo;
		uint32_t albedo_texture_id;
	};
	using Instance = Scene::Instance;

private:
	std::vector<Instance> m_instances;
	myvk::Ptr<myvk::Queue> m_queue_ptr;
	myvk::Ptr<myvk::Buffer> m_vertex_buffer, m_vertex_index_buffer;
	myvk::Ptr<myvk::Buffer> m_texcoord_buffer, m_texcoord_index_buffer;
	myvk::Ptr<myvk::Buffer> m_material_id_buffer, m_material_buffer;
	std::vector<myvk::Ptr<myvk::ImageView>> m_textures;

	static_assert(sizeof(Material) == 4 * sizeof(float));

	void load_textures(const Scene &scene, auto &&set_material_texture_id);

	std::vector<Material> make_materials(const Scene &scene);
	void upload_buffers(const Scene &scene, std::span<const Material> materials);

public:
	inline uint32_t GetVertexCount() const { return m_vertex_buffer->GetSize() / sizeof(glm::vec3); }
	inline uint32_t GetInstanceCount() const { return m_instances.size(); }
	inline auto GetInstanceRange() const { return std::views::iota((uint32_t)0, GetInstanceCount()); }
	VkScene(const myvk::Ptr<myvk::Queue> &queue, const Scene &scene);
	VkAccelerationStructureGeometryKHR GetBLASGeometry() const;
	VkAccelerationStructureBuildRangeInfoKHR GetInstanceBLASBuildRange(uint32_t instance_id) const;

	inline const auto &GetQueuePtr() const { return m_queue_ptr; }
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const final { return m_queue_ptr->GetDevicePtr(); }
};

#endif // VKNRC_VKSCENE_HPP
