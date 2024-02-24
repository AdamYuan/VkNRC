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
#include <myvk/CommandBuffer.hpp>
#include <myvk/ImageView.hpp>
#include <myvk/Queue.hpp>

class VkScene final : public myvk::DeviceObjectBase {
public:
	struct Material {
		glm::vec3 albedo;
		uint32_t albedo_texture_id;
	};
	struct Transform {
		glm::mat3 rotate;
		glm::vec3 translate;
	};
	static_assert(sizeof(Transform) == 12 * sizeof(float));

	using Instance = Scene::Instance;

private:
	std::vector<Instance> m_instances;
	myvk::Ptr<myvk::Queue> m_queue_ptr;
	myvk::Ptr<myvk::Buffer> m_vertex_buffer, m_vertex_index_buffer;
	myvk::Ptr<myvk::Buffer> m_texcoord_buffer, m_texcoord_index_buffer;
	myvk::Ptr<myvk::Buffer> m_material_buffer, m_material_id_buffer;
	std::vector<myvk::Ptr<myvk::ImageView>> m_textures;
	std::vector<Transform> m_transforms;

	static_assert(sizeof(Material) == 4 * sizeof(float));

	struct TexLoad {
		std::filesystem::path Scene::Material::*p_path;
		uint32_t Material::*p_id;
	};
	template <TexLoad... Loads> void load_textures(const Scene &scene, auto &&get_material);

	std::vector<Material> make_materials(const Scene &scene);
	void upload_buffers(const Scene &scene, std::span<const Material> materials);

public:
	VkScene(const myvk::Ptr<myvk::Queue> &queue, const Scene &scene);
	inline ~VkScene() final = default;

	inline uint32_t GetVertexCount() const { return m_vertex_buffer->GetSize() / sizeof(glm::vec3); }
	inline uint32_t GetInstanceCount() const { return m_instances.size(); }
	inline Instance GetInstance(uint32_t instance_id) const { return m_instances[instance_id]; }
	inline auto GetInstanceRange() const { return std::views::iota((uint32_t)0, GetInstanceCount()); }
	VkAccelerationStructureGeometryKHR GetBLASGeometry() const;
	VkAccelerationStructureBuildRangeInfoKHR GetInstanceBLASBuildRange(uint32_t instance_id) const;
	inline const auto &GetTransform(uint32_t instance_id) const { return m_transforms[instance_id]; }
	inline auto &GetTransform(uint32_t instance_id) { return m_transforms[instance_id]; }
	VkTransformMatrixKHR GetVkTransform(uint32_t instance_id) const;

	inline const auto &GetTextures() const { return m_textures; }

	inline const auto &GetQueuePtr() const { return m_queue_ptr; }
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const final { return m_queue_ptr->GetDevicePtr(); }

	inline const auto &GetVertexBuffer() const { return m_vertex_buffer; }
	inline const auto &GetVertexIndexBuffer() const { return m_vertex_index_buffer; }
	inline const auto &GetTexcoordBuffer() const { return m_texcoord_buffer; }
	inline const auto &GetTexcoordIndexBuffer() const { return m_texcoord_index_buffer; }
	inline const auto &GetMaterialBuffer() const { return m_material_buffer; }
	inline const auto &GetMaterialIDBuffer() const { return m_material_id_buffer; }

	myvk::Ptr<myvk::Buffer> MakeTransformBuffer(VkBufferUsageFlags usages) const;
	void UpdateTransformBuffer(const myvk::Ptr<myvk::Buffer> &transform_buffer) const;
};

#endif // VKNRC_VKSCENE_HPP
