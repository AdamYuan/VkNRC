//
// Created by adamyuan on 2/18/24.
//

#pragma once
#ifndef VKNRC_SCENE_HPP
#define VKNRC_SCENE_HPP

#include <filesystem>
#include <glm/glm.hpp>
#include <vector>

class Scene {
public:
	struct Material {
		glm::vec3 diffuse;
		std::filesystem::path diffuse_texture;
		glm::vec3 specular;
		std::filesystem::path specular_texture;
		float metallic, roughness, ior;
	};
	struct Instance {
		uint32_t first_index, index_count;
	};

private:
	static_assert(sizeof(glm::vec2) == 2 * sizeof(float));
	static_assert(sizeof(glm::vec3) == 3 * sizeof(float));

	std::vector<glm::vec3> m_vertices;
	std::vector<glm::vec2> m_texcoords;
	std::vector<Material> m_materials;

	std::vector<Instance> m_instances;
	std::vector<uint32_t> m_vertex_indices, m_texcoord_indices; // Indices on each triangle vertex
	std::vector<uint32_t> m_material_ids; // Material IDs on each triangle

	bool obj_load(const std::filesystem::path &filename, auto &&make_instance);
	void obj_single_instance(auto &&shapes);
	void obj_shape_instance(auto &&shapes);
	void obj_sah_shape_instance(auto &&shapes, uint32_t max_level);

public:
	static Scene LoadOBJSingleInstance(const std::filesystem::path &filename);
	static Scene LoadOBJShapeInstance(const std::filesystem::path &filename);
	static Scene LoadOBJShapeInstanceSAH(const std::filesystem::path &filename, uint32_t max_level);

	inline bool Empty() const { return m_instances.empty(); }
	inline explicit operator bool() const { return !Empty(); }

	inline const auto &GetVertices() const { return m_vertices; }
	inline const auto &GetTexcoords() const { return m_texcoords; }
	inline const auto &GetVertexIndices() const { return m_vertex_indices; }
	inline const auto &GetTexcoordIndices() const { return m_texcoord_indices; }

	inline const auto &GetMaterials() const { return m_materials; }
	inline const auto &GetInstances() const { return m_instances; }
	inline const auto &GetMaterialIDs() const { return m_material_ids; }
};

#endif // VKNRC_SCENE_HPP
