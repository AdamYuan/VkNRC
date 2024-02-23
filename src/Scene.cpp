//
// Created by adamyuan on 2/18/24.
//

#include "Scene.hpp"

#include <algorithm>

#include "AABB.hpp"
#include <spdlog/spdlog.h>
#include <tiny_obj_loader.h>

bool Scene::obj_load(const std::filesystem::path &filename, auto &&make_instance) {
	tinyobj::ObjReaderConfig reader_config;
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filename.string(), reader_config)) {
		if (!reader.Error().empty())
			spdlog::error("TinyObjReader: {}", reader.Error());
		return false;
	}
	if (!reader.Warning().empty())
		spdlog::warn("TinyObjReader: {}", reader.Warning());

	const tinyobj::attrib_t &attrib = reader.GetAttrib();
	const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();
	const std::vector<tinyobj::material_t> &materials = reader.GetMaterials();

	// Read Vertices
	m_vertices.resize(attrib.vertices.size() / 3);
	std::ranges::copy(attrib.vertices, (float *)m_vertices.data());
	{
		AABB obj_aabb{};
		for (const auto &vertex : m_vertices)
			obj_aabb.Expand(vertex);
		glm::vec3 obj_extent = obj_aabb.GetExtent(), obj_center = obj_aabb.GetCenter();
		float inv_max_extent = 2.0f / glm::max(obj_extent.x, glm::max(obj_extent.y, obj_extent.z));
		for (auto &vertex : m_vertices)
			vertex = (vertex - obj_center) * inv_max_extent;
	}

	// Read Texcoords
	m_texcoords.resize(attrib.texcoords.size() / 2);
	std::ranges::copy(attrib.texcoords, (float *)m_texcoords.data());
	for (auto &texcoord : m_texcoords)
		texcoord.y = -texcoord.y; // Flip Y-Coord

	// Read Materials
	m_materials.reserve(materials.size());
	for (const auto &material : materials)
		m_materials.push_back({
		    .albedo = {material.diffuse[0], material.diffuse[1], material.diffuse[2]},
		    .albedo_texture = material.diffuse_texname.empty() ? std::filesystem::path{}
		                                                       : filename.parent_path() / material.diffuse_texname,
		});

	for (const tinyobj::shape_t &shape : shapes)
		// Check Non-triangle faces
		if (std::ranges::any_of(shape.mesh.num_face_vertices,
		                        [](unsigned num_face_vertex) { return num_face_vertex != 3; })) {
			spdlog::error("Non-triangle faces not supported");
			return false;
		}

	make_instance(shapes);

	return true;
}

void Scene::obj_single_instance(auto &&shapes) {
	for (const tinyobj::shape_t &shape : shapes) {
		for (tinyobj::index_t index : shape.mesh.indices) {
			m_vertex_indices.push_back(index.vertex_index);
			m_texcoord_indices.push_back(index.texcoord_index);
		}
		for (uint32_t material_id : shape.mesh.material_ids)
			m_material_ids.push_back(material_id);
	}
	m_instances.push_back({.first_index = 0, .index_count = (uint32_t)m_vertex_indices.size()});
}
void Scene::obj_shape_instance(auto &&shapes) {
	for (const tinyobj::shape_t &shape : shapes) {
		m_instances.push_back(
		    {.first_index = (uint32_t)m_vertex_indices.size(), .index_count = (uint32_t)shape.mesh.indices.size()});
		for (tinyobj::index_t index : shape.mesh.indices) {
			m_vertex_indices.push_back(index.vertex_index);
			m_texcoord_indices.push_back(index.texcoord_index);
		}
		for (uint32_t material_id : shape.mesh.material_ids)
			m_material_ids.push_back(material_id);
	}
}
namespace obj_sah_shape_instance {
struct SAHSplit {
	int axis{};
	uint32_t left_size{};
	float sah{std::numeric_limits<float>::infinity()};
};
struct Reference {
	AABB aabb;
	std::vector<uint32_t> vertex_indices, texcoord_indices, material_ids;

	inline uint32_t GetTriangleCount() const { return material_ids.size(); }
	template <int Axis> static AABB _AxisSplitSAH(std::span<Reference> refs, SAHSplit *p_split) {
		// assert(refs.size() >= 2);
		std::ranges::sort(refs, [](const Reference &l, const Reference &r) {
			return l.aabb.GetDimCenter<Axis>() < r.aabb.GetDimCenter<Axis>();
		});
		std::vector<AABB> right_aabbs(refs.size());
		right_aabbs.back() = refs.back().aabb;
		for (uint32_t i = refs.size() - 2; i >= 1; --i)
			right_aabbs[i] = AABB(refs[i].aabb, right_aabbs[i + 1]);

		AABB left_aabb = refs.front().aabb;
		for (uint32_t left_size = 1; left_size < refs.size(); ++left_size) {
			const AABB &right_aabb = right_aabbs[left_size];
			float sah =
			    float(left_size) * left_aabb.GetHalfArea() + float(refs.size() - left_size) * right_aabb.GetHalfArea();

			if (sah < p_split->sah) {
				*p_split = {
				    .axis = Axis,
				    .left_size = left_size,
				    .sah = sah,
				};
			}
			left_aabb.Expand(refs[left_size].aabb);
		}
		return AABB{left_aabb}; // left_aabb is the full AABB at last
	}
	static std::tuple<std::span<Reference>, std::span<Reference>> SplitSAH(std::span<Reference> refs) {
		SAHSplit sah_split{};
		_AxisSplitSAH<0>(refs, &sah_split);
		_AxisSplitSAH<1>(refs, &sah_split);
		AABB aabb = _AxisSplitSAH<2>(refs, &sah_split);

		if (sah_split.sah >= aabb.GetHalfArea() * float(refs.size()))
			return {{}, {}};

		std::span<Reference> left_refs{refs.begin(), refs.begin() + sah_split.left_size};
		std::span<Reference> right_refs{refs.begin() + sah_split.left_size, refs.end()};
		std::ranges::nth_element(refs, left_refs.end() - 1,
		                         [axis = sah_split.axis](const Reference &l, const Reference &r) {
			                         return l.aabb.GetDimCenter(axis) < r.aabb.GetDimCenter(axis);
		                         });
		return {left_refs, right_refs};
	}
	static bool CanSplit(std::span<const Reference> refs) {
		return refs.size() > 1 || refs.size() == 1 && refs[0].GetTriangleCount() > 1;
	}
};
} // namespace obj_sah_shape_instance
void Scene::obj_sah_shape_instance(auto &&shapes, uint32_t max_level) {
	using obj_sah_shape_instance::Reference;

	std::vector<Reference> references;
	references.reserve(shapes.size());

	for (const tinyobj::shape_t &shape : shapes) {
		references.emplace_back();
		auto &ref = references.back();
		for (tinyobj::index_t index : shape.mesh.indices) {
			ref.vertex_indices.push_back(index.vertex_index);
			ref.aabb.Expand(GetVertices()[index.vertex_index]);
			ref.texcoord_indices.push_back(index.texcoord_index);
		}
		for (uint32_t material_id : shape.mesh.material_ids)
			ref.material_ids.push_back(material_id);
	}

	const auto ref_to_triangles = [this](const Reference &ref) -> std::vector<Reference> {
		std::vector<Reference> tri_refs(ref.material_ids.size());
		for (uint32_t i = 0; i < ref.material_ids.size(); ++i) {
			auto &tri_ref = tri_refs[i];
			tri_ref.material_ids = {ref.material_ids[i]};
			tri_ref.vertex_indices = {
			    ref.vertex_indices[i * 3 + 0],
			    ref.vertex_indices[i * 3 + 1],
			    ref.vertex_indices[i * 3 + 2],
			};
			auto &vertices = m_vertices;
			tri_ref.aabb = {vertices[tri_ref.vertex_indices[0]], vertices[tri_ref.vertex_indices[1]]};
			tri_ref.aabb.Expand(vertices[tri_ref.vertex_indices[2]]);
			tri_ref.texcoord_indices = {
			    ref.texcoord_indices[i * 3 + 0],
			    ref.texcoord_indices[i * 3 + 1],
			    ref.texcoord_indices[i * 3 + 2],
			};
		}
		return tri_refs;
	};

	const auto ref_push = [this](std::span<const Reference> refs) -> void {
		uint32_t first_index = m_vertex_indices.size(), index_count = 0;
		for (const Reference &ref : refs) {
			index_count += ref.vertex_indices.size();
			m_vertex_indices.insert(m_vertex_indices.end(), ref.vertex_indices.begin(), ref.vertex_indices.end());
			m_texcoord_indices.insert(m_texcoord_indices.end(), ref.texcoord_indices.begin(),
			                          ref.texcoord_indices.end());
			m_material_ids.insert(m_material_ids.end(), ref.material_ids.begin(), ref.material_ids.end());
		}
		m_instances.push_back({first_index, index_count});
	};

	const auto partition_sah = [&](std::span<Reference> refs) -> void {
		const auto partition_sah_impl = [&](std::span<Reference> refs, uint32_t level, auto &&partition_sah) -> void {
			if (refs.empty())
				return;
			if (level == max_level || !Reference::CanSplit(refs))
				return ref_push(refs);
			if (refs.size() == 1) {
				std::vector<Reference> tri_refs = ref_to_triangles(refs[0]);
				return partition_sah(tri_refs, level, partition_sah);
			}
			auto [left_refs, right_refs] = Reference::SplitSAH(refs);
			if (left_refs.empty() || right_refs.empty()) // No good SAH partition
				return ref_push(refs);
			partition_sah(left_refs, level + 1, partition_sah);
			partition_sah(right_refs, level + 1, partition_sah);
		};
		partition_sah_impl(refs, 0, partition_sah_impl);
	};
	partition_sah(references);
}

Scene Scene::LoadOBJSingleInstance(const std::filesystem::path &filename) {
	Scene scene = {};
	if (!scene.obj_load(filename, [&](auto &&shapes) { scene.obj_single_instance(shapes); }))
		return {};
	return scene;
}
Scene Scene::LoadOBJShapeInstance(const std::filesystem::path &filename) {
	Scene scene = {};
	if (!scene.obj_load(filename, [&](auto &&shapes) { scene.obj_shape_instance(shapes); }))
		return {};
	return scene;
}
Scene Scene::LoadOBJShapeInstanceSAH(const std::filesystem::path &filename, uint32_t max_level) {
	Scene scene = {};
	if (!scene.obj_load(filename, [&](auto &&shapes) { scene.obj_sah_shape_instance(shapes, max_level); }))
		return {};
	return scene;
}
