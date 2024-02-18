//
// Created by adamyuan on 2/18/24.
//

#include "Scene.hpp"
#include <algorithm>
#include <iostream>
#include <ranges>

#include "tiny_obj_loader.h"

Scene Scene::LoadOBJ(const std::filesystem::path &filename) {
	Scene scene = {};

	tinyobj::ObjReaderConfig reader_config;
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(filename, reader_config)) {
		if (!reader.Error().empty())
			std::cerr << "TinyObjReader: " << reader.Error() << std::endl;
		return {};
	}
	if (!reader.Warning().empty())
		std::cout << "TinyObjReader: " << reader.Warning() << std::endl;

	const tinyobj::attrib_t &attrib = reader.GetAttrib();
	const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();
	const std::vector<tinyobj::material_t> &materials = reader.GetMaterials();

	for (const tinyobj::shape_t &shape : shapes) {
		// Push IndexBegin
		scene.m_instances.push_back({.index_begin = (uint32_t)scene.m_vertex_indices.size()});

		// Check Non-triangle faces
		if (std::ranges::any_of(shape.mesh.num_face_vertices,
		                        [](unsigned num_face_vertex) { return num_face_vertex != 3; })) {
			std::cerr << "Non-triangle faces not supported" << std::endl;
			return {};
		}

		assert(shape.mesh.num_face_vertices.size() == shape.mesh.material_ids.size());

		// Push Per-Triangle IDs
		for (const auto &material_id : shape.mesh.material_ids) {
			scene.m_material_ids.push_back(material_id);
		}

		// Push Per-Vertex Indices
		for (tinyobj::index_t index : shape.mesh.indices) {
			scene.m_vertex_indices.push_back(index.vertex_index);
			scene.m_texcoord_indices.push_back(index.texcoord_index);
		}

		// Set IndexEnd
		scene.m_instances.back().index_end = scene.m_vertex_indices.size();
	}

	// Read Vertices
	scene.m_vertices.resize(attrib.vertices.size() / 3);
	std::ranges::copy(attrib.vertices, (float *)scene.m_vertices.data());

	// Read Texcoords
	scene.m_texcoords.resize(attrib.texcoords.size() / 2);
	std::ranges::copy(attrib.texcoords, (float *)scene.m_texcoords.data());
	for (auto &texcoord : scene.m_texcoords)
		texcoord.y = -texcoord.y; // Flip Y-Coord

	// Read Materials
	scene.m_materials.reserve(materials.size());
	for (const auto &material : materials) {
		scene.m_materials.push_back({
		    .albedo = {material.diffuse[0], material.diffuse[1], material.diffuse[2]},
		    .albedo_texture = material.diffuse_texname,
		});
	}

	return scene;
}
