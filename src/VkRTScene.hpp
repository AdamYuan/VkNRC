//
// Created by adamyuan on 2/18/24.
//

#pragma once
#ifndef VKNRC_VKRTSCENE_HPP
#define VKNRC_VKRTSCENE_HPP

#include "Scene.hpp"

#include <span>

#include <myvk/ASBuffer.hpp>
#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>
#include <myvk/Queue.hpp>

class VkRTScene {
private:
	myvk::Ptr<myvk::ASBuffer> m_vertex_buffer, m_vertex_index_buffer;
	myvk::Ptr<myvk::Buffer> m_texcoord_buffer, m_texcoord_index_buffer;
	myvk::Ptr<myvk::Buffer> m_material_id_buffer, m_material_buffer;
	std::vector<myvk::Ptr<myvk::ImageView>> m_textures;

	struct Material {
		glm::vec3 albedo;
		uint32_t albedo_texture_id;
	};
	static_assert(sizeof(Material) == 4 * sizeof(float));

	void load_textures(const myvk::Ptr<myvk::Queue> &queue, const Scene &scene, auto &&set_material_texture_id);

	std::vector<Material> make_materials(const myvk::Ptr<myvk::Queue> &queue, const Scene &scene);
	void upload_buffers(const myvk::Ptr<myvk::Queue> &queue, const Scene &scene, std::span<const Material> materials);

public:
	VkRTScene(const myvk::Ptr<myvk::Queue> &queue, const Scene &scene);
};

#endif // VKNRC_VKRTSCENE_HPP
