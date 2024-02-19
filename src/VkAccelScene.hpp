//
// Created by adamyuan on 2/19/24.
//

#pragma once
#ifndef VKNRC_VKACCELSCENE_HPP
#define VKNRC_VKACCELSCENE_HPP

#include "VkScene.hpp"

class VkAccelScene {
private:
	myvk::Ptr<VkScene> m_scene_ptr;

	std::vector<myvk::Ptr<myvk::AccelerationStructure>> m_blas_s; // Bottom-Level AS s
	myvk::Ptr<myvk::AccelerationStructure> m_tlas;                // Top-Level AS

	void create_blas_s();
	void create_tlas();

public:
	explicit VkAccelScene(const myvk::Ptr<VkScene> &scene_ptr);
};

#endif // VKNRC_VKACCELSCENE_HPP
