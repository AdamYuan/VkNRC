//
// Created by adamyuan on 2/20/24.
//

#pragma once
#ifndef VKNRC_VKSCENEVIEW_HPP
#define VKNRC_VKSCENEVIEW_HPP

#include "VkSceneTLAS.hpp"

#include <myvk/DescriptorSet.hpp>

class VkSceneView final : public myvk::DeviceObjectBase {
private:
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<VkSceneTLAS> m_scene_tlas_ptr;

	myvk::Ptr<myvk::AccelerationStructure> m_tlas;

	myvk::Ptr<myvk::DescriptorSetLayout> m_descriptor_set_layout;
	myvk::Ptr<myvk::DescriptorSet> m_descriptor_set;

	void create_descriptor();

public:
	explicit VkSceneView(const myvk::Ptr<VkSceneTLAS> &scene_tlas_ptr);
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const { return m_scene_ptr->GetDevicePtr(); }
	inline const auto &GetDescriptorSetLayout() const { return m_descriptor_set_layout; }
};

#endif // VKNRC_VKSCENEVIEW_HPP
