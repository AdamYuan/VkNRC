//
// Created by adamyuan on 2/19/24.
//

#pragma once
#ifndef VKNRC_VKSCENEBLAS_HPP
#define VKNRC_VKSCENEBLAS_HPP

#include "VkScene.hpp"
#include <myvk/CommandBuffer.hpp>

class VkSceneBLAS final : public myvk::DeviceObjectBase {
private:
	myvk::Ptr<VkScene> m_scene_ptr;
	std::vector<myvk::Ptr<myvk::AccelerationStructure>> m_blas_s;

	void create_blas_s();

public:
	explicit VkSceneBLAS(const myvk::Ptr<VkScene> &scene_ptr);
	const myvk::Ptr<VkScene> &GetScenePtr() const { return m_scene_ptr; }
	const auto &GetBLASs() const { return m_blas_s; }
	const myvk::Ptr<myvk::Device> &GetDevicePtr() const final { return m_scene_ptr->GetDevicePtr(); }
};

#endif // VKNRC_VKSCENEBLAS_HPP
