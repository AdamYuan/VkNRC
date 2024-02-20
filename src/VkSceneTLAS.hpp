//
// Created by adamyuan on 2/20/24.
//

#pragma once
#ifndef VKNRC_VKSCENETLAS_HPP
#define VKNRC_VKSCENETLAS_HPP

#include "VkSceneBLAS.hpp"

class VkSceneTLAS final : public myvk::DeviceObjectBase {
private:
	myvk::Ptr<VkScene> m_scene_ptr;
	myvk::Ptr<VkSceneBLAS> m_scene_blas_ptr;
	myvk::Ptr<myvk::AccelerationStructure> m_tlas;
	myvk::Ptr<myvk::Buffer> m_instance_buffer;
	VkAccelerationStructureInstanceKHR *m_p_instances;

	void create_instance_buffer();
	void create_tlas();

public:
	explicit VkSceneTLAS(const myvk::Ptr<VkSceneBLAS> &scene_blas_ptr);
	const myvk::Ptr<myvk::Device> &GetDevicePtr() const final { return m_scene_ptr->GetDevicePtr(); }
	inline void UpdateTLAS() { create_tlas(); }
	const auto &GetTLAS() const { return m_tlas; }
};

#endif // VKNRC_VKSCENETLAS_HPP
