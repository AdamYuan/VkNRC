//
// Created by adamyuan on 2/23/24.
//

#pragma once
#ifndef VKNRC_VKNRCSTATE_HPP
#define VKNRC_VKNRCSTATE_HPP

#include "Sobol.hpp"

#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>

class VkNRCState final : public myvk::DeviceObjectBase {
private:
	myvk::Ptr<myvk::Queue> m_queue_ptr;
	myvk::Ptr<myvk::ImageView> m_noise_view, m_result_view;
	Sobol m_sobol;

	void create_noise_image();
	void create_result_image(VkExtent2D extent);

public:
	inline explicit VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr) : m_queue_ptr(queue_ptr) {
		create_noise_image();
	}
	inline VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent) : VkNRCState(queue_ptr) {
		Resize(extent);
	}
	inline ~VkNRCState() final = default;

	myvk::Ptr<myvk::Buffer> MakeSobolBuffer(VkBufferUsageFlags usages) const;
	void UpdateSobolBuffer(const myvk::Ptr<myvk::Buffer> &sobol_buffer) const;

	inline void Resize(VkExtent2D extent) { create_result_image(extent); }
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const { return m_queue_ptr->GetDevicePtr(); }
};

#endif // VKNRC_VKNRCSTATE_HPP
