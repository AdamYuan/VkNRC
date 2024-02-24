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
	VkExtent2D m_extent{};
	uint32_t m_samples{};

	void create_noise_image();
	void create_result_image();

public:
	inline VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent) : m_queue_ptr(queue_ptr) {
		create_noise_image();
		SetExtent(extent);
	}
	inline ~VkNRCState() final = default;

	myvk::Ptr<myvk::Buffer> MakeSobolBuffer(VkBufferUsageFlags usages) const;
	void UpdateSobolBuffer(const myvk::Ptr<myvk::Buffer> &sobol_buffer) const;

	inline const auto &GetNoiseImageView() const { return m_noise_view; }
	inline const auto &GetResultImageView() const { return m_result_view; }

	inline uint32_t GetSampleCount() const { return m_samples; }

	inline void Next() {
		++m_samples;
		m_sobol.Next();
	}
	inline void Reset() {
		m_samples = 0;
		m_sobol.Reset();
	}

	inline void SetExtent(VkExtent2D extent) {
		if (std::tie(m_extent.width, m_extent.height) == std::tie(extent.width, extent.height))
			return;
		m_extent = extent;
		Reset();
		create_result_image();
	}
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const { return m_queue_ptr->GetDevicePtr(); }
};

#endif // VKNRC_VKNRCSTATE_HPP
