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
	inline static constexpr uint32_t kNNHiddenLayers = 5, kNNWidth = 64;
	myvk::Ptr<myvk::Queue> m_queue_ptr;
	myvk::Ptr<myvk::Buffer> m_weights;
	myvk::Ptr<myvk::ImageView> m_result_view;
	VkExtent2D m_extent{};
	uint32_t m_samples{};

	void create_result_image();
	void create_weight_buffer();

public:
	inline VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent) : m_queue_ptr(queue_ptr) {
		SetExtent(extent);
	}
	inline ~VkNRCState() final = default;

	inline const auto &GetResultImageView() const { return m_result_view; }

	inline uint32_t GetSampleCount() const { return m_samples; }

	inline void Next() { ++m_samples; }
	inline void Reset() { m_samples = 0; }

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
