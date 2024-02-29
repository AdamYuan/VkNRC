//
// Created by adamyuan on 2/23/24.
//

#pragma once
#ifndef VKNRC_VKNRCSTATE_HPP
#define VKNRC_VKNRCSTATE_HPP

#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>
#include <random>

class VkNRCState final : public myvk::DeviceObjectBase {
private:
	myvk::Ptr<myvk::Queue> m_queue_ptr;
	myvk::Ptr<myvk::Buffer> m_weights;
	myvk::Ptr<myvk::ImageView> m_result_view;
	VkExtent2D m_extent{};
	uint32_t m_samples{}, m_seed{};
	std::mt19937 m_rng{std::random_device{}()};

	void create_result_image();
	void create_weight_buffer();

public:
	inline VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent) : m_queue_ptr(queue_ptr) {
		SetExtent(extent);
		create_weight_buffer();
	}
	inline ~VkNRCState() final = default;

	inline const auto &GetResultImageView() const { return m_result_view; }

	inline uint32_t GetSampleCount() const { return m_samples; }
	inline uint32_t GetSeed() const { return m_seed; }

	inline void Next() {
		++m_samples;
		m_seed = std::uniform_int_distribution<uint32_t>{0, 0xFFFFFFFFu}(m_rng);
	}
	inline void Reset() { m_samples = 0; }

	inline void SetExtent(VkExtent2D extent) {
		if (std::tie(m_extent.width, m_extent.height) == std::tie(extent.width, extent.height))
			return;
		m_extent = extent;
		Reset();
		create_result_image();
	}
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const { return m_queue_ptr->GetDevicePtr(); }
	static VkDeviceSize GetEvalRecordBufferSize(VkExtent2D extent);
	static VkDeviceSize GetTrainBatchRecordBufferSize();
	static uint32_t GetTrainBatchCount();
};

#endif // VKNRC_VKNRCSTATE_HPP
