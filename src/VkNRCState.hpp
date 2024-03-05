//
// Created by adamyuan on 2/23/24.
//

#pragma once
#ifndef VKNRC_VKNRCSTATE_HPP
#define VKNRC_VKNRCSTATE_HPP

#include <glm/glm.hpp>
#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>
#include <random>

class VkNRCState final : public myvk::DeviceObjectBase {
private:
	inline static constexpr uint32_t kNNHiddenLayers = 5, kNNWidth = 64, kPaddedNNOutWidth = 16,
	                                 kTrainBatchSize = 16384, kTrainBatchCount = 4;
	inline static constexpr glm::vec2 kAdamBeta{0.9, 0.999};

	myvk::Ptr<myvk::Queue> m_queue_ptr;
	myvk::Ptr<myvk::Buffer> m_weights, m_adam_mv;
	myvk::Ptr<myvk::ImageView> m_result_view;
	VkExtent2D m_extent{};
	uint32_t m_samples{}, m_seed{};
	glm::vec2 m_adam_t;
	std::mt19937 m_rng{std::random_device{}()};

	void create_result_image();
	void create_weight_buffer();
	void create_adam_buffer();

public:
	inline VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent)
	    : m_queue_ptr(queue_ptr), m_adam_t(kAdamBeta) {
		SetExtent(extent);
		create_weight_buffer();
		create_adam_buffer();
	}
	inline ~VkNRCState() final = default;

	inline const auto &GetResultImageView() const { return m_result_view; }
	inline const auto &GetWeightBuffer() const { return m_weights; }
	inline const auto &GetAdamMVBuffer() const { return m_adam_mv; }

	inline uint32_t GetSampleCount() const { return m_samples; }
	inline uint32_t GetSeed() const { return m_seed; }
	std::array<glm::vec2, kTrainBatchCount> GetAdamTs() {
		std::array<glm::vec2, kTrainBatchCount> ret = {m_adam_t};
		for (std::size_t i = 1; i < kTrainBatchCount; ++i)
			ret[i] = ret[i - 1] * kAdamBeta;
		return ret;
	}

	inline void Next() {
		++m_samples;
		m_seed = std::uniform_int_distribution<uint32_t>{0, 0xFFFFFFFFu}(m_rng);
		for (std::size_t i = 0; i < kTrainBatchCount; ++i)
			m_adam_t *= kAdamBeta;
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

	static constexpr uint32_t GetTrainBatchCount() { return kTrainBatchCount; }
	static constexpr uint32_t GetTrainBatchSize() { return kTrainBatchSize; }
	static constexpr uint32_t GetWeightCount() {
		return kNNWidth * kNNWidth * kNNHiddenLayers + kNNWidth * kPaddedNNOutWidth;
	}
	static constexpr glm::vec2 GetAdamBeta() { return kAdamBeta; }
};

#endif // VKNRC_VKNRCSTATE_HPP
