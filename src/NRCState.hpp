//
// Created by adamyuan on 2/23/24.
//

#pragma once
#ifndef VKNRC_NRCSTATE_HPP
#define VKNRC_NRCSTATE_HPP

#include <glm/glm.hpp>
#include <half.hpp>
#include <myvk/Buffer.hpp>
#include <myvk/ExportBuffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>
#include <random>
#include <span>

class NRCState {
public:
	enum Method { kNone, kNRC, kCache };

private:
	inline static constexpr float kDefaultTrainProbability = 0.03f;
	inline static constexpr uint32_t kNNHiddenLayers = 5, kNNWidth = 64, kNNOutWidth = 3, kTrainBatchSize = 16384,
	                                 kTrainBatchCount = 4;
	inline static constexpr uint32_t kNNWeighCount = kNNWidth * kNNWidth * kNNHiddenLayers + kNNWidth * kNNOutWidth;

	uint32_t m_seed{};
	std::mt19937 m_rng{std::random_device{}()};
	Method m_left_method{kNRC}, m_right_method{kNRC};
	bool m_accumulate{false};
	uint32_t m_accumulate_count{0};
	float m_train_probability{kDefaultTrainProbability};

public:
	inline Method GetLeftMethod() const { return m_left_method; }
	inline Method GetRightMethod() const { return m_right_method; }
	inline bool IsAccumulate() const { return m_accumulate; }
	inline uint32_t GetAccumulateCount() const { return m_accumulate_count; }
	inline float GetTrainProbability() const { return m_train_probability; }

	inline void SetLeftMethod(Method method) { m_left_method = method; }
	inline void SetRightMethod(Method method) { m_right_method = method; }
	inline void SetAccumulate(bool accumulate) {
		m_accumulate = accumulate;
		if (!accumulate)
			m_accumulate_count = 0;
	}
	inline void ResetAccumulateCount() { m_accumulate_count = 0; }
	inline void SetTrainProbability(float train_probability) { m_train_probability = train_probability; }

	inline uint32_t GetSeed() const { return m_seed; }

	inline void NextFrame() {
		if (m_accumulate)
			++m_accumulate_count;
		m_seed = std::uniform_int_distribution<uint32_t>{0}(m_rng);
	}

	static constexpr uint32_t GetInferenceCount(VkExtent2D extent, uint32_t block_count) {
		uint32_t count = extent.width * extent.height + kTrainBatchSize * kTrainBatchCount;
		count = (count + block_count - 1u) / block_count;
		count *= block_count;
		return count;
	}
	static constexpr uint32_t GetTrainBatchCount() { return kTrainBatchCount; }
	static constexpr uint32_t GetTrainBatchSize() { return kTrainBatchSize; }
	static constexpr uint32_t GetWeightCount() { return kNNWeighCount; }
	static constexpr float GetDefaultTrainProbability() { return kDefaultTrainProbability; }
};

#endif // VKNRC_NRCSTATE_HPP
