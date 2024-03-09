//
// Created by adamyuan on 2/23/24.
//

#pragma once
#ifndef VKNRC_VKNRCSTATE_HPP
#define VKNRC_VKNRCSTATE_HPP

#include <glm/glm.hpp>
#include <half.hpp>
#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>
#include <random>
#include <span>

class VkNRCState final : public myvk::DeviceObjectBase {
public:
	enum Method { kNone, kNRC, kCache };

private:
	inline static constexpr uint32_t kNNHiddenLayers = 5, kNNWidth = 64, kNNOutWidth = 3, kTrainBatchSize = 16384,
	                                 kTrainBatchCount = 4;
	inline static constexpr uint32_t kNNWeighCount = kNNWidth * kNNWidth * kNNHiddenLayers + kNNWidth * kNNOutWidth;

	myvk::Ptr<myvk::Queue> m_queue_ptr;
	// use_weights: weights used by actual renderer
	myvk::Ptr<myvk::Buffer> m_weights, m_use_weights, m_optimizer_state, m_optimizer_entries;
	myvk::Ptr<myvk::ImageView> m_result_view;
	VkExtent2D m_extent{};
	uint32_t m_seed{};
	std::mt19937 m_rng{std::random_device{}()};
	Method m_left_method{kNRC}, m_right_method{kNRC};
	bool m_accumulate{false};
	uint32_t m_accumulate_count{0};
	bool m_use_ema_weights{false};

	void initialize_weights(std::span<float, kNNWeighCount> weights);
	void create_result_image();
	void create_mlp_buffer();

public:
	inline VkNRCState(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent) : m_queue_ptr(queue_ptr) {
		SetExtent(extent);
		create_mlp_buffer();
	}
	inline ~VkNRCState() final = default;

	inline const auto &GetResultImageView() const { return m_result_view; }
	inline const auto &GetWeightBuffer() const { return m_weights; }
	inline const auto &GetUseWeightBuffer() const { return m_use_weights; }
	inline const auto &GetOptimizerEntryBuffer() const { return m_optimizer_entries; }
	inline const auto &GetOptimizerStateBuffer() const { return m_optimizer_state; }

	inline Method GetLeftMethod() const { return m_left_method; }
	inline Method GetRightMethod() const { return m_right_method; }
	inline bool IsAccumulate() const { return m_accumulate; }
	inline uint32_t GetAccumulateCount() const { return m_accumulate_count; }
	inline bool IsUseEMAWeights() const { return m_use_ema_weights; }

	inline void SetLeftMethod(Method method) {
		if (method != m_left_method) {
			m_left_method = method;
			ResetAccumulate();
		}
	}
	inline void SetRightMethod(Method method) {
		if (method != m_right_method) {
			m_right_method = method;
			ResetAccumulate();
		}
	}
	inline void SetAccumulate(bool accumulate) {
		m_accumulate = accumulate;
		if (!accumulate)
			m_accumulate_count = 0;
	}
	inline void ResetAccumulate() { m_accumulate_count = 0; }
	inline void SetUseEMAWeights(bool use_ema_weights) { m_use_ema_weights = use_ema_weights; }

	inline uint32_t GetSeed() const { return m_seed; }

	inline void Next() {
		if (m_accumulate)
			++m_accumulate_count;
		m_seed = std::uniform_int_distribution<uint32_t>{0}(m_rng);
	}

	inline void SetExtent(VkExtent2D extent) {
		if (std::tie(m_extent.width, m_extent.height) != std::tie(extent.width, extent.height)) {
			m_extent = extent;
			create_result_image();
			ResetAccumulate();
		}
	}
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const { return m_queue_ptr->GetDevicePtr(); }
	static VkDeviceSize GetEvalRecordBufferSize(VkExtent2D extent);
	static VkDeviceSize GetBatchTrainRecordBufferSize();
	static constexpr uint32_t GetTrainBatchCount() { return kTrainBatchCount; }
	static constexpr uint32_t GetTrainBatchSize() { return kTrainBatchSize; }
	static constexpr uint32_t GetWeightCount() { return kNNWeighCount; }
};

#endif // VKNRC_VKNRCSTATE_HPP
