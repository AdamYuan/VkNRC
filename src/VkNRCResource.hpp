//
// Created by adamyuan on 4/2/24.
//

#pragma once
#ifndef VKNRC_VKNRCRESOURCE_HPP
#define VKNRC_VKNRCRESOURCE_HPP

#include "CuVkBuffer.hpp"
#include "NRCState.hpp"
#include <myvk/Buffer.hpp>
#include <myvk/ExportBuffer.hpp>
#include <myvk/Image.hpp>
#include <myvk/ImageView.hpp>

class VkNRCResource final : public myvk::DeviceObjectBase {
public:
	struct Frame {
		std::array<myvk::Ptr<myvk::Buffer>, NRCState::GetTrainBatchCount()> batch_train_counts, batch_train_factors;
		myvk::Ptr<myvk::Buffer> inference_count, inference_dst;
		myvk::Ptr<myvk::ImageView> bias_factor_r, factor_gb;
	};

private:
	myvk::Ptr<myvk::Queue> m_queue_ptr;

	std::vector<Frame> m_frames;
	std::unique_ptr<CuVkBuffer> m_inference_inputs, m_inference_outputs;
	std::array<std::unique_ptr<CuVkBuffer>, NRCState::GetTrainBatchCount()> m_batch_train_inputs, m_batch_train_targets;
	myvk::Ptr<myvk::ImageView> m_accumulate_view;

	void create_fixed();

public:
	inline VkNRCResource(const myvk::Ptr<myvk::Queue> &queue_ptr, VkExtent2D extent, uint32_t frame_count)
	    : m_queue_ptr(queue_ptr), m_frames(frame_count) {
		create_fixed();
		Resize(extent);
	}
	inline ~VkNRCResource() final = default;
	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const final { return m_queue_ptr->GetDevicePtr(); }

	void Resize(VkExtent2D extent);

	inline const auto &GetAccumulateImageView() const { return m_accumulate_view; }
	inline const auto &GetInferenceInputBuffer() const { return m_inference_inputs; }
	inline const auto &GetInferenceOutputBuffer() const { return m_inference_outputs; }
	inline const auto &GetBatchTrainInputBufferArray() const { return m_batch_train_inputs; }
	inline const auto &GetBatchTrainTargetBufferArray() const { return m_batch_train_targets; }
	inline const auto &GetFrameResources(uint32_t frame_index) const { return m_frames[frame_index]; }
};

#endif // VKNRC_VKNRCRESOURCE_HPP
