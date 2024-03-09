//
// Created by adamyuan on 2/24/24.
//

#pragma once
#ifndef VKNRC_NRCRESOURCES_HPP
#define VKNRC_NRCRESOURCES_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

struct NRCResources {
	myvk_rg::Image accumulate;
	myvk_rg::Buffer weights, use_weights, optimizer_state, optimizer_entries;
	myvk_rg::Buffer eval_records, eval_count;
	std::array<myvk_rg::Buffer, VkNRCState::GetTrainBatchCount()> batch_train_records, batch_train_counts;
};

} // namespace rg

#endif // VKNRC_NRCRESOURCES_HPP
