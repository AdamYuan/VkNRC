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
	myvk_rg::Buffer weights;
	myvk_rg::Buffer eval_records, eval_record_count;
	myvk_rg::Buffer train_batch_records, train_batch_record_counts;
};

} // namespace rg

#endif // VKNRC_NRCRESOURCES_HPP
