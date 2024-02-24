//
// Created by adamyuan on 2/24/24.
//

#pragma once
#ifndef VKNRC_NRCRESOURCES_HPP
#define VKNRC_NRCRESOURCES_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace rg {

struct NRCResources {
	myvk_rg::Buffer sobol;
	myvk_rg::Image noise, result;
	myvk::Ptr<myvk::Sampler> noise_sampler;
};

} // namespace rg

#endif // VKNRC_NRCRESOURCES_HPP
