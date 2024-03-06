//
// Created by adamyuan on 3/5/24.
//

#pragma once
#ifndef VKNRC_NNGRADIENTSHADER_HPP
#define VKNRC_NNGRADIENTSHADER_HPP

#include <myvk/ShaderModule.hpp>

namespace rg {

struct NNGradientShader {
	static std::tuple<myvk::Ptr<myvk::ShaderModule>, VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>
	Create(const myvk::Ptr<myvk::Device> &device);
};

} // namespace rg

#endif // VKNRC_NNGRADIENTSHADER_HPP
