//
// Created by adamyuan on 3/5/24.
//

#pragma once
#ifndef VKNRC_NNGRADIENTSHADER_HPP
#define VKNRC_NNGRADIENTSHADER_HPP

#include <myvk/ShaderModule.hpp>

namespace rg {

class NNGradientShader {
	static myvk::Ptr<myvk::ShaderModule> Create(const myvk::Ptr<myvk::Device> &device);
};

}

#endif // VKNRC_NNGRADIENTSHADER_HPP
