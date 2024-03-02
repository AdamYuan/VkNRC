//
// Created by adamyuan on 3/2/24.
//

#pragma once
#ifndef VKNRC_NNINFERENCESHADER_HPP
#define VKNRC_NNINFERENCESHADER_HPP

#include <myvk/ShaderModule.hpp>

namespace rg {

struct NNInferenceShader {
	static myvk::Ptr<myvk::ShaderModule> Create(const myvk::Ptr<myvk::Device> &device);
};

} // namespace rg

#endif // VKNRC_NNINFERENCESHADER_HPP
