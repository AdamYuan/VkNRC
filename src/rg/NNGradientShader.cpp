//
// Created by adamyuan on 3/5/24.
//

#include "NNGradientShader.hpp"

#include <spdlog/spdlog.h>

namespace rg {

myvk::Ptr<myvk::ShaderModule> NNGradientShader::Create(const myvk::Ptr<myvk::Device> &device) {
	auto subgroup_size = device->GetPhysicalDevicePtr()->GetProperties().vk11.subgroupSize;
	if (subgroup_size == 16) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_16.comp.u32>
		};
		return myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	} else if (subgroup_size == 32) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_32.comp.u32>
		};
		return myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	} else if (subgroup_size == 64) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_64.comp.u32>
		};
		return myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	}
	spdlog::error("Unsupported subgroup size {}, only 16, 32, 64 are supported", subgroup_size);
	return nullptr;
}

} // namespace rg
