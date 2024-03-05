//
// Created by adamyuan on 3/2/24.
//

#include "NNInferenceShader.hpp"

#include <spdlog/spdlog.h>

namespace rg {

myvk::Ptr<myvk::ShaderModule> NNInferenceShader::Create(const myvk::Ptr<myvk::Device> &device) {
	auto subgroup_size = device->GetPhysicalDevicePtr()->GetProperties().vk11.subgroupSize;
	if (subgroup_size == 16) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_inference_16.comp.u32>
		};
		return myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	} else if (subgroup_size == 32) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_inference_32.comp.u32>
		};
		return myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	} else if (subgroup_size == 64) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_inference_64.comp.u32>
		};
		return myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv));
	}
	spdlog::error("Unsupported subgroup size {}, only 16, 32, 64 are supported", subgroup_size);
	return nullptr;
}

} // namespace rg
