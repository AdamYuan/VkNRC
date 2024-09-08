//
// Created by adamyuan on 3/5/24.
//

#include "NNGradientShader.hpp"

#include <spdlog/spdlog.h>

namespace rg {

std::tuple<myvk::Ptr<myvk::ShaderModule>, VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>
NNGradientShader::Create(const myvk::Ptr<myvk::Device> &device) {
	auto subgroup_size = device->GetPhysicalDevicePtr()->GetProperties().vk11.subgroupSize;
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo info{
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
	    .requiredSubgroupSize = subgroup_size};
	if (subgroup_size == 16) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_16.comp.u32>

		};
		return {myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv)), info};
	} else if (subgroup_size == 32) {
		constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_32.comp.u32>

		};
		return {myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv)), info};
	} /* else if (subgroup_size == 64) {
	    constexpr uint32_t kCompSpv[] = {
#include <shader/nrc_gradient_64.comp.u32>
	    };
	    return {myvk::ShaderModule::Create(device, kCompSpv, sizeof(kCompSpv)), info};
	} */
	spdlog::error("Unsupported subgroup size {}, only 16, 32, 64 are supported", subgroup_size);
	return {nullptr, info};
}

} // namespace rg
