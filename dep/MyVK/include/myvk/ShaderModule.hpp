#ifndef MYVK_SHADER_MODULE_HPP
#define MYVK_SHADER_MODULE_HPP

#include "DeviceObjectBase.hpp"
#include "volk.h"
#include <memory>

namespace myvk {
class ShaderModule : public DeviceObjectBase {
private:
	Ptr<Device> m_device_ptr;
	VkShaderModule m_shader_module{VK_NULL_HANDLE};
	VkShaderStageFlagBits m_stage;

	std::vector<uint32_t> m_specialization_data;
	std::vector<VkSpecializationMapEntry> m_specialization_entries;
	mutable VkSpecializationInfo m_specialization_info;

	/* uint32_t spec_data[] = {m_tile_size, m_subgroup_size, m_shared_size};
	VkSpecializationMapEntry spec_entries[] = {
	    {.constantID = 0, .offset = 0, .size = sizeof(uint32_t)},
	    {.constantID = 1, .offset = sizeof(uint32_t), .size = sizeof(uint32_t)},
	    {.constantID = 2, .offset = 2 * sizeof(uint32_t), .size = sizeof(uint32_t)},
	};
	VkSpecializationInfo spec_info = {3, spec_entries, sizeof(spec_data), spec_data}; */

public:
	static Ptr<ShaderModule> Create(const Ptr<Device> &device, const uint32_t *code, uint32_t code_size);

	VkShaderStageFlagBits GetStage() const { return m_stage; }
	VkShaderModule GetHandle() const { return m_shader_module; }

	template <typename T> inline void AddSpecialization(uint32_t constant_id, T value) {
		static_assert(sizeof(T) == sizeof(uint32_t));
		m_specialization_entries.push_back({.constantID = constant_id,
		                                    .offset = uint32_t(m_specialization_data.size() * sizeof(uint32_t)),
		                                    .size = sizeof(uint32_t)});
		m_specialization_data.push_back(*(const uint32_t *)(&value));
	}

	inline const VkSpecializationInfo *GetSpecializationInfoPtr() const {
		m_specialization_info = {.mapEntryCount = (uint32_t)m_specialization_entries.size(),
		                         .pMapEntries = m_specialization_entries.data(),
		                         .dataSize = (uint32_t)m_specialization_data.size() * sizeof(uint32_t),
		                         .pData = m_specialization_data.data()};
		return m_specialization_data.empty() ? nullptr : &m_specialization_info;
	}

	const Ptr<Device> &GetDevicePtr() const override { return m_device_ptr; }

	VkPipelineShaderStageCreateInfo GetPipelineShaderStageCreateInfo(VkShaderStageFlagBits stage) const;

	~ShaderModule() override;
};
} // namespace myvk

#endif
