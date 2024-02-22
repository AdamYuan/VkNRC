//
// Created by adamyuan on 2/15/24.
//

#pragma once
#ifndef MYVK_RG_DEF_EXE_VKDESCRIPTOR_HPP
#define MYVK_RG_DEF_EXE_VKDESCRIPTOR_HPP

#include "VkAllocation.hpp"

#include "../Hash.hpp"
#include "../VkHelper.hpp"

#include <span>

namespace myvk_rg_executor {

class VkDescriptor {
private:
	struct Args {
		const RenderGraphBase &render_graph;
		const Collection &collection;
		const Dependency &dependency;
		const Metadata &metadata;
		const VkAllocation &vk_allocation;
	};

	myvk::Ptr<myvk::Device> m_device_ptr;

	static auto &get_desc_info(const PassBase *p_pass) { return GetPassInfo(p_pass).vk_descriptor; }

	static void collect_pass_bindings(const PassBase *p_pass);
	void create_vk_sets(const Args &args);
	void vk_update_internal(std::span<const PassBase *const> passes);

public:
	static VkDescriptor Create(const myvk::Ptr<myvk::Device> &device_ptr, const Args &args);
	void VkUpdateExternal(std::span<const PassBase *const> passes) const;
	static const myvk::Ptr<myvk::DescriptorSet> &GetVkDescriptorSet(const PassBase *p_pass) {
		return get_desc_info(p_pass).myvk_set;
	}
	static const myvk::Ptr<myvk::DescriptorSetLayout> &GetVkDescriptorSetLayout(const PassBase *p_pass) {
		return get_desc_info(p_pass).myvk_layout;
	}
};

} // namespace myvk_rg_executor

#endif
