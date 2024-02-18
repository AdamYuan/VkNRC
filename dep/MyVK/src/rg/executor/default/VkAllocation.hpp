//
// Created by adamyuan on 2/8/24.
//

#pragma once
#ifndef MYVK_RG_EXE_DEF_ALLOCATOR_HPP
#define MYVK_RG_EXE_DEF_ALLOCATOR_HPP

#include "Metadata.hpp"

#include <myvk/Device.hpp>

namespace myvk_rg_executor {

class VkAllocation {
private:
	struct Args {
		const RenderGraphBase &render_graph;
		const Collection &collection;
		const Dependency &dependency;
		const Metadata &metadata;
	};

	myvk::Ptr<myvk::Device> m_device_ptr;

	Relation m_resource_alias_relation;

	static auto &get_vk_alloc(const ResourceBase *p_resource) { return GetResourceInfo(p_resource).vk_allocation; }

	void init_alias_relation(const Args &args);
	void check_double_buffer(const Args &args);
	void create_vk_resources(const Args &args);
	static std::tuple<VkDeviceSize, uint32_t> fetch_memory_requirements(std::ranges::input_range auto &&resources);
	void alloc_naive(std::ranges::input_range auto &&resources, const VmaAllocationCreateInfo &create_info);
	void alloc_optimal(const Args &args, std::ranges::input_range auto &&resources,
	                   const VmaAllocationCreateInfo &create_info);
	void create_vk_allocations(const Args &args);
	void bind_vk_resources(const Args &args);
	void create_vk_image_views(const Args &args);
	void set_lf_vk_resources(const Args &args);

public:
	static VkAllocation Create(const myvk::Ptr<myvk::Device> &device_ptr, const Args &args);

	// Resource Alias Relationship
	inline bool IsAliased(const ResourceBase *p_l, const ResourceBase *p_r) const {
		return m_resource_alias_relation.Get(Dependency::GetResourceRootID(p_l), Dependency::GetResourceRootID(p_r));
	}

	static bool IsDoubleBuffered(const ResourceBase *p_resource) {
		return get_vk_alloc(Metadata::GetAllocResource(p_resource)).double_buffer;
	}

	static const myvk::Ptr<myvk::ImageView> &GetVkImageView(const InternalImage auto *p_image, bool flip) {
		return get_vk_alloc(p_image).image.myvk_image_views[flip];
	}
	static const myvk::Ptr<myvk::BufferBase> &GetVkBuffer(const InternalBuffer auto *p_buffer, bool flip) {
		return get_vk_alloc(p_buffer).buffer.myvk_buffers[flip];
	}
	static void *GetMappedData(const InternalBuffer auto *p_buffer, bool flip) {
		return get_vk_alloc(p_buffer).buffer.mapped_ptrs[flip];
	}
};

} // namespace myvk_rg_executor

#endif // MYVK_RG_EXE_DEF_ALLOCATOR_HPP
