//
// Created by adamyuan on 2/5/24.
//

#pragma once
#ifndef MYVK_INFO_HPP
#define MYVK_INFO_HPP

#include <array>
#include <myvk_rg/interface/RenderGraph.hpp>

#include "../Hash.hpp"

namespace myvk_rg_executor {

using namespace myvk_rg::interface;
using namespace myvk_rg::executor;

struct InputInfo {
	// Dependency
	struct {
		friend class Dependency;

		const PassBase *p_pass{};
		const ResourceBase *p_resource{};
	} dependency{};
};

using DescriptorBindingMap =
    std::unordered_map<DescriptorIndex, const InputBase *,
                       U32PairHash<DescriptorIndex, &DescriptorIndex::binding, &DescriptorIndex::array_element>>;

struct PassInfo {
	// Dependency
	struct {
		friend class Dependency;

	private:
		std::size_t topo_id{};
		std::vector<const InputBase *> inputs;
	} dependency{};

	// Metadata
	struct {
		friend class Metadata;

	private:
		RenderPassArea render_area;
	} metadata{};

	// Schedule
	struct {
		friend class Schedule;

	private:
		std::size_t group_id{}, subpass_id{};
	} schedule{};

	// VkDescriptor
	struct {
		friend class VkDescriptor;

	private:
		bool double_buffer{false};
		DescriptorBindingMap bindings, static_bindings, dynamic_bindings;

		std::array<myvk::Ptr<myvk::DescriptorSet>, 2> myvk_sets;
		myvk::Ptr<myvk::DescriptorSetLayout> myvk_layout;
	} vk_descriptor;

	// VkCommand
	struct {
		friend class VkCommand;

	private:
		bool update_pipeline{true};
	} vk_command{};
};

class RGMemoryAllocation;

struct ResourceInfo {
	// Dependency
	struct {
		friend class Dependency;

	private:
		std::size_t root_id{};
		const ResourceBase *p_root_resource{}, *p_lf_resource{};
	} dependency{};

	// Metadata
	struct {
		friend class Metadata;

	private:
		const ResourceBase *p_alloc_resource{}, *p_view_resource{};

		struct {
			VkImageType vk_type{};
			VkFormat vk_format{};
			VkImageUsageFlags vk_usages{};
		} image_alloc{};
		struct {
			VkBufferUsageFlags vk_usages{};
		} buffer_alloc;
		struct {
			SubImageSize size{};
			uint32_t base_layer{};
		} image_view{};
		struct {
			VkDeviceSize size{};
		} buffer_view;
	} metadata{};

	// Schedule
	struct {
		friend class Schedule;

	private:
		std::vector<const InputBase *> first_inputs, last_inputs;
	} schedule{};

	// VkAllocation
	struct {
		friend class VkAllocation;

	private:
		bool double_buffer{};
		struct {
			std::array<myvk::Ptr<myvk::ImageBase>, 2> myvk_images{};
			std::array<myvk::Ptr<myvk::ImageView>, 2> myvk_image_views{};
		} image{};
		struct {
			std::array<myvk::Ptr<myvk::BufferBase>, 2> myvk_buffers{};
			std::array<void *, 2> mapped_ptrs{};
		} buffer{};
		VkMemoryRequirements vk_mem_reqs{};
		myvk::Ptr<RGMemoryAllocation> myvk_mem_alloc{};
		std::array<VkDeviceSize, 2> mem_offsets{};
	} vk_allocation{};

	struct {
		friend class VkCommand;

	private:
	} vk_command{};
};

inline PassInfo &GetPassInfo(const PassBase *p_pass) { return *p_pass->__GetPExecutorInfo<PassInfo>(); }
inline InputInfo &GetInputInfo(const InputBase *p_input) { return *p_input->__GetPExecutorInfo<InputInfo>(); }
inline ResourceInfo &GetResourceInfo(const ResourceBase *p_resource) {
	return *p_resource->__GetPExecutorInfo<ResourceInfo>();
}

} // namespace myvk_rg_executor

#endif // MYVK_INFO_HPP
