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
		std::unordered_map<DescriptorIndex, const InputBase *,
		                   U32PairHash<DescriptorIndex, &DescriptorIndex::binding, &DescriptorIndex::array_element>>
		    bindings, ext_bindings;

		myvk::Ptr<myvk::DescriptorSet> myvk_set;
		myvk::Ptr<myvk::DescriptorSetLayout> myvk_layout;
	} vk_descriptor;

	// VkCommand
	struct {
		friend class VkCommand;

	private:
		bool update_pipeline{true};
		myvk::Ptr<myvk::PipelineBase> vk_pipeline;
	} vk_command{};
};

class RGMemoryAllocation;

struct ResourceInfo {
	// Dependency
	struct {
		friend class Dependency;

	private:
		std::size_t root_id{};
		const ResourceBase *p_root_resource{};
	} dependency{};

	// Metadata
	struct {
		friend class Metadata;

	private:
		struct {
			VkImageType vk_type{};
			VkFormat vk_format{};
			VkImageUsageFlags vk_usages{};
		} image_alloc{};
		struct {
			bool mapped{false};
			VkBufferUsageFlags vk_usages{};
		} buffer_alloc;
		struct {
			SubImageSize size{};
			uint32_t base_layer{};
		} image_view{};
		struct {
			VkDeviceSize offset{}, size{};
		} buffer_view;
	} metadata{};

	// Schedule
	struct {
		friend class Schedule;

	private:
		std::vector<const InputBase *> first_inputs, last_inputs;
		bool ext_read_only{true};
	} schedule{};

	// VkAllocation
	struct {
		friend class VkAllocation;

	private:
		struct {
			myvk::Ptr<myvk::ImageBase> myvk_image{};
			myvk::Ptr<myvk::ImageView> myvk_image_view{};
		} image{};
		struct {
			myvk::Ptr<myvk::BufferBase> myvk_buffer{};
			BufferView buffer_view{};
			void *p_mapped{};
		} buffer{};
		VkMemoryRequirements vk_mem_reqs{};
		myvk::Ptr<RGMemoryAllocation> myvk_mem_alloc{};
		VkDeviceSize mem_offset{};
	} vk_allocation{};

	// VkRunner
	struct {
		friend class VkRunner;

	private:
		myvk::Ptr<myvk::ImageView> ext_image_view_cache{};
		BufferView ext_buffer_view_cache{};
		bool ext_changed{};
	} vk_runner{};
};

inline PassInfo &GetPassInfo(const PassBase *p_pass) { return *p_pass->__GetPExecutorInfo<PassInfo>(); }
inline InputInfo &GetInputInfo(const InputBase *p_input) { return *p_input->__GetPExecutorInfo<InputInfo>(); }
inline ResourceInfo &GetResourceInfo(const ResourceBase *p_resource) {
	return *p_resource->__GetPExecutorInfo<ResourceInfo>();
}

} // namespace myvk_rg_executor

#endif // MYVK_INFO_HPP
