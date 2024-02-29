//
// Created by adamyuan on 2/15/24.
//

#include "VkDescriptor.hpp"
#include "VkRunner.hpp" // For IsExtChanged

#include <list>
#include <myvk/AccelerationStructure.hpp>

namespace myvk_rg_executor {

struct DescriptorWriter {
	std::vector<VkWriteDescriptorSet> writes;
	std::list<VkWriteDescriptorSetAccelerationStructureKHR> as_writes;
	std::list<VkDescriptorImageInfo> image_infos;
	std::list<VkDescriptorBufferInfo> buffer_infos;

	inline void PushBufferWrite(const myvk::Ptr<myvk::DescriptorSet> &set, DescriptorIndex index,
	                            const BufferView &buffer_view, const InputBase *p_input) {
		VkDescriptorType descriptor_type = UsageGetDescriptorType(p_input->GetUsage());
		if (descriptor_type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
			auto accel_struct = std::dynamic_pointer_cast<myvk::AccelerationStructure>(buffer_view.data);
			assert(accel_struct);
			as_writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
			                     .accelerationStructureCount = 1,
			                     .pAccelerationStructures = &accel_struct->GetHandle()});
			writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			                  .pNext = &as_writes.back(),
			                  .dstSet = set->GetHandle(),
			                  .dstBinding = index.binding,
			                  .dstArrayElement = index.array_element,
			                  .descriptorCount = 1u,
			                  .descriptorType = descriptor_type});
		} else {
			buffer_infos.push_back(
			    {.buffer = buffer_view.buffer->GetHandle(), .offset = buffer_view.offset, .range = buffer_view.size});
			writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			                  .dstSet = set->GetHandle(),
			                  .dstBinding = index.binding,
			                  .dstArrayElement = index.array_element,
			                  .descriptorCount = 1u,
			                  .descriptorType = descriptor_type,
			                  .pBufferInfo = &buffer_infos.back()});
		}
	}
	inline void PushImageWrite(const myvk::Ptr<myvk::DescriptorSet> &set, DescriptorIndex index,
	                           const myvk::Ptr<myvk::ImageView> &image_view, const InputBase *p_input) {
		image_infos.push_back(
		    {.imageView = image_view->GetHandle(), .imageLayout = UsageGetImageLayout(p_input->GetUsage())});
		writes.push_back({.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		                  .dstSet = set->GetHandle(),
		                  .dstBinding = index.binding,
		                  .dstArrayElement = index.array_element,
		                  .descriptorCount = 1u,
		                  .descriptorType = UsageGetDescriptorType(p_input->GetUsage()),
		                  .pImageInfo = &image_infos.back()});
	}
};

void VkDescriptor::collect_pass_bindings(const PassBase *p_pass) {
	for (const InputBase *p_input : Dependency::GetPassInputs(p_pass)) {
		std::optional<DescriptorIndex> opt_index = p_input->GetOptDescriptorIndex();
		if (!opt_index)
			continue;
		auto index = *opt_index;

		// Check whether the index exists
		if (get_desc_info(p_pass).bindings.contains(index))
			Throw(error::DupDescriptorIndex{.key = p_input->GetGlobalKey()});
		get_desc_info(p_pass).bindings[index] = p_input;

		// External Bindings might need updates
		if (Dependency::GetInputResource(p_input)->GetState() == ResourceState::kExternal)
			get_desc_info(p_pass).ext_bindings[index] = p_input;
	}
}

namespace create_vk_sets {
struct BindingInfo {
	VkDescriptorType type;
	VkShaderStageFlags shader_stages;
	inline auto operator<=>(const BindingInfo &) const = default;
};
} // namespace create_vk_sets

void VkDescriptor::create_vk_sets(const VkDescriptor::Args &args) {
	using create_vk_sets::BindingInfo;

	// Pass with Descriptors
	auto desc_pass_range = args.dependency.GetPasses() | std::views::filter([](const PassBase *p_pass) {
		                       return !get_desc_info(p_pass).bindings.empty();
	                       });

	// Count VkDescriptorType
	std::unordered_map<VkDescriptorType, uint32_t> vk_desc_type_counts;

	// Descriptor Set Layouts
	std::vector<myvk::Ptr<myvk::DescriptorSetLayout>> batch_myvk_set_layouts;

	// Create Set Layouts, Fetch VkDescriptorType Counts
	for (const PassBase *p_pass : desc_pass_range) {
		auto &desc_info = get_desc_info(p_pass);

		std::unordered_map<uint32_t, std::vector<const InputBase *>> binding_array;

		for (auto [index, p_input] : desc_info.bindings) {
			auto &array = binding_array[index.binding];
			if (array.size() <= index.array_element)
				array.resize(index.array_element + 1);
			array[index.array_element] = p_input;
		}

		std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
		layout_bindings.reserve(binding_array.size());
		std::vector<std::vector<VkSampler>> immutable_samplers;

		for (const auto &[binding, array] : binding_array) {
			// Get Type and ShaderStages of the Binding, also Validate
			if (array[0] == nullptr)
				Throw(error::InvalidDescriptorArray{.key = p_pass->GetGlobalKey()});

			const auto get_binding_info = [](const InputBase *p_input) -> BindingInfo {
				return {UsageGetDescriptorType(p_input->GetUsage()),
				        VkShaderStagesFromVkPipelineStages(p_input->GetPipelineStages())};
			};
			auto [type, shader_stages] = get_binding_info(array[0]);
			for (const InputBase *p_input : array)
				if (!p_input || get_binding_info(p_input) != BindingInfo{type, shader_stages})
					Throw(error::InvalidDescriptorArray{.key = p_pass->GetGlobalKey()});

			// Update Descriptor Type Counts
			vk_desc_type_counts[type] += array.size();
			// Fetch Immutable Samplers
			VkSampler *p_immutable_samplers = nullptr;
			if (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || type == VK_DESCRIPTOR_TYPE_SAMPLER) {
				immutable_samplers.emplace_back(array.size());
				p_immutable_samplers = immutable_samplers.back().data();

				for (std::size_t i = 0; const InputBase *p_input : array) {
					assert(p_input->GetType() == ResourceType::kImage);
					const auto &myvk_sampler = static_cast<const ImageInput *>(p_input)->GetVkSampler();
					p_immutable_samplers[i++] = myvk_sampler ? myvk_sampler->GetHandle() : VK_NULL_HANDLE;
				}
			}
			// Push Bindings
			layout_bindings.push_back({.binding = binding,
			                           .descriptorType = type,
			                           .descriptorCount = (uint32_t)array.size(),
			                           .stageFlags = shader_stages,
			                           .pImmutableSamplers = p_immutable_samplers});
		}

		// Create Layout
		auto myvk_layout = myvk::DescriptorSetLayout::Create(m_device_ptr, layout_bindings);
		desc_info.myvk_layout = myvk_layout;

		// Push VkDescriptorLayouts for Batch Creation
		batch_myvk_set_layouts.push_back(myvk_layout);
	}

	if (batch_myvk_set_layouts.empty())
		return;

	// Create Descriptor Pool
	myvk::Ptr<myvk::DescriptorPool> myvk_descriptor_pool;
	{
		std::vector<VkDescriptorPoolSize> pool_sizes;
		pool_sizes.reserve(vk_desc_type_counts.size());
		for (auto [type, count] : vk_desc_type_counts)
			pool_sizes.push_back({.type = type, .descriptorCount = count});
		myvk_descriptor_pool = myvk::DescriptorPool::Create(m_device_ptr, args.dependency.GetPassCount(), pool_sizes);
	}

	// Create Descriptor Sets
	auto batch_myvk_sets = myvk::DescriptorSet::CreateMultiple(myvk_descriptor_pool, batch_myvk_set_layouts);
	for (std::size_t counter = 0; const PassBase *p_pass : desc_pass_range) {
		auto &desc_info = get_desc_info(p_pass);
		desc_info.myvk_set = std::move(batch_myvk_sets[counter++]);
	}
}

void VkDescriptor::vk_update_internal(std::span<const PassBase *const> passes) {
	DescriptorWriter writer{};

	for (auto p_pass : passes) {
		auto &desc_info = get_desc_info(p_pass);
		for (const auto &[index, p_input] : desc_info.bindings) {
			Dependency::GetInputResource(p_input)->Visit(overloaded(
			    [&](const InternalImage auto *p_int_image) {
				    writer.PushImageWrite(desc_info.myvk_set, index, VkAllocation::GetVkImageView(p_int_image),
				                          p_input);
			    },
			    [&](const InternalBuffer auto *p_int_buffer) {
				    writer.PushBufferWrite(desc_info.myvk_set, index, p_int_buffer->GetBufferView(), p_input);
			    },
			    [](auto &&) {}));
		}
	}

	if (writer.writes.empty())
		return;

	vkUpdateDescriptorSets(m_device_ptr->GetHandle(), writer.writes.size(), writer.writes.data(), 0, nullptr);
}

void VkDescriptor::VkUpdateExternal(std::span<const PassBase *const> passes) const {
	DescriptorWriter writer{};

	for (auto p_pass : passes) {
		auto &desc_info = get_desc_info(p_pass);
		for (const auto &[index, p_input] : desc_info.ext_bindings) {
			const ResourceBase *p_resource = Dependency::GetInputResource(p_input);
			if (!VkRunner::IsExtChanged(p_resource))
				continue;
			Dependency::GetInputResource(p_input)->Visit(overloaded(
			    [&](const ExternalImageBase *p_ext_image) {
				    writer.PushImageWrite(desc_info.myvk_set, index, p_ext_image->GetVkImageView(), p_input);
			    },
			    [&](const ExternalBufferBase *p_ext_buffer) {
				    writer.PushBufferWrite(desc_info.myvk_set, index, p_ext_buffer->GetBufferView(), p_input);
			    },
			    [](auto &&) {}));
		}
	}

	if (writer.writes.empty())
		return;

	vkUpdateDescriptorSets(m_device_ptr->GetHandle(), writer.writes.size(), writer.writes.data(), 0, nullptr);
}

VkDescriptor VkDescriptor::Create(const myvk::Ptr<myvk::Device> &device_ptr, const Args &args) {
	args.collection.ClearInfo(&PassInfo::vk_descriptor);

	VkDescriptor vk_desc = {};
	vk_desc.m_device_ptr = device_ptr;
	for (const PassBase *p_pass : args.dependency.GetPasses())
		collect_pass_bindings(p_pass);
	vk_desc.create_vk_sets(args);
	vk_desc.vk_update_internal(args.dependency.GetPasses());
	return vk_desc;
}

} // namespace myvk_rg_executor