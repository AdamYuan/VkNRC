//
// Created by adamyuan on 2/10/24.
//

#include "Metadata.hpp"

#include "../VkHelper.hpp"

namespace myvk_rg_executor {

Metadata Metadata::Create(const Args &args) {
	args.collection.ClearInfo(&ResourceInfo::metadata, &PassInfo::metadata);

	Metadata r = {};
	r.classify_resources(args);
	r.fetch_external_infos(args);
	r.fetch_alloc_sizes(args);
	fetch_alloc_usages(args);
	r.propagate_alloc_info(args);
	fetch_render_areas(args);
	return r;
}

void Metadata::classify_resources(const Metadata::Args &args) {
	for (const ResourceBase *p_resource : args.dependency.GetResources()) {
		if (p_resource->GetState() == myvk_rg::interface::ResourceState::kExternal) {
			assert(Dependency::IsRootResource(p_resource));
			m_external_resources.push_back(p_resource);
		} else {
			m_internal_resources.push_back(p_resource);
			if (Dependency::IsRootResource(p_resource))
				m_internal_root_resources.push_back(p_resource);
		}
	}
}

void Metadata::fetch_external_infos(const Args &args) {
	for (const ResourceBase *p_resource : m_external_resources)
		p_resource->Visit(overloaded(
		    [](const ExternalImageBase *p_ext_image) {
			    const auto &myvk_view = p_ext_image->GetVkImageView();
			    const auto &myvk_image = myvk_view->GetImagePtr();
			    const auto &vk_sub_range = myvk_view->GetSubresourceRange();
			    get_view(p_ext_image) = {.size = SubImageSize(myvk_image->GetExtent(), vk_sub_range.layerCount,
			                                                  vk_sub_range.baseMipLevel, vk_sub_range.levelCount),
			                             .base_layer = myvk_view->GetSubresourceRange().baseArrayLayer};
			    get_alloc(p_ext_image) = {.vk_type = myvk_image->GetType(),
			                              .vk_format = myvk_image->GetFormat(),
			                              .vk_usages = myvk_image->GetUsage()};
		    },
		    [](const ExternalBufferBase *p_ext_buffer) {
			    const auto &view = p_ext_buffer->GetBufferView();
			    get_view(p_ext_buffer) = {.offset = view.offset, .size = view.size};
			    get_alloc(p_ext_buffer) = {.vk_usages = view.buffer->GetUsage()};
		    },
		    [](auto &&) {}));
}

void Metadata::propagate_alloc_info(const Args &args) {
	for (const ResourceBase *p_resource : m_internal_resources)
		if (!Dependency::IsRootResource(p_resource))
			p_resource->Visit([](const auto *p_resource) {
				get_alloc(p_resource) = get_alloc(Dependency::GetRootResource(p_resource));
			});
}

auto Metadata::get_size(const Args &args, const auto &size_variant) {
	const auto get_size_visitor = overloaded(
	    [&](const std::invocable<VkExtent2D> auto &size_func) { return size_func(args.render_graph.GetCanvasSize()); },
	    [](const auto &size) { return size; });

	return std::visit(get_size_visitor, size_variant);
}

void Metadata::combine_image(const Metadata::Args &args, const InternalImage auto *p_alloc_image) {
	auto &alloc = get_alloc(p_alloc_image);

	const auto combine_impl = [&](const InternalImage auto *p_view_image, auto &&combine) {
		auto &view = get_view(p_view_image);
		UpdateVkImageTypeFromVkImageViewType(&alloc.vk_type, p_view_image->GetViewType());
		view.size = overloaded(
		    // Combined Image
		    [&](const CombinedImage *p_combined_image) -> SubImageSize {
			    SubImageSize size = {};
			    VkFormat format = VK_FORMAT_UNDEFINED;

			    for (auto [p_sub, _, _1] : args.dependency.GetResourceGraph().GetOutEdges(p_combined_image))
				    // Foreach Sub-Image
				    p_sub->Visit(overloaded(
				        [&](const InternalImage auto *p_sub) {
					        combine(p_sub, combine);

					        const auto &sub_size = get_view(p_sub).size;
					        if (!size.Merge(sub_size))
						        Throw(error::ImageNotMerge{.key = p_combined_image->GetGlobalKey()});

					        // Base Layer (Offset)
					        get_view(p_sub).base_layer = size.GetArrayLayers() - sub_size.GetArrayLayers();
				        },
				        [](auto &&) {}));

			    return size;
		    },
		    // Managed Image
		    [&](const ManagedImage *p_managed_image) -> SubImageSize {
			    // Maintain VkFormat
			    VkFormat &alloc_format = alloc.vk_format;
			    if (alloc_format != VK_FORMAT_UNDEFINED && alloc_format != p_managed_image->GetFormat())
				    Throw(error::ImageNotMerge{.key = p_managed_image->GetGlobalKey()});
			    alloc_format = p_managed_image->GetFormat();

			    return get_size(args, p_managed_image->GetSize());
		    })(p_view_image);
	};
	combine_impl(p_alloc_image, combine_impl);

	// Base Mip Level should be 0
	if (get_view(p_alloc_image).size.GetBaseMipLevel() != 0)
		Throw(error::ImageNotMerge{.key = p_alloc_image->GetGlobalKey()});

	// Accumulate Base Layer Offsets
	const auto accumulate_base_impl = overloaded(
	    [&](const CombinedImage *p_combined_image, auto &&accumulate_base) -> void {
		    for (auto [p_sub, _, _1] : args.dependency.GetResourceGraph().GetOutEdges(p_combined_image)) {
			    p_sub->Visit(overloaded(
			        [&](const InternalImage auto *p_sub) {
				        get_view(p_sub).base_layer += get_view(p_combined_image).base_layer;
			        },
			        [](auto &&) {}));
			    p_sub->Visit(overloaded([&](const CombinedImage *p_sub) { accumulate_base(p_sub, accumulate_base); },
			                            [](auto &&) {}));
		    }
	    },
	    [](auto &&, auto &&) {});
	accumulate_base_impl(p_alloc_image, accumulate_base_impl);
}

void Metadata::combine_buffer(const Metadata::Args &args, const InternalBuffer auto *p_alloc_buffer) {
	auto &alloc = get_alloc(p_alloc_buffer);

	const auto combine_impl = [&](const InternalBuffer auto *p_view_buffer, VkDeviceSize offset, auto &&combine) {
		auto &view = get_view(p_view_buffer);
		view.offset = offset;
		view.size = overloaded(
		    // Combined Image
		    [&](const CombinedBuffer *p_combined_buffer) -> VkDeviceSize {
			    VkDeviceSize size = 0;
			    for (auto [p_sub, _, _1] : args.dependency.GetResourceGraph().GetOutEdges(p_combined_buffer))
				    // Foreach Sub-Image
				    p_sub->Visit(overloaded(
				        [&](const InternalBuffer auto *p_sub) {
					        combine(p_sub, offset + size, combine);
					        size += get_view(p_sub).size;
				        },
				        [](auto &&) {}));
			    return size;
		    },
		    // Managed Buffer
		    [&](const ManagedBuffer *p_managed_buffer) -> VkDeviceSize {
			    alloc.mapped |= p_managed_buffer->IsMapped();
			    return get_size(args, p_managed_buffer->GetSize());
		    })(p_view_buffer);
	};
	combine_impl(p_alloc_buffer, 0, combine_impl);
}

void Metadata::fetch_alloc_sizes(const Args &args) {
	for (const ResourceBase *p_resource : m_internal_root_resources)
		// Collect Size
		p_resource->Visit(overloaded([&](const InternalImage auto *p_image) { combine_image(args, p_image); },
		                             [&](const InternalBuffer auto *p_buffer) { combine_buffer(args, p_buffer); },
		                             [](auto &&) {}));
}

void Metadata::fetch_alloc_usages(const Args &args) {
	for (const auto *p_pass : args.dependency.GetPasses()) {
		for (const InputBase *p_input : Dependency::GetPassInputs(p_pass))
			Dependency::GetInputResource(p_input)->Visit(overloaded(
			    [&](const InternalResource auto *p_resource) {
				    get_alloc(Dependency::GetRootResource(p_resource)).vk_usages |=
				        UsageGetCreationUsages(p_input->GetUsage());
			    },
			    [](auto &&) {}));
	}
}

void Metadata::fetch_render_areas(const Metadata::Args &args) {
	auto graphics_pass_visitor = [&](const GraphicsPassBase *p_graphics_pass) {
		const auto &opt_area = p_graphics_pass->GetOptRenderArea();
		if (opt_area) {
			get_meta(p_graphics_pass).render_area =
			    std::visit(overloaded(
			                   [&](const std::invocable<VkExtent2D> auto &area_func) {
				                   return area_func(args.render_graph.GetCanvasSize());
			                   },
			                   [](const RenderPassArea &area) { return area; }),
			               *opt_area);
		} else {
			// Loop through Pass Inputs and find the largest attachment
			RenderPassArea area = {};
			for (const InputBase *p_input : Dependency::GetPassInputs(p_graphics_pass)) {
				if (!UsageIsAttachment(p_input->GetUsage()))
					continue;
				Dependency::GetInputResource(p_input)->Visit(overloaded(
				    [&](const ImageResource auto *p_image) {
					    const auto &size = Metadata::GetViewInfo(p_image).size;
					    area.layers = std::max(area.layers, size.GetArrayLayers());
					    auto [width, height, _] = size.GetBaseMipExtent();
					    area.extent.width = std::max(area.extent.width, width);
					    area.extent.height = std::max(area.extent.height, height);
				    },
				    [](auto &&) {}));
			}
			get_meta(p_graphics_pass).render_area = area;
		}
	};
	for (const PassBase *p_pass : args.dependency.GetPasses())
		p_pass->Visit(overloaded(graphics_pass_visitor, [](auto &&) {}));
}

} // namespace myvk_rg_executor
