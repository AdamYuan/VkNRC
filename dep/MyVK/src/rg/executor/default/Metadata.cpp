//
// Created by adamyuan on 2/10/24.
//

#include "Metadata.hpp"

#include "../VkHelper.hpp"

namespace myvk_rg_executor {

Metadata Metadata::Create(const Args &args) {
	args.collection.ClearInfo(&ResourceInfo::metadata, &PassInfo::metadata);

	Metadata r = {};
	r.tag_resources(args);
	r.fetch_alloc_sizes(args);
	fetch_alloc_usages(args);
	propagate_resource_meta(args);
	fetch_render_areas(args);
	return r;
}

void Metadata::tag_resources(const Args &args) {
	for (const ResourceBase *p_resource : args.dependency.GetResources()) {
		// Skip External Resources
		if (p_resource->GetState() == ResourceState::kExternal) {
			get_meta(p_resource).p_alloc_resource = nullptr;
			get_meta(p_resource).p_view_resource = nullptr;
			// Set External View and "Alloc" data
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
				    const auto &myvk_buffer = p_ext_buffer->GetVkBuffer();
				    get_view(p_ext_buffer) = {.size = myvk_buffer->GetSize()};
				    // get_alloc(p_ext_buffer) = {.vk_usages = myvk_buffer->GetUsa}; TODO: External buffer usage
			    },
			    [](auto &&) {}));
			continue;
		}

		auto p_alloc = Dependency::GetRootResource(p_resource), p_view = p_resource;
		if (p_resource->GetState() == ResourceState::kLastFrame)
			p_alloc = p_view = Dependency::GetLFResource(p_alloc);

		auto &alloc_info = get_meta(p_resource);

		alloc_info.p_alloc_resource = p_alloc;
		alloc_info.p_view_resource = p_view;

		if (IsAllocResource(p_resource))
			m_alloc_resources.push_back(p_resource);
		if (IsViewResource(p_resource))
			m_view_resources.push_back(p_resource);
	}
}

void Metadata::propagate_resource_meta(const Args &args) {
	for (const ResourceBase *p_resource : args.dependency.GetResources()) {
		if (!IsAllocResource(p_resource) && GetAllocResource(p_resource)) {
			p_resource->Visit(
			    [](const auto *p_resource) { get_alloc(p_resource) = get_alloc(GetAllocResource(p_resource)); });
		}
		if (!IsViewResource(p_resource) && GetViewResource(p_resource)) {
			p_resource->Visit(
			    [](const auto *p_resource) { get_view(p_resource) = get_view(GetViewResource(p_resource)); });
		}
	}
}

void Metadata::fetch_alloc_sizes(const Args &args) {
	const auto get_size = [&](const auto &size_variant) {
		const auto get_size_visitor = overloaded(
		    [&](const std::invocable<VkExtent2D> auto &size_func) {
			    return size_func(args.render_graph.GetCanvasSize());
		    },
		    [](const auto &size) { return size; });

		return std::visit(get_size_visitor, size_variant);
	};

	const auto combine_size = [&](const LocalInternalImage auto *p_alloc_image) {
		auto &alloc = get_alloc(p_alloc_image);

		const auto combine_size_impl = [&](const LocalInternalImage auto *p_view_image, auto &&combine_size) {
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
					        [&](const LocalInternalImage auto *p_sub) {
						        combine_size(p_sub, combine_size);

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

				    return get_size(p_managed_image->GetSize());
			    })(p_view_image);
		};
		combine_size_impl(p_alloc_image, combine_size_impl);

		// Base Mip Level should be 0
		if (get_view(p_alloc_image).size.GetBaseMipLevel() != 0)
			Throw(error::ImageNotMerge{.key = p_alloc_image->GetGlobalKey()});

		// Accumulate Base Layer Offsets
		const auto accumulate_base_impl = overloaded(
		    [&](const CombinedImage *p_combined_image, auto &&accumulate_base) -> void {
			    for (auto [p_sub, _, _1] : args.dependency.GetResourceGraph().GetOutEdges(p_combined_image)) {

				    p_sub->Visit(overloaded(
				        [&](const LocalInternalImage auto *p_sub) {
					        get_view(p_sub).base_layer += get_view(p_combined_image).base_layer;
				        },
				        [](auto &&) {}));

				    p_sub->Visit(overloaded(
				        [&](const CombinedImage *p_sub) { accumulate_base(p_sub, accumulate_base); }, [](auto &&) {}));
			    }
		    },
		    [](auto &&, auto &&) {});
		accumulate_base_impl(p_alloc_image, accumulate_base_impl);
	};

	for (const ResourceBase *p_resource : m_alloc_resources)
		// Collect Size
		p_resource->Visit(overloaded([&](const LocalInternalImage auto *p_image) { combine_size(p_image); },
		                             [&](const ManagedBuffer *p_managed_buffer) {
			                             get_view(p_managed_buffer).size = get_size(p_managed_buffer->GetSize());
		                             },
		                             [](auto &&) {}));
}

void Metadata::fetch_alloc_usages(const Args &args) {
	for (const auto *p_pass : args.dependency.GetPasses()) {
		for (const InputBase *p_input : Dependency::GetPassInputs(p_pass))
			Dependency::GetInputResource(p_input)->Visit([&](const auto *p_resource) {
				if (GetAllocResource(p_resource))
					get_alloc(GetAllocResource(p_resource)).vk_usages |= UsageGetCreationUsages(p_input->GetUsage());
			});
	}
	for (const auto *p_resource : args.dependency.GetLFResources()) {
		p_resource->Visit(overloaded(
		    [&](const LastFrameImage *p_lf_image) {
			    if (p_lf_image->GetInitTransferFunc())
				    get_alloc(GetAllocResource(p_lf_image)).vk_usages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		    },
		    [&](const LastFrameBuffer *p_lf_buffer) {
			    if (p_lf_buffer->GetInitTransferFunc())
				    get_alloc(GetAllocResource(p_lf_buffer)).vk_usages |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
