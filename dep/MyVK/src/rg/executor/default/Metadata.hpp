//
// Created by adamyuan on 2/10/24.
//

#pragma once
#ifndef MYVK_DEF_EXE_RESOURCEMETA_HPP
#define MYVK_DEF_EXE_RESOURCEMETA_HPP

#include "Dependency.hpp"

namespace myvk_rg_executor {

class Metadata {
private:
	struct Args {
		const RenderGraphBase &render_graph;
		const Collection &collection;
		const Dependency &dependency;
	};

	std::vector<const ResourceBase *> m_alloc_resources, m_view_resources;

	void tag_resources(const Args &args);
	void fetch_alloc_sizes(const Args &p_view_image);
	static void fetch_alloc_usages(const Args &args);
	static void propagate_resource_meta(const Args &args);
	static void fetch_render_areas(const Args &args);

	static auto &get_meta(const PassBase *p_pass) { return GetPassInfo(p_pass).metadata; }
	static auto &get_meta(const ResourceBase *p_resource) { return GetResourceInfo(p_resource).metadata; }
	static auto &get_alloc(const ImageBase *p_image) { return get_meta(p_image).image_alloc; }
	static auto &get_alloc(const BufferBase *p_buffer) { return get_meta(p_buffer).buffer_alloc; }
	static auto &get_view(const ImageBase *p_image) { return get_meta(p_image).image_view; }
	static auto &get_view(const BufferBase *p_buffer) { return get_meta(p_buffer).buffer_view; }

public:
	static Metadata Create(const Args &args);

	// Alloc ID (Internal & Local & Physical Resources)
	inline std::size_t GetResourceAllocCount() const { return m_alloc_resources.size(); }
	inline const auto &GetAllocResources() const { return m_alloc_resources; }
	static const ResourceBase *GetAllocResource(const ResourceBase *p_resource) {
		return get_meta(p_resource).p_alloc_resource;
	}
	static const ImageBase *GetAllocResource(const ImageBase *p_image) {
		return static_cast<const ImageBase *>(get_meta(p_image).p_alloc_resource);
	}
	static const BufferBase *GetAllocResource(const BufferBase *p_buffer) {
		return static_cast<const BufferBase *>(get_meta(p_buffer).p_alloc_resource);
	}
	static bool IsAllocResource(const ResourceBase *p_resource) {
		return get_meta(p_resource).p_alloc_resource == p_resource;
	}
	static const auto &GetAllocInfo(const ImageBase *p_image) { return get_alloc(p_image); }
	static const auto &GetAllocInfo(const BufferBase *p_buffer) { return get_alloc(p_buffer); }

	// View ID (Internal & Local Resources)
	inline std::size_t GetResourceViewCount() const { return m_view_resources.size(); }
	inline const ResourceBase *GetViewResource(std::size_t view_id) const { return m_view_resources[view_id]; }
	inline const auto &GetViewResources() const { return m_view_resources; }
	static const ResourceBase *GetViewResource(const ResourceBase *p_resource) {
		return get_meta(p_resource).p_view_resource;
	}
	static const ImageBase *GetViewResource(const ImageBase *p_image) {
		return static_cast<const ImageBase *>(get_meta(p_image).p_view_resource);
	}
	static const BufferBase *GetViewResource(const BufferBase *p_buffer) {
		return static_cast<const BufferBase *>(get_meta(p_buffer).p_view_resource);
	}
	static bool IsViewResource(const ResourceBase *p_resource) {
		return get_meta(p_resource).p_view_resource == p_resource;
	}

	static const auto &GetViewInfo(const ImageBase *p_image) { return get_view(p_image); }
	static const auto &GetViewInfo(const BufferBase *p_buffer) { return get_view(p_buffer); }

	// Pass RenderArea
	static RenderPassArea GetPassRenderArea(const PassBase *p_pass) { return get_meta(p_pass).render_area; }
};

} // namespace myvk_rg_executor

#endif
