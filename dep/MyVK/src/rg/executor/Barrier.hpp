//
// Created by adamyuan on 2/16/24.
//

#pragma once
#ifndef MYVK_RG_EXE_BARRIER_HPP
#define MYVK_RG_EXE_BARRIER_HPP

#include "VkHelper.hpp"
#include <myvk_rg/interface/Input.hpp>
#include <myvk_rg/interface/Usage.hpp>

namespace myvk_rg::executor {

struct BarrierCmd {
	const interface::ResourceBase *p_resource;
	VkPipelineStageFlags2 src_stage_mask;
	VkPipelineStageFlags2 dst_stage_mask;
	VkAccessFlags2 src_access_mask;
	VkAccessFlags2 dst_access_mask;
	VkImageLayout old_layout;
	VkImageLayout new_layout;
};

struct Barrier {
	VkPipelineStageFlags2 src_stage_mask;
	VkPipelineStageFlags2 dst_stage_mask;
	VkAccessFlags2 src_access_mask;
	VkAccessFlags2 dst_access_mask;
	VkImageLayout old_layout;
	VkImageLayout new_layout;
};

inline static void SetVkLayout(VkImageLayout *p_dst, VkImageLayout src) {
	assert(*p_dst == VK_IMAGE_LAYOUT_UNDEFINED || src == VK_IMAGE_LAYOUT_UNDEFINED || *p_dst == src);
	if (*p_dst == VK_IMAGE_LAYOUT_UNDEFINED)
		*p_dst = src;
}

struct State {
	VkPipelineStageFlags2 stage_mask{};
	VkAccessFlags2 access_mask{};
	VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};

	inline State &operator|=(const State &r) {
		stage_mask |= r.stage_mask;
		access_mask |= r.access_mask;
		SetVkLayout(&layout, r.layout);
		return *this;
	}
	inline State operator|(const State &r) const {
		State l = *this;
		l |= r;
		return l;
	}
};

inline static State GetSrcState(const interface::InputBase *p_src) {
	return {.stage_mask = p_src->GetPipelineStages(),
	        .access_mask = interface::UsageGetWriteAccessFlags(p_src->GetUsage()),
	        .layout = interface::UsageGetImageLayout(p_src->GetUsage())};
}
inline static State GetSrcState(std::ranges::input_range auto src_s) {
	State state = {};
	for (const interface::InputBase *p_src : src_s)
		state |= GetSrcState(p_src);
	return state;
}
inline static State GetValidateSrcState(std::ranges::input_range auto src_s) {
	State state = {};
	for (const interface::InputBase *p_src : src_s)
		state |= {.stage_mask = p_src->GetPipelineStages()};
	return state;
}
inline static State GetDstState(const interface::InputBase *p_dst) {
	return {.stage_mask = p_dst->GetPipelineStages(),
	        .access_mask = interface::UsageGetAccessFlags(p_dst->GetUsage()),
	        .layout = interface::UsageGetImageLayout(p_dst->GetUsage())};
}
inline static State GetDstState(std::ranges::input_range auto dst_s) {
	State state = {};
	for (const interface::InputBase *p_dst : dst_s)
		state |= GetDstState(p_dst);
	return state;
}

template <typename T> inline static void AddSrcBarrier(T *p_barrier, const State &state) {
	p_barrier->src_access_mask |= state.access_mask;
	p_barrier->src_stage_mask |= state.stage_mask;
	if constexpr (requires(T t) { t.old_layout; })
		SetVkLayout(&p_barrier->old_layout, state.layout);
}

template <typename T> inline static void AddDstBarrier(T *p_barrier, const State &state) {
	p_barrier->dst_access_mask |= state.access_mask;
	p_barrier->dst_stage_mask |= state.stage_mask;
	if constexpr (requires(T t) { t.new_layout; })
		SetVkLayout(&p_barrier->new_layout, state.layout);
}
template <typename T> inline static void AddBarrier(T *p_barrier, const State &src_state, const State &dst_state) {
	AddSrcBarrier(p_barrier, src_state);
	AddDstBarrier(p_barrier, dst_state);
}

template <typename T> inline static bool IsValidBarrier(const T &barrier) {
	bool layout_transition = false;
	if constexpr (requires(T t) {
		              t.old_layout;
		              t.new_layout;
	              })
		layout_transition = barrier.old_layout != barrier.new_layout && barrier.new_layout != VK_IMAGE_LAYOUT_UNDEFINED;

	return (barrier.src_stage_mask | barrier.src_access_mask) && (barrier.dst_stage_mask | barrier.dst_access_mask) ||
	       layout_transition;
}
template <typename Src_T, typename Dst_T> inline static void CopyBarrier(const Src_T &src, Dst_T *p_dst) {
	p_dst->src_stage_mask = src.src_stage_mask;
	p_dst->dst_stage_mask = src.dst_stage_mask;
	p_dst->src_access_mask = src.src_access_mask;
	p_dst->dst_access_mask = src.dst_access_mask;
	if constexpr (requires(Src_T src_t, Dst_T dst_t) {
		              src_t.old_layout;
		              src_t.new_layout;
		              dst_t.old_layout;
		              dst_t.new_layout;
	              }) {
		p_dst->old_layout = src.old_layout;
		p_dst->new_layout = src.new_layout;
	}
}

template <typename Src_T, typename Dst_T> inline static void CopyVkBarrier(const Src_T &src, Dst_T *p_dst) {
	p_dst->srcStageMask = src.src_stage_mask;
	p_dst->dstStageMask = src.dst_stage_mask;
	p_dst->srcAccessMask = src.src_access_mask;
	p_dst->dstAccessMask = src.dst_access_mask;
	if constexpr (requires(Src_T src_t, Dst_T dst_t) {
		              src_t.old_layout;
		              src_t.new_layout;
		              dst_t.oldLayout;
		              dst_t.newLayout;
	              }) {
		p_dst->oldLayout = src.old_layout;
		p_dst->newLayout = src.new_layout == VK_IMAGE_LAYOUT_UNDEFINED ? src.old_layout : src.new_layout;
	}
}

} // namespace myvk_rg::executor

#endif // MYVK_BARRIER_HPP
