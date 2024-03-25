//
// Created by adamyuan on 2/12/24.
//

#pragma once
#ifndef MYVK_RG_DEF_EXE_VKCOMMAND_HPP
#define MYVK_RG_DEF_EXE_VKCOMMAND_HPP

#include "../Barrier.hpp"
#include "Schedule.hpp"
#include "VkAllocation.hpp"

namespace myvk_rg_executor {

class VkCommand {
public:
	struct PassCmd {
		std::span<const PassBase *const> subpasses; // pointed to subpasses in Schedule::PassGroup
		std::vector<BarrierCmd> prior_barriers;

		myvk::Ptr<myvk::RenderPass> myvk_render_pass;
		myvk::Ptr<myvk::ImagelessFramebuffer> myvk_framebuffer;
		std::vector<const ImageBase *> attachments;
		bool has_clear_values;
	};

private:
	struct Args {
		const RenderGraphBase &render_graph;
		const Collection &collection;
		const Dependency &dependency;
		const Metadata &metadata;
		const Schedule &schedule;
		const VkAllocation &vk_allocation;
	};
	class Builder;

	std::vector<PassCmd> m_pass_commands;
	std::vector<BarrierCmd> m_post_barriers;

public:
	static VkCommand Create(const myvk::Ptr<myvk::Device> &device_ptr, const Args &args);
	inline const auto &GetPassCommands() const { return m_pass_commands; }
	inline const auto &GetPostBarriers() const { return m_post_barriers; }
	static void CreatePipeline(const PassBase *p_pass) {
		if (GetPassInfo(p_pass).vk_command.update_pipeline) {
			GetPassInfo(p_pass).vk_command.update_pipeline = false;
			GetPassInfo(p_pass).vk_command.vk_pipeline = p_pass->Visit(overloaded(
			    [](PassWithPipeline auto *p_pipeline_pass) -> myvk::Ptr<myvk::PipelineBase> {
				    return p_pipeline_pass->CreatePipeline();
			    },
			    [](auto &&) -> myvk::Ptr<myvk::PipelineBase> { return nullptr; }));
		}
	}
	static const myvk::Ptr<myvk::PipelineBase> &GetVkPipeline(const PassBase *p_pass) {
		return GetPassInfo(p_pass).vk_command.vk_pipeline;
	}
	static void UpdatePipeline(const PassBase *p_pass) { GetPassInfo(p_pass).vk_command.update_pipeline = true; }
};

} // namespace myvk_rg_executor

#endif
