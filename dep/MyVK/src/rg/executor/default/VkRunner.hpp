//
// Created by adamyuan on 2/16/24.
//

#pragma once
#ifndef MYVK_VKRUNNER_HPP
#define MYVK_VKRUNNER_HPP

#include "VkCommand.hpp"
#include "VkDescriptor.hpp"

#include <span>

namespace myvk_rg_executor {

class VkRunner {
private:
	struct Args {
		const RenderGraphBase &render_graph;
		const Collection &collection;
		const Dependency &dependency;
		const Metadata &metadata;
		const Schedule &schedule;
		const VkAllocation &vk_allocation;
		const VkCommand &vk_command;
		const VkDescriptor &vk_descriptor;
	};

	static auto &get_runner_cache(const ResourceBase *p_resource) { return GetResourceInfo(p_resource).vk_runner; }

	static void cmd_pipeline_barriers(const myvk::Ptr<myvk::CommandBuffer> &command_buffer,
	                                  std::span<const BarrierCmd> barrier_cmds);
	static void update_ext_cache(const Args &args);

public:
	static VkRunner Create(const Args &args);
	static void Run(const myvk::Ptr<myvk::CommandBuffer> &command_buffer, const Args &args);
	static bool IsExtChanged(const ResourceBase *p_resource) { return get_runner_cache(p_resource).ext_changed; }
};

} // namespace myvk_rg_executor

#endif // MYVK_VKRUNNER_HPP
