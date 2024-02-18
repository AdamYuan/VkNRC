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
	static void cmd_pipeline_barriers(const myvk::Ptr<myvk::CommandBuffer> &command_buffer,
	                                  std::span<const BarrierCmd> barrier_cmds);

public:
	static void LastFrameInit(const myvk::Ptr<myvk::Queue> &queue, const Dependency &dependency);
	static void Run(const myvk::Ptr<myvk::CommandBuffer> &command_buffer, const VkCommand &vk_command,
	                const VkDescriptor &vk_descriptor, bool flip);
};

} // namespace myvk_rg_executor

#endif // MYVK_VKRUNNER_HPP
