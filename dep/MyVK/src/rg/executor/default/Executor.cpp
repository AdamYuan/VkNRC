#include <myvk_rg/executor/Executor.hpp>

#include "Collection.hpp"
#include "Dependency.hpp"
#include "Metadata.hpp"
#include "Schedule.hpp"
#include "VkAllocation.hpp"
#include "VkCommand.hpp"
#include "VkDescriptor.hpp"
#include "VkRunner.hpp"

namespace myvk_rg::executor {

enum CompileFlag : uint8_t {
	kCollection = 1u,
	kDependency = 2u,
	kMetadata = 4u,
	kSchedule = 8u,
	kVkAllocation = 16u,
	kVkDescriptor = 32u,
	kVkCommand = 64u,
};

using interface::overloaded;
using myvk_rg_executor::Collection;
using myvk_rg_executor::Dependency;
using myvk_rg_executor::Metadata;
using myvk_rg_executor::Schedule;
using myvk_rg_executor::VkAllocation;
using myvk_rg_executor::VkCommand;
using myvk_rg_executor::VkDescriptor;
using myvk_rg_executor::VkRunner;

struct Executor::CompileInfo {
	Collection collection;
	Dependency dependency;
	Metadata metadata;
	Schedule schedule;
	VkAllocation vk_allocation;
	VkCommand vk_command;
	VkDescriptor vk_descriptor;
};

Executor::Executor(interface::Parent parent) : interface::ObjectBase(parent), m_p_compile_info{new CompileInfo{}} {}
Executor::~Executor() { delete m_p_compile_info; }

void Executor::OnEvent(interface::ObjectBase *p_object, interface::Event event) {
	using interface::Event;
	switch (event) {
	case Event::kPassChanged:
	case Event::kResourceChanged:
	case Event::kInputChanged:
		m_compile_flags |= kCollection;
		break;
	case Event::kResultChanged:
		m_compile_flags |= kDependency;
		break;
	case Event::kCanvasResized:
	case Event::kBufferResized:
	case Event::kBufferMappedChanged:
	case Event::kImageResized:
	case Event::kRenderAreaChanged:
		m_compile_flags |= kMetadata;
		break;
	case Event::kAttachmentChanged:
		m_compile_flags |= kSchedule;
		break;
	case Event::kDescriptorChanged:
		m_compile_flags |= kVkDescriptor;
		break;
	case Event::kExternalImageLayoutChanged:
	case Event::kExternalAccessChanged:
	case Event::kExternalStageChanged:
	case Event::kExternalSyncChanged:
	case Event::kImageLoadOpChanged:
		m_compile_flags |= kVkCommand;
		break;
	case Event::kUpdatePipeline:
		VkCommand::UpdatePipeline(static_cast<const interface::PassBase *>(p_object));
		break;
	}
}

void Executor::compile(const interface::RenderGraphBase *p_render_graph, const myvk::Ptr<myvk::Queue> &queue) {
	/* digraph {
	    Collection -> Dependency;
	    Dependency -> Metadata;
	    Metadata -> Schedule;
	    Metadata -> VkAllocation;
	    VkAllocation -> VkCommand;
	    Schedule -> VkCommand;
	    VkAllocation -> VkDescriptor;
	} */
	if (m_compile_flags == 0)
		return;

	uint8_t exe_compile_flags = m_compile_flags;
	if (m_compile_flags & kCollection)
		exe_compile_flags |= kDependency | kMetadata | kSchedule | kVkAllocation | kVkCommand | kVkDescriptor;
	if (m_compile_flags & kDependency)
		exe_compile_flags |= kMetadata | kSchedule | kVkAllocation | kVkCommand | kVkDescriptor;
	if (m_compile_flags & kMetadata)
		exe_compile_flags |= kSchedule | kVkAllocation | kVkCommand | kVkDescriptor;
	if (m_compile_flags & kSchedule)
		exe_compile_flags |= kVkCommand;
	if (m_compile_flags & kVkAllocation)
		exe_compile_flags |= kVkCommand | kVkDescriptor;
	m_compile_flags = 0u;

	if (exe_compile_flags & kCollection)
		m_p_compile_info->collection = Collection::Create(*p_render_graph);
	if (exe_compile_flags & kDependency)
		m_p_compile_info->dependency =
		    Dependency::Create({.render_graph = *p_render_graph, .collection = m_p_compile_info->collection});
	if (exe_compile_flags & kMetadata)
		m_p_compile_info->metadata = Metadata::Create({.render_graph = *p_render_graph,
		                                               .collection = m_p_compile_info->collection,
		                                               .dependency = m_p_compile_info->dependency});
	if (exe_compile_flags & kSchedule)
		m_p_compile_info->schedule = Schedule::Create({.render_graph = *p_render_graph,
		                                               .collection = m_p_compile_info->collection,
		                                               .dependency = m_p_compile_info->dependency,
		                                               .metadata = m_p_compile_info->metadata});
	if (exe_compile_flags & kVkAllocation)
		m_p_compile_info->vk_allocation =
		    VkAllocation::Create(queue->GetDevicePtr(), {.render_graph = *p_render_graph,
		                                                 .collection = m_p_compile_info->collection,
		                                                 .dependency = m_p_compile_info->dependency,
		                                                 .metadata = m_p_compile_info->metadata});
	if (exe_compile_flags & kVkDescriptor)
		m_p_compile_info->vk_descriptor =
		    VkDescriptor::Create(queue->GetDevicePtr(), {.render_graph = *p_render_graph,
		                                                 .collection = m_p_compile_info->collection,
		                                                 .dependency = m_p_compile_info->dependency,
		                                                 .metadata = m_p_compile_info->metadata,
		                                                 .vk_allocation = m_p_compile_info->vk_allocation});
	if (exe_compile_flags & kVkCommand)
		m_p_compile_info->vk_command =
		    VkCommand::Create(queue->GetDevicePtr(), {.render_graph = *p_render_graph,
		                                              .collection = m_p_compile_info->collection,
		                                              .dependency = m_p_compile_info->dependency,
		                                              .metadata = m_p_compile_info->metadata,
		                                              .schedule = m_p_compile_info->schedule,
		                                              .vk_allocation = m_p_compile_info->vk_allocation});

	VkRunner::Create({.render_graph = *p_render_graph,
	                  .collection = m_p_compile_info->collection,
	                  .dependency = m_p_compile_info->dependency,
	                  .metadata = m_p_compile_info->metadata,
	                  .schedule = m_p_compile_info->schedule,
	                  .vk_allocation = m_p_compile_info->vk_allocation,
	                  .vk_command = m_p_compile_info->vk_command,
	                  .vk_descriptor = m_p_compile_info->vk_descriptor});
}

void Executor::CmdExecute(const interface::RenderGraphBase *p_render_graph,
                          const myvk::Ptr<myvk::CommandBuffer> &command_buffer) {
	const auto &queue = command_buffer->GetCommandPoolPtr()->GetQueuePtr();
	compile(p_render_graph, queue);
	p_render_graph->PreExecute();
	VkRunner::Run(command_buffer, {.render_graph = *p_render_graph,
	                               .collection = m_p_compile_info->collection,
	                               .dependency = m_p_compile_info->dependency,
	                               .metadata = m_p_compile_info->metadata,
	                               .schedule = m_p_compile_info->schedule,
	                               .vk_allocation = m_p_compile_info->vk_allocation,
	                               .vk_command = m_p_compile_info->vk_command,
	                               .vk_descriptor = m_p_compile_info->vk_descriptor});
}

const myvk::Ptr<myvk::ImageView> &Executor::GetVkImageView(const interface::ManagedImage *p_managed_image) {
	return VkAllocation::GetVkImageView(p_managed_image);
}
const myvk::Ptr<myvk::ImageView> &Executor::GetVkImageView(const interface::CombinedImage *p_combined_image) {
	return VkAllocation::GetVkImageView(p_combined_image);
}

const interface::BufferView &Executor::GetBufferView(const interface::ManagedBuffer *p_managed_buffer) {
	return VkAllocation::GetBufferView(p_managed_buffer);
}
const interface::BufferView &Executor::GetBufferView(const interface::CombinedBuffer *p_combined_buffer) {
	return VkAllocation::GetBufferView(p_combined_buffer);
}

uint32_t Executor::GetSubpass(const interface::PassBase *p_pass) { return Schedule::GetU32SubpassID(p_pass); }
const myvk::Ptr<myvk::RenderPass> &Executor::GetVkRenderPass(const interface::PassBase *p_pass) const {
	return m_p_compile_info->vk_command.GetPassCommands()[Schedule::GetGroupID(p_pass)].myvk_render_pass;
}
const myvk::Ptr<myvk::DescriptorSetLayout> &Executor::GetVkDescriptorSetLayout(const interface::PassBase *p_pass) {
	return VkDescriptor::GetVkDescriptorSetLayout(p_pass);
}
const myvk::Ptr<myvk::DescriptorSet> &Executor::GetVkDescriptorSet(const interface::PassBase *p_pass) {
	return VkDescriptor::GetVkDescriptorSet(p_pass);
}
const interface::ImageBase *Executor::GetInputImage(const interface::InputBase *p_input) {
	assert(Dependency::GetInputResource(p_input)->GetType() == interface::ResourceType::kImage);
	return static_cast<const interface::ImageBase *>(Dependency::GetInputResource(p_input));
}
const interface::BufferBase *Executor::GetInputBuffer(const interface::InputBase *p_input) {
	assert(Dependency::GetInputResource(p_input)->GetType() == interface::ResourceType::kBuffer);
	return static_cast<const interface::BufferBase *>(Dependency::GetInputResource(p_input));
}

void *Executor::GetMappedData(const interface::ManagedBuffer *p_managed_buffer) {
	return VkAllocation::GetMappedData(p_managed_buffer);
}
void *Executor::GetMappedData(const interface::CombinedBuffer *p_combined_buffer) {
	return VkAllocation::GetMappedData(p_combined_buffer);
}
const myvk::Ptr<myvk::PipelineBase> &Executor::GetVkPipeline(const interface::PassBase *p_pass) {
	return VkCommand::GetVkPipeline(p_pass);
}

} // namespace myvk_rg::executor