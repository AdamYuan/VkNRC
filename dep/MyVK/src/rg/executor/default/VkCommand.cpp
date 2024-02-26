//
// Created by adamyuan on 2/12/24.
//

#include "VkCommand.hpp"

#include "../VkHelper.hpp"

#include <cassert>

namespace myvk_rg_executor {

inline static State GetAttachmentStoreOpState(const ResourceBase *p_attachment, VkAttachmentStoreOp vk_store_op) {
	auto vk_format = Metadata::GetAllocInfo(static_cast<const ImageBase *>(p_attachment)).vk_format;
	return {.stage_mask = VkAttachmentStoreOpStages(vk_format),
	        .access_mask = VkAttachmentStoreOpAccesses(vk_store_op, vk_format)};
}
inline static State GetAttachmentLoadOpState(const ResourceBase *p_attachment, VkAttachmentLoadOp vk_load_op) {
	auto vk_format = Metadata::GetAllocInfo(static_cast<const ImageBase *>(p_attachment)).vk_format;
	return {.stage_mask = VkAttachmentLoadOpStages(vk_format),
	        .access_mask = VkAttachmentLoadOpAccesses(vk_load_op, vk_format)};
}
inline static State GetLastInputSrcState(const ResourceBase *p_resource) {
	return GetSrcState(Schedule::GetLastInputs(p_resource));
}
inline static State GetLastInputSrcState(const ResourceBase *p_resource, VkAttachmentStoreOp vk_store_op) {
	State src_state = GetLastInputSrcState(p_resource);
	// If LastInput is Attachment, add StoreOp States
	if (UsageIsAttachment(Schedule::GetLastInputs(p_resource)[0]->GetUsage()))
		src_state |= GetAttachmentStoreOpState(p_resource, vk_store_op);
	return src_state;
}
inline static State GetLastInputValidateSrcState(const ResourceBase *p_resource) {
	return GetValidateSrcState(Schedule::GetLastInputs(p_resource));
}
inline static State GetLastInputValidateSrcState(const ResourceBase *p_resource, VkAttachmentStoreOp vk_store_op) {
	State src_state = GetLastInputValidateSrcState(p_resource);
	// If LastInput is Attachment, add StoreOp Stages
	if (UsageIsAttachment(Schedule::GetLastInputs(p_resource)[0]->GetUsage()))
		src_state.stage_mask |= GetAttachmentStoreOpState(p_resource, vk_store_op).stage_mask;
	return src_state;
}

class VkCommand::Builder {
private:
	struct SubpassPair {
		uint32_t src_subpass, dst_subpass;
		inline auto operator<=>(const SubpassPair &r) const = default;
	};
	struct SubpassDependency {
		VkPipelineStageFlags2 src_stage_mask{};
		VkAccessFlags2 src_access_mask{};
		VkPipelineStageFlags2 dst_stage_mask{};
		VkAccessFlags2 dst_access_mask{};
	};
	struct AttachmentData {
		VkImageLayout initial_layout{}, final_layout{};
		VkAttachmentLoadOp load_op{VK_ATTACHMENT_LOAD_OP_DONT_CARE};
		VkAttachmentStoreOp store_op{VK_ATTACHMENT_STORE_OP_NONE};
		bool may_alias{false};
		uint32_t id{};
		uint32_t first_subpass{}, last_subpass{};
	};
	struct PassData {
		std::span<const PassBase *const> subpasses; // pointed to subpasses in Schedule::PassGroup
		std::unordered_map<const ResourceBase *, Barrier> prior_barriers;
		std::unordered_map<SubpassPair, SubpassDependency,
		                   U32PairHash<SubpassPair, &SubpassPair::src_subpass, &SubpassPair::dst_subpass>>
		    by_region_subpass_deps, subpass_deps;
		std::unordered_map<const ImageBase *, AttachmentData> attachment_data_s;
	};

	std::vector<PassData> m_pass_data_s;
	std::unordered_map<const ResourceBase *, Barrier> m_post_barriers;

	void make_pass_data(const Args &args) {
		m_pass_data_s.reserve(args.schedule.GetPassGroups().size());
		for (const auto &pass_group : args.schedule.GetPassGroups()) {
			m_pass_data_s.emplace_back();
			auto &pass_data = m_pass_data_s.back();
			pass_data.subpasses = pass_group.subpasses;
			// Push Internal Subpass Dependencies and Attachments
			for (const auto &subpass_dep : pass_group.subpass_deps) {
				const PassBase *p_src_pass = Dependency::GetInputPass(subpass_dep.p_src),
				               *p_dst_pass = Dependency::GetInputPass(subpass_dep.p_dst);
				uint32_t src_subpass = Schedule::GetU32SubpassID(p_src_pass),
				         dst_subpass = Schedule::GetU32SubpassID(p_dst_pass);
				AddBarrier(&pass_data.by_region_subpass_deps[{src_subpass, dst_subpass}],
				           GetSrcState(subpass_dep.p_src), GetDstState(subpass_dep.p_dst));
				pass_data.attachment_data_s[subpass_dep.p_attachment];
			}
		}
	}

	inline PassData *get_p_pass_data(const InputBase *p_input) {
		std::size_t group_id = Schedule::GetGroupID(Dependency::GetInputPass(p_input));
		return &m_pass_data_s[group_id];
	}
	inline std::tuple<PassData *, AttachmentData *, uint32_t>
	get_p_pass_att_data(const ResourceBase *p_resource, const std::span<const InputBase *const> &inputs) {
		if (UsageIsAttachment(inputs[0]->GetUsage())) {
			assert(inputs.size() == 1);
			assert(p_resource->GetType() == ResourceType::kImage);
			auto p_image = static_cast<const ImageBase *>(p_resource);
			auto p_pass_data = get_p_pass_data(inputs[0]);
			return {p_pass_data, &p_pass_data->attachment_data_s[p_image],
			        Schedule::GetU32SubpassID(Dependency::GetInputPass(inputs[0]))};
		}
		return {nullptr, nullptr, VK_SUBPASS_EXTERNAL};
	}
	inline auto get_src_p_pass_att_data(const Schedule::PassBarrier &pass_barrier) {
		return get_p_pass_att_data(pass_barrier.p_resource, pass_barrier.src_s);
	}
	inline auto get_dst_p_pass_att_data(const Schedule::PassBarrier &pass_barrier) {
		return get_p_pass_att_data(pass_barrier.p_resource, pass_barrier.dst_s);
	}
	Barrier *get_p_barrier_data(const Schedule::PassBarrier &pass_barrier) {
		if (pass_barrier.dst_s.empty())
			return &m_post_barriers[pass_barrier.p_resource];
		// It is guaranteed dst_s[0] has the smallest GroupID
		return &get_p_pass_data(pass_barrier.dst_s[0])->prior_barriers[pass_barrier.p_resource];
	}

	void add_local_barrier(const Schedule::PassBarrier &pass_barrier) {
		State src_state = GetSrcState(pass_barrier.src_s), dst_state = GetDstState(pass_barrier.dst_s);
		if (auto [_, p_src_att_data, _1] = get_src_p_pass_att_data(pass_barrier); p_src_att_data) {
			p_src_att_data->store_op = VK_ATTACHMENT_STORE_OP_STORE;
			p_src_att_data->final_layout = UsageGetImageLayout(pass_barrier.src_s[0]->GetUsage());
			src_state |= GetAttachmentStoreOpState(pass_barrier.p_resource, VK_ATTACHMENT_STORE_OP_STORE);
		}
		if (auto [_, p_dst_att_data, _1] = get_dst_p_pass_att_data(pass_barrier); p_dst_att_data) {
			p_dst_att_data->load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
			p_dst_att_data->initial_layout = UsageGetImageLayout(pass_barrier.dst_s[0]->GetUsage());
			dst_state |= GetAttachmentLoadOpState(pass_barrier.p_resource, VK_ATTACHMENT_LOAD_OP_LOAD);
		}
		AddBarrier(get_p_barrier_data(pass_barrier), src_state, dst_state);
	}

	void add_validate_barrier(const Args &args, const Schedule::PassBarrier &pass_barrier, VkAttachmentLoadOp load_op) {
		std::vector<const ResourceBase *> alias_resources;
		for (const ResourceBase *p_resource : args.dependency.GetRootResources()) {
			if (args.dependency.IsResourceLess(p_resource, pass_barrier.p_resource) &&
			    args.vk_allocation.IsAliased(p_resource, pass_barrier.p_resource))
				alias_resources.push_back(p_resource);
		}

		State dst_state = GetDstState(pass_barrier.dst_s);

		if (auto [p_dst_pass_data, p_dst_att_data, dst_subpass] = get_dst_p_pass_att_data(pass_barrier);
		    p_dst_att_data) {
			// Dst is a RenderPass, so no need for explicit layout transition
			// p_dst_att_data->initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;
			dst_state |= GetAttachmentLoadOpState(pass_barrier.p_resource, load_op);
			p_dst_att_data->load_op = load_op;

			for (const ResourceBase *p_resource : alias_resources) {
				SubpassDependency *p_subpass_dep;
				if (auto [p_src_pass_data, p_src_att_data, src_subpass] =
				        get_p_pass_att_data(p_resource, Schedule::GetLastInputs(p_resource));
				    p_src_att_data && p_src_pass_data == p_dst_pass_data) {
					// From the same RenderPass
					p_subpass_dep = &p_dst_pass_data->subpass_deps[{src_subpass, dst_subpass}];
					AddBarrier(p_subpass_dep, GetLastInputValidateSrcState(p_resource), dst_state);
				} else {
					// Not the same RenderPass
					p_subpass_dep = &p_dst_pass_data->subpass_deps[{VK_SUBPASS_EXTERNAL, dst_subpass}];
					AddBarrier(p_subpass_dep, GetLastInputValidateSrcState(p_resource, VK_ATTACHMENT_STORE_OP_NONE),
					           dst_state);
				}
			}
		} else {
			auto *p_barrier = get_p_barrier_data(pass_barrier);
			// Might need explicit initial image layout transition
			AddDstBarrier(p_barrier, dst_state);
			for (const ResourceBase *p_resource : alias_resources)
				AddSrcBarrier(p_barrier, GetLastInputValidateSrcState(p_resource, VK_ATTACHMENT_STORE_OP_NONE));
		}
	}

	void add_input_barrier(const Schedule::PassBarrier &pass_barrier, const State &src_state,
	                       VkAttachmentLoadOp load_op) {
		State dst_state = GetDstState(pass_barrier.dst_s);

		if (auto [p_dst_pass_data, p_dst_att_data, dst_subpass] = get_dst_p_pass_att_data(pass_barrier);
		    p_dst_att_data) {
			dst_state |= GetAttachmentLoadOpState(pass_barrier.p_resource, load_op);
			p_dst_att_data->load_op = load_op; // Input ==> Load
			p_dst_att_data->initial_layout = src_state.layout;

			SubpassDependency *p_subpass_dep = &p_dst_pass_data->subpass_deps[{VK_SUBPASS_EXTERNAL, dst_subpass}];
			AddBarrier(p_subpass_dep, src_state, dst_state);
		} else {
			auto *p_barrier = get_p_barrier_data(pass_barrier);
			AddBarrier(p_barrier, src_state, dst_state);
		}
	}

	void add_output_barrier(const Schedule::PassBarrier &pass_barrier, const State &dst_state) {
		State src_state = GetSrcState(pass_barrier.src_s);

		if (auto [p_src_pass_data, p_src_att_data, src_subpass] = get_src_p_pass_att_data(pass_barrier);
		    p_src_att_data) {
			VkAttachmentStoreOp store_op = dst_state.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_ATTACHMENT_STORE_OP_NONE
			                                                                             : VK_ATTACHMENT_STORE_OP_STORE;
			src_state |= GetAttachmentStoreOpState(pass_barrier.p_resource, store_op);
			p_src_att_data->store_op = store_op;
			p_src_att_data->final_layout = dst_state.layout;

			SubpassDependency *p_subpass_dep = &p_src_pass_data->subpass_deps[{src_subpass, VK_SUBPASS_EXTERNAL}];
			AddBarrier(p_subpass_dep, src_state, dst_state);
		} else {
			auto *p_barrier = get_p_barrier_data(pass_barrier);
			AddBarrier(p_barrier, src_state, dst_state);
		}
	}

	void make_barriers(const Args &args) {
		const auto get_load_op =
		    overloaded([](const AttachmentImage auto *p_att_image) { return p_att_image->GetLoadOp(); },
		               [](auto &&) { return VK_ATTACHMENT_LOAD_OP_DONT_CARE; });
		const auto get_ext_src = overloaded(
		    [](const ExternalResource auto *p_ext_resource) -> State {
			    switch (p_ext_resource->GetSyncType()) {
			    case myvk_rg::ExternalSyncType::kLastFrame:
				    return Schedule::IsExtReadOnly(p_ext_resource)
				               ? State{.layout = Schedule::GetLastVkLayout(p_ext_resource)}
				               : GetLastInputSrcState(p_ext_resource, VK_ATTACHMENT_STORE_OP_STORE);
			    case myvk_rg::ExternalSyncType::kCustom:
				    return {.stage_mask = p_ext_resource->GetSrcPipelineStages(),
				            .access_mask = p_ext_resource->GetSrcAccessFlags(),
				            .layout = p_ext_resource->GetSrcLayout()};
			    }
			    return {};
		    },
		    [](auto &&) -> State { return {}; });
		const auto get_ext_dst = overloaded(
		    [](const ExternalResource auto *p_ext_resource) -> State {
			    switch (p_ext_resource->GetSyncType()) {
			    case myvk_rg::ExternalSyncType::kLastFrame:
				    return {.layout = Schedule::GetLastVkLayout(p_ext_resource)};
			    case myvk_rg::ExternalSyncType::kCustom:
				    return {.stage_mask = p_ext_resource->GetDstPipelineStages(),
				            .access_mask = p_ext_resource->GetDstAccessFlags(),
				            .layout = p_ext_resource->GetDstLayout()};
			    }
			    return {};
		    },
		    [](auto &&) -> State { return {}; });

		for (const auto &pass_barrier : args.schedule.GetPassBarriers()) {
			switch (pass_barrier.type) {
			case Schedule::BarrierType::kLocal:
				add_local_barrier(pass_barrier);
				break;
			case Schedule::BarrierType::kIntValidate:
				add_validate_barrier(args, pass_barrier, pass_barrier.p_resource->Visit(get_load_op));
				break;
			case Schedule::BarrierType::kExtInput:
				add_input_barrier(pass_barrier, pass_barrier.p_resource->Visit(get_ext_src),
				                  pass_barrier.p_resource->Visit(get_load_op));
				break;
			case Schedule::BarrierType::kExtOutput:
				add_output_barrier(pass_barrier, pass_barrier.p_resource->Visit(get_ext_dst));
				break;
			}
		}
	}

	void finalize_attachments(const Args &args) {
		for (auto &pass_data : m_pass_data_s) {
			auto &att_data_s = pass_data.attachment_data_s;
			// Assign Attachment ID
			for (uint32_t att_id = 0; auto &[_, att_data] : att_data_s)
				att_data.id = att_id++;

			// Assign First/Last Subpass
			for (auto &[p_image, att_data] : att_data_s) {
				if (att_data.load_op == VK_ATTACHMENT_LOAD_OP_LOAD)
					att_data.first_subpass = 0;
				else {
					assert(Schedule::GetFirstInputs(p_image).size() == 1 &&
					       Schedule::GetGroupID(Dependency::GetInputPass(Schedule::GetFirstInputs(p_image)[0])) ==
					           &pass_data - m_pass_data_s.data());
					// Attachment is first use during the RenderPass
					att_data.first_subpass =
					    Schedule::GetU32SubpassID(Dependency::GetInputPass(Schedule::GetFirstInputs(p_image)[0]));
				}
				if (att_data.store_op == VK_ATTACHMENT_STORE_OP_STORE)
					att_data.last_subpass = pass_data.subpasses.size();
				else {
					assert(Schedule::GetLastInputs(p_image).size() == 1 &&
					       Schedule::GetGroupID(Dependency::GetInputPass(Schedule::GetLastInputs(p_image)[0])) ==
					           &pass_data - m_pass_data_s.data());
					// Attachment is not used after the RenderPass
					att_data.last_subpass =
					    Schedule::GetU32SubpassID(Dependency::GetInputPass(Schedule::GetLastInputs(p_image)[0]));
				}
			}

			// Check Resource Aliasing
			for (auto l_it = att_data_s.begin(); l_it != att_data_s.end(); ++l_it) {
				auto r_it = l_it;
				++r_it;

				for (; r_it != att_data_s.end(); ++r_it) {
					if (args.vk_allocation.IsAliased(l_it->first, r_it->first))
						l_it->second.may_alias = r_it->second.may_alias = true;
				}
			}

			// Deal-with UNDEFINED final_layout
			for (auto &[p_image, att_data] : att_data_s)
				if (att_data.final_layout == VK_IMAGE_LAYOUT_UNDEFINED)
					att_data.final_layout = Schedule::GetLastVkLayout(p_image);
		}
	}

	inline static void pop_barriers(const std::unordered_map<const ResourceBase *, Barrier> &in,
	                                std::vector<BarrierCmd> *p_out) {
		for (const auto &[p_resource, barrier] : in)
			if (IsValidBarrier(barrier)) {
				p_out->push_back({.p_resource = p_resource});
				CopyBarrier(barrier, &p_out->back());
			}
	};

	inline static void pop_pass(const myvk::Ptr<myvk::Device> &device_ptr, const PassData &in, PassCmd *p_out);

	void pop_pass_commands(const myvk::Ptr<myvk::Device> &device_ptr, VkCommand *p_target) const {
		p_target->m_pass_commands.reserve(m_pass_data_s.size());
		for (const auto &pass_data : m_pass_data_s) {
			p_target->m_pass_commands.emplace_back();
			pop_pass(device_ptr, pass_data, &p_target->m_pass_commands.back());
		}
	}

public:
	inline explicit Builder(const Args &args) {
		make_pass_data(args);
		make_barriers(args);
		finalize_attachments(args);
	}

	void PopResult(const myvk::Ptr<myvk::Device> &device_ptr, VkCommand *p_target) const {
		pop_pass_commands(device_ptr, p_target);
		pop_barriers(m_post_barriers, &p_target->m_post_barriers);
	}
};

namespace pop_pass {
struct SubpassDesciptionInfo {
	std::vector<VkAttachmentReference2> input_attachments, color_attachments, depth_attachments;
	std::vector<uint32_t> preserve_attachments;
};
} // namespace pop_pass

void VkCommand::Builder::pop_pass(const myvk::Ptr<myvk::Device> &device_ptr, const PassData &in, PassCmd *p_out) {
	pop_barriers(in.prior_barriers, &p_out->prior_barriers);
	p_out->subpasses = in.subpasses;

	// Skip if not RenderPass
	if (in.subpasses[0]->GetType() != PassType::kGraphics)
		return;

	// Fill Attachment Image Data
	p_out->attachments.resize(in.attachment_data_s.size());
	for (const auto &[p_attachment, att_data] : in.attachment_data_s)
		p_out->attachments[att_data.id] = p_attachment;

	// Subpass Dependencies
	std::vector<VkSubpassDependency2> vk_subpass_dependencies;
	std::vector<VkMemoryBarrier2> vk_subpass_dep_barriers;
	{
		std::size_t subpass_dep_count = in.subpass_deps.size() + in.by_region_subpass_deps.size();
		vk_subpass_dependencies.reserve(subpass_dep_count);
		vk_subpass_dep_barriers.reserve(subpass_dep_count);
	}
	{
		const auto pop_subpass_deps = [&](const auto &subpass_deps, VkDependencyFlags dep_flags) {
			for (const auto &[pair, dep] : subpass_deps)
				if (IsValidBarrier(dep)) {
					vk_subpass_dep_barriers.push_back({VK_STRUCTURE_TYPE_MEMORY_BARRIER_2});
					VkMemoryBarrier2 &vk_barrier = vk_subpass_dep_barriers.back();
					CopyVkBarrier(dep, &vk_barrier);
					vk_subpass_dependencies.push_back({.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
					                                   .pNext = &vk_barrier,
					                                   .srcSubpass = pair.src_subpass,
					                                   .dstSubpass = pair.dst_subpass,
					                                   .dependencyFlags = dep_flags});
				}
		};
		pop_subpass_deps(in.subpass_deps, 0);
		pop_subpass_deps(in.by_region_subpass_deps, VK_DEPENDENCY_BY_REGION_BIT);
	}

	// Attachment Descriptions
	std::vector<VkAttachmentDescription2> vk_attachment_descriptions;
	vk_attachment_descriptions.reserve(in.attachment_data_s.size());
	for (const auto &[p_image, att_data] : in.attachment_data_s) {
		vk_attachment_descriptions.push_back({
		    .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
		    .flags = att_data.may_alias ? VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT : 0u,
		    .format = Metadata::GetAllocInfo(p_image).vk_format,
		    .samples = VK_SAMPLE_COUNT_1_BIT,
		    .loadOp = att_data.load_op,
		    .storeOp = att_data.store_op,
		    .stencilLoadOp = att_data.load_op,
		    .stencilStoreOp = att_data.store_op,
		    .initialLayout = att_data.initial_layout,
		    .finalLayout = att_data.final_layout,
		});

		if (att_data.load_op == VK_ATTACHMENT_LOAD_OP_CLEAR)
			p_out->has_clear_values = true;
	}

	// Subpass Description Info
	using pop_pass::SubpassDesciptionInfo;
	std::vector<SubpassDesciptionInfo> subpass_desc_infos(in.subpasses.size());

	for (uint32_t subpass = 0; subpass < in.subpasses.size(); ++subpass) {
		const PassBase *p_pass = in.subpasses[subpass];
		auto &vk_desc = subpass_desc_infos[subpass];

		std::unordered_set<const ImageBase *> used_attachments;
		for (const InputBase *p_input : Dependency::GetPassInputs(p_pass)) {
			if (!UsageIsAttachment(p_input->GetUsage()))
				continue;
			assert(p_input->GetType() == ResourceType::kImage);
			auto p_image_input = static_cast<const ImageInput *>(p_input);
			auto p_attachment = static_cast<const ImageBase *>(Dependency::GetInputResource(p_input));
			assert(!used_attachments.contains(p_attachment));
			used_attachments.insert(p_attachment);

			auto &att_data = in.attachment_data_s.at(p_attachment);
			const auto push_vk_att_ref = [&](std::vector<VkAttachmentReference2> *p_vec, uint32_t index) {
				if (index >= p_vec->size())
					p_vec->resize(index + 1, {.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
					                          .attachment = VK_ATTACHMENT_UNUSED});
				auto &vk_ref = (*p_vec)[index];
				if (vk_ref.attachment != VK_ATTACHMENT_UNUSED)
					Throw(error::DupAttachmentIndex{.key = p_input->GetGlobalKey()});
				vk_ref.attachment = att_data.id;
				vk_ref.layout = UsageGetImageLayout(p_input->GetUsage());
				vk_ref.aspectMask = VkImageAspectFlagsFromVkFormat(Metadata::GetAllocInfo(p_attachment).vk_format);
			};

			if (UsageIsColorAttachment(p_input->GetUsage()))
				push_vk_att_ref(&vk_desc.color_attachments, *p_image_input->GetOptAttachmentIndex());
			else if (myvk_rg::Usage::kInputAttachment == p_input->GetUsage())
				push_vk_att_ref(&vk_desc.input_attachments, *p_image_input->GetOptAttachmentIndex());
			else if (UsageIsDepthAttachment(p_input->GetUsage()))
				push_vk_att_ref(&vk_desc.depth_attachments, 0);
		}

		// Check Preserved Attachments
		for (const auto &[p_attachment, att_data] : in.attachment_data_s) {
			if (used_attachments.contains(p_attachment))
				continue;
			if (att_data.first_subpass <= subpass && subpass <= att_data.last_subpass)
				vk_desc.preserve_attachments.push_back(att_data.id);
		}
	}

	// Subpass Descriptions
	std::vector<VkSubpassDescription2> vk_subpass_descriptions;
	vk_subpass_descriptions.reserve(subpass_desc_infos.size());
	for (auto &info : subpass_desc_infos)
		vk_subpass_descriptions.push_back({
		    .sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
		    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		    .inputAttachmentCount = (uint32_t)info.input_attachments.size(),
		    .pInputAttachments = info.input_attachments.data(),
		    .colorAttachmentCount = (uint32_t)info.color_attachments.size(),
		    .pColorAttachments = info.color_attachments.data(),
		    .pDepthStencilAttachment = info.depth_attachments.empty() ? nullptr : info.depth_attachments.data(),
		    .preserveAttachmentCount = (uint32_t)info.preserve_attachments.size(),
		    .pPreserveAttachments = info.preserve_attachments.data(),
		});

	// Create RenderPass
	VkRenderPassCreateInfo2 render_pass_create_info = {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
	    .attachmentCount = (uint32_t)vk_attachment_descriptions.size(),
	    .pAttachments = vk_attachment_descriptions.data(),
	    .subpassCount = (uint32_t)vk_subpass_descriptions.size(),
	    .pSubpasses = vk_subpass_descriptions.data(),
	    .dependencyCount = (uint32_t)vk_subpass_dependencies.size(),
	    .pDependencies = vk_subpass_dependencies.data(),
	};
	p_out->myvk_render_pass = myvk::RenderPass::Create(device_ptr, render_pass_create_info);

	// Create Imageless Framebuffer
	std::vector<VkFramebufferAttachmentImageInfo> vk_fb_att_image_infos;
	vk_fb_att_image_infos.reserve(in.attachment_data_s.size());
	std::vector<VkFormat> vk_att_formats;
	vk_att_formats.reserve(in.attachment_data_s.size());
	for (const auto &[p_attachment, att_data] : in.attachment_data_s) {
		vk_att_formats.push_back(Metadata::GetAllocInfo(p_attachment).vk_format);
		const auto &size = Metadata::GetViewInfo(p_attachment).size;
		auto extent = size.GetBaseMipExtent();

		vk_fb_att_image_infos.push_back({
		    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
		    .usage = Metadata::GetAllocInfo(p_attachment).vk_usages,
		    .width = extent.width,
		    .height = extent.height,
		    .layerCount = size.GetArrayLayers(),
		    .viewFormatCount = 1,
		    .pViewFormats = &vk_att_formats.back(),
		});
	}
	const auto &area = Metadata::GetPassRenderArea(in.subpasses[0]);
	p_out->myvk_framebuffer =
	    myvk::ImagelessFramebuffer::Create(p_out->myvk_render_pass, vk_fb_att_image_infos, area.extent, area.layers);
}

VkCommand VkCommand::Create(const myvk::Ptr<myvk::Device> &device_ptr, const Args &args) {
	args.collection.ClearInfo(&PassInfo::vk_command);

	VkCommand command = {};
	Builder builder{args};
	builder.PopResult(device_ptr, &command);
	return command;
}

} // namespace myvk_rg_executor
