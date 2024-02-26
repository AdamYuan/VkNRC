//
// Created by adamyuan on 2/4/24.
//

#include "Schedule.hpp"

#include <algorithm>
#include <cassert>

namespace myvk_rg_executor {

Schedule Schedule::Create(const Args &args) {
	args.collection.ClearInfo(&PassInfo::schedule, &ResourceInfo::schedule);

	Schedule s = {};
	s.make_pass_groups(args, merge_passes(args));
	s.make_barriers(args);
	finalize_last_inputs(args);
	s.make_output_barriers(args);
	s.check_ext_read_only(args);
	return s;
}

inline static bool is_image_read_grouped(const InputBase *p_l, const InputBase *p_r) {
	myvk_rg::Usage ul = p_l->GetUsage(), ur = p_r->GetUsage();
	return !UsageIsAttachment(ul) && !UsageIsAttachment(ur) && UsageGetImageLayout(ul) == UsageGetImageLayout(ur);
}
inline static bool is_access_mergeable(const InputBase *p_l, const InputBase *p_r) {
	return UsageIsAttachment(p_l->GetUsage()) && UsageIsAttachment(p_r->GetUsage()) &&
	       Dependency::GetInputResource(p_l) == Dependency::GetInputResource(p_r);
	// Image Combine is not allowed to merge
}

std::vector<std::size_t> Schedule::merge_passes(const Args &args) {
	// Compute Merge Size

	// Calculate merge_size, Complexity: O(N + M)
	// merge_size == 0: The pass is not a graphics pass
	// merge_size == 1: The pass is a graphics pass, but can't be merged
	// merge_size >  1: The pass is a graphics pass, and it can be merged to a group of merge_size with the passes
	// before

	std::vector<std::size_t> merge_sizes(args.dependency.GetPassCount());

	// Initial Merge Sizes
	merge_sizes[0] = args.dependency.GetTopoIDPass(0)->GetType() == PassType::kGraphics;
	for (std::size_t i = 1; i < args.dependency.GetPassCount(); ++i) {
		const PassBase *p_pass = args.dependency.GetTopoIDPass(i);
		if (p_pass->GetType() == PassType::kGraphics) {
			// Both are RenderPass and have equal RenderArea
			const PassBase *p_prev_pass = args.dependency.GetTopoIDPass(i - 1);
			merge_sizes[i] = p_prev_pass->GetType() == PassType::kGraphics &&
			                         Metadata::GetPassRenderArea(p_pass) == Metadata::GetPassRenderArea(p_prev_pass)
			                     ? merge_sizes[i - 1] + 1
			                     : 1;
		} else
			merge_sizes[i] = 0;
	}

	// Exclude nullptr Pass
	const auto node_filter = [](const PassBase *p_pass) -> bool { return p_pass; };
	auto barrier_view = args.dependency.GetPassGraph().MakeView(
	    node_filter, Dependency::kPassEdgeFilter<Dependency::PassEdgeType::kBarrier>);
	auto image_read_view = args.dependency.GetPassGraph().MakeView(
	    node_filter, Dependency::kPassEdgeFilter<Dependency::PassEdgeType::kImageRead>);

	for (std::size_t i = 0; i < args.dependency.GetPassCount(); ++i) {
		if (i == 0)
			continue;

		const PassBase *p_pass = args.dependency.GetTopoIDPass(i);
		std::size_t &size = merge_sizes[i];

		for (auto [from, e, _] : barrier_view.GetInEdges(p_pass)) {
			std::size_t from_topo_id = Dependency::GetPassTopoID(from);
			if (is_access_mergeable(e.opt_p_src_input, e.p_dst_input))
				size = std::min(size, i - from_topo_id + merge_sizes[from_topo_id]);
			else
				size = std::min(size, i - from_topo_id);
		}
		for (auto [from, e, _] : image_read_view.GetInEdges(p_pass)) {
			std::size_t from_topo_id = Dependency::GetPassTopoID(from);
			if (is_access_mergeable(e.opt_p_src_input, e.p_dst_input) ||
			    is_image_read_grouped(e.opt_p_src_input, e.p_dst_input))
				// Image Reads with same Usage can merge into the same RenderPass
				size = std::min(size, i - from_topo_id + merge_sizes[from_topo_id]);
			else
				size = std::min(size, i - from_topo_id);
		}
	}

	// Regularize Merge Sizes
	for (std::size_t i = 0, prev_size = 0; i < args.dependency.GetPassCount(); ++i) {
		const PassBase *p_pass = args.dependency.GetTopoIDPass(i);
		auto &size = merge_sizes[i];
		if (size > prev_size)
			size = prev_size + 1;
		else
			size = p_pass->GetType() == PassType::kGraphics ? 1 : 0;

		prev_size = size;
	}

	return merge_sizes;
}

void Schedule::make_pass_groups(const Args &args, const std::vector<std::size_t> &merge_sizes) {
	for (std::size_t topo_id = 0; const PassBase *p_pass : args.dependency.GetPasses()) {
		std::size_t merge_size = merge_sizes[topo_id];
		if (merge_size <= 1) {
			get_sched_info(p_pass).group_id = m_pass_groups.size();
			get_sched_info(p_pass).subpass_id = 0;
			m_pass_groups.emplace_back();
		} else {
			get_sched_info(p_pass).group_id = m_pass_groups.size() - 1;
			get_sched_info(p_pass).subpass_id = merge_size - 1;
		}
		m_pass_groups.back().subpasses.push_back(p_pass);

		++topo_id;
	}
}

Schedule::BarrierType Schedule::get_valid_barrier_type(const ResourceBase *p_valid_resource) {
	switch (p_valid_resource->GetState()) {
	case ResourceState::kExternal:
		return BarrierType::kExtInput;
	default:
		return BarrierType::kIntValidate;
	}
}

void Schedule::push_wrw_barriers(const Args &args, const ResourceBase *p_resource, const InputBase *p_write,
                                 std::span<const InputBase *> reads, const InputBase *p_next_write) {
	assert(!reads.empty());

	auto write_span = p_write ? std::span<const InputBase *const>{&p_write, 1} : std::span<const InputBase *const>{};
	auto next_write_span =
	    p_next_write ? std::span<const InputBase *const>{&p_next_write, 1} : std::span<const InputBase *const>{};

	std::ranges::sort(reads, [&](auto l_read, auto r_read) {
		return Dependency::GetPassTopoID(Dependency::GetInputPass(l_read)) <
		       Dependency::GetPassTopoID(Dependency::GetInputPass(r_read));
	});

	if (p_resource->GetType() == ResourceType::kBuffer) {
		push_read_barrier(args, p_resource, write_span, reads);
		push_read_barrier(args, p_resource, reads, next_write_span);
	} else {
		std::span<const InputBase *const> prev_span = write_span;
		for (std::size_t i = 0; i < reads.size();) {
			std::size_t j = i + 1;
			while (j < reads.size() && is_image_read_grouped(reads[j - 1], reads[j]))
				++j;

			std::span<const InputBase *const> span = {reads.data() + i, reads.data() + j};
			push_read_barrier(args, p_resource, prev_span, span);
			prev_span = span;

			i = j;
		}
		push_read_barrier(args, p_resource, prev_span, next_write_span);
	}
}

// Push a Barrier including read accesses
void Schedule::push_read_barrier(const Args &args, const ResourceBase *p_resource,
                                 std::span<const InputBase *const> src_s, std::span<const InputBase *const> dst_s) {
	if (dst_s.empty())
		return;

	bool validate = src_s.empty();

	update_last_inputs(p_resource, dst_s);
	if (validate)
		update_first_inputs(p_resource, dst_s);

	if (std::size_t group; !src_s.empty() && (group = GetGroupID(Dependency::GetInputPass(src_s[0]))) ==
	                                             GetGroupID(Dependency::GetInputPass(dst_s[0]))) {
		// Subpass Barrier
		assert(src_s.size() == 1 && dst_s.size() == 1);
		assert(UsageIsAttachment(src_s[0]->GetUsage()) && UsageIsAttachment(dst_s[0]->GetUsage()));
		m_pass_groups[group].subpass_deps.push_back(SubpassBarrier{
		    .p_attachment = static_cast<const ImageBase *>(p_resource),
		    .p_src = src_s[0],
		    .p_dst = dst_s[0],
		});
	} else {
		// Ignore Write-after-Read barriers when p_next_write is null
		m_pass_barriers.push_back({.p_resource = p_resource,
		                           .src_s = std::vector<const InputBase *>{src_s.begin(), src_s.end()},
		                           .dst_s = std::vector<const InputBase *>{dst_s.begin(), dst_s.end()},
		                           .type = validate ? get_valid_barrier_type(p_resource) : BarrierType::kLocal});
	}
}

// Push a Write-Only Barrier
void Schedule::push_write_barrier(const Args &args, const ResourceBase *p_resource, const InputBase *p_write,
                                  const InputBase *p_next_write) {
	if (p_next_write == nullptr)
		return;

	bool validate = p_write == nullptr;

	// Update resource info
	update_last_inputs(p_resource, {&p_next_write, 1});
	if (validate)
		update_first_inputs(p_resource, {&p_next_write, 1});

	if (std::size_t group; p_write && (group = GetGroupID(Dependency::GetInputPass(p_write))) ==
	                                      GetGroupID(Dependency::GetInputPass(p_next_write))) {
		// Subpass Barrier
		assert(p_resource->GetType() == ResourceType::kImage && UsageIsAttachment(p_write->GetUsage()) &&
		       UsageIsAttachment(p_next_write->GetUsage()));
		m_pass_groups[group].subpass_deps.push_back(SubpassBarrier{
		    .p_attachment = static_cast<const ImageBase *>(p_resource),
		    .p_src = p_write,
		    .p_dst = p_next_write,
		});
	} else {
		m_pass_barriers.push_back({
		    .p_resource = p_resource,
		    .src_s = validate ? std::vector<const InputBase *>{} : std::vector<const InputBase *>{p_write},
		    .dst_s = {p_next_write},
		    .type = validate ? get_valid_barrier_type(p_resource) : BarrierType::kLocal,
		});
	}
}

namespace make_barriers {
struct ReadInfo {
	std::vector<const InputBase *> reads;
	const InputBase *p_next_write{};
};
} // namespace make_barriers
void Schedule::make_barriers(const Schedule::Args &args) {
	using make_barriers::ReadInfo;

	std::unordered_map<const InputBase *, ReadInfo> local_reads;
	std::unordered_map<const ResourceBase *, ReadInfo> valid_reads;

	// Fetch p_next_write from Indirect WAW Graph, set to local_reads and valid_reads
	auto indirect_waw_view = args.dependency.GetPassGraph().MakeView(
	    Dependency::kAnyFilter, Dependency::kPassEdgeFilter<Dependency::PassEdgeType::kIndirectWAW>);
	for (auto [from, to, e, _] : indirect_waw_view.GetEdges()) {
		ReadInfo &info = from == nullptr ? valid_reads[e.p_resource] : local_reads[e.opt_p_src_input];
		info.p_next_write = e.p_dst_input;
	}
	// Write-After-Write or Write validation Barriers are directly added, WAR and RAW Barriers are remained to be
	// processed (push to local_reads and valid_reads)
	auto barrier_view = args.dependency.GetPassGraph().MakeView(
	    Dependency::kAnyFilter, Dependency::kPassEdgeFilter<Dependency::PassEdgeType::kBarrier>);
	for (auto [from, to, e, _] : barrier_view.GetEdges()) {
		// from == nullptr <==> Validation
		if (from == nullptr) {
			auto [_, p_dst_input, p_resource, _1] = e;
			// Not Local Dependency, should be Valid / LFValid / ExtValid
			bool dst_write = !UsageIsReadOnly(p_dst_input->GetUsage());
			if (dst_write)
				// Write Validation, push directly
				push_write_barrier(args, p_resource, nullptr, p_dst_input);
			else
				// Read Validation, push to Valid Read Edges
				valid_reads[p_resource].reads.push_back(p_dst_input);
		} else {
			auto [p_src_input, p_dst_input, p_resource, _1] = e;
			bool src_write = !UsageIsReadOnly(p_src_input->GetUsage()),
			     dst_write = !UsageIsReadOnly(p_dst_input->GetUsage());
			assert(src_write || dst_write); // Should have a write access
			if (src_write && dst_write)
				// Both are write access, push directly
				push_write_barrier(args, p_resource, p_src_input, p_dst_input);
			else if (src_write)
				// Read after Write, push to RAW edges
				local_reads[p_src_input].reads.push_back(p_dst_input);
		}
	}
	// Push barriers with read accesses
	for (auto &[p_input, read_info] : local_reads) {
		auto p_resource = Dependency::GetInputResource(p_input);
		push_wrw_barriers(args, p_resource, p_input, read_info.reads, read_info.p_next_write);
	}
	for (auto &[p_resource, read_info] : valid_reads)
		push_wrw_barriers(args, p_resource, nullptr, read_info.reads, read_info.p_next_write);
}

void Schedule::update_first_inputs(const ResourceBase *p_resource, std::span<const InputBase *const> accesses) {
	assert(get_sched_info(p_resource).first_inputs.empty());
	get_sched_info(p_resource).first_inputs = {accesses.begin(), accesses.end()};
}

void Schedule::update_last_inputs(const ResourceBase *p_resource, std::span<const InputBase *const> accesses) {
	const auto get_input_topo = [](const auto *p_input) {
		return Dependency::GetPassTopoID(Dependency::GetInputPass(p_input));
	};
	p_resource = Dependency::GetRootResource(p_resource);
	auto &last_accesses = get_sched_info(p_resource).last_inputs;
	if (last_accesses.empty() || get_input_topo(last_accesses[0]) < get_input_topo(accesses[0])) {
		last_accesses.resize(accesses.size());
		std::ranges::copy(accesses, last_accesses.begin());
	}
}

void Schedule::finalize_last_inputs(const Schedule::Args &args) {
	for (const ResourceBase *p_resource : args.dependency.GetResources()) {
		if (Dependency::IsRootResource(p_resource)) {
			// Check whether Last Inputs are on the root resource
			for (const InputBase *p_input : get_sched_info(p_resource).last_inputs)
				if (Dependency::GetInputResource(p_input) != p_resource)
					Throw(error::ResourceLastInputNotRoot{.key = p_resource->GetGlobalKey()});
		} else {
			get_sched_info(p_resource).last_inputs =
			    get_sched_info(Dependency::GetRootResource(p_resource)).last_inputs;
		}
	}
}

void Schedule::make_output_barriers(const Args &args) {
	// External Resources
	for (const ResourceBase *p_resource : args.metadata.GetExtResources())
		m_pass_barriers.push_back({.p_resource = p_resource,
		                           .src_s = get_sched_info(p_resource).last_inputs,
		                           .dst_s = {},
		                           .type = BarrierType::kExtOutput});
}

void Schedule::check_ext_read_only(const Schedule::Args &args) {
	for (const PassBase *p_pass : args.dependency.GetPasses())
		for (const InputBase *p_input : Dependency::GetPassInputs(p_pass)) {
			auto usage = p_input->GetUsage();
			const ResourceBase *p_resource = Dependency::GetInputResource(p_input);
			// Attachment and Layout Transition means implicit Writes
			if (!UsageIsReadOnly(usage) || UsageIsAttachment(usage) ||
			    UsageGetImageLayout(usage) != UsageGetImageLayout(GetLastInputs(p_resource)[0]->GetUsage())) {
				// Set ext_read_only flag, only valid for External Resources
				get_sched_info(p_resource).ext_read_only = false;
			}
		}
}

} // namespace myvk_rg_executor