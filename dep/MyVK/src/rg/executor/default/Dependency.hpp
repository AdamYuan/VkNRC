#ifndef MYVK_RG_EXE_DEFAULT_GRAPH_HPP
#define MYVK_RG_EXE_DEFAULT_GRAPH_HPP

#include "../Graph.hpp"
#include "Collection.hpp"

#include <unordered_map>
#include <variant>

namespace myvk_rg_executor {

using namespace myvk_rg::interface;
using namespace myvk_rg::executor;

class Dependency {
public:
	struct Args {
		const RenderGraphBase &render_graph;
		const Collection &collection;
	};

	enum class PassEdgeType { kBarrier, kIndirectWAW, kImageRead };
	struct PassEdge {
		const InputBase *opt_p_src_input{}, *p_dst_input{};
		const ResourceBase *p_resource{};
		PassEdgeType type{PassEdgeType::kBarrier};
	};
	struct ResourceEdge {};

private:
	Graph<const PassBase *, PassEdge> m_pass_graph;
	Graph<const ResourceBase *, ResourceEdge> m_resource_graph;
	std::vector<const PassBase *> m_passes;
	std::vector<const ResourceBase *> m_resources, m_root_resources;

	Relation m_pass_relation, m_resource_relation;

	void traverse_pass(const Args &args, const PassBase *p_pass);
	const InputBase *traverse_output_alias(const Dependency::Args &args, const OutputAlias auto &output_alias);
	void add_war_edges(); // Write-After-Read Edges
	void sort_passes();
	void tag_resources(const Args &args);
	void get_pass_relation();
	void get_resource_relation();
	void add_image_read_edges(); // Edges for scheduler

	static auto &get_dep_info(const PassBase *p_pass) { return GetPassInfo(p_pass).dependency; }
	static auto &get_dep_info(const InputBase *p_input) { return GetInputInfo(p_input).dependency; }
	static auto &get_dep_info(const ResourceBase *p_resource) { return GetResourceInfo(p_resource).dependency; }

public:
	static Dependency Create(const Args &args);

	template <PassEdgeType... Types>
	inline static const auto kPassEdgeFilter = [](const auto &e) { return ((e.type == Types) || ...); };
	inline static const auto kAnyFilter = [](auto &&) { return true; };

	// Graph
	inline const auto &GetResourceGraph() const { return m_resource_graph; }
	inline const auto &GetPassGraph() const { return m_pass_graph; }

	// Input
	static const ResourceBase *GetInputResource(const InputBase *p_input) { return get_dep_info(p_input).p_resource; }
	static const PassBase *GetInputPass(const InputBase *p_input) { return get_dep_info(p_input).p_pass; }
	static std::vector<const InputBase *> &GetPassInputs(const PassBase *p_pass) { return get_dep_info(p_pass).inputs; }

	// Counts
	inline std::size_t GetPassCount() const { return m_passes.size(); }
	inline std::size_t GetRootResourceCount() const { return m_root_resources.size(); }

	// Topological Ordered ID for Passes
	static std::size_t GetPassTopoID(const PassBase *p_pass) { return GetPassInfo(p_pass).dependency.topo_id; }
	const PassBase *GetTopoIDPass(std::size_t topo_order) const { return m_passes[topo_order]; }
	const auto &GetPasses() const { return m_passes; }

	// Root ID for Resources
	static std::size_t GetResourceRootID(const ResourceBase *p_resource) { return get_dep_info(p_resource).root_id; }
	const ResourceBase *GetRootIDResource(std::size_t root_id) const { return m_root_resources[root_id]; }
	const auto &GetRootResources() const { return m_root_resources; }
	const auto &GetResources() const { return m_resources; }

	// Resource Pointers
	static const ResourceBase *GetRootResource(const ResourceBase *p_resource) {
		return get_dep_info(p_resource).p_root_resource;
	}
	static const ImageBase *GetRootResource(const ImageBase *p_resource) {
		return static_cast<const ImageBase *>(get_dep_info(p_resource).p_root_resource);
	}
	static const BufferBase *GetRootResource(const BufferBase *p_resource) {
		return static_cast<const BufferBase *>(get_dep_info(p_resource).p_root_resource);
	}
	static bool IsRootResource(const ResourceBase *p_resource) {
		return get_dep_info(p_resource).p_root_resource == p_resource;
	}

	// Relations
	inline bool IsPassLess(std::size_t topo_id_l, std::size_t topo_id_r) const {
		return m_pass_relation.Get(topo_id_l, topo_id_r);
	}
	inline bool IsPassLess(const PassBase *p_l, const PassBase *p_r) const {
		return IsPassLess(GetPassTopoID(p_l), GetPassTopoID(p_r));
	}
	inline bool IsResourceLess(std::size_t root_id_l, std::size_t root_id_r) const {
		return m_resource_relation.Get(root_id_l, root_id_r);
	}
	inline bool IsResourceLess(const ResourceBase *p_l, const ResourceBase *p_r) const {
		return IsResourceLess(GetResourceRootID(p_l), GetResourceRootID(p_r));
	}
	inline bool IsResourceConflicted(std::size_t root_id_0, std::size_t root_id_1) const {
		return !IsResourceLess(root_id_0, root_id_1) && !IsResourceLess(root_id_1, root_id_0);
	}
	inline bool IsResourceConflicted(const ResourceBase *p_0, const ResourceBase *p_1) const {
		return IsResourceConflicted(GetResourceRootID(p_0), GetResourceRootID(p_1));
	}
};

} // namespace myvk_rg_executor

#endif // MYVK_GRAPH_HPP
