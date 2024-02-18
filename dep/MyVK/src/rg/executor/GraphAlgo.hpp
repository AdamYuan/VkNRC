//
// Created by adamyuan on 2/7/24.
//

#pragma once
#ifndef MYVK_GRAPHALGO_HPP
#define MYVK_GRAPHALGO_HPP

#include "Relation.hpp"

#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace myvk_rg::executor {

template <typename Derived, typename VertexID_T, typename Edge_T> class GraphAlgo {
private:
	const Derived *get() const { return static_cast<const Derived *>(this); }

public:
	void WriteGraphViz(std::ostream &out, auto &&vertex_name, auto &&edge_label) const {
		out << "digraph{" << std::endl;
		for (auto vertex : get()->GetVertices())
			out << '\"' << vertex_name(vertex) << "\";" << std::endl;
		for (auto [from, to, e, id] : get()->GetEdges()) {
			out << '\"' << vertex_name(from) << "\"->\"" << vertex_name(to) << "\"[label=\"" << edge_label(e) << "\"];"
			    << std::endl;
		}
		out << "}" << std::endl;
	}

	struct KahnTopologicalSortResult {
		std::vector<VertexID_T> sorted;
		bool is_dag;
	};
	KahnTopologicalSortResult KahnTopologicalSort() const {
		std::vector<VertexID_T> sorted;

		std::unordered_map<VertexID_T, std::size_t> in_degrees;
		std::size_t vertex_count = 0;
		for (auto vertex : get()->GetVertices()) {
			in_degrees[vertex] = 0;
			++vertex_count;
		}
		sorted.reserve(vertex_count);

		for (auto [_, to, _1, _2] : get()->GetEdges())
			++in_degrees[to];

		std::queue<VertexID_T> queue;
		for (auto [vert, in_deg] : in_degrees)
			if (in_deg == 0)
				queue.push(vert);

		while (!queue.empty()) {
			VertexID_T vertex = queue.front();
			sorted.push_back(vertex);
			queue.pop();

			for (auto [to, _, _1] : get()->GetOutEdges(vertex)) {
				if (--in_degrees[to] == 0)
					queue.push(to);
			}
		}

		bool is_dag = sorted.size() == vertex_count;
		return {
		    .sorted = std::move(sorted),
		    .is_dag = is_dag,
		};
	}

	struct FindTreesResult {
		std::vector<VertexID_T> roots;
		bool is_forest;
	};
	FindTreesResult FindTrees(auto &&visitor) const {
		std::vector<VertexID_T> roots;
		std::unordered_set<VertexID_T> visit_set;

		const auto visit_tree = [&](VertexID_T root) -> bool {
			const auto visit_tree_impl = [&](VertexID_T vertex, auto &&visit_tree_impl) -> bool {
				if (visit_set.contains(vertex))
					return false;
				visit_set.insert(vertex);

				visitor(root, vertex);

				for (auto [to, _, _1] : get()->GetOutEdges(vertex))
					if (visit_tree_impl(to, visit_tree_impl) == false)
						return false;

				return true;
			};
			return visit_tree_impl(root, visit_tree_impl);
		};

		std::size_t vertex_count = 0;
		for (auto vertex : get()->GetVertices()) {
			++vertex_count;

			auto in_edges = get()->GetInEdges(vertex);
			std::size_t in_degree = std::distance(in_edges.begin(), in_edges.end());

			if (in_degree == 0) {
				roots.push_back(vertex);
				if (visit_tree(vertex) == false)
					return FindTreesResult{.is_forest = false};
			}
		}

		if (visit_set.size() != vertex_count)
			return FindTreesResult{.is_forest = false};

		return FindTreesResult{
		    .roots = std::move(roots),
		    .is_forest = true,
		};
	}

	Relation TransitiveClosure(auto &&get_vertex_topo_order, auto &&get_topo_order_vertex) const {
		std::size_t vertex_count = 0;
		for (auto _ : get()->GetVertices())
			++vertex_count;

		Relation relation{vertex_count, vertex_count};

		for (std::size_t topo_order = vertex_count - 1; ~topo_order; --topo_order) {
			for (auto [from, _, _1] : get()->GetInEdges(get_topo_order_vertex(topo_order))) {
				std::size_t from_topo_order = get_vertex_topo_order(from);
				// assert(from_topo_order < topo_order)
				relation.Add(from_topo_order, topo_order);   // from -> cur
				relation.Apply(topo_order, from_topo_order); // forall x, cur -> x ==> from -> x
			}
		}

		return relation;
	}
};

} // namespace myvk_rg::executor

#endif // MYVK_GRAPHALGO_HPP
