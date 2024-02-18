//
// Created by adamyuan on 2/4/24.
//

#pragma once
#ifndef MYVK_GRAPH_HPP
#define MYVK_GRAPH_HPP

#include <cinttypes>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <vector>

#include "GraphAlgo.hpp"

namespace myvk_rg::executor {

template <typename VertexID_T, typename Edge_T, typename VertexFilter, typename EdgeFilter> class GraphView;

template <typename VertexID_T, typename Edge_T>
class Graph : public GraphAlgo<Graph<VertexID_T, Edge_T>, VertexID_T, Edge_T> {
private:
	struct VertexInfo {
		std::vector<std::size_t> in, out;
	};
	struct EdgeInfo {
		Edge_T e;
		VertexID_T from, to;
	};
	std::unordered_map<VertexID_T, VertexInfo> m_vertices;
	std::vector<EdgeInfo> m_edges;

public:
	struct InEdgeIterator {
		VertexID_T from;
		const Edge_T &e;
		std::size_t edge_id;
	};
	struct OutEdgeIterator {
		VertexID_T to;
		const Edge_T &e;
		std::size_t edge_id;
	};
	struct EdgeIterator {
		VertexID_T from, to;
		const Edge_T &e;
		std::size_t edge_id;
	};

	void AddVertex(VertexID_T vertex) { m_vertices.insert({vertex, VertexInfo{}}); }
	std::size_t AddEdge(VertexID_T from, VertexID_T to, Edge_T edge) {
		std::size_t edge_id = m_edges.size();
		m_edges.emplace_back(EdgeInfo{
		    .e = std::move(edge),
		    .from = from,
		    .to = to,
		});
		m_vertices[from].out.push_back(edge_id);
		m_vertices[to].in.push_back(edge_id);
		return edge_id;
	}
	bool HasVertex(VertexID_T vertex) const { return m_vertices.count(vertex); }

	auto GetOutEdges(VertexID_T vertex) const {
		return m_vertices.at(vertex).out | std::views::transform([this](std::size_t edge_id) -> OutEdgeIterator {
			       return OutEdgeIterator{m_edges[edge_id].to, m_edges[edge_id].e, edge_id};
		       });
	}
	auto GetInEdges(VertexID_T vertex) const {
		return m_vertices.at(vertex).in | std::views::transform([this](std::size_t edge_id) -> InEdgeIterator {
			       return InEdgeIterator{m_edges[edge_id].from, m_edges[edge_id].e, edge_id};
		       });
	}
	auto GetEdges() const {
		return m_edges | std::views::transform([this](const EdgeInfo &edge_info) -> EdgeIterator {
			       std::size_t edge_id = &edge_info - m_edges.data();
			       return EdgeIterator{edge_info.from, edge_info.to, edge_info.e, edge_id};
		       });
	}

	VertexID_T GetFromVertex(std::size_t edge_id) const { return m_edges[edge_id].from; }
	VertexID_T GetToVertex(std::size_t edge_id) const { return m_edges[edge_id].to; }
	const Edge_T &GetEdge(std::size_t edge_id) const { return m_edges[edge_id].e; }
	Edge_T &GetEdge(std::size_t edge_id) { return m_edges[edge_id].e; }

	auto GetVertices() const {
		return m_vertices | std::views::transform(
		                        [](const std::pair<VertexID_T, VertexInfo> &pair) -> VertexID_T { return pair.first; });
	}

	template <typename VertexFilter, typename EdgeFilter>
	GraphView<VertexID_T, Edge_T, VertexFilter, EdgeFilter> MakeView(VertexFilter &&vertex_filter,
	                                                                 EdgeFilter &&edge_filter) const {
		return GraphView<VertexID_T, Edge_T, VertexFilter, EdgeFilter>(*this, std::forward<VertexFilter>(vertex_filter),
		                                                               std::forward<EdgeFilter>(edge_filter));
	}
};

template <typename VertexID_T, typename Edge_T, typename VertexFilter, typename EdgeFilter>
class GraphView : public GraphAlgo<GraphView<VertexID_T, Edge_T, VertexFilter, EdgeFilter>, VertexID_T, Edge_T> {
private:
	const Graph<VertexID_T, Edge_T> &m_graph_ref;
	VertexFilter m_vertex_filter;
	EdgeFilter m_edge_filter;

	using EdgeIterator = typename Graph<VertexID_T, Edge_T>::EdgeIterator;
	using InEdgeIterator = typename Graph<VertexID_T, Edge_T>::InEdgeIterator;
	using OutEdgeIterator = typename Graph<VertexID_T, Edge_T>::OutEdgeIterator;

public:
	GraphView(const Graph<VertexID_T, Edge_T> &graph_ref, VertexFilter &&vertex_filter, EdgeFilter &&edge_filter)
	    : m_graph_ref{graph_ref}, m_vertex_filter{std::forward<VertexFilter>(vertex_filter)},
	      m_edge_filter{std::forward<EdgeFilter>(edge_filter)} {}
	auto GetVertices() const {
		return m_graph_ref.GetVertices() |
		       std::views::filter([this](VertexID_T vertex) { return m_vertex_filter(vertex); });
	}
	auto GetEdges() const {
		return m_graph_ref.GetEdges() | std::views::filter([this](const EdgeIterator &it) {
			       return m_vertex_filter(it.from) && m_vertex_filter(it.to) && m_edge_filter(it.e);
		       });
	}
	auto GetOutEdges(VertexID_T vertex) const {
		return m_graph_ref.GetOutEdges(vertex) | std::views::filter([this, vertex](const OutEdgeIterator &it) {
			       return m_vertex_filter(vertex) && m_vertex_filter(it.to) && m_edge_filter(it.e);
		       });
	}
	auto GetInEdges(VertexID_T vertex) const {
		return m_graph_ref.GetInEdges(vertex) | std::views::filter([this, vertex](const InEdgeIterator &it) {
			       return m_vertex_filter(vertex) && m_vertex_filter(it.from) && m_edge_filter(it.e);
		       });
	}
};

} // namespace myvk_rg::executor

#endif // MYVK_GRAPH_HPP
