#pragma once
#ifndef MYVK_RG_COLLECTOR_HPP
#define MYVK_RG_COLLECTOR_HPP

#include <map>

#include "Error.hpp"
#include <myvk_rg/interface/RenderGraph.hpp>

#include "Info.hpp"

namespace myvk_rg_executor {

using namespace myvk_rg::interface;

class Collection {
private:
	std::map<GlobalKey, const PassBase *> m_passes;
	std::map<GlobalKey, const InputBase *> m_inputs;
	std::map<GlobalKey, const ResourceBase *> m_resources;
	mutable std::vector<PassInfo> m_pass_infos;
	mutable std::vector<InputInfo> m_input_infos;
	mutable std::vector<ResourceInfo> m_resource_infos;

	template <typename Container> void collect_resources(const Container &pool);
	template <typename Container> void collect_passes(const Container &pool);
	template <typename Container> void collect_inputs(const Container &pool);
	void make_infos();

public:
	static Collection Create(const RenderGraphBase &rg);

	inline const PassBase *FindPass(const GlobalKey &key) const {
		auto it = m_passes.find(key);
		if (it == m_passes.end())
			Throw(error::PassNotFound{.key = key});
		return it->second;
	}
	inline const InputBase *FindInput(const GlobalKey &key) const {
		auto it = m_inputs.find(key);
		if (it == m_inputs.end())
			Throw(error::InputNotFound{.key = key});
		return it->second;
	}
	inline const ResourceBase *FindResource(const GlobalKey &key) const {
		auto it = m_resources.find(key);
		if (it == m_resources.end())
			Throw(error::ResourceNotFound{.key = key});
		return it->second;
	}

	void ClearInfo() const {}
	template <typename Info_T, typename Member_T, typename... Args>
	void ClearInfo(Member_T Info_T::*p_member, Args &&...args) const {
		static_assert(std::convertible_to<Info_T, PassInfo> || std::convertible_to<Info_T, ResourceInfo> ||
		              std::convertible_to<Info_T, InputInfo>);

		if constexpr (std::convertible_to<Info_T, PassInfo>) {
			for (PassInfo &pass_info : m_pass_infos)
				pass_info.*p_member = {};
		} else if constexpr (std::convertible_to<Info_T, ResourceInfo>) {
			for (ResourceInfo &resource_info : m_resource_infos)
				resource_info.*p_member = {};
		} else if constexpr (std::convertible_to<Info_T, InputInfo>) {
			for (InputInfo &input_info : m_input_infos)
				input_info.*p_member = {};
		}

		ClearInfo(std::forward<Args>(args)...);
	}
};

} // namespace myvk_rg_executor

#endif
