#include "Collection.hpp"

namespace myvk_rg_executor {

Collection Collection::Create(const RenderGraphBase &rg) {
	Collection c;
	c.collect_resources(rg);
	c.collect_passes(rg);
	c.make_infos();
	return c;
}

void Collection::make_infos() {
	m_resource_infos.reserve(m_resources.size());
	for (const auto &[_, p_resource] : m_resources) {
		m_resource_infos.emplace_back();
		p_resource->__SetPExecutorInfo(&m_resource_infos.back());
	}

	m_input_infos.reserve(m_inputs.size());
	for (const auto &[_, p_input] : m_inputs) {
		m_input_infos.emplace_back();
		p_input->__SetPExecutorInfo(&m_input_infos.back());
	}

	m_pass_infos.reserve(m_passes.size());
	for (const auto &[_, p_pass] : m_passes) {
		m_pass_infos.emplace_back();
		p_pass->__SetPExecutorInfo(&m_pass_infos.back());
	}
}

template <typename Container> void Collection::collect_resources(const Container &pool) {
	for (const auto &[_, pool_data] : pool.GetResourcePoolData()) {
		const auto *p_resource = pool_data.template Get<ResourceBase>();
		if (p_resource == nullptr)
			Throw(error::NullResource{.parent = pool.GetGlobalKey()});
		m_resources[p_resource->GetGlobalKey()] = p_resource;
	}
}

template <typename Container> void Collection::collect_inputs(const Container &pool) {
	for (const auto &[_, pool_data] : pool.GetInputPoolData()) {
		const auto *p_input = pool_data.template Get<InputBase>();
		if (p_input == nullptr)
			Throw(error::NullInput{.parent = pool.GetGlobalKey()});
		m_inputs[p_input->GetGlobalKey()] = p_input;
	}
}

template <typename Container> void Collection::collect_passes(const Container &pool) {
	for (const auto &[_, pool_data] : pool.GetPassPoolData()) {
		const PassBase *p_pass = pool_data.template Get<PassBase>();
		if (p_pass == nullptr)
			Throw(error::NullPass{.parent = pool.GetGlobalKey()});

		p_pass->Visit(overloaded(
		    [this](const PassGroupBase *p_pass) {
			    collect_resources(*p_pass);
			    collect_passes(*p_pass);
		    },
		    [this](const auto *p_pass) {
			    collect_resources(*p_pass);
			    collect_inputs(*p_pass);
			    m_passes[p_pass->GetGlobalKey()] = p_pass;
		    }));
	}
}

} // namespace myvk_rg_executor
