#ifndef MYVK_RG_BASE_HPP
#define MYVK_RG_BASE_HPP

#include "Event.hpp"
#include "Key.hpp"

#include <variant>

namespace myvk_rg::interface {

template <class... Ts> struct overloaded : Ts... {
	using Ts::operator()...;
};
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

class RenderGraphBase;
class ObjectBase;
struct Parent {
	const PoolKey *p_pool_key;
	std::variant<RenderGraphBase *, const ObjectBase *> p_var_parent;
};

class ObjectBase {
private:
	RenderGraphBase *m_p_render_graph{};
	const ObjectBase *m_p_parent_object{};
	const PoolKey *m_p_key{};
	mutable void *m_p_executor_info{};

public:
	inline explicit ObjectBase(Parent parent) : m_p_key{parent.p_pool_key} {
		std::visit(overloaded([this](RenderGraphBase *p_rg) { m_p_render_graph = p_rg; },
		                      [this](const ObjectBase *p_obj) {
			                      m_p_render_graph = p_obj->GetRenderGraphPtr();
			                      m_p_parent_object = p_obj;
		                      }),
		           parent.p_var_parent);
	}
	inline virtual ~ObjectBase() = default;

	inline RenderGraphBase *GetRenderGraphPtr() const { return m_p_render_graph; }
	inline const PoolKey &GetKey() const { return *m_p_key; }
	inline GlobalKey GetGlobalKey() const {
		return m_p_parent_object ? GlobalKey{m_p_parent_object->GetGlobalKey(), *m_p_key} : GlobalKey{*m_p_key};
	}
	void EmitEvent(Event event) ;

	inline void __SetPExecutorInfo(void *p_info) const { m_p_executor_info = p_info; }
	template <typename T> inline T *__GetPExecutorInfo() const { return (T *)m_p_executor_info; }
};

} // namespace myvk_rg::interface

#endif
