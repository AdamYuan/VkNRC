#ifndef MYVK_RG_DETAILS_RENDER_GRAPH_HPP
#define MYVK_RG_DETAILS_RENDER_GRAPH_HPP

#include "../executor/Executor.hpp"
#include "Object.hpp"
#include "Pass.hpp"
#include "Resource.hpp"
#include "ResultPool.hpp"
#include <memory>
#include <myvk/Queue.hpp>

namespace myvk_rg::interface {

class RenderGraphBase : public ObjectBase,
                        public PassPool<RenderGraphBase>,
                        public ResourcePool<RenderGraphBase>,
                        public ResultPool<RenderGraphBase>,
                        public myvk::DeviceObjectBase {
private:
	inline static const PoolKey kRGKey = {"[RG]"};
	inline static const interface::PoolKey kEXEKey = {"[EXE]"};

	VkExtent2D m_canvas_size{};
	myvk::UPtr<executor::Executor> m_executor{};
	myvk::Ptr<myvk::Device> m_device_ptr;

	friend class ObjectBase;

public:
	inline explicit RenderGraphBase(myvk::Ptr<myvk::Device> device_ptr)
	    : ObjectBase({.p_pool_key = &kRGKey, .p_var_parent = this}), m_device_ptr{std::move(device_ptr)},
	      m_executor(myvk::MakeUPtr<executor::Executor>(Parent{.p_pool_key = &kEXEKey, .p_var_parent = this})) {}
	inline ~RenderGraphBase() override = default;

	RenderGraphBase(const RenderGraphBase &) = delete;
	RenderGraphBase &operator=(const RenderGraphBase &) = delete;
	RenderGraphBase(RenderGraphBase &&) = delete;
	RenderGraphBase &operator=(RenderGraphBase &&) = delete;

	inline void SetCanvasSize(const VkExtent2D &canvas_size) {
		if (canvas_size.width != m_canvas_size.width || canvas_size.height != m_canvas_size.height) {
			m_canvas_size = canvas_size;
			EmitEvent(Event::kCanvasResized);
		}
	}
	inline const VkExtent2D &GetCanvasSize() const { return m_canvas_size; }
	template <std::floating_point Float = float> inline Float GetCanvasAspectRatio() const {
		return Float(m_canvas_size.width) / Float(m_canvas_size.height);
	}
	inline const executor::Executor *GetExecutor() const { return m_executor.get(); }

	virtual void PreExecute() const {}
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) {
		m_executor->CmdExecute(this, command_buffer);
	}

	inline const myvk::Ptr<myvk::Device> &GetDevicePtr() const final { return m_device_ptr; }
};

} // namespace myvk_rg::interface

#endif
