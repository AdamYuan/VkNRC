#ifndef MYVK_RG_PASS_HPP
#define MYVK_RG_PASS_HPP

#include <concepts>

#include "InputPool.hpp"
#include "ResourcePool.hpp"

#include <myvk/CommandBuffer.hpp>
#include <myvk/ComputePipeline.hpp>
#include <myvk/GraphicsPipeline.hpp>

namespace myvk_rg::interface {

enum class PassType : uint8_t { kGraphics, kCompute, kTransfer, kGroup };

struct RenderPassArea {
	VkExtent2D extent{};
	uint32_t layers{};
	inline bool operator==(const RenderPassArea &r) const {
		return std::tie(extent.width, extent.height, layers) == std::tie(r.extent.width, r.extent.height, r.layers);
	}
};
using RenderPassAreaFunc = std::function<RenderPassArea(const VkExtent2D &)>;

class GraphicsPassBase;
class PassBase : public ObjectBase {
private:
	PassType m_type{};

public:
	inline PassBase(Parent parent, PassType type) : ObjectBase(parent), m_type{type} {};
	inline ~PassBase() override = default;

	inline PassType GetType() const { return m_type; }

	template <typename Visitor> inline std::invoke_result_t<Visitor, GraphicsPassBase *> Visit(Visitor &&visitor) const;

	virtual void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const = 0;
};

template <typename Derived> class PassPool : public Pool<Derived, PassBase> {
private:
	using PoolBase = Pool<Derived, PassBase>;

public:
	inline PassPool() = default;
	inline ~PassPool() override = default;

	inline const auto &GetPassPoolData() const { return PoolBase::GetPoolData(); }

protected:
	template <typename PassType, typename... Args>
	inline PassType *CreatePass(const PoolKey &pass_key, Args &&...args) {
		static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kPassChanged);
		return PoolBase::template Construct<PassType, Args...>(pass_key, std::forward<Args>(args)...);
	}
	template <typename PassType = PassBase> inline PassType *GetPass(const PoolKey &pass_key) const {
		return PoolBase::template Get<PassType>(pass_key);
	}
	inline void ClearPasses() {
		PoolBase::Clear();
		static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kPassChanged);
	}
};

class GraphicsPassBase : public PassBase,
                         public InputPool<GraphicsPassBase>,
                         public ResourcePool<GraphicsPassBase>,
                         public AttachmentInputSlot<GraphicsPassBase>,
                         public DescriptorInputSlot<GraphicsPassBase> {
private:
	std::optional<RenderPassArea> m_opt_area = std::nullopt;
	std::optional<RenderPassAreaFunc> m_opt_area_func = std::nullopt;

public:
	inline constexpr PassType GetType() const { return PassType::kGraphics; }

	inline GraphicsPassBase(Parent parent) : PassBase(parent, PassType::kGraphics) {}
	inline ~GraphicsPassBase() override = default;

	uint32_t GetSubpass() const;
	const myvk::Ptr<myvk::RenderPass> &GetVkRenderPass() const;

	const myvk::Ptr<myvk::DescriptorSetLayout> &GetVkDescriptorSetLayout() const;
	const myvk::Ptr<myvk::DescriptorSet> &GetVkDescriptorSet() const;

	const ImageBase *GetInputImage(const PoolKey &input_key) const;
	const BufferBase *GetInputBuffer(const PoolKey &input_key) const;

	inline void SetRenderArea(VkExtent2D extent, uint32_t layer = 1) {
		if (m_opt_area_func || !m_opt_area || m_opt_area != RenderPassArea{extent, layer})
			EmitEvent(Event::kRenderAreaChanged);
		m_opt_area = RenderPassArea{extent, layer};
		m_opt_area_func = std::nullopt;
	}
	inline void SetRenderArea(const RenderPassAreaFunc &area_func) {
		if (m_opt_area || !m_opt_area_func)
			EmitEvent(Event::kRenderAreaChanged);
		m_opt_area = std::nullopt;
		m_opt_area_func = area_func;
	}
	inline void ClearRenderArea() {
		if (m_opt_area || m_opt_area_func)
			EmitEvent(Event::kRenderAreaChanged);
		m_opt_area = std::nullopt;
		m_opt_area_func = std::nullopt;
	}
	inline std::optional<std::variant<RenderPassArea, RenderPassAreaFunc>> GetOptRenderArea() const {
		if (m_opt_area)
			return *m_opt_area;
		if (m_opt_area_func)
			return *m_opt_area_func;
		return std::nullopt;
	}

	inline void UpdatePipeline() { EmitEvent(Event::kUpdatePipeline); }
	virtual myvk::Ptr<myvk::GraphicsPipeline> CreatePipeline() const = 0;
	const myvk::Ptr<myvk::PipelineBase> &GetVkPipeline() const;
};

class ComputePassBase : public PassBase,
                        public InputPool<ComputePassBase>,
                        public ResourcePool<ComputePassBase>,
                        public DescriptorInputSlot<ComputePassBase> {
public:
	inline constexpr PassType GetType() const { return PassType::kCompute; }

	inline ComputePassBase(Parent parent) : PassBase(parent, PassType::kCompute) {}
	inline ~ComputePassBase() override = default;

	virtual myvk::Ptr<myvk::ComputePipeline> CreatePipeline() const = 0;
	inline void UpdatePipeline() { EmitEvent(Event::kUpdatePipeline); }
	const myvk::Ptr<myvk::PipelineBase> &GetVkPipeline() const;

	const ImageBase *GetInputImage(const PoolKey &input_key) const;
	const BufferBase *GetInputBuffer(const PoolKey &input_key) const;

	const myvk::Ptr<myvk::DescriptorSetLayout> &GetVkDescriptorSetLayout() const;
	const myvk::Ptr<myvk::DescriptorSet> &GetVkDescriptorSet() const;
};

class TransferPassBase : public PassBase, public InputPool<TransferPassBase>, public ResourcePool<TransferPassBase> {
public:
	inline constexpr PassType GetType() const { return PassType::kTransfer; }

	const ImageBase *GetInputImage(const PoolKey &input_key) const;
	const BufferBase *GetInputBuffer(const PoolKey &input_key) const;

	inline TransferPassBase(Parent parent) : PassBase(parent, PassType::kTransfer) {}
	inline ~TransferPassBase() override = default;
};

class PassGroupBase : public PassBase, public ResourcePool<PassGroupBase>, public PassPool<PassGroupBase> {
public:
	inline constexpr PassType GetType() const { return PassType::kGroup; }

	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &) const final {}

	inline PassGroupBase(Parent parent) : PassBase(parent, PassType::kGroup) {}
	inline ~PassGroupBase() override = default;
};

template <typename Visitor> std::invoke_result_t<Visitor, GraphicsPassBase *> PassBase::Visit(Visitor &&visitor) const {
	switch (GetType()) {
	case PassType::kGraphics:
		return visitor(static_cast<const GraphicsPassBase *>(this));
	case PassType::kCompute:
		return visitor(static_cast<const ComputePassBase *>(this));
	case PassType::kTransfer:
		return visitor(static_cast<const TransferPassBase *>(this));
	case PassType::kGroup:
		return visitor(static_cast<const PassGroupBase *>(this));
	default:
		assert(false);
	}
	return visitor(static_cast<const GraphicsPassBase *>(nullptr));
}

template <typename T>
concept PassWithInput = !std::derived_from<T, PassGroupBase>;
template <typename T>
concept PassWithPipeline = std::derived_from<T, ComputePassBase> || std::derived_from<T, GraphicsPassBase>;

} // namespace myvk_rg::interface

#endif
