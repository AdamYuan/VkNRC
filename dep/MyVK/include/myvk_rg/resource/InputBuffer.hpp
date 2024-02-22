#ifndef MYVK_RG_STATIC_BUFFER_HPP
#define MYVK_RG_STATIC_BUFFER_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace myvk_rg {
class InputBuffer final : public myvk_rg::ExternalBufferBase {
private:
	BufferView m_buffer_view;

public:
	inline InputBuffer(myvk_rg::Parent parent, const myvk::Ptr<myvk::BufferBase> &buffer)
	    : myvk_rg::ExternalBufferBase(parent) {
		SetBufferView(buffer);
		SetSyncType(ExternalSyncType::kLastFrame);
	}
	inline InputBuffer(myvk_rg::Parent parent, BufferView buffer_view) : myvk_rg::ExternalBufferBase(parent) {
		SetBufferView(std::move(buffer_view));
		SetSyncType(ExternalSyncType::kLastFrame);
	}
	inline InputBuffer(myvk_rg::Parent parent) : myvk_rg::ExternalBufferBase(parent) {
		SetSyncType(ExternalSyncType::kLastFrame);
	}
	inline ~InputBuffer() final = default;
	inline void SetBufferView(const myvk::Ptr<myvk::BufferBase> &buffer) {
		m_buffer_view = {.buffer = buffer, .offset = 0, .size = buffer->GetSize()};
	}
	inline void SetBufferView(BufferView buffer_view) { m_buffer_view = std::move(buffer_view); }
	inline const BufferView &GetBufferView() const final { return m_buffer_view; }
};
} // namespace myvk_rg

#endif
