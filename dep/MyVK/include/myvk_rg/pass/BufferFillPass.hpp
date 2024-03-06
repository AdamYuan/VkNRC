//
// Created by adamyuan on 2/28/24.
//

#pragma once
#ifndef MYVK_RG_CLEAR_BUFFER_PASS_HPP
#define MYVK_RG_CLEAR_BUFFER_PASS_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace myvk_rg {
class BufferFillPass final : public TransferPassBase {
private:
	uint32_t m_data{};

public:
	inline BufferFillPass(Parent parent, const myvk_rg::Buffer &dst, uint32_t data = 0)
	    : TransferPassBase(parent), m_data{data} {
		AddInput<myvk_rg::Usage::kTransferBufferDst, VK_PIPELINE_STAGE_2_CLEAR_BIT>({"dst"}, dst);
	}
	inline ~BufferFillPass() final = default;

	inline void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final {
		const auto &buffer_view = GetInputBuffer({"dst"})->GetBufferView();
		vkCmdFillBuffer(command_buffer->GetHandle(), buffer_view.buffer->GetHandle(), buffer_view.offset,
		                buffer_view.size, m_data);
	}
	inline auto GetDstOutput() { return MakeBufferOutput({"dst"}); }
};
} // namespace myvk_rg

#endif
