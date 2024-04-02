//
// Created by adamyuan on 4/2/24.
//

#pragma once
#ifndef VKNRC_RECONSTRUCTPASS_HPP
#define VKNRC_RECONSTRUCTPASS_HPP

#include "../NRCState.hpp"
#include <myvk_rg/RenderGraph.hpp>

namespace rg {

class ReconstructPass final : public myvk_rg::ComputePassBase {
public:
	struct Args {
		const myvk_rg::Buffer &eval_dests, &eval_outputs;
		const myvk_rg::Image &bias_factor_r, &factor_gb;
		std::span<const myvk_rg::Buffer, NRCState::GetTrainBatchCount()> batch_train_biases, batch_train_factors;
	};

private:
	uint32_t m_count{};

public:
	ReconstructPass(myvk_rg::Parent parent, const Args &args);
	inline ~ReconstructPass() final = default;
	inline void SetCount(uint32_t count) { m_count = count; }
	myvk::Ptr<myvk::ComputePipeline> CreatePipeline() const final;
	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final;
	inline auto GetColorOutput() const { return MakeImageOutput({"bias_factor_r"}); }
	inline auto GetBatchTrainBiasesOutput(uint32_t batch_index) const {
		return MakeBufferOutput({"batch_train_biases", batch_index});
	}
};

} // namespace rg

#endif // VKNRC_RECONSTRUCTPASS_HPP
