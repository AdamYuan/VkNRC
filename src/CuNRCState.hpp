//
// Created by adamyuan on 4/1/24.
//

#pragma once
#ifndef VKNRC_CUNRCSTATE_HPP
#define VKNRC_CUNRCSTATE_HPP

#include <memory>

inline constexpr uint32_t kCuNRCInputDims = 14, kCuNRCOutputDims = 3;
template <uint32_t Dims> struct CuNRCData {
	float *p_data{nullptr};
	uint32_t count = 0, stride = 0;
};

using CuNRCInput = CuNRCData<kCuNRCInputDims>;
using CuNRCOutput = CuNRCData<kCuNRCOutputDims>;

class CuNRCState {
public:
private:
	struct TCNNImpl;
	TCNNImpl *m_p_tcnn_impl;

public:
	CuNRCState();
	~CuNRCState();

	void Inference(const CuNRCInput &inputs, const CuNRCOutput &outputs) const;
	void Train(const CuNRCInput &inputs, const CuNRCOutput &targets);
	void Synchronize() const;
};

#endif // VKNRC_CUNRCSTATE_HPP
