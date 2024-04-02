//
// Created by adamyuan on 4/1/24.
//

#pragma once
#ifndef VKNRC_CUNRCNETWORK_HPP
#define VKNRC_CUNRCNETWORK_HPP

#include "CuVkBuffer.hpp"

inline constexpr uint32_t kCuNRCInputDims = 14, kCuNRCOutputDims = 3;
inline constexpr uint32_t kTCNNBlockCount = 256;

class CuNRCNetwork {
private:
	struct CudaImpl;
	CudaImpl *m_p_cuda_impl;

public:
	CuNRCNetwork();
	~CuNRCNetwork();

	void Inference(const CuVkBuffer &inputs, const CuVkBuffer &outputs, uint32_t count) const;
	void Train(const CuVkBuffer &inputs, const CuVkBuffer &targets, uint32_t count);
	void Reset();
	void Synchronize() const;
};

#endif // VKNRC_CUNRCNETWORK_HPP
