//
// Created by adamyuan on 4/1/24.
//

#include "CuNRCNetwork.hpp"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>

struct CuNRCNetwork::CudaImpl {
	std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> loss;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> network;
	std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> trainer;
	cudaStream_t stream;
};

CuNRCNetwork::~CuNRCNetwork() {
	cudaStreamDestroy(m_p_cuda_impl->stream);
	delete m_p_cuda_impl;
}

CuNRCNetwork::CuNRCNetwork() {
	tcnn::json config = {
	    {"loss", {{"otype", "RelativeL2Luminance"}}},
	    {"optimizer",
	     {
	         {"otype", "Adam"},
	         {"learning_rate", 0.002},
	         {"beta1", 0.9},
	         {"beta2", 0.999},
	         {"epsilon", 1e-8},
	         {"l2_reg", 1e-8},
	     }},
	    {"encoding",
	     {
	         {"otype", "NRC"},
	     }},
	    {"network",
	     {
	         {"otype", "FullyFusedMLP"},
	         {"n_neurons", 64},
	         {"n_hidden_layers", 5},
	         {"activation", "ReLU"},
	         {"output_activation", "None"},
	     }},
	};
	auto model = tcnn::create_from_config(kCuNRCInputDims, kCuNRCOutputDims, config);

	cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

	m_p_cuda_impl = new CudaImpl{
	    .loss = std::move(model.loss),
	    .optimizer = std::move(model.optimizer),
	    .network = std::move(model.network),
	    .trainer = std::move(model.trainer),
	    .stream = stream,
	};
}

void CuNRCNetwork::Reset() { m_p_cuda_impl->trainer->initialize_params(); }

void CuNRCNetwork::Inference(const CuVkBuffer &inputs, const CuVkBuffer &outputs, uint32_t count) const {
	count = (count + kTCNNBlockCount - 1u) / kTCNNBlockCount * kTCNNBlockCount;
	tcnn::GPUMatrix<float> input_mat{inputs.GetMappedPtr<float>(), kCuNRCInputDims, count};
	tcnn::GPUMatrix<float> output_mat{outputs.GetMappedPtr<float>(), kCuNRCOutputDims, count};
	m_p_cuda_impl->network->inference(m_p_cuda_impl->stream, input_mat, output_mat);
}
void CuNRCNetwork::Train(const CuVkBuffer &inputs, const CuVkBuffer &targets, uint32_t count) {
	count = count / kTCNNBlockCount * kTCNNBlockCount;
	if (count == 0)
		return;
	tcnn::GPUMatrix<float> input_mat{inputs.GetMappedPtr<float>(), kCuNRCInputDims, count};
	tcnn::GPUMatrix<float> target_mat{targets.GetMappedPtr<float>(), kCuNRCOutputDims, count};
	m_p_cuda_impl->trainer->training_step(m_p_cuda_impl->stream, input_mat, target_mat);
}

void CuNRCNetwork::Synchronize() const { cudaStreamSynchronize(m_p_cuda_impl->stream); }
