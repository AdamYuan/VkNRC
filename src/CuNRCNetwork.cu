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

template <uint32_t Dims> tcnn::GPUMatrix<float> to_tcnn_gpu_matrix(const CuNRCData<Dims> &data) {
	return tcnn::GPUMatrix<float>(data.p_data, Dims, data.count, data.stride);
}

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
	         {"learning_rate", 1e-3},
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

void CuNRCNetwork::Inference(const CuNRCInput &inputs, const CuNRCOutput &outputs) const {
	auto output_gpu_matrix = to_tcnn_gpu_matrix(outputs);
	m_p_cuda_impl->network->inference(m_p_cuda_impl->stream, to_tcnn_gpu_matrix(inputs), output_gpu_matrix);
}

void CuNRCNetwork::Train(const CuNRCInput &inputs, const CuNRCOutput &targets) {
	m_p_cuda_impl->trainer->training_step(m_p_cuda_impl->stream, to_tcnn_gpu_matrix(inputs),
	                                      to_tcnn_gpu_matrix(targets));
}

void CuNRCNetwork::Synchronize() const { cudaStreamSynchronize(m_p_cuda_impl->stream); }
