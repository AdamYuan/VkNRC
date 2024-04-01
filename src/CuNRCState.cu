//
// Created by adamyuan on 4/1/24.
//

#include "CuNRCState.hpp"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>

struct CuNRCState::TCNNImpl {
	std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> loss;
	std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> optimizer;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> network;
	std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> trainer;
};

CuNRCState::CuNRCState() {
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
	auto model = tcnn::create_from_config(14, 3, config);

	m_p_tcnn_impl = std::make_unique<TCNNImpl>(TCNNImpl{
	    .loss = std::move(model.loss),
	    .optimizer = std::move(model.optimizer),
	    .network = std::move(model.network),
	    .trainer = std::move(model.trainer),
	});
}
