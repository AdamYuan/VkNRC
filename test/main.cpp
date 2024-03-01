#include <Eigen/Eigen>
#include <cinttypes>
#include <half.hpp>
#include <random>
#include <vuda_runtime.hpp>

using half_float::half;
static_assert(sizeof(half) == sizeof(uint16_t));

std::vector<half> Evaluate(std::span<half> weights, std::span<half> inputs) {
	using ActivateMatrix = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	auto act_matrix = Eigen::Map<ActivateMatrix>(inputs.data(), (Eigen::Index)64, (Eigen::Index)(inputs.size() / 64));

	using WeightMatrix = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	for (int i = 0; i < 5; ++i) {
		auto weight_matrix = Eigen::Map<WeightMatrix>(weights.data() + i * 64 * 64, (Eigen::Index)64, (Eigen::Index)64);
		ActivateMatrix next_act_matrix = weight_matrix * act_matrix;
		next_act_matrix.unaryExpr([](const half &x) -> half { return half_float::fmax(x, half{0}); });
		act_matrix = next_act_matrix;
	}
	auto weight_matrix = Eigen::Map<WeightMatrix>(weights.data() + 5 * 64 * 64, (Eigen::Index)4, (Eigen::Index)64);
	ActivateMatrix out_matrix = weight_matrix * act_matrix;
	out_matrix.unaryExpr([](const half &x) -> half { return half_float::fmax(x, half{0}); });
	std::vector<half> output(out_matrix.data(), out_matrix.data() + out_matrix.size());
	return output;
}

template <typename Func> inline double ms(Func &&func) {
	auto begin = std::chrono::high_resolution_clock::now();
	func();
	auto end = std::chrono::high_resolution_clock::now();
	return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000000.0;
}

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc == 0)
		return EXIT_FAILURE;
	std::size_t blocks = 1;
	for (int i = 0; i < argc; ++i)
		blocks *= std::stoull(argv[i]);
	blocks /= 128;

	std::mt19937 random{std::random_device{}()};
	std::vector<half> weights(64 * 64 * 5 + 64 * 16), inputs(128 * blocks * 64);

	for (auto &w : weights)
		w = std::uniform_real_distribution<float>{0, 0.1}(random);
	for (auto &i : inputs)
		i = std::uniform_real_distribution<float>{0, 0.1}(random);

	std::vector<half> comp_outputs(128 * blocks * 4);
	{
		cudaSetDevice(0);
		int *gpu_weights, *gpu_inputs, *gpu_outputs;
		cudaMalloc((void **)&gpu_weights, weights.size() * sizeof(half));
		cudaMalloc((void **)&gpu_inputs, inputs.size() * sizeof(half));
		cudaMalloc((void **)&gpu_outputs, comp_outputs.size() * sizeof(half));

		cudaMemcpy(gpu_weights, weights.data(), weights.size() * sizeof(half), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_inputs, inputs.data(), inputs.size() * sizeof(half), cudaMemcpyHostToDevice);
		cudaStreamSynchronize(0);
		// SubgroupSize = 32 for NVIDIA
		double time = ms([&]() {
			vuda::launchKernel("evaluate_32.spv", "main", 0, (int)blocks, 128, gpu_weights, gpu_inputs, gpu_outputs);
			cudaStreamSynchronize(0);
		});
		std::cout << "GLSL: " << time << " ms" << std::endl;
		// copy result to host
		cudaMemcpy(comp_outputs.data(), gpu_outputs, comp_outputs.size() * sizeof(half), cudaMemcpyDeviceToHost);
	}

	auto real_outputs = Evaluate(weights, inputs);
	std::cout << real_outputs.size() << std::endl;
	for (std::size_t i = 0; i < 128 * blocks; ++i) {
		std::cout << "REAL: " << real_outputs[i * 4 + 0] << ",\t" << real_outputs[i * 4 + 1] << ",\t"
		          << real_outputs[i * 4 + 2] << std::endl;
		std::cout << "COMP: " << comp_outputs[i * 4 + 0] << ",\t" << comp_outputs[i * 4 + 1] << ",\t"
		          << comp_outputs[i * 4 + 2] << std::endl
		          << std::endl;
	}

	return 0;
}