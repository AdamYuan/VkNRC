#include <Eigen/Eigen>
#include <cinttypes>
#include <random>
#include <vuda_runtime.hpp>

constexpr uint32_t kWorkgroupSize = 128, kOutputCount = 3;

using half = _Float16;
static_assert(sizeof(half) == sizeof(uint16_t));

std::vector<half> Evaluate(std::span<half> weights, std::span<half> inputs) {
	using ActivateMatrix = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	auto act_matrix = Eigen::Map<ActivateMatrix>(inputs.data(), (Eigen::Index)64, (Eigen::Index)(inputs.size() / 64));

	using WeightMatrix = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	for (int i = 0; i < 5; ++i) {
		auto weight_matrix = Eigen::Map<WeightMatrix>(weights.data() + i * 64 * 64, (Eigen::Index)64, (Eigen::Index)64);
		ActivateMatrix next_act_matrix =
		    (weight_matrix * act_matrix).unaryExpr([](const half &x) -> half { return std::max(x, half{0}); });
		act_matrix = next_act_matrix;
	}
	auto weight_matrix =
	    Eigen::Map<WeightMatrix>(weights.data() + 5 * 64 * 64, (Eigen::Index)kOutputCount, (Eigen::Index)64);
	ActivateMatrix out_matrix = weight_matrix * act_matrix;
	std::vector<half> output(out_matrix.data(), out_matrix.data() + out_matrix.size());
	return output;
}

std::vector<float> Train(std::span<half> weights, std::span<half> inputs, std::span<half> targets) {
	if (inputs.size() / 64 != targets.size() / 3) {
		printf("Invalid\n");
		return {};
	}
	using ActivateMatrix = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	using WeightMatrix = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	std::vector<WeightMatrix> weight_matrices;
	weight_matrices.reserve(6);
	for (int i = 0; i < 5; ++i)
		weight_matrices.emplace_back(
		    Eigen::Map<WeightMatrix>(weights.data() + i * 64 * 64, (Eigen::Index)64, (Eigen::Index)64));
	weight_matrices.emplace_back(
	    Eigen::Map<WeightMatrix>(weights.data() + 5 * 64 * 64, (Eigen::Index)kOutputCount, (Eigen::Index)64));

	std::vector<ActivateMatrix> act_matrices;
	act_matrices.reserve(7);
	act_matrices.emplace_back(
	    Eigen::Map<ActivateMatrix>(inputs.data(), (Eigen::Index)64, (Eigen::Index)(inputs.size() / 64)));
	for (int i = 1; i < 7; ++i) {
		act_matrices.emplace_back(
		    (weight_matrices[i - 1] * act_matrices[i - 1]).eval().unaryExpr([](const half &x) -> half {
			    return std::max(x, half{0});
		    }));
	}
	ActivateMatrix target_matrix = Eigen::Map<WeightMatrix>(targets.data(), (Eigen::Index)kOutputCount,
	                                                        (Eigen::Index)(targets.size() / kOutputCount));

	using DWMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using DATMatrix = Eigen::Matrix<half, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

	std::vector<float> dw(5 * 64 * 64 + kOutputCount * 64);
	DATMatrix dat_matrix = (act_matrices[6] - target_matrix).transpose(); // L2 Loss
	// std::cout << dat_matrix << std::endl;
	for (int i = 5; i >= 0; --i) {
		DWMatrix dw_matrix = (act_matrices[i] * dat_matrix).eval().transpose().cast<float>();
		printf("%lu %lu\n", dw_matrix.rows(), dw_matrix.cols());
		std::copy(dw_matrix.data(), dw_matrix.data() + dw_matrix.size(), dw.data() + i * 64 * 64);
		dat_matrix = (dat_matrix * weight_matrices[i]).eval().unaryExpr([](const half &x) -> half {
			return x >= 0 ? half(1) : half(0);
		});
		// std::cout << dat_matrix << std::endl;
	}
	return dw;
}

template <typename Func> inline double ms(Func &&func) {
	auto begin = std::chrono::high_resolution_clock::now();
	func();
	auto end = std::chrono::high_resolution_clock::now();
	return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000000.0;
}

double error_3(const half *p_l, const half *p_r) {
	double e0 = p_l[0] - p_r[0];
	double e1 = p_l[1] - p_r[1];
	double e2 = p_l[2] - p_r[2];
	e0 *= e0, e1 *= e1, e2 *= e2;
	return std::sqrt(e0 + e1 + e2);
}

void test_inference(std::size_t blocks) {
	std::mt19937 random{std::random_device{}()};
	std::vector<half> weights(64 * 64 * 5 + 64 * kOutputCount), inputs(kWorkgroupSize * blocks * 64);

	for (int x = 0; auto &w : weights)
		w = half(std::uniform_real_distribution<float>{-0.02, 0.02}(random));
	// for (int x = 0; auto &w : weights)
	// 		w = half(0.00001 * (x++)); // std::uniform_real_distribution<float>{0, 0.25}(random);
	// w = std::uniform_real_distribution<float>{0, 0.1}(random);
	for (int x = 0; auto &i : inputs)
		i = half(std::uniform_real_distribution<float>{0, 1.0}(random));

	std::vector<half> comp_outputs(kWorkgroupSize * blocks * kOutputCount);
	{
		int *gpu_weights, *gpu_inputs, *gpu_outputs;
		cudaMalloc((void **)&gpu_weights, weights.size() * sizeof(half));
		cudaMalloc((void **)&gpu_inputs, inputs.size() * sizeof(half));
		cudaMalloc((void **)&gpu_outputs, comp_outputs.size() * sizeof(half));

		cudaMemcpy(gpu_weights, weights.data(), weights.size() * sizeof(half), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_inputs, inputs.data(), inputs.size() * sizeof(half), cudaMemcpyHostToDevice);
		cudaStreamSynchronize(0);
		// SubgroupSize = 32 for NVIDIA
		double time_0 = ms([&]() {
			// vuda::launchKernel("empty.spv", "main", 0, (int)blocks, kWorkgroupSize);
			vuda::launchKernel("evaluate_32.spv", "main", 0, (int)blocks, kWorkgroupSize, gpu_weights, gpu_inputs,
			                   gpu_outputs);
			cudaStreamSynchronize(0);
		});
		cudaStreamSynchronize(0);
		double time = ms([&]() {
			vuda::launchKernel("evaluate_32.spv", "main", 0, (int)blocks, kWorkgroupSize, gpu_weights, gpu_inputs,
			                   gpu_outputs);
			cudaStreamSynchronize(0);
		});
		std::cout << "GLSL: " << time << " ms" << std::endl;
		// copy result to host
		cudaMemcpy(comp_outputs.data(), gpu_outputs, comp_outputs.size() * sizeof(half), cudaMemcpyDeviceToHost);
	}

	double total_error = 0.0;
	auto real_outputs = Evaluate(weights, inputs);
	std::cout << real_outputs.size() << std::endl;
	for (std::size_t i = 0; i < kWorkgroupSize * blocks; ++i) {
		std::cout << "# " << i << std::endl;
		std::cout << "REAL: " << float(real_outputs[i * kOutputCount + 0]) << ",\t"
		          << float(real_outputs[i * kOutputCount + 1]) << ",\t" << float(real_outputs[i * kOutputCount + 2])
		          << std::endl;
		std::cout << "COMP: " << float(comp_outputs[i * kOutputCount + 0]) << ",\t"
		          << float(comp_outputs[i * kOutputCount + 1]) << ",\t" << float(comp_outputs[i * kOutputCount + 2])
		          << std::endl
		          << std::endl;

		total_error += error_3(real_outputs.data() + i * kOutputCount, comp_outputs.data() + i * kOutputCount);
	}
	total_error /= double(kWorkgroupSize * blocks);
	printf("Avg MSE: %f\n", total_error);
}

void test_train(std::size_t blocks) {
	std::mt19937 random{std::random_device{}()};
	std::vector<half> weights(64 * 64 * 5 + 64 * kOutputCount), inputs(kWorkgroupSize * blocks * 64),
	    targets(kWorkgroupSize * blocks * kOutputCount);

	for (int x = 0; auto &w : weights)
		w = half(std::uniform_real_distribution<float>{-0.02, 0.02}(random));
	// w = 0.1 - 0.00001 * (x++); // std::uniform_real_distribution<float>{0, 0.25}(random);
	// w = std::uniform_real_distribution<float>{0, 0.02}(random);
	for (auto &i : inputs)
		i = half(std::uniform_real_distribution<float>{0, 1.0}(random));
	for (auto &i : targets)
		i = half(std::uniform_real_distribution<float>{0, 1.0}(random));

	std::vector<float> comp_dw(5 * 64 * 64 + kOutputCount * 64);
	{
		int *gpu_weights, *gpu_dw, *gpu_dw_1, *gpu_inputs, *gpu_targets;
		cudaMalloc((void **)&gpu_weights, weights.size() * sizeof(half));
		cudaMalloc((void **)&gpu_dw, comp_dw.size() * sizeof(float));
		cudaMalloc((void **)&gpu_dw_1, comp_dw.size() * sizeof(float));
		cudaMalloc((void **)&gpu_inputs, inputs.size() * sizeof(half));
		cudaMalloc((void **)&gpu_targets, targets.size() * sizeof(half));

		cudaMemcpy(gpu_weights, weights.data(), weights.size() * sizeof(half), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_dw, comp_dw.data(), comp_dw.size() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_inputs, inputs.data(), inputs.size() * sizeof(half), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_targets, targets.data(), targets.size() * sizeof(half), cudaMemcpyHostToDevice);
		cudaStreamSynchronize(0);
		// SubgroupSize = 32 for NVIDIA
		double time_0 = ms([&]() {
			vuda::launchKernel("train_32.spv", "main", 0, (int)blocks, kWorkgroupSize, gpu_weights, gpu_dw_1,
			                   gpu_inputs, gpu_targets);
			cudaStreamSynchronize(0);
		});
		cudaStreamSynchronize(0);
		double time = ms([&]() {
			vuda::launchKernel("train_32.spv", "main", 0, (int)blocks, kWorkgroupSize, gpu_weights, gpu_dw, gpu_inputs,
			                   gpu_targets);
			cudaStreamSynchronize(0);
		});
		cudaStreamSynchronize(0);
		double time_2 = ms([&]() {
			vuda::launchKernel("train_32.spv", "main", 0, (int)blocks, kWorkgroupSize, gpu_weights, gpu_dw_1,
			                   gpu_inputs, gpu_targets);
			cudaStreamSynchronize(0);
		});
		std::cout << "GLSL: " << time << "," << time_2 << " ms" << std::endl;
		// copy result to host
		cudaMemcpy(comp_dw.data(), gpu_dw, comp_dw.size() * sizeof(float), cudaMemcpyDeviceToHost);
	}

	auto real_dw = Train(weights, inputs, targets);
	double error = 0.0, r_error = 0.0;
	for (std::size_t i = 0; i < 64 * 5 + kOutputCount; ++i) {
		std::cout << "# " << i << std::endl;
		std::cout << "REAL: " << std::endl;
		for (std::size_t j = 0; j < 64; ++j)
			std::cout << real_dw[i * 64 + j] << ", ";
		printf("\n");
		std::cout << "COMP: " << std::endl;
		for (std::size_t j = 0; j < 64; ++j)
			std::cout << comp_dw[i * 64 + j] << ", ";
		printf("\n");

		for (std::size_t j = 0; j < 64; ++j) {
			double diff = std::abs(double(comp_dw[i * 64 + j] - real_dw[i * 64 + j]));
			double r_diff = diff / real_dw[i * 64 + j];
			if (std::isfinite(diff))
				error += diff;
			if (std::isfinite(r_diff))
				r_error += r_diff;
		}
	}
	error /= double(64 * 5 + kOutputCount) * double(64);
	r_error /= double(64 * 5 + kOutputCount) * double(64);
	printf("Avg ERROR: %f\n", error);
	printf("Avg ERROR / BATCH_SIZE: %f\n", error / double(blocks * kWorkgroupSize));
	printf("Avg Relative ERROR: %f\n", r_error);
}

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc == 0)
		return EXIT_FAILURE;
	std::size_t blocks = 1;
	for (int i = 0; i < argc; ++i)
		blocks *= std::stoull(argv[i]);
	blocks /= kWorkgroupSize;

	cudaSetDevice(0);

	// test_inference(blocks);
	test_train(blocks);

	return 0;
}
