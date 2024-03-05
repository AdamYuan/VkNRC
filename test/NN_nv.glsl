#ifndef NN_NV_GLSL
#define NN_NV_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_NV_cooperative_matrix : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_atomic_float : enable

#if WORKGROUP_SIZE != 128
#error WORKGROUP_SIZE must be 128
#endif

#ifndef NN_SET
#define NN_SET 0
#endif

layout(std430, set = NN_SET, binding = WEIGHTS_BINDING) readonly buffer uuWeights { uvec4 uWeights[]; };
#if defined(NN_BACKPROPAGATION)
layout(std430, set = NN_SET, binding = DWEIGHTS_BINDING) buffer uuDWeights { float uDWeights[]; };
#endif

#define SUBGROUP_COUNT (WORKGROUP_SIZE / SUBGROUP_SIZE)
#define FP16_PER_UV4 8

// Counts
#define FP_X 64
#define ACT_Y WORKGROUP_SIZE
#define ACT_COUNT (FP_X * ACT_Y)
#define WEIGHT_64_COUNT (FP_X * 64)
#define WEIGHT_16_COUNT (FP_X * 16)
// UVec4 Counts
#define UV4_X (FP_X / FP16_PER_UV4)                          // 8
#define ACT_UV4_COUNT (ACT_COUNT / FP16_PER_UV4)             // 1024
#define WEIGHT_64_UV4_COUNT (WEIGHT_64_COUNT / FP16_PER_UV4) // 512
#define WEIGHT_16_UV4_COUNT (WEIGHT_16_COUNT / FP16_PER_UV4) // 128
// Cooperative Matrix Counts
#define COOPMAT_X (FP_X / 16)                                 // 4
#define ACT_COOPMAT_COUNT (ACT_COUNT / (16 * 16))             // 32
#define ACT_COOPMAT_Y (ACT_Y / 16)                            // 8
#define WEIGHT_64_COOPMAT_COUNT (WEIGHT_64_COUNT / (16 * 16)) // 16
// Subgroup
#define SUBGROUP_COOPMAT_ROW (COOPMAT_X / SUBGROUP_COUNT)
// Thread Weight Counts
#define THREAD_WEIGHT_64_UV4_COUNT (WEIGHT_64_UV4_COUNT / WORKGROUP_SIZE)
// Subgroup Activates Counts
#define SUBGROUP_ACT_COOPMAT_Y (ACT_COOPMAT_Y / SUBGROUP_COUNT)
// Matrix Strides & Major
#define MAT64_COOPMAT_STRIDE UV4_X
#define MAT64_COOPMAT_ELEMENT(X, Y) ((X) * (16 / FP16_PER_UV4) + (Y)*16 * MAT64_COOPMAT_STRIDE)
#define WEIGHT_COOPMAT_MAJOR false // gl_CooperativeMatrixLayoutRowMajor
#define ACT_COOPMAT_MAJOR true     // gl_CooperativeMatrixLayoutColumnMajor
#define COOPMAT_MAJOR_T(x) (!(x))

#ifdef NN_BACKPROPAGATION
#define SHARED_BUFFER_SIZE \
	((WEIGHT_64_UV4_COUNT * SUBGROUP_COUNT) > ACT_UV4_COUNT ? (WEIGHT_64_UV4_COUNT * SUBGROUP_COUNT) : ACT_UV4_COUNT)
#else
#define SHARED_BUFFER_SIZE (WEIGHT_64_UV4_COUNT > ACT_UV4_COUNT ? WEIGHT_64_UV4_COUNT : ACT_UV4_COUNT)
#endif

shared uvec4 SHARED_BUFFER[SHARED_BUFFER_SIZE];

void _nn_load_weight_64(in const uint layer) {
	const uint kSharedUV4Base = gl_LocalInvocationID.x * THREAD_WEIGHT_64_UV4_COUNT;
	const uint kWeightUV4Base = layer * WEIGHT_64_UV4_COUNT + kSharedUV4Base;
	[[unroll]] for (uint i = 0; i < THREAD_WEIGHT_64_UV4_COUNT; ++i)
		SHARED_BUFFER[kSharedUV4Base + i] = uWeights[kWeightUV4Base + i];
	barrier();
}

void _nn_load_weight_16(in const uint layer) {
	// assert(WEIGHT_16_UV4_COUNT == 128 == WORKGROUP_SIZE);
	const uint kLayerUV4Base = layer * WEIGHT_64_UV4_COUNT;
	SHARED_BUFFER[gl_LocalInvocationID.x] = uWeights[kLayerUV4Base + gl_LocalInvocationID.x];
	barrier();
}

void NNLoadInput(in const uvec4 inputs[UV4_X],
                 inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	[[unroll]] for (uint x = 0; x < UV4_X; ++x)
		SHARED_BUFFER[(gl_LocalInvocationID.x * UV4_X) | x] = inputs[x];
	barrier();
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatLoadNV(act_coopmats[x][y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, workgroup_y),
			              MAT64_COOPMAT_STRIDE, ACT_COOPMAT_MAJOR);
		}
	}
	barrier();
}

void NNForward64_ReLU(in const uint layer,
                      in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
                      inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	_nn_load_weight_64(layer);
	// Zero Initialize
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x)
			dst_coopmats[x][y] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
	}
	// MMA
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, w_y), MAT64_COOPMAT_STRIDE,
			              WEIGHT_COOPMAT_MAJOR);
			[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
				dst_coopmats[w_y][a_y] = coopMatMulAddNV(weight_coopmat, src_coopmats[x][a_y], dst_coopmats[w_y][a_y]);
			}
		}
	}
	barrier();
	// ReLU
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			for (uint k = 0; k < dst_coopmats[x][y].length(); ++k)
				dst_coopmats[x][y][k] = max(dst_coopmats[x][y][k], float16_t(0));
		}
	}
}

void NNForward16(in const uint layer,
                 in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
                 inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_coopmats[SUBGROUP_ACT_COOPMAT_Y]) {
	_nn_load_weight_16(layer);
	// Zero Initialize
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y)
		dst_coopmats[y] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
	// MMA
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, 0), MAT64_COOPMAT_STRIDE,
		              WEIGHT_COOPMAT_MAJOR);
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			dst_coopmats[y] = coopMatMulAddNV(weight_coopmat, src_coopmats[x][y], dst_coopmats[y]);
		}
	}
	barrier();
}

uvec2 NNOutputUV2(in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[SUBGROUP_ACT_COOPMAT_Y]) {
	// Store
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		coopMatStoreNV(act_coopmats[y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(0, workgroup_y), MAT64_COOPMAT_STRIDE,
		               ACT_COOPMAT_MAJOR);
	}
	barrier();
	return SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X].rg;
}

#ifdef NN_BACKPROPAGATION
// http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf
void NNLoadDA16_L2Loss(in const uvec2 predict,
                       in const uvec2 target,
                       inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> da_coopmats_t[SUBGROUP_ACT_COOPMAT_Y]) {
	vec4 output_v4 = vec4(unpackHalf2x16(predict.x).xy, unpackHalf2x16(predict.y).x, 0);
	vec4 target_v4 = vec4(unpackHalf2x16(target.x).xy, unpackHalf2x16(target.y).x, 0);
	vec4 d_l2_v4 = output_v4 - target_v4;
	uvec2 d_l2 = uvec2(packHalf2x16(d_l2_v4.xy), packHalf2x16(d_l2_v4.zw));
	SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X].rg = d_l2;
	barrier();
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		coopMatLoadNV(da_coopmats_t[y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(0, workgroup_y), MAT64_COOPMAT_STRIDE,
		              COOPMAT_MAJOR_T(ACT_COOPMAT_MAJOR));
	}
	barrier();
}

void NNBackwardDA16_ReLU(
    in const uint layer,
    in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_da_coopmats_t[SUBGROUP_ACT_COOPMAT_Y],
    inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {

	_nn_load_weight_16(layer);
	// da_coopmats^T (128, 16) x weights (16, 64) = dA^T (128, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, 0), MAT64_COOPMAT_STRIDE,
		              WEIGHT_COOPMAT_MAJOR);
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			// Zero Initialize
			dst_da_coopmats_t[x][y] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
			// MMA
			dst_da_coopmats_t[x][y] = coopMatMulAddNV(src_da_coopmats_t[y], weight_coopmat, dst_da_coopmats_t[x][y]);
			// Inv ReLU
			for (uint k = 0; k < dst_da_coopmats_t[x][y].length(); ++k)
				dst_da_coopmats_t[x][y][k] = dst_da_coopmats_t[x][y][k] > 0.0 ? float16_t(1.0) : float16_t(0.0);
		}
	}
	barrier();
}

void NNBackwardDA64_ReLU(
    in const uint layer,
    in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
    inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	// Zero Initialize
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			dst_da_coopmats_t[x][y] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
		}
	}
	// da_coopmats^T (128, 64) x weights (64, 64) = dA^T (128, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
	// MMA
	_nn_load_weight_64(layer);
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, w_y), MAT64_COOPMAT_STRIDE,
			              WEIGHT_COOPMAT_MAJOR);
			[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
				dst_da_coopmats_t[w_y][a_y] =
				    coopMatMulAddNV(src_da_coopmats_t[x][a_y], weight_coopmat, dst_da_coopmats_t[w_y][a_y]);
			}
		}
	}
	barrier();
	// Inv ReLU
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			for (uint k = 0; k < dst_da_coopmats_t[x][y].length(); ++k)
				dst_da_coopmats_t[x][y][k] = dst_da_coopmats_t[x][y][k] > 0.0 ? float16_t(1.0) : float16_t(0.0);
		}
	}
}

void NNUpdateDW16(in const uint layer,
                  in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> da_coopmats_t[SUBGROUP_ACT_COOPMAT_Y],
                  in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	// (act_coopmats (64, 128) x da_coopmats^T (128, 16))^T = dW (16, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dw_coopmats[COOPMAT_X], act_coopmat_t;
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		dw_coopmats[x] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			dw_coopmats[x] = coopMatMulAddNV(act_coopmats[x][y], da_coopmats_t[y], dw_coopmats[x]);
		}
	}
	barrier();
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		coopMatStoreNV(dw_coopmats[x], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, 0), MAT64_COOPMAT_STRIDE,
		               COOPMAT_MAJOR_T(WEIGHT_COOPMAT_MAJOR));
	}
	barrier();
	uvec4 d_w_uv4 = SHARED_BUFFER[gl_LocalInvocationID.x];
	const uint kWeightFP16Base = layer * WEIGHT_64_COUNT + gl_LocalInvocationID.x * FP16_PER_UV4;
	[[unroll]] for (uint i = 0; i < (FP16_PER_UV4 / 2); ++i) {
		vec2 d_w_2 = unpackHalf2x16(d_w_uv4[i]);
		atomicAdd(uDWeights[kWeightFP16Base + (i << 1u)], d_w_2.x);
		atomicAdd(uDWeights[kWeightFP16Base + (i << 1u | 1u)], d_w_2.x);
	}
	barrier();
}

void NNUpdateDW64(in const uint layer,
                  in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
                  in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	// (act_coopmats (64, 128) x da_coopmats^T (128, 64))^T = dW (64, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dw_coopmats[COOPMAT_X][COOPMAT_X], act_coopmat_t;
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			dw_coopmats[w_y][x] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
			[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
				dw_coopmats[w_y][x] =
				    coopMatMulAddNV(act_coopmats[x][a_y], da_coopmats_t[w_y][a_y], dw_coopmats[w_y][x]);
			}
		}
	}
	barrier();
	[[unroll]] for (uint y = 0; y < COOPMAT_X; ++y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatStoreNV(dw_coopmats[y][x], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, y), MAT64_COOPMAT_STRIDE,
			               COOPMAT_MAJOR_T(WEIGHT_COOPMAT_MAJOR));
		}
	}
	barrier();
	const uint kSharedUV4Base = gl_LocalInvocationID.x * THREAD_WEIGHT_64_UV4_COUNT;
	const uint kWeightUV4Base = layer * WEIGHT_64_UV4_COUNT + kSharedUV4Base;
	[[unroll]] for (uint u = 0; u < THREAD_WEIGHT_64_UV4_COUNT; ++u) {
		uvec4 d_w_uv4 = SHARED_BUFFER[kSharedUV4Base + u];
		const uint kWeightFP16Base = (kWeightUV4Base + u) * FP16_PER_UV4;
		[[unroll]] for (uint i = 0; i < (FP16_PER_UV4 / 2); ++i) {
			vec2 d_w_2 = unpackHalf2x16(d_w_uv4[i]);
			atomicAdd(uDWeights[kWeightFP16Base + (i << 1u)], d_w_2.x);
			atomicAdd(uDWeights[kWeightFP16Base + (i << 1u | 1u)], d_w_2.x);
		}
	}
	barrier();
}
#endif

#endif
