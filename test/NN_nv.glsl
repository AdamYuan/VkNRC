#ifndef NN_NV_GLSL
#define NN_NV_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
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
#define WEIGHT_3_COUNT (FP_X * 3)
// UVec4 Counts
#define UV4_X (FP_X / FP16_PER_UV4)                          // 8
#define ACT_UV4_COUNT (ACT_COUNT / FP16_PER_UV4)             // 1024
#define WEIGHT_64_UV4_COUNT (WEIGHT_64_COUNT / FP16_PER_UV4) // 512
#define WEIGHT_3_UV4_COUNT (WEIGHT_3_COUNT / FP16_PER_UV4)   // 24
#if WEIGHT_3_UV4_COUNT > WORKGROUP_SIZE
#error
#endif
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

void _nn_load_weight_3(in const uint layer) {
	const uint kLayerUV4Base = layer * WEIGHT_64_UV4_COUNT;
	SHARED_BUFFER[gl_LocalInvocationID.x] =
	    gl_LocalInvocationID.x < WEIGHT_3_UV4_COUNT ? uWeights[kLayerUV4Base + gl_LocalInvocationID.x] : uvec4(0u);
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

void NNForward3(in const uint layer,
                in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
                inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_coopmats[SUBGROUP_ACT_COOPMAT_Y]) {
	_nn_load_weight_3(layer);
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

vec3 NNOutput3(in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[SUBGROUP_ACT_COOPMAT_Y]) {
	// Store
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		coopMatStoreNV(act_coopmats[y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(0, workgroup_y), MAT64_COOPMAT_STRIDE,
		               ACT_COOPMAT_MAJOR);
	}
	uvec2 uv2 = SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X].rg;
	barrier();
	return vec3(unpackHalf2x16(uv2.x), unpackHalf2x16(uv2.y).x);
}

#ifdef NN_BACKPROPAGATION
// http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf
void NNLoadDA3_L2Loss(in const vec3 predict,
                      in const vec3 target,
                      in const float loss_scale,
                      inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> da_coopmats_t[SUBGROUP_ACT_COOPMAT_Y]) {
	vec3 d_l2 = 2.0 * (predict - target) * loss_scale;
	SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X] = uvec4(packHalf2x16(d_l2.xy), packHalf2x16(vec2(d_l2.z, 0)), 0u, 0u);
	[[unroll]] for (uint i = 1; i < 16 / FP16_PER_UV4; ++i)
		SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X + i] = uvec4(0u);
	barrier();
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		coopMatLoadNV(da_coopmats_t[y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(0, workgroup_y), MAT64_COOPMAT_STRIDE,
		              COOPMAT_MAJOR_T(ACT_COOPMAT_MAJOR));
	}
	barrier();
}
void NNLoadDA3_RelativeL2LuminanceLoss(
    in const vec3 predict,
    in const vec3 target,
    in const float loss_scale,
    inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> da_coopmats_t[SUBGROUP_ACT_COOPMAT_Y]) {
	float predict_luminance = dot(vec3(0.299, 0.587, 0.114), max(predict, vec3(0)));
	vec3 d_loss = 2.0 * loss_scale * (predict - target) / (predict_luminance * predict_luminance + 0.01);
	SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X] =
	    uvec4(packHalf2x16(d_loss.xy), packHalf2x16(vec2(d_loss.z, 0)), 0u, 0u);
	[[unroll]] for (uint i = 1; i < 16 / FP16_PER_UV4; ++i)
		SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X + i] = uvec4(0u);
	barrier();
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		coopMatLoadNV(da_coopmats_t[y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(0, workgroup_y), MAT64_COOPMAT_STRIDE,
		              COOPMAT_MAJOR_T(ACT_COOPMAT_MAJOR));
	}
	barrier();
}

void _nn_act_64_relu_mask_t(
    inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatStoreNV(coopmats[x][y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, workgroup_y), MAT64_COOPMAT_STRIDE,
			               ACT_COOPMAT_MAJOR);
		}
	}
	barrier();
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint workgroup_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatLoadNV(coopmats[x][y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, workgroup_y), MAT64_COOPMAT_STRIDE,
			              COOPMAT_MAJOR_T(ACT_COOPMAT_MAJOR));
			// NAN for zero
			for (uint k = 0; k < coopmats[x][y].length(); ++k)
				coopmats[x][y][k] = coopmats[x][y][k] > 0.0 ? float16_t(0.0) : uint16BitsToHalf(uint16_t(0x7FFF));
		}
	}
	barrier();
}

void NNBackwardDA3_ReLU(
    in const uint layer,
    in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_da_coopmats_t[SUBGROUP_ACT_COOPMAT_Y],
    inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {

	// For Inv ReLU
	_nn_act_64_relu_mask_t(dst_da_coopmats_t);

	_nn_load_weight_3(layer);
	// da_coopmats^T (128, 16) x weights (16, 64) = dA^T (128, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, 0), MAT64_COOPMAT_STRIDE,
		              WEIGHT_COOPMAT_MAJOR);
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			// MMA
			dst_da_coopmats_t[x][y] = coopMatMulAddNV(src_da_coopmats_t[y], weight_coopmat, dst_da_coopmats_t[x][y]);
			for (uint k = 0; k < dst_da_coopmats_t[x][y].length(); ++k)
				dst_da_coopmats_t[x][y][k] =
				    isnan(dst_da_coopmats_t[x][y][k]) ? float16_t(0) : dst_da_coopmats_t[x][y][k];
		}
	}
	barrier();
}

void NNBackwardDA64_ReLU(
    in const uint layer,
    in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
    inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {

	// For Inv ReLU
	_nn_act_64_relu_mask_t(dst_da_coopmats_t);

	// da_coopmats^T (128, 64) x weights (64, 64) = dA^T (128, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
	// MMA
	_nn_load_weight_64(layer);
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, w_y), MAT64_COOPMAT_STRIDE,
			              WEIGHT_COOPMAT_MAJOR);
			[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
				dst_da_coopmats_t[x][a_y] =
				    coopMatMulAddNV(src_da_coopmats_t[w_y][a_y], weight_coopmat, dst_da_coopmats_t[x][a_y]);
			}
		}
	}
	barrier();
	// Inv ReLU
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y)
			for (uint k = 0; k < dst_da_coopmats_t[x][y].length(); ++k)
				dst_da_coopmats_t[x][y][k] =
				    isnan(dst_da_coopmats_t[x][y][k]) ? float16_t(0) : dst_da_coopmats_t[x][y][k];
	}
}

void NNUpdateDW3(in const uint layer,
                 in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> da_coopmats_t[SUBGROUP_ACT_COOPMAT_Y],
                 in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	// (act_coopmats (64, 128) x da_coopmats^T (128, 16))^T = dW (16, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dw_coopmats[COOPMAT_X];
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		dw_coopmats[w_y] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
		[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
			dw_coopmats[w_y] = coopMatMulAddNV(act_coopmats[w_y][a_y], da_coopmats_t[a_y], dw_coopmats[w_y]);
		}
	}
	barrier();
	[[unroll]] for (uint y = 0; y < COOPMAT_X; ++y) {
		coopMatStoreNV(dw_coopmats[y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(y, gl_SubgroupID), MAT64_COOPMAT_STRIDE,
		               COOPMAT_MAJOR_T(WEIGHT_COOPMAT_MAJOR));
	}
	barrier();
	// Reduce dW across subgroups
	vec2 d_w[FP16_PER_UV4 / 2];
	[[unroll]] for (uint i = 0; i < FP16_PER_UV4 / 2; ++i)
		d_w[i] = vec2(0);
	[[unroll]] for (uint s = 0; s < SUBGROUP_COUNT; ++s) {
		uvec4 d_w_uv4 = SHARED_BUFFER[MAT64_COOPMAT_ELEMENT(0, s) + gl_LocalInvocationID.x];
		[[unroll]] for (uint i = 0; i < FP16_PER_UV4 / 2; ++i)
			d_w[i] += unpackHalf2x16(d_w_uv4[i]);
	}
	barrier();
	// Store
	if (gl_LocalInvocationID.x < WEIGHT_3_UV4_COUNT) {
		const uint kWeightFP16Base = layer * WEIGHT_64_COUNT + gl_LocalInvocationID.x * FP16_PER_UV4;
		[[unroll]] for (uint i = 0; i < FP16_PER_UV4 / 2; ++i) {
			atomicAdd(uDWeights[kWeightFP16Base + (i << 1u)], d_w[i].x, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer,
			          gl_SemanticsRelaxed);
			atomicAdd(uDWeights[kWeightFP16Base + (i << 1u | 1u)], d_w[i].y, gl_ScopeQueueFamily,
			          gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
		}
	}
}

void NNUpdateDW64(in const uint layer,
                  in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> da_coopmats_t[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
                  in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y]) {
	// (act_coopmats (64, 128) x da_coopmats^T (128, 64))^T = dW (64, 64)
	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dw_coopmats[COOPMAT_X][COOPMAT_X];
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			dw_coopmats[w_y][x] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
			[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
				dw_coopmats[w_y][x] =
				    coopMatMulAddNV(act_coopmats[w_y][a_y], da_coopmats_t[x][a_y], dw_coopmats[w_y][x]);
			}
		}
	}
	barrier();
	[[unroll]] for (uint y = 0; y < COOPMAT_X; ++y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopMatStoreNV(dw_coopmats[y][x], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(y, x + COOPMAT_X * gl_SubgroupID),
			               MAT64_COOPMAT_STRIDE, COOPMAT_MAJOR_T(WEIGHT_COOPMAT_MAJOR));
		}
	}
	barrier();
	// Reduce dW across subgroups
	vec2 d_w[THREAD_WEIGHT_64_UV4_COUNT][FP16_PER_UV4 / 2];
	[[unroll]] for (uint u = 0; u < THREAD_WEIGHT_64_UV4_COUNT; ++u) {
		[[unroll]] for (uint i = 0; i < (FP16_PER_UV4 / 2); ++i)
			d_w[u][i] = vec2(0);
	}
	const uint kSharedUV4Base = gl_LocalInvocationID.x * THREAD_WEIGHT_64_UV4_COUNT;
	[[unroll]] for (uint s = 0; s < SUBGROUP_COUNT; ++s) {
		const uint kSubgroupUV4Base = MAT64_COOPMAT_ELEMENT(0, COOPMAT_X * s) + kSharedUV4Base;
		[[unroll]] for (uint u = 0; u < THREAD_WEIGHT_64_UV4_COUNT; ++u) {
			uvec4 d_w_uv4 = SHARED_BUFFER[kSubgroupUV4Base + u];
			[[unroll]] for (uint i = 0; i < (FP16_PER_UV4 / 2); ++i)
				d_w[u][i] += unpackHalf2x16(d_w_uv4[i]);
		}
	}
	barrier();
	const uint kWeightUV4Base = layer * WEIGHT_64_UV4_COUNT + kSharedUV4Base;
	[[unroll]] for (uint u = 0; u < THREAD_WEIGHT_64_UV4_COUNT; ++u) {
		const uint kWeightFP16Base = (kWeightUV4Base + u) * FP16_PER_UV4;
		[[unroll]] for (uint i = 0; i < (FP16_PER_UV4 / 2); ++i) {
			atomicAdd(uDWeights[kWeightFP16Base + (i << 1u)], d_w[u][i].x, gl_ScopeQueueFamily,
			          gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
			atomicAdd(uDWeights[kWeightFP16Base + (i << 1u | 1u)], d_w[u][i].y, gl_ScopeQueueFamily,
			          gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
		}
	}
}
#endif

#endif
