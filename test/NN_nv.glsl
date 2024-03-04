#ifndef NN_NV_GLSL
#define NN_NV_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_NV_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

#if WORKGROUP_SIZE != 128
#error WORKGROUP_SIZE must be 128
#endif

#ifndef WEIGHTS_SET
#define WEIGHTS_SET 0
#endif

layout(std430, set = WEIGHTS_SET, binding = WEIGHTS_BINDING) readonly buffer uuWeights { uvec4 uWeights[]; };

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
#define MAT64_COOPMAT_STRIDE (64 / FP16_PER_UV4)
#define MAT64_COOPMAT_ELEMENT(X, Y) ((X) * (16 / FP16_PER_UV4) + (Y)*16 * MAT64_COOPMAT_STRIDE)
#define WEIGHT_COOPMAT_MAJOR false // gl_CooperativeMatrixLayoutRowMajor
#define ACT_COOPMAT_MAJOR true     // gl_CooperativeMatrixLayoutColumnMajor

#define SHARED_BUFFER_SIZE (WEIGHT_64_UV4_COUNT > ACT_UV4_COUNT ? WEIGHT_64_UV4_COUNT : ACT_UV4_COUNT)

shared uvec4 SHARED_BUFFER[SHARED_BUFFER_SIZE];

void NNLoadInput(in const uvec4 inputs[UV4_X],
                 inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[SUBGROUP_ACT_COOPMAT_Y][COOPMAT_X]) {
	[[unroll]] for (uint x = 0; x < UV4_X; ++x)
		SHARED_BUFFER[(gl_LocalInvocationID.x * UV4_X) | x] = inputs[x];
	barrier();
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint w_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x)
			coopMatLoadNV(act_coopmats[y][x], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, w_y), MAT64_COOPMAT_STRIDE,
			              ACT_COOPMAT_MAJOR);
	}
	barrier();
}

void NNLoadWeight64(in const uint layer) {
	const uint kLayerUV4Base = layer * WEIGHT_64_UV4_COUNT;
	const uint kLocalThreadUV4Base = gl_LocalInvocationID.x * THREAD_WEIGHT_64_UV4_COUNT;
	const uint kThreadUV4Base = kLayerUV4Base + kLocalThreadUV4Base;
	[[unroll]] for (uint i = 0; i < THREAD_WEIGHT_64_UV4_COUNT; ++i)
		SHARED_BUFFER[kLocalThreadUV4Base + i] = uWeights[kThreadUV4Base + i];
	barrier();
}

void NNLoadWeight16(in const uint layer) {
	// assert(WEIGHT_16_UV4_COUNT == 128 == WORKGROUP_SIZE);
	const uint kLayerUV4Base = layer * WEIGHT_64_UV4_COUNT;
	SHARED_BUFFER[gl_LocalInvocationID.x] = uWeights[kLayerUV4Base + gl_LocalInvocationID.x];
	barrier();
}

void NNForward64ReLU(in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_coopmats[SUBGROUP_ACT_COOPMAT_Y][COOPMAT_X],
                     inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_coopmats[SUBGROUP_ACT_COOPMAT_Y][COOPMAT_X]) {
	// Zero Initialize
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x)
			dst_coopmats[y][x] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
	}
	// MMA
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
			coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, w_y), MAT64_COOPMAT_STRIDE,
			              WEIGHT_COOPMAT_MAJOR);
			[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
				dst_coopmats[a_y][w_y] = coopMatMulAddNV(weight_coopmat, src_coopmats[a_y][x], dst_coopmats[a_y][w_y]);
			}
		}
	}
	barrier();
	// ReLU
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			for (uint k = 0; k < dst_coopmats[y][x].length(); ++k)
				dst_coopmats[y][x][k] = max(dst_coopmats[y][x][k], float16_t(0));
		}
	}
}

void NNForward16(in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> src_coopmats[SUBGROUP_ACT_COOPMAT_Y][COOPMAT_X],
                 inout fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> dst_coopmats[COOPMAT_X]) {
	// Zero Initialize
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y)
		dst_coopmats[y] = fcoopmatNV<16, gl_ScopeSubgroup, 16, 16>(0);
	// MMA
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> weight_coopmat;
		coopMatLoadNV(weight_coopmat, SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(x, 0), MAT64_COOPMAT_STRIDE,
		              WEIGHT_COOPMAT_MAJOR);
		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			dst_coopmats[y] = coopMatMulAddNV(weight_coopmat, src_coopmats[y][x], dst_coopmats[y]);
		}
	}
	barrier();
}

uvec2 NNOutputUV2(in const fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[COOPMAT_X]) {
	// Store
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint w_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		coopMatStoreNV(act_coopmats[y], SHARED_BUFFER, MAT64_COOPMAT_ELEMENT(0, w_y), MAT64_COOPMAT_STRIDE,
		               ACT_COOPMAT_MAJOR);
	}
	barrier();
	return SHARED_BUFFER[gl_LocalInvocationID.x * UV4_X].rg;
}

#endif
