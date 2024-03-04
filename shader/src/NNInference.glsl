#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#define WORKGROUP_SIZE 128
#define SUBGROUP_COUNT (WORKGROUP_SIZE / SUBGROUP_SIZE)

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#include "NRCRecord.glsl"

#define SCENE_TEXTURE_BINDING 7
#define SCENE_TEXTURE_NUM_CONST_ID 0
#define SCENE_BUFFERS_FIRST_BINDING 0
#include "Scene.glsl"

layout(std430, binding = 8) readonly buffer uuEvalRecords { NRCEvalRecord uEvalRecords[]; };
layout(binding = 9) uniform uuEvalCount { uint uEvalCount; };
layout(std430, binding = 10) readonly buffer uuWeights { uvec4 uWeights[]; };

layout(binding = 11, rgba32f) uniform image2D uColor;

UnpackedNRCInput UnpackNRCInput(in const PackedNRCInput packed_input) {
	uint primitive_id = packed_input.primitive_id;
	uint instance_id = packed_input.flip_bit_instance_id & 0x7FFFFFFFu;
	bool flip = bool(packed_input.flip_bit_instance_id >> 31u);
	vec2 texcoord_0 = GetSceneTexcoord(primitive_id, 0);
	vec2 texcoord_1 = GetSceneTexcoord(primitive_id, 1);
	vec2 texcoord_2 = GetSceneTexcoord(primitive_id, 2);
	vec3 vertex_0 = GetSceneVertex(instance_id, primitive_id, 0);
	vec3 vertex_1 = GetSceneVertex(instance_id, primitive_id, 1);
	vec3 vertex_2 = GetSceneVertex(instance_id, primitive_id, 2);
	vec3 normal = normalize(cross(vertex_1 - vertex_0, vertex_2 - vertex_0));
	normal = flip ? -normal : normal;

	vec3 barycentric = vec3(0, unpackUnorm2x16(packed_input.barycentric_2x16U));
	barycentric.x = 1.0 - barycentric.y - barycentric.z;

	Material mat = GetSceneMaterial(primitive_id);

	UnpackedNRCInput unpacked_input;
	unpacked_input.position = mat3(vertex_0, vertex_1, vertex_2) * barycentric;
	unpacked_input.scattered_dir = unpackUnorm2x16(packed_input.scattered_dir_2x16U);
	unpacked_input.normal = NRCSphEncode(normal);
	unpacked_input.roughness = mat.roughness;
	vec2 texcoord = mat3x2(texcoord_0, texcoord_1, texcoord_2) * barycentric;
	unpacked_input.diffuse = GetSceneDiffuse(mat, texcoord);
	unpacked_input.specular = GetSceneSpecular(mat, texcoord);
	return unpacked_input;
}

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
// Thread Weight Counts
#define THREAD_WEIGHT_64_COUNT (WEIGHT_64_COUNT / WORKGROUP_SIZE)
#define THREAD_WEIGHT_64_UV4_COUNT (WEIGHT_64_UV4_COUNT / WORKGROUP_SIZE)
// Subgroup Activates Counts
#define SUBGROUP_ACT_COOPMAT_Y (ACT_COOPMAT_Y / SUBGROUP_COUNT)
// Matrix Strides & Major
#define MAT64_COOPMAT_STRIDE (64 / FP16_PER_UV4)
#define MAT64_COOPMAT_ELEMENT(X, Y) ((X) * (16 / FP16_PER_UV4) + (Y)*16 * MAT64_COOPMAT_STRIDE)
#define WEIGHT_COOPMAT_MAJOR gl_CooperativeMatrixLayoutRowMajor
#define ACT_COOPMAT_MAJOR gl_CooperativeMatrixLayoutColumnMajor

shared uvec4 sSharedBuffer[WEIGHT_64_UV4_COUNT > ACT_UV4_COUNT ? WEIGHT_64_UV4_COUNT : ACT_UV4_COUNT];
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> subgroup_act_coopmats[SUBGROUP_ACT_COOPMAT_Y][COOPMAT_X];

uint LoadFirstActs() {
	NRCEvalRecord eval_record;
	if (gl_GlobalInvocationID.x < uEvalCount) {
		eval_record = uEvalRecords[gl_GlobalInvocationID.x];
		uint base = gl_LocalInvocationID.x * UV4_X;
		NRCInputEncode(UnpackNRCInput(eval_record.packed_input),
		               sSharedBuffer[base],     //
		               sSharedBuffer[base | 1], //
		               sSharedBuffer[base | 2], //
		               sSharedBuffer[base | 3], //
		               sSharedBuffer[base | 4], //
		               sSharedBuffer[base | 5], //
		               sSharedBuffer[base | 6], //
		               sSharedBuffer[base | 7]);
	} else
		eval_record.pixel_x_y = -1u; // -1 means invalid
	barrier();

	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint w_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x)
			coopMatLoad(subgroup_act_coopmats[y][x], sSharedBuffer, MAT64_COOPMAT_ELEMENT(x, w_y), MAT64_COOPMAT_STRIDE,
			            ACT_COOPMAT_MAJOR);
	}
	barrier();
	return eval_record.pixel_x_y;
}
void LoadW64Weights(in const uint layer) {
	const uint kLayerUV4Base = layer * WEIGHT_64_UV4_COUNT;
	const uint kLocalThreadUV4Base = gl_LocalInvocationID.x * THREAD_WEIGHT_64_UV4_COUNT;
	const uint kThreadUV4Base = kLayerUV4Base + kLocalThreadUV4Base;
	[[unroll]] for (uint i = 0; i < THREAD_WEIGHT_64_UV4_COUNT; ++i)
		sSharedBuffer[kLocalThreadUV4Base + i] = uWeights[kThreadUV4Base + i];
	barrier();
}
void LoadW16Weights(in const uint layer) {
#if WEIGHT_16_UV4_COUNT != WORKGROUP_SIZE
#error
#endif
	const uint kLayerUV4Base = layer * WEIGHT_64_UV4_COUNT;
	sSharedBuffer[gl_LocalInvocationID.x] = uWeights[kLayerUV4Base + gl_LocalInvocationID.x];
	barrier();
}

void Forward64() {
	coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> subgroup_dst_coopmats[SUBGROUP_ACT_COOPMAT_Y]
	                                                                                           [COOPMAT_X];
	// Zero Initialize
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x)
			subgroup_dst_coopmats[y][x] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
	}
	// MMA
	[[unroll]] for (uint w_y = 0; w_y < COOPMAT_X; ++w_y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> weight_coopmat;
			coopMatLoad(weight_coopmat, sSharedBuffer, MAT64_COOPMAT_ELEMENT(x, w_y), MAT64_COOPMAT_STRIDE,
			            WEIGHT_COOPMAT_MAJOR);

			[[unroll]] for (uint a_y = 0; a_y < SUBGROUP_ACT_COOPMAT_Y; ++a_y) {
				subgroup_dst_coopmats[a_y][w_y] =
				    coopMatMulAdd(weight_coopmat, subgroup_act_coopmats[a_y][x], subgroup_dst_coopmats[a_y][w_y]);
			}
		}
	}
	barrier();
	// ReLU & Store
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
			// assert(subgroup_dst_coopmats[y][x].length() == 16 * 16 / SUBGROUP_SIZE)
			// TODO: This is not guarenteed
			[[unroll]] for (uint k = 0; k < (256 / SUBGROUP_SIZE); ++k)
				subgroup_act_coopmats[y][x][k] = max(subgroup_dst_coopmats[y][x][k], float16_t(0));
		}
	}
}

void Forward16Store() {
	coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> subgroup_dst_coopmats[SUBGROUP_ACT_COOPMAT_Y];
	// Zero Initialize
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y)
		subgroup_dst_coopmats[y] = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0);
	// MMA
	[[unroll]] for (uint x = 0; x < COOPMAT_X; ++x) {
		coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> weight_coopmat;
		coopMatLoad(weight_coopmat, sSharedBuffer, MAT64_COOPMAT_ELEMENT(x, 0), MAT64_COOPMAT_STRIDE,
		            WEIGHT_COOPMAT_MAJOR);

		[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
			subgroup_dst_coopmats[y] =
			    coopMatMulAdd(weight_coopmat, subgroup_act_coopmats[y][x], subgroup_dst_coopmats[y]);
		}
	}
	barrier();
	// Store
	[[unroll]] for (uint y = 0; y < SUBGROUP_ACT_COOPMAT_Y; ++y) {
		uint w_y = gl_SubgroupID * SUBGROUP_ACT_COOPMAT_Y + y;
		// coopMatStore(subgroup_dst_coopmats[y], sSharedBuffer, w_y * (16 * 16) / FP16_PER_UV4, 16 / FP16_PER_UV4,
		//              ACT_COOPMAT_MAJOR);
		coopMatStore(subgroup_dst_coopmats[y], sSharedBuffer, MAT64_COOPMAT_ELEMENT(0, w_y), MAT64_COOPMAT_STRIDE,
		             ACT_COOPMAT_MAJOR);
	}
	barrier();
}

void OutputPixel(in const uint pixel_x_y) {
	if (pixel_x_y == -1u)
		return;
	uvec2 rgba_fp16 = sSharedBuffer[gl_LocalInvocationID.x * UV4_X].rg;
	ivec2 coord = ivec2(pixel_x_y & 0xFFFF, pixel_x_y >> 16);
	vec3 radiance = vec3(unpackUnorm2x16(rgba_fp16.x), unpackUnorm2x16(rgba_fp16.y).x);
	radiance = max(radiance, vec3(0));
	vec3 color = imageLoad(uColor, coord).rgb;
	color *= radiance;
	// imageStore(uColor, coord, vec4(color, 0));
}

void main() {
	uint pixel_x_y = LoadFirstActs();
	LoadW64Weights(0);
	Forward64();
	LoadW64Weights(1);
	Forward64();
	LoadW64Weights(2);
	Forward64();
	LoadW64Weights(3);
	Forward64();
	LoadW64Weights(4);
	Forward64();
	LoadW16Weights(5);
	Forward16Store();
	OutputPixel(pixel_x_y);
}
