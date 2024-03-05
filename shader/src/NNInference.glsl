#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#define WORKGROUP_SIZE 128

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#include "NRCRecord.glsl"

#define SCENE_TEXTURE_BINDING 7
#define SCENE_TEXTURE_NUM_CONST_ID 0
#define SCENE_BUFFERS_FIRST_BINDING 0
#include "Scene.glsl"

layout(std430, binding = 8) readonly buffer uuEvalRecords { NRCEvalRecord uEvalRecords[]; };
layout(binding = 9) uniform uuEvalCount { uint uEvalCount; };

#define WEIGHTS_BINDING 10
#include "NN_nv.glsl"

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

void main() {
	uint pixel_x_y = -1u;
	uvec4 nn_inputs[8];

	if (gl_GlobalInvocationID.x < uEvalCount) {
		NRCEvalRecord eval_record = uEvalRecords[gl_GlobalInvocationID.x];
		NRCInputEncode(UnpackNRCInput(eval_record.packed_input), nn_inputs);
		pixel_x_y = eval_record.pixel_x_y;
	}

	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[2][COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y];
	NNLoadInput(nn_inputs, act_coopmats[0]);
	NNForward64_ReLU(0, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(1, act_coopmats[1], act_coopmats[0]);
	NNForward64_ReLU(2, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(3, act_coopmats[1], act_coopmats[0]);
	NNForward64_ReLU(4, act_coopmats[0], act_coopmats[1]);
	NNForward16(5, act_coopmats[1], act_coopmats[0][0]);
	uvec2 predict = NNOutputUV2(act_coopmats[0][0]);

	if (pixel_x_y != -1u) {
		ivec2 coord = ivec2(pixel_x_y & 0xFFFF, pixel_x_y >> 16);
		vec3 radiance = vec3(unpackUnorm2x16(predict.x), unpackUnorm2x16(predict.y).x);
		radiance = max(radiance, vec3(0));
		vec3 color = imageLoad(uColor, coord).rgb;
		color *= radiance;
		// imageStore(uColor, coord, vec4(color, 0));
	}
}
