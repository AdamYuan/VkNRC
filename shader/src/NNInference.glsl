#define WORKGROUP_SIZE 128

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#define SCENE_TEXTURE_BINDING 7
#define SCENE_TEXTURE_NUM_CONST_ID 0
#define SCENE_BUFFERS_FIRST_BINDING 0
#include "Scene.glsl"

#define NRC_SCENE_UNPACK
#include "NRCRecord.glsl"

layout(std430, binding = 8) readonly buffer uuEvalRecords { NRCEvalRecord uEvalRecords[]; };
layout(binding = 9) uniform uuEvalCount { uint uEvalCount; };

#define WEIGHTS_BINDING 10
#include "NN_nv.glsl"

layout(binding = 11, rgba32f) uniform image2D uColor;

void main() {
	uint pixel_x_y = -1u;
	uvec4 inputs[8];

	if (gl_GlobalInvocationID.x < uEvalCount) {
		NRCEvalRecord eval_record = uEvalRecords[gl_GlobalInvocationID.x];
		NRCInputEncode(UnpackNRCInput(eval_record.packed_input), inputs);
		pixel_x_y = eval_record.pixel_x_y;

		// For testing
		inputs = uvec4[8](uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0),
		                  uvec4(0, 0, 0, packHalf2x16(vec2(1.0))));
	}

	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[2][COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y];
	NNLoadInput(inputs, act_coopmats[0]);
	NNForward64_ReLU(0, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(1, act_coopmats[1], act_coopmats[0]);
	NNForward64_ReLU(2, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(3, act_coopmats[1], act_coopmats[0]);
	NNForward64_ReLU(4, act_coopmats[0], act_coopmats[1]);
	NNForward3(5, act_coopmats[1], act_coopmats[0][0]);
	vec3 predict = NNOutput3(act_coopmats[0][0]);

	if (pixel_x_y != -1u) {
		ivec2 coord = ivec2(pixel_x_y & 0xFFFF, pixel_x_y >> 16);
		vec3 color = imageLoad(uColor, coord).rgb * max(predict, vec3(0));
		imageStore(uColor, coord, vec4(color, 0));
	}
}
