#define WORKGROUP_SIZE 128

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#include "Constant.glsl"

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

layout(binding = 11, rgba32f) uniform image2D uBase_ExtraR;
layout(binding = 12, rg32f) readonly uniform image2D uExtraGB;
layout(std430, binding = 13) buffer uuBatchTrainRecords { NRCTrainRecord records[]; }
uBatchTrainRecords[NRC_TRAIN_BATCH_COUNT];

void main() {
	uint screen_x16_y16 = -1u, train_l14_r14_b4 = -1u;
	uvec4 inputs[8];

	if (gl_GlobalInvocationID.x < uEvalCount) {
		NRCEvalRecord eval_record = uEvalRecords[gl_GlobalInvocationID.x];
		NRCInputEncode(UnpackNRCInput(eval_record.packed_input), inputs);
		screen_x16_y16 = eval_record.screen_x16_y16;
		train_l14_r14_b4 = eval_record.train_l14_r14_b4;
	}

	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[2][COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y];
	NNLoadInput(inputs, act_coopmats[0]);
	NNForward64_ReLU(0, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(1, act_coopmats[1], act_coopmats[0]);
	NNForward64_ReLU(2, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(3, act_coopmats[1], act_coopmats[0]);
	NNForward64_ReLU(4, act_coopmats[0], act_coopmats[1]);
	NNForward3(5, act_coopmats[1], act_coopmats[0][0]);
	vec3 predict = max(NNOutput3(act_coopmats[0][0]), vec3(0));

	// Write to Screen
	if (screen_x16_y16 != -1u) {
		ivec2 coord = ivec2(screen_x16_y16 & 0xFFFF, screen_x16_y16 >> 16);
		vec4 base_extra_r = imageLoad(uBase_ExtraR, coord);
		vec2 extra_gb = imageLoad(uExtraGB, coord).rg;
		vec3 color = base_extra_r.rgb + vec3(base_extra_r.a, extra_gb) * predict;
		imageStore(uBase_ExtraR, coord, vec4(color, 0));
	}
	// Write to Train Records
	if (train_l14_r14_b4 != -1u) {
		uint b = train_l14_r14_b4 >> 28u, l = train_l14_r14_b4 & 0x3FFFu, r = (train_l14_r14_b4 >> 14u) & 0x3FFFu;
		[[unroll]] for (uint i = l; i <= r; ++i) {
			NRCTrainRecord train_record = uBatchTrainRecords[b].records[i];
			vec3 base = vec3(train_record.base_r, train_record.base_g, train_record.base_b);
			vec3 extra = vec3(train_record.extra_r, train_record.extra_g, train_record.extra_b);
			vec3 color = base + extra * predict;
			uBatchTrainRecords[b].records[i].base_r = color.r;
			uBatchTrainRecords[b].records[i].base_g = color.g;
			uBatchTrainRecords[b].records[i].base_b = color.b;
		}
	}
}
