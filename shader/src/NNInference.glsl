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

layout(binding = 11, rgba32f) uniform image2D uBias_FactorR;
layout(binding = 12, rg32f) readonly uniform image2D uFactorGB;
layout(std430, binding = 13) buffer uuBatchTrainRecords { NRCTrainRecord records[]; }
uBatchTrainRecords[NRC_TRAIN_BATCH_COUNT];

void main() {
	uint dst = NRC_EVAL_INVALID_DST;
	uvec4 inputs[8];

	if (gl_GlobalInvocationID.x < uEvalCount) {
		NRCEvalRecord eval_record = uEvalRecords[gl_GlobalInvocationID.x];
		NRCInputEncode(UnpackNRCInput(eval_record.packed_input), inputs);
		dst = eval_record.dst;
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

	if (dst == NRC_EVAL_INVALID_DST)
		return;

	if (GetNRCEvalDstType(dst) == NRC_EVAL_DST_SCREEN) {
		// Write to Screen
		ivec2 coord = ivec2(DecodeNRCEvalDstScreen(dst));
		vec4 bias_factor_r = imageLoad(uBias_FactorR, coord);
		vec2 factor_gb = imageLoad(uFactorGB, coord).rg;
		vec3 color = bias_factor_r.rgb + vec3(bias_factor_r.a, factor_gb) * predict;
		imageStore(uBias_FactorR, coord, vec4(color, 0));
	} else {
		// Write to Train Records
		uint b, l, r;
		DecodeNRCEvalDstTrain(dst, b, l, r);
		[[unroll]] for (uint i = l; i <= r; ++i) {
			NRCTrainRecord train_record = uBatchTrainRecords[b].records[i];
			vec3 bias = vec3(train_record.bias_r, train_record.bias_g, train_record.bias_b);
			vec3 factor = vec3(train_record.factor_r, train_record.factor_g, train_record.factor_b);
			vec3 color = bias + factor * predict;
			uBatchTrainRecords[b].records[i].bias_r = color.r;
			uBatchTrainRecords[b].records[i].bias_g = color.g;
			uBatchTrainRecords[b].records[i].bias_b = color.b;
		}
	}
}
