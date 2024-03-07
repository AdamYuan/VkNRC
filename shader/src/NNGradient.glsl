#define WORKGROUP_SIZE 128
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#define SCENE_TEXTURE_BINDING 7
#define SCENE_TEXTURE_NUM_CONST_ID 0
#define SCENE_BUFFERS_FIRST_BINDING 0
#include "Scene.glsl"

#define NRC_SCENE_UNPACK
#include "NRCRecord.glsl"

layout(std430, binding = 8) readonly buffer uuBatchTrainRecords { NRCTrainRecord uBatchTrainRecords[]; };
layout(binding = 9) uniform uuBatchTrainCounts { uint uBatchTrainCount; };

#define NN_BACKPROPAGATION
#define WEIGHTS_BINDING 10
#define DWEIGHTS_BINDING 11
#include "NN_nv.glsl"

void main() {
	uvec4 inputs[8] = uvec4[8](uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0));
	vec3 target = vec3(0);
	if (gl_GlobalInvocationID.x < uBatchTrainCount) {
		NRCTrainRecord train_record = uBatchTrainRecords[gl_GlobalInvocationID.x];
		NRCInputEncode(UnpackNRCInput(train_record.packed_input), inputs);
		target = vec3(train_record.base_r, train_record.base_g, train_record.base_b);

		// For testing
		/* inputs = uvec4[8](uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0),
		                  uvec4(0, 0, 0, packHalf2x16(vec2(1.0))));
		target = vec3(1); */
	}

	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[6][COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
	    out_coopmats[SUBGROUP_ACT_COOPMAT_Y];
	NNLoadInput(inputs, act_coopmats[0]);
	NNForward64_ReLU(0, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(1, act_coopmats[1], act_coopmats[2]);
	NNForward64_ReLU(2, act_coopmats[2], act_coopmats[3]);
	NNForward64_ReLU(3, act_coopmats[3], act_coopmats[4]);
	NNForward64_ReLU(4, act_coopmats[4], act_coopmats[5]);
	NNForward3(5, act_coopmats[5], out_coopmats);
	vec3 predict = NNOutput3(out_coopmats);
	NNLoadDA3_RelativeL2LuminanceLoss(predict, target, out_coopmats);
	NNUpdateDW3(5, out_coopmats, act_coopmats[5]);
	NNBackwardDA3_ReLU(5, out_coopmats, act_coopmats[5]);
	NNUpdateDW64(4, act_coopmats[5], act_coopmats[4]);
	NNBackwardDA64_ReLU(4, act_coopmats[5], act_coopmats[4]);
	NNUpdateDW64(3, act_coopmats[4], act_coopmats[3]);
	NNBackwardDA64_ReLU(3, act_coopmats[4], act_coopmats[3]);
	NNUpdateDW64(2, act_coopmats[3], act_coopmats[2]);
	NNBackwardDA64_ReLU(2, act_coopmats[3], act_coopmats[2]);
	NNUpdateDW64(1, act_coopmats[2], act_coopmats[1]);
	NNBackwardDA64_ReLU(1, act_coopmats[2], act_coopmats[1]);
	NNUpdateDW64(0, act_coopmats[1], act_coopmats[0]);
}
