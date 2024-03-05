#define WORKGROUP_SIZE 128
layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#define SCENE_TEXTURE_BINDING 7
#define SCENE_TEXTURE_NUM_CONST_ID 0
#define SCENE_BUFFERS_FIRST_BINDING 0
#include "Scene.glsl"

#define NRC_SCENE_UNPACK
#include "NRCRecord.glsl"

layout(constant_id = 1) const uint kBatchIndex = 0;
layout(std430, binding = 8) readonly buffer uuBatchTrainRecords { NRCTrainRecord uBatchTrainRecords[]; };
layout(binding = 9) uniform uuBatchTrainCounts { uint uBatchTrainCounts[kBatchIndex + 1]; };

#define NN_BACKPROPAGATION
#define WEIGHTS_BINDING 10
#define DWEIGHTS_BINDING 11
#include "NN_nv.glsl"

void main() {
	uvec4 inputs[8] = uvec4[8](uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0), uvec4(0));
	uvec2 target = uvec2(0);
	if (gl_GlobalInvocationID.x < uBatchTrainCounts[kBatchIndex]) {
		NRCTrainRecord train_record = uBatchTrainRecords[kBatchIndex * NRC_TRAIN_BATCH_SIZE + gl_GlobalInvocationID.x];
		NRCInputEncode(UnpackNRCInput(train_record.packed_input), inputs);
		target = uvec2(train_record.radiance_RG, train_record.radiance_B);
	}

	fcoopmatNV<16, gl_ScopeSubgroup, 16, 16> act_coopmats[6][COOPMAT_X][SUBGROUP_ACT_COOPMAT_Y],
	    out_coopmats[SUBGROUP_ACT_COOPMAT_Y];
	NNLoadInput(inputs, act_coopmats[0]);
	NNForward64_ReLU(0, act_coopmats[0], act_coopmats[1]);
	NNForward64_ReLU(1, act_coopmats[1], act_coopmats[2]);
	NNForward64_ReLU(2, act_coopmats[2], act_coopmats[3]);
	NNForward64_ReLU(3, act_coopmats[3], act_coopmats[4]);
	NNForward64_ReLU(4, act_coopmats[4], act_coopmats[5]);
	NNForward16(5, act_coopmats[5], out_coopmats);
	uvec2 predict = NNOutputUV2(out_coopmats);
	NNLoadDA16_L2Loss(predict, target, out_coopmats);
	NNUpdateDW16(5, out_coopmats, act_coopmats[5]);
	NNBackwardDA16_ReLU(5, out_coopmats, act_coopmats[5]);
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
