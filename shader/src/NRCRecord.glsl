#ifndef NRC_RECORD_GLSL
#define NRC_RECORD_GLSL

#include "Constant.glsl"

struct NRCInput {
	float pos[3];
	float scat[2], norm[2];
	float roughness;
	float diff[3], spec[3];
};
struct NRCOutput {
	float rgb[3];
};

#define NRC_EVAL_INVALID_DST (-1u)
#define NRC_EVAL_DST_SCREEN 0u
#define NRC_EVAL_DST_TRAIN 1u
uint EncodeNRCEvalDstScreen(in const uvec2 xy15) { return (xy15.x | (xy15.y << 15u)) << 1u; }
uint EncodeNRCEvalDstTrain(in const uint b2, in const uint l14, in const uint r14) {
	return (b2 | (l14 << 2u) | (r14 << 16u)) << 1u | 1u;
}
uint GetNRCEvalDstType(in const uint e) { return e & 1u; }
uvec2 DecodeNRCEvalDstScreen(uint e) {
	e >>= 1u;
	return uvec2(e & 0x7FFFu, e >> 15u);
}
void DecodeNRCEvalDstTrain(uint e, out uint b2, out uint l14, out uint r14) {
	e >>= 1u;
	b2 = e & 3u;
	l14 = (e >> 2u) & 0x3FFFu;
	r14 = e >> 16u;
}

vec2 NRCSphEncode(in const vec3 d) {
	return vec2(d.xy == vec2(0) ? 0.5 : 0.5 + atan(d.y, d.x) / (2.0 * M_PI), acos(clamp(d.z, -1, 1)) / M_PI);
}

#endif
