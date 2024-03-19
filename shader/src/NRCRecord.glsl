#ifndef NRC_RECORD_GLSL
#define NRC_RECORD_GLSL

#include "Constant.glsl"

struct PackedNRCInput {
	uint primitive_id, flip_bit_instance_id;
	uint barycentric_2x16U;
	uint scattered_dir_2x16U;
};

struct NRCEvalRecord {
#define NRC_EVAL_INVALID_DST (-1u)
#define NRC_EVAL_DST_SCREEN 0u
#define NRC_EVAL_DST_TRAIN 1u
	uint dst;
	PackedNRCInput packed_input;
};
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

struct NRCTrainRecord {
	float bias_r, bias_g, bias_b, factor_r, factor_g, factor_b;
	PackedNRCInput packed_input;
};

struct UnpackedNRCInput {
	vec3 position;
	vec2 scattered_dir, normal;
	float roughness;
	vec3 diffuse, specular;
};

vec2 NRCSphEncode(in const vec3 d) {
	return vec2(d.xy == vec2(0) ? 0.5 : 0.5 + atan(d.y, d.x) / (2.0 * M_PI), acos(clamp(d.z, -1, 1)) / M_PI);
}

vec4 _quartic_cdf(in const vec4 x, in const float inv_radius) {
	vec4 u = x * inv_radius;
	vec4 u2 = u * u;
	vec4 u4 = u2 * u2;
	return clamp((15.0 / 16.0) * u * (1 - (2.0 / 3.0) * u2 + (1.0 / 5.0) * u4) + 0.5, vec4(0), vec4(1));
}

// x in [0, 1]
vec4 NRCOneBlob4Encode(in const float x) {
	// Quartic Function
	vec4 l = vec4(0, 0.25, 0.5, 0.75), r = vec4(0.25, 0.5, 0.75, 1);
	return _quartic_cdf(r - x, 4) - _quartic_cdf(l - x, 4);
}

vec4 _nrc_tri(in const vec4 x) {
	// return sin(M_PI * x);
	return 2.0 * abs(mod(x - 0.5, 2.0) - 1.0) - 1.0;
}
mat3x4 NRCFrequencyEncode(in const float p) {
	mat3x4 f = mat3x4(vec4(1, 2, 4, 8), vec4(16, 32, 64, 128), vec4(256, 512, 1024, 2048)) * p;
	return mat3x4(_nrc_tri(f[0]), _nrc_tri(f[1]), _nrc_tri(f[2]));
}

uvec4 NRCPackHalf8x16(in const vec4 a, in const vec4 b) {
	return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}

void NRCInputEncode(in const UnpackedNRCInput unpacked_input, inout uvec4 o[8]) {
	mat3x4 pos_freq_0 = NRCFrequencyEncode(unpacked_input.position.x);
	mat3x4 pos_freq_1 = NRCFrequencyEncode(unpacked_input.position.y);
	mat3x4 pos_freq_2 = NRCFrequencyEncode(unpacked_input.position.z);
	vec4 scat_ob_0 = NRCOneBlob4Encode(unpacked_input.scattered_dir.x),
	     scat_ob_1 = NRCOneBlob4Encode(unpacked_input.scattered_dir.y);
	vec4 norm_ob_0 = NRCOneBlob4Encode(unpacked_input.normal.x), norm_ob_1 = NRCOneBlob4Encode(unpacked_input.normal.y);
	vec4 r_ob = NRCOneBlob4Encode(1 - exp(-unpacked_input.roughness));
	o[0] = NRCPackHalf8x16(pos_freq_0[0], pos_freq_0[1]);
	o[1] = NRCPackHalf8x16(pos_freq_0[2], pos_freq_1[0]);
	o[2] = NRCPackHalf8x16(pos_freq_1[1], pos_freq_1[2]);
	o[3] = NRCPackHalf8x16(pos_freq_2[0], pos_freq_2[1]);
	o[4] = NRCPackHalf8x16(pos_freq_2[2], scat_ob_0);
	o[5] = NRCPackHalf8x16(scat_ob_1, norm_ob_0);
	o[6] = NRCPackHalf8x16(norm_ob_1, r_ob);
	o[7] = NRCPackHalf8x16(vec4(unpacked_input.diffuse, unpacked_input.specular.r),
	                       vec4(unpacked_input.specular.gb, 1, 1));
}

#ifdef NRC_SCENE_UNPACK
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
#endif

#endif
