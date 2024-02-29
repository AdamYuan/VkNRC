#ifndef NRC_RECORD_GLSL
#define NRC_RECORD_GLSL

#include "Constant.glsl"

struct PackedNRCInput {
	uint primitive_id, flip_bit_instance_id;
	uint barycentric_2x16U;
	uint scattered_dir_2x16U;
};

struct NRCEvalRecord {
	uint pixel_x_y;
	PackedNRCInput packed_input;
};

struct NRCTrainRecord {
	uint radiance_RG, radiance_B;
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
// x in [0, 1]
vec4 NRCOneBlob4Encode(in const float mu) {
	// (S)igma = 1 / 4
#define INV_S 4
#define INV_S2 16
#define INV_SQRT_2_PI inversesqrt(2.0 * M_PI)
	vec4 x_minus_mu = vec4(0.125, 0.375, 0.625, 0.875) - mu;
	return INV_S * INV_SQRT_2_PI * exp(-0.5 * INV_S2 * x_minus_mu * x_minus_mu);
#undef INV_S
#undef INV_S2
#undef INV_SQRT_2_PI
}

mat3x4 NRCFrequencyEncode(in const float p) {
	mat3x4 f = mat3x4(vec4(1, 2, 4, 8), vec4(16, 32, 64, 128), vec4(256, 512, 1024, 2048)) * (M_PI * p);
	return mat3x4(sin(f[0]), sin(f[1]), sin(f[2]));
}

uvec4 NRCPackHalf8x16(in const vec4 a, in const vec4 b) {
	return uvec4(packHalf2x16(a.xy), packHalf2x16(a.zw), packHalf2x16(b.xy), packHalf2x16(b.zw));
}

void NRCInputEncode(in const UnpackedNRCInput unpacked_input,
                    out uvec4 o_0,
                    out uvec4 o_8,
                    out uvec4 o_16,
                    out uvec4 o_24,
                    out uvec4 o_32,
                    out uvec4 o_40,
                    out uvec4 o_48,
                    out uvec4 o_56) {
	mat3x4 pos_freq_0 = NRCFrequencyEncode(unpacked_input.position.x);
	mat3x4 pos_freq_1 = NRCFrequencyEncode(unpacked_input.position.y);
	mat3x4 pos_freq_2 = NRCFrequencyEncode(unpacked_input.position.z);
	vec4 scat_ob_0 = NRCOneBlob4Encode(unpacked_input.scattered_dir.x),
	     scat_ob_1 = NRCOneBlob4Encode(unpacked_input.scattered_dir.y);
	vec4 norm_ob_0 = NRCOneBlob4Encode(unpacked_input.normal.x), norm_ob_1 = NRCOneBlob4Encode(unpacked_input.normal.y);
	vec4 r_ob = NRCOneBlob4Encode(1 - exp(-unpacked_input.roughness));
	o_0 = NRCPackHalf8x16(pos_freq_0[0], pos_freq_0[1]);
	o_8 = NRCPackHalf8x16(pos_freq_0[2], pos_freq_1[0]);
	o_16 = NRCPackHalf8x16(pos_freq_1[1], pos_freq_1[2]);
	o_24 = NRCPackHalf8x16(pos_freq_2[0], pos_freq_2[1]);
	o_32 = NRCPackHalf8x16(pos_freq_2[2], scat_ob_0);
	o_40 = NRCPackHalf8x16(scat_ob_1, norm_ob_0);
	o_48 = NRCPackHalf8x16(norm_ob_1, r_ob);
	o_56 = NRCPackHalf8x16(vec4(unpacked_input.diffuse, unpacked_input.specular.r),
	                       vec4(unpacked_input.specular.gb, 0, 0));
}

#endif
