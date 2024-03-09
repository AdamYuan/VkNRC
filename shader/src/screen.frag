#version 450

layout(input_attachment_index = 0, binding = 0) uniform subpassInput uColor;
layout(binding = 1, rgba32f) uniform image2D uAccumulate;

layout(location = 0) out vec4 oScreen;

layout(push_constant) uniform uuPushConstant { uint uIsAccumulate, uAccumulateCount; };

vec3 ToneMapFilmic_Hejl2015(in const vec3 hdr, in const float white_pt) {
	vec4 vh = vec4(hdr, white_pt);
	vec4 va = (1.425 * vh) + 0.05;
	vec4 vf = (vh * va + 0.004) / ((vh * (va + 0.55) + 0.0491)) - 0.0821;
	return vf.rgb / vf.w;
}

void main() {
	vec3 color = subpassLoad(uColor).rgb;

	ivec2 coord = ivec2(gl_FragCoord.xy);
	if (uIsAccumulate != 0) {
		if (uAccumulateCount != 0) {
			color += imageLoad(uAccumulate, coord).rgb * float(uAccumulateCount);
			color /= float(uAccumulateCount + 1);
		}
		imageStore(uAccumulate, coord, vec4(color, 0));
	}

	vec3 screen = pow(ToneMapFilmic_Hejl2015(color, 3.2), vec3(1 / 2.2));
	oScreen = vec4(screen, 1.0);
}
