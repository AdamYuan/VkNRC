#ifndef SAMPLE_GLSL
#define SAMPLE_GLSL

#include "Constant.glsl"
#include "RNG.glsl"

vec3 AlignDir(in const vec3 n, in const vec3 d) {
	vec3 u = normalize(cross(abs(n.x) > .01 ? vec3(0, 1, 0) : vec3(1, 0, 0), n));
	vec3 v = cross(n, u);
	return d.x * u + d.y * v + d.z * n;
}

vec4 SampleCosineWeighted(in const vec3 normal) {
	// cosine hemisphere sampling
	vec2 sample2 = vec2(RNGNext(), RNGNext());
	float r = sqrt(sample2.x), phi = 2 * M_PI * sample2.y;
	vec3 d = vec3(r * cos(phi), r * sin(phi), sqrt(1.0 - sample2.x));
	// calculate pdf (dot(n, d) / M_PI)
	float pdf = d.z / M_PI;
	return vec4(AlignDir(normal, d), pdf);
}
float PDFCosineWeighted(in const vec3 normal, in const vec3 dir) { return dot(normal, dir) / M_PI; }

#endif
