#ifndef SAMPLE_GLSL
#define SAMPLE_GLSL

#include "Constant.glsl"
#include "RNG.glsl"

vec3 align_to_normal(in const vec3 normal, in const vec3 dir) {
	vec3 u = normalize(cross(abs(normal.x) > .01 ? vec3(0, 1, 0) : vec3(1, 0, 0), normal));
	vec3 v = cross(normal, u);
	return dir.x * u + dir.y * v + dir.z * normal;
}

vec3 SampleCosineWeighted(in const vec3 normal, out float o_pdf) {
	// cosine hemisphere sampling
	vec2 sample2 = vec2(RNGNext(), RNGNext());
	float r = sqrt(sample2.x), phi = 2 * M_PI * sample2.y;
	vec3 d = vec3(r * cos(phi), r * sin(phi), sqrt(1.0 - sample2.x));
	// calculate pdf (dot(n, d) / M_PI)
	o_pdf = d.z / M_PI;
	return align_to_normal(normal, d);
}
float PDFCosineWeighted(in const vec3 normal, in const vec3 dir) { return dot(normal, dir) / M_PI; }

#endif
