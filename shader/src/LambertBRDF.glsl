#ifndef LAMBERT_BRDF
#define LAMBERT_BRDF

#include "Constant.glsl"
#include "Sample.glsl"

struct LambertBRDFArgs {
	vec3 diffuse;
};

// all dirs' origins are their hit points
// L: Light (Incident), V: View (Out)
vec3 LambertBRDF(in const LambertBRDFArgs args, in const vec3 l, in const vec3 v, in const vec3 n) {
	return args.diffuse / M_PI;
}
vec4 LambertSample(in const LambertBRDFArgs args, in const vec3 v, in const vec3 n) { return SampleCosineWeighted(n); }
float LambertPDF(in const LambertBRDFArgs args, in const vec3 l, in const vec3 v, in const vec3 n) {
	return PDFCosineWeighted(n, l);
}

#endif
