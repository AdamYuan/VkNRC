#ifndef LAMBERT_BRDF
#define LAMBERT_BRDF

#include "Constant.glsl"
#include "Sample.glsl"

struct LambertBRDFArgs {
	vec3 albedo;
};

// all dirs' origins are their hit points
vec3 LambertBRDF(in const LambertBRDFArgs args, in const vec3 in_dir, in const vec3 out_dir, in const vec3 normal) {
	return args.albedo / M_PI;
}
vec3 LambertSample(in const LambertBRDFArgs args, in const vec3 dir, in const vec3 normal, out float o_pdf) {
	return SampleCosineWeighted(normal, o_pdf);
}
float LambertPDF(in const LambertBRDFArgs args, in const vec3 dir, in const vec3 normal, in const vec3 sample_dir) {
	return PDFCosineWeighted(normal, sample_dir);
}

#endif
