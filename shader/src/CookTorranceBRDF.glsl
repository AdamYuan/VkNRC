#ifndef COOK_TORRANCE_BRDF
#define COOK_TORRANCE_BRDF

#include "Constant.glsl"
#include "RNG.glsl"
#include "Sample.glsl"

struct CookTorranceBRDFArgs {
	vec3 diffuse, specular;
	float roughness, ior;
};

float CT_Luminance(in const vec3 c) { return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z; }
float CT_Specular_Prob(in const CookTorranceBRDFArgs args) {
	float diffuse_lumi = CT_Luminance(args.diffuse);
	float specular_lumi = CT_Luminance(args.specular);
	return specular_lumi / (diffuse_lumi + specular_lumi);
}
float Walter_G1(in const float v_dot_h, in const float v_dot_n, in const float a_b_2) {
	float v_dot_n_2 = v_dot_n * v_dot_n;
	float a2 = 1.0 / (a_b_2 * (1 - v_dot_n_2) / (v_dot_n_2)), a = sqrt(a2);
	return max(v_dot_h / v_dot_n, 0) * (a >= 1.6 ? 1.0 : (3.535 * a + 2.181 * a2) / (1.0 + 2.276 * a + 2.577 * a2));
}
float Smith_G(in const float G1_l, in const float G1_v) { return G1_l * G1_v; }

float Beckmann_D(in const float n_dot_h, in const float a2) {
	float n_dot_h_2 = n_dot_h * n_dot_h, n_dot_h_4 = n_dot_h_2 * n_dot_h_2;
	return exp((n_dot_h_2 - 1) / (a2 * n_dot_h_2)) / (M_PI * a2 * n_dot_h_4);
}
float Beckmann_D_Sample_N_DOT_H(in const float a2) {
	float tan_2_nh = -a2 * log(1.0 - RNGNext());
	if (isnan(tan_2_nh))
		tan_2_nh = 0.0;
	float n_dot_h_2 = 1.0 / (1.0 + tan_2_nh);
	return sqrt(n_dot_h_2);
}
float Beckmann_D_PDF(in const float n_dot_h, in const float a2) { return Beckmann_D(n_dot_h, a2) * n_dot_h; }

float Schlick_Fresnel(in const float v_dot_h, in const float ior) {
	float f0 = (ior - 1.0) / (ior + 1.0);
	f0 *= f0;
	return f0 + (1 - f0) * pow(1.0 - v_dot_h, 5.0);
}

// all dirs' origins are their hit points
// L: Light (Incident), V: View (Out)
vec3 CookTorranceBRDF(in const CookTorranceBRDFArgs args, in const vec3 l, in const vec3 v, in const vec3 n) {
	vec3 h = normalize(l + v);

	float n_dot_h = max(dot(n, h), 1e-6);
	float v_dot_h = max(dot(v, h), 1e-6), l_dot_h = v_dot_h;
	float n_dot_l = clamp(dot(n, l), 1e-6, 1.0 - 1e-6);
	float n_dot_v = clamp(dot(n, v), 1e-6, 1.0 - 1e-6);

	float a = args.roughness, a2 = a * a;

	// Geometric
	float G = Smith_G(Walter_G1(v_dot_h, n_dot_v, a2), Walter_G1(l_dot_h, n_dot_l, a2));
	// Normal
	float D = Beckmann_D(n_dot_h, a2);
	// Fresnel
	float F = Schlick_Fresnel(v_dot_h, args.ior);

	vec3 specular = args.specular * D * G * F / (4.0 * n_dot_l * n_dot_v);
	vec3 diffuse = args.diffuse / M_PI;

	return diffuse + specular;
}

// https://agraphicsguynotes.com/posts/sample_microfacet_brdf/
float CookTorrancePDF(in const CookTorranceBRDFArgs args, in const vec3 l, in const vec3 v, in const vec3 n) {
	float n_dot_l = dot(l, n);
	if (n_dot_l < 0)
		return 0;
	vec3 h = normalize(l + v);
	float a = args.roughness;
	float p_h = Beckmann_D_PDF(dot(n, h), a * a);
	float p_cook_torrance = p_h / (4 * dot(v, h));
	float p_lambert = n_dot_l / M_PI;

	float p = mix(p_lambert, p_cook_torrance, CT_Specular_Prob(args));
	return p;
}
vec4 CookTorranceSample(in const CookTorranceBRDFArgs args, in const vec3 v, in const vec3 n) {
	float a = args.roughness;
	float n_dot_h = Beckmann_D_Sample_N_DOT_H(a * a);
	float phi = 2 * M_PI * RNGNext();
	float r = sqrt(1.0 - n_dot_h * n_dot_h);
	vec3 h = AlignDir(n, vec3(r * cos(phi), r * sin(phi), n_dot_h));
	if (dot(v, h) < 0)
		h = -h;
	vec3 l = reflect(-v, h);
	if (RNGNext() > CT_Specular_Prob(args))
		l = SampleCosineWeighted(n).xyz;
	return vec4(l, CookTorrancePDF(args, l, v, n));
}

#endif
