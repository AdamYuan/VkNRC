#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

#include "CookTorranceBRDF.glsl"
#include "NRCRecord.glsl"

#define SCENE_TLAS_BINDING 0
#define SCENE_TEXTURE_BINDING 8
#define SCENE_TEXTURE_NUM_CONST_ID 0
#define SCENE_BUFFERS_FIRST_BINDING 1
#include "Scene.glsl"

layout(push_constant) uniform uuPushConstant {
	vec3 uOrigin, uLook, uSide, uUp;
	uint uSampleCount;
};

layout(location = 0) in vec3 vDir;
layout(location = 0) out vec4 oColor;

// V-Buffer
layout(input_attachment_index = 0, binding = 9) uniform usubpassInput uPrimitiveID_InstanceID;
// NRC
layout(binding = 10, rgba32f) uniform image2D uResult;

struct Ray {
	vec3 o, d;
};

struct Hit {
	vec3 position, normal;
	vec2 texcoord;
	Material material;
};

Hit GetVBufferHit(in const uint primitive_id, in const uint instance_id, in const Ray ray) {
	vec2 texcoord_0 = GetSceneTexcoord(primitive_id, 0);
	vec2 texcoord_1 = GetSceneTexcoord(primitive_id, 1);
	vec2 texcoord_2 = GetSceneTexcoord(primitive_id, 2);
	vec3 vertex_0 = GetSceneVertex(instance_id, primitive_id, 0);
	vec3 vertex_1 = GetSceneVertex(instance_id, primitive_id, 1);
	vec3 vertex_2 = GetSceneVertex(instance_id, primitive_id, 2);
	vec3 v01 = vertex_1 - vertex_0, v02 = vertex_2 - vertex_0, v0o = ray.o - vertex_0;
	vec3 n = cross(v01, v02);
	vec3 q = cross(v0o, ray.d);
	float d = 1.0 / dot(ray.d, n);
	float u = d * dot(-q, v02);
	float v = d * dot(q, v01);
	vec3 barycentric = vec3(1.0 - u - v, u, v);

	n = normalize(n);

	Hit hit;
	hit.normal = dot(n, ray.d) < 0 ? n : -n;
	hit.position = mat3(vertex_0, vertex_1, vertex_2) * barycentric;
	hit.texcoord = mat3x2(texcoord_0, texcoord_1, texcoord_2) * barycentric;
	hit.material = GetSceneMaterial(primitive_id);
	return hit;
}

Hit GetRayQueryHit(in const rayQueryEXT ray_query, in const Ray ray) {
	// Base PrimitiveID + Instance PrimitiveID
	uint primitive_id = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true) +
	                    rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
	uint instance_id = rayQueryGetIntersectionInstanceIdEXT(ray_query, true);

	vec2 texcoord_0 = GetSceneTexcoord(primitive_id, 0);
	vec2 texcoord_1 = GetSceneTexcoord(primitive_id, 1);
	vec2 texcoord_2 = GetSceneTexcoord(primitive_id, 2);
	vec3 vertex_0 = GetSceneVertex(instance_id, primitive_id, 0);
	vec3 vertex_1 = GetSceneVertex(instance_id, primitive_id, 1);
	vec3 vertex_2 = GetSceneVertex(instance_id, primitive_id, 2);
	vec3 normal = normalize(cross(vertex_1 - vertex_0, vertex_2 - vertex_0));

	vec3 barycentric = vec3(0, rayQueryGetIntersectionBarycentricsEXT(ray_query, true));
	barycentric.x = 1.0 - barycentric.y - barycentric.z;

	Hit hit;
	hit.normal = dot(normal, ray.d) < 0 ? normal : -normal;
	hit.position = mat3(vertex_0, vertex_1, vertex_2) * barycentric;
	hit.texcoord = mat3x2(texcoord_0, texcoord_1, texcoord_2) * barycentric;
	hit.material = GetSceneMaterial(primitive_id);
	return hit;
}

vec3 GetHitDiffuse(in const Hit hit) { return GetSceneDiffuse(hit.material, hit.texcoord); }
vec3 GetHitSpecular(in const Hit hit) { return GetSceneSpecular(hit.material, hit.texcoord); }

CookTorranceBRDFArgs GetHitBRDFArgs(in const Hit hit) {
	CookTorranceBRDFArgs args;
	args.diffuse = GetHitDiffuse(hit);
	args.specular = GetHitSpecular(hit);
	args.roughness = max(hit.material.roughness, 0.001);
	args.ior = max(hit.material.ior, 1.5);
	return args;
}

const vec3 kConstLight = vec3(10, 10, 10);

bool IsValidRGB(in const vec3 rgb) { return !any(isnan(rgb)) && !any(isinf(rgb)) && !any(lessThan(rgb, vec3(0))); }

bool PathTraceBRDFStep(in const Hit hit, inout Ray io_ray, inout vec3 io_accumulate) {
	CookTorranceBRDFArgs brdf_args = GetHitBRDFArgs(hit);
	vec4 sample_dir_pdf = CookTorranceSample(brdf_args, -io_ray.d, hit.normal);
	if (sample_dir_pdf.w == 0)
		return false;
	vec3 brdf = CookTorranceBRDF(brdf_args, sample_dir_pdf.xyz, -io_ray.d, hit.normal); // sample_dir is incidence dir
	io_accumulate *= brdf * abs(dot(hit.normal, sample_dir_pdf.xyz)) / sample_dir_pdf.w;
	if (!IsValidRGB(io_accumulate))
		return false;
	io_ray = Ray(hit.position, sample_dir_pdf.xyz);
	return true;
}

#define T_MIN 1e-6
#define T_MAX 4.0
vec3 PathTrace(Hit hit, Ray ray) {
	vec3 accumulate = vec3(1), irradiance = vec3(0);

	if (!PathTraceBRDFStep(hit, ray, accumulate))
		return irradiance;

	for (uint bounce = 0; bounce < 4; ++bounce) {
		rayQueryEXT ray_query;
		rayQueryInitializeEXT(ray_query, uTLAS, gl_RayFlagsOpaqueEXT, 0xFF, ray.o, T_MIN, ray.d, T_MAX);
		while (rayQueryProceedEXT(ray_query))
			;

		if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
			hit = GetRayQueryHit(ray_query, ray);
			if (!PathTraceBRDFStep(hit, ray, accumulate))
				break;
		} else {
			irradiance += accumulate * kConstLight;
			break;
		}
	}
	return irradiance;
}

vec3 ToneMapFilmic_Hejl2015(in const vec3 hdr, in const float white_pt) {
	vec4 vh = vec4(hdr, white_pt);
	vec4 va = (1.425 * vh) + 0.05;
	vec4 vf = (vh * va + 0.004) / ((vh * (va + 0.55) + 0.0491)) - 0.0821;
	return vf.rgb / vf.w;
}

void main() {
	uvec2 primitive_id_instance_id = subpassLoad(uPrimitiveID_InstanceID).rg;
	uint primitive_id = primitive_id_instance_id.x, instance_id = primitive_id_instance_id.y;

	ivec2 coord = ivec2(gl_FragCoord.xy);

	vec3 color;
	if (primitive_id == -1u)
		color = kConstLight;
	else {
		Ray ray = Ray(uOrigin, normalize(vDir));
		Hit hit = GetVBufferHit(primitive_id, instance_id, ray);
		// Seed on 256 x 256 Tile
		uint seed = (((coord.y & 0xFF) << 8) | (coord.x & 0xFF)) + uSampleCount * ((1 << 16u) + 1);
		RNGSetState(seed);

		color = PathTrace(hit, ray);
	}

	color = IsValidRGB(color) ? color : vec3(0);
	if (uSampleCount != 0) {
		color += imageLoad(uResult, coord).rgb * float(uSampleCount);
		color /= float(uSampleCount + 1);
	}
	imageStore(uResult, coord, vec4(color, 0));

	color = pow(ToneMapFilmic_Hejl2015(color, 3.2), vec3(1 / 2.2));
	oColor = vec4(color, 1.0);
}
