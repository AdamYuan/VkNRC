#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

#include "CookTorranceBRDF.glsl"

layout(constant_id = 0) const uint kTextureNum = 1024;

layout(push_constant) uniform uuPushConstant {
	vec3 uOrigin, uLook, uSide, uUp;
	uint uSampleCount;
};

struct Vertex {
	float x, y, z;
};
struct Material {
	vec3 diffuse;
	uint diffuse_texture_id;
	vec3 specular;
	uint specular_texture_id;
	float metallic, roughness, ior;
};
struct Ray {
	vec3 o, d;
};
struct Hit {
	vec3 position, normal;
	vec2 texcoord;
	Material material;
};

layout(location = 0) in vec3 vDir;
layout(location = 0) out vec4 oColor;

// Scene
layout(binding = 0) uniform accelerationStructureEXT uTLAS;
layout(std430, binding = 1) readonly buffer uuVertices { Vertex uVertices[]; };
layout(std430, binding = 2) readonly buffer uuVertexIndices { uint uVertexIndices[]; };
layout(std430, binding = 3) readonly buffer uuTexcoords { vec2 uTexcoords[]; };
layout(std430, binding = 4) readonly buffer uuTexcoordIndices { uint uTexcoordIndices[]; };
layout(std430, binding = 5) readonly buffer uuMaterials { Material uMaterials[]; };
layout(std430, binding = 6) readonly buffer uuMaterialIDs { uint uMaterialIDs[]; }; // Per-Primitive
layout(std430, binding = 7) readonly buffer uuTransforms { mat3x4 uTransforms[]; }; // Per-Instance
layout(binding = 8) uniform sampler2D uTextures[kTextureNum];
// V-Buffer
layout(input_attachment_index = 0, binding = 9) uniform usubpassInput uPrimitiveID_InstanceID;
// NRC
layout(binding = 10, rgba32f) uniform image2D uResult;

mat3x4 GetTransform(in const uint instance_id) { return uTransforms[instance_id]; }
vec3 GetVertex(in const uint primitive_id, in const uint vert_id) {
	Vertex v = uVertices[uVertexIndices[primitive_id * 3 + vert_id]];
	return vec3(v.x, v.y, v.z);
}
vec3 GetTransformedVertex(in const uint instance_id, in const uint primitive_id, in const uint vert_id) {
	return vec4(GetVertex(primitive_id, vert_id), 1.0) * GetTransform(instance_id);
}
vec2 GetTexcoord(in const uint primitive_id, in const uint vert_id) {
	return uTexcoords[uTexcoordIndices[primitive_id * 3 + vert_id]];
}
Material GetMaterial(in const uint primitive_id) { return uMaterials[uMaterialIDs[primitive_id]]; }

Hit GetVBufferHit(in const uint primitive_id, in const uint instance_id, in const Ray ray) {
	vec2 texcoord_0 = GetTexcoord(primitive_id, 0);
	vec2 texcoord_1 = GetTexcoord(primitive_id, 1);
	vec2 texcoord_2 = GetTexcoord(primitive_id, 2);
	vec3 vertex_0 = GetTransformedVertex(instance_id, primitive_id, 0);
	vec3 vertex_1 = GetTransformedVertex(instance_id, primitive_id, 1);
	vec3 vertex_2 = GetTransformedVertex(instance_id, primitive_id, 2);
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
	hit.material = GetMaterial(primitive_id);
	return hit;
}

Hit GetRayQueryHit(in const rayQueryEXT ray_query, in const Ray ray) {
	// Base PrimitiveID + Instance PrimitiveID
	uint primitive_id = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true) +
	                    rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
	uint instance_id = rayQueryGetIntersectionInstanceIdEXT(ray_query, true);

	vec2 texcoord_0 = GetTexcoord(primitive_id, 0);
	vec2 texcoord_1 = GetTexcoord(primitive_id, 1);
	vec2 texcoord_2 = GetTexcoord(primitive_id, 2);
	vec3 vertex_0 = GetTransformedVertex(instance_id, primitive_id, 0);
	vec3 vertex_1 = GetTransformedVertex(instance_id, primitive_id, 1);
	vec3 vertex_2 = GetTransformedVertex(instance_id, primitive_id, 2);
	vec3 normal = normalize(cross(vertex_1 - vertex_0, vertex_2 - vertex_0));

	vec3 barycentric = vec3(0, rayQueryGetIntersectionBarycentricsEXT(ray_query, true));
	barycentric.x = 1.0 - barycentric.y - barycentric.z;

	Hit hit;
	hit.normal = dot(normal, ray.d) < 0 ? normal : -normal;
	hit.position = mat3(vertex_0, vertex_1, vertex_2) * barycentric;
	hit.texcoord = mat3x2(texcoord_0, texcoord_1, texcoord_2) * barycentric;
	hit.material = GetMaterial(primitive_id);
	return hit;
}

vec3 GetHitDiffuse(in const Hit hit) {
	return hit.material.diffuse_texture_id == -1
	           ? hit.material.diffuse
	           : texture(uTextures[hit.material.diffuse_texture_id], hit.texcoord).rgb;
}
vec3 GetHitSpecular(in const Hit hit) {
	return hit.material.specular_texture_id == -1
	           ? hit.material.specular
	           : texture(uTextures[hit.material.specular_texture_id], hit.texcoord).rgb;
}

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
