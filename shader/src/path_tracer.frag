#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#define M_PI 3.1415926535897932384626433832795

layout(constant_id = 0) const uint kTextureNum = 1024;

layout(push_constant) uniform uuPushConstant {
	vec3 uOrigin, uLook, uSide, uUp;
	uint uSampleCount;
};

struct Vertex {
	float x, y, z;
};
struct Material {
	vec3 albedo;
	uint albedo_texture_id;
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

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float RNGNext(inout uint rng_state) {
	// Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
	rng_state = rng_state * 747796405 + 1;
	uint word = ((rng_state >> ((rng_state >> 28) + 4)) ^ rng_state) * 277803737;
	word = (word >> 22) ^ word;
	return float(word) / 4294967295.0f;
}

Hit GetVBufferHit(in const uint primitive_id, in const uint instance_id, in const vec3 ray_o, in const vec3 ray_d) {
	vec2 texcoord_0 = GetTexcoord(primitive_id, 0);
	vec2 texcoord_1 = GetTexcoord(primitive_id, 1);
	vec2 texcoord_2 = GetTexcoord(primitive_id, 2);
	vec3 vertex_0 = GetTransformedVertex(instance_id, primitive_id, 0);
	vec3 vertex_1 = GetTransformedVertex(instance_id, primitive_id, 1);
	vec3 vertex_2 = GetTransformedVertex(instance_id, primitive_id, 2);
	vec3 v01 = vertex_1 - vertex_0, v02 = vertex_2 - vertex_0, v0o = ray_o - vertex_0;
	vec3 n = cross(v01, v02);
	vec3 q = cross(v0o, ray_d);
	float d = 1.0 / dot(ray_d, n);
	float u = d * dot(-q, v02);
	float v = d * dot(q, v01);
	vec3 barycentric = vec3(1.0 - u - v, u, v);

	n = normalize(n);

	Hit hit;
	hit.normal = dot(n, ray_d) < 0 ? n : -n;
	hit.position = mat3(vertex_0, vertex_1, vertex_2) * barycentric;
	hit.texcoord = mat3x2(texcoord_0, texcoord_1, texcoord_2) * barycentric;
	hit.material = GetMaterial(primitive_id);
	return hit;
}

Hit GetRayQueryHit(in const rayQueryEXT ray_query, in const vec3 ray_o, in const vec3 ray_d) {
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
	hit.normal = dot(normal, ray_d) < 0 ? normal : -normal;
	hit.position = mat3(vertex_0, vertex_1, vertex_2) * barycentric;
	hit.texcoord = mat3x2(texcoord_0, texcoord_1, texcoord_2) * barycentric;
	hit.material = GetMaterial(primitive_id);
	return hit;
}

vec3 GetMaterialAlbedo(in const Material mat, in const vec2 texcoord) {
	return mat.albedo_texture_id == -1 ? mat.albedo : texture(uTextures[mat.albedo_texture_id], texcoord).rgb;
}

vec3 AlignDirection(in const vec3 dir, in const vec3 target) {
	vec3 u = normalize(cross(abs(target.x) > .01 ? vec3(0, 1, 0) : vec3(1, 0, 0), target));
	vec3 v = cross(target, u);
	return dir.x * u + dir.y * v + dir.z * target;
}

#define DIFFUSE_BRDF (1.0 / M_PI)
vec3 SampleDiffuse(in const vec3 normal, in const vec2 samp, out float pdf) {
	// cosine hemisphere sampling
	float r = sqrt(samp.x), phi = 2 * M_PI * samp.y;
	vec3 d = vec3(r * cos(phi), r * sin(phi), sqrt(1.0 - samp.x));
	// calculate pdf (dot(n, d) / M_PI)
	pdf = d.z / M_PI;
	return AlignDirection(d, normal);
}
float GetDiffusePDF(in const float norm_dot_dir) { return norm_dot_dir * DIFFUSE_BRDF; }

const vec3 kConstLight = vec3(10, 8, 8);

#define T_MIN 1e-6
#define T_MAX 4.0
vec3 PathTrace(in const Hit start_hit, uint rng_state) {
	vec3 acc_color = vec3(1), irradiance = vec3(0);
	vec3 origin = start_hit.position, normal = start_hit.normal;

	acc_color *= GetMaterialAlbedo(start_hit.material, start_hit.texcoord);

	float bsdf_pdf;
	for (uint bounce = 0; bounce < 4; ++bounce) {
		vec2 sample2 = vec2(RNGNext(rng_state), RNGNext(rng_state));

		float pdf;
		vec3 dir = SampleDiffuse(normal, sample2, pdf);

		rayQueryEXT ray_query;
		rayQueryInitializeEXT(ray_query, uTLAS, gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, T_MIN, dir, T_MAX);
		while (rayQueryProceedEXT(ray_query))
			;

		if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
			Hit hit = GetRayQueryHit(ray_query, origin, dir);
			vec3 albedo = GetMaterialAlbedo(hit.material, hit.texcoord);
			acc_color *= albedo; // * ndd * DIFFUSE_BRDF / pdf;
			origin = hit.position;
			normal = hit.normal;
		} else {
			irradiance = acc_color * kConstLight; // * ndd * DIFFUSE_BRDF / pdf;
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
		Hit hit = GetVBufferHit(primitive_id, instance_id, uOrigin, normalize(vDir));
		// Seed on 256 x 256 Tile
		uint seed = (((coord.y & 0xFF) << 8) | (coord.x & 0xFF)) + uSampleCount * ((1 << 16u) + 1);
		color = PathTrace(hit, seed);
	}

	color += imageLoad(uResult, coord).rgb * float(uSampleCount);
	color /= float(uSampleCount + 1);
	imageStore(uResult, coord, vec4(color, 0));

	color = pow(ToneMapFilmic_Hejl2015(color, 3.2), vec3(1 / 2.2));
	oColor = vec4(color, 1.0);
}
