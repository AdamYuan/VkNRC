#version 450
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable

layout(constant_id = 0) const uint kTextureNum = 1024;

layout(push_constant) uniform uuPushConstant { vec3 uOrigin; };

struct Vertex {
	float x, y, z;
};
struct Material {
	vec3 albedo;
	uint albedo_texture_id;
};

layout(location = 0) in vec3 vDir;
layout(location = 0) out vec4 oColor;

layout(binding = 0) uniform accelerationStructureEXT uTLAS;
layout(std430, binding = 1) readonly buffer uuVertices { Vertex uVertices[]; };
layout(std430, binding = 2) readonly buffer uuVertexIndices { uint uVertexIndices[]; };
layout(std430, binding = 3) readonly buffer uuTexcoords { vec2 uTexcoords[]; };
layout(std430, binding = 4) readonly buffer uuTexcoordIndices { uint uTexcoordIndices[]; };
layout(std430, binding = 5) readonly buffer uuMaterials { Material uMaterials[]; };
layout(std430, binding = 6) readonly buffer uuMaterialIDs { uint uMaterialIDs[]; }; // Per-Primitive
layout(std430, binding = 7) readonly buffer uuTransforms { mat3x4 uTransforms[]; }; // Per-Instance
layout(binding = 8) uniform sampler2D uTextures[kTextureNum];
layout(input_attachment_index = 0, binding = 9) uniform usubpassInput uPrimitiveID_InstanceID; // Visibility Buffer

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

void ResolvePrimaryRay(in const uint primitive_id,
                       in const uint instance_id,
                       in const vec3 ray_o,
                       in const vec3 ray_d,
                       out vec3 o_position,
                       out vec3 o_normal,
                       out vec2 o_texcoord) {
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
	o_normal = normalize(n);
	o_position = mat3(vertex_0, vertex_1, vertex_2) * barycentric;
	o_texcoord = mat3x2(texcoord_0, texcoord_1, texcoord_2) * barycentric;
}

vec3 GetMaterialAlbedo(in const Material mat, in const vec2 texcoord) {
	return mat.albedo_texture_id == -1 ? mat.albedo : texture(uTextures[mat.albedo_texture_id], texcoord).rgb;
}

void main() {
	uvec2 primitive_id_instance_id = subpassLoad(uPrimitiveID_InstanceID).rg;
	uint primitive_id = primitive_id_instance_id.x, instance_id = primitive_id_instance_id.y;
	if (primitive_id == -1u) {
		oColor = vec4(0, 0, 0, 1);
		return;
	}

	vec3 position, normal;
	vec2 texcoord;
	ResolvePrimaryRay(primitive_id, instance_id, uOrigin, normalize(vDir), position, normal, texcoord);
	Material mat = GetMaterial(primitive_id);

	oColor = vec4(GetMaterialAlbedo(mat, texcoord), 1.0);
}
