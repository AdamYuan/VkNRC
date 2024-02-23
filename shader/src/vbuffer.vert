#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in mat3x4 aModelT;

layout(location = 0) out flat uint vInstanceID;

layout(push_constant) uniform uuPushConstant {
	mat4 uViewProj;
	uint uPrimitiveBase;
};

void main() {
	gl_Position = uViewProj * vec4(vec4(aPosition, 1.0) * aModelT, 1.0);
	vInstanceID = gl_InstanceIndex;
}
