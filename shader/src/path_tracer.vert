#version 450

layout(push_constant) uniform uuPushConstant {
	float _1, _2, _3, uLookX, uLookY, uLookZ, uSideX, uSideY, uSideZ, uUpX, uUpY, uUpZ;
};

layout(location = 0) out vec3 vDir;

void main() {
	gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0, 0.0, 1.0);
	vDir = vec3(uLookX, uLookY, uLookZ) + gl_Position.x * vec3(uSideX, uSideY, uSideZ) +
	       gl_Position.y * vec3(uUpX, uUpY, uUpZ);
}
